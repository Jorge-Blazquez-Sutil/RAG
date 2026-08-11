"""Implementación del índice semántico sobre ChromaDB (despliegue local).

Dos detalles de configuración que condicionan el resto del sistema:

* La colección se crea con `hnsw:space=cosine`. Chroma usa distancia euclídea por
  defecto, y el documento de diseño especifica similitud del coseno; con el
  espacio equivocado el orden de los resultados cambia sin dar ningún error.
* Nunca se delega la vectorización en Chroma: los vectores se pasan siempre
  explícitos. Su función de embedding por defecto descargaría un modelo ONNX de
  internet la primera vez, lo que rompería en un despliegue aislado y además
  usaría un modelo distinto al de la ingesta.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.chunk import EmbeddedChunk
from app.storage.base import (
    DocumentSummary,
    Filters,
    MetadataValue,
    SearchResult,
    VectorStore,
)
from app.storage.errors import DimensionMismatchError, VectorStoreError

logger = get_logger(__name__)

#: Metadatos reservados: los escribe el sistema y no se pueden sobrescribir con
#: las etiquetas del usuario.
RESERVED_KEYS = frozenset({"doc_id", "page", "chunk_index", "token_count", "model"})


class ChromaVectorStore(VectorStore):
    """Índice persistente en disco (o en memoria, para tests)."""

    def __init__(
        self,
        *,
        path: Path | None = None,
        collection_name: str | None = None,
        client: Any | None = None,
    ) -> None:
        settings = get_settings()
        self._collection_name = collection_name or settings.collection_name
        self._client = client if client is not None else self._build_client(path)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    @staticmethod
    def _build_client(path: Path | None) -> Any:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - dependencia declarada
            raise VectorStoreError(
                "El paquete 'chromadb' no está instalado: ejecuta `uv sync`."
            ) from exc

        target = path or get_settings().chroma_path
        target.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(target))

    # ---------- Escritura ----------

    def upsert(
        self,
        embedded: Sequence[EmbeddedChunk],
        *,
        metadata: Filters | None = None,
    ) -> int:
        if not embedded:
            return 0

        self._check_dimensions(embedded)
        tags = self._clean_tags(metadata)

        self._collection.upsert(
            ids=[item.chunk.chunk_id for item in embedded],
            embeddings=[list(item.embedding) for item in embedded],
            documents=[item.chunk.text for item in embedded],
            metadatas=[self._metadata_for(item, tags) for item in embedded],
        )
        return len(embedded)

    def _check_dimensions(self, embedded: Sequence[EmbeddedChunk]) -> None:
        """Comprueba que todos los vectores comparten dimensión con la colección."""
        sizes = {item.dimensions for item in embedded}
        if len(sizes) > 1:
            raise DimensionMismatchError(expected=min(sizes), received=max(sizes))

        stored = self._stored_dimensions()
        incoming = sizes.pop()
        if stored is not None and stored != incoming:
            raise DimensionMismatchError(expected=stored, received=incoming)

    def _stored_dimensions(self) -> int | None:
        """Dimensión de los vectores ya almacenados, o None si la colección está vacía."""
        if self._collection.count() == 0:
            return None
        sample = self._collection.get(limit=1, include=["embeddings"])
        embeddings = sample.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return None
        return len(embeddings[0])

    @staticmethod
    def _clean_tags(metadata: Filters | None) -> Filters:
        if not metadata:
            return {}
        collisions = RESERVED_KEYS & set(metadata)
        if collisions:
            raise VectorStoreError(
                f"Metadatos reservados por el sistema: {sorted(collisions)}. "
                "Usa otros nombres para tus etiquetas."
            )
        return dict(metadata)

    @staticmethod
    def _metadata_for(item: EmbeddedChunk, tags: Filters) -> dict[str, MetadataValue]:
        chunk = item.chunk
        return {
            "doc_id": chunk.filename,
            "page": chunk.page_number,
            "chunk_index": chunk.index,
            "token_count": chunk.token_count,
            "model": item.model,
            **tags,
        }

    # ---------- Lectura ----------

    def search(
        self,
        embedding: Sequence[float],
        *,
        top_k: int,
        filters: Filters | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise VectorStoreError(f"top_k debe ser positivo, recibido {top_k}")
        if self._collection.count() == 0:
            return []

        response = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=top_k,
            where=self._build_where(filters),
            include=["documents", "metadatas", "distances"],
        )

        ids = _first(response, "ids")
        documents = _first(response, "documents")
        metadatas = _first(response, "metadatas")
        distances = _first(response, "distances")

        return [
            SearchResult(
                chunk=self.build_chunk(str(chunk_id), text, dict(meta)),
                score=self.to_similarity(distance),
            )
            for chunk_id, text, meta, distance in zip(
                ids, documents, metadatas, distances, strict=True
            )
        ]

    @staticmethod
    def to_similarity(distance: float) -> float:
        """Convierte la distancia que devuelve Chroma en similitud del coseno.

        Con espacio coseno, similitud = 1 - distancia. Se acota a [-1, 1] porque
        el cálculo en coma flotante se sale del rango por unas millonésimas —un
        vector idéntico llega como distancia -1e-6— y eso bastaría para romper la
        validación del modelo de respuesta.
        """
        return round(min(1.0, max(-1.0, 1.0 - float(distance))), 6)

    @staticmethod
    def _build_where(filters: Filters | None) -> dict[str, Any] | None:
        """Traduce los filtros planos de la API al formato `where` de Chroma."""
        if not filters:
            return None
        if len(filters) == 1:
            key, value = next(iter(filters.items()))
            return {key: value}
        return {"$and": [{key: value} for key, value in filters.items()]}

    def list_documents(self) -> list[DocumentSummary]:
        if self._collection.count() == 0:
            return []

        stored = self._collection.get(include=["metadatas"])
        chunks_by_doc: dict[str, int] = {}
        pages_by_doc: dict[str, set[int]] = {}

        for meta in stored.get("metadatas") or []:
            doc_id = str(meta["doc_id"])
            chunks_by_doc[doc_id] = chunks_by_doc.get(doc_id, 0) + 1
            pages_by_doc.setdefault(doc_id, set()).add(int(meta["page"]))

        return [
            DocumentSummary(
                doc_id=doc_id,
                chunk_count=count,
                page_count=len(pages_by_doc[doc_id]),
            )
            for doc_id, count in sorted(chunks_by_doc.items())
        ]

    def count(self) -> int:
        return int(self._collection.count())

    # ---------- Borrado ----------

    def delete_document(self, doc_id: str) -> int:
        existing = self._collection.get(where={"doc_id": doc_id}, include=[])
        ids = existing.get("ids") or []
        if not ids:
            return 0
        self._collection.delete(ids=ids)
        logger.info("Eliminados %d fragmentos de '%s'", len(ids), doc_id)
        return len(ids)

    def reset(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )


def _first(response: dict[str, Any], key: str) -> list[Any]:
    """Chroma envuelve cada campo en una lista por consulta; solo mandamos una."""
    values = response.get(key)
    if not values:
        return []
    return list(values[0])
