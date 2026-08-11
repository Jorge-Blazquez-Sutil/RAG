"""Contrato de la base de datos vectorial (módulo 3.2).

Aísla el resto del sistema del motor concreto: la recuperación (módulo 3.3) habla
con esta interfaz, así que cambiar ChromaDB por pgvector o Pinecone —el escenario
distribuido del documento de diseño— no obliga a tocar el pipeline.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from pydantic import BaseModel, Field

from app.schemas.chunk import Chunk, EmbeddedChunk

#: Valores admitidos como metadato. Los motores vectoriales solo almacenan
#: escalares: nada de listas ni diccionarios anidados.
MetadataValue = str | int | float | bool
Filters = dict[str, MetadataValue]


class SearchResult(BaseModel):
    """Fragmento recuperado con su grado de parecido con la consulta."""

    chunk: Chunk
    score: float = Field(
        description="Similitud del coseno en [0, 1]; 1 es idéntico",
        ge=-1.0,
        le=1.0,
    )


class DocumentSummary(BaseModel):
    """Documento indexado, tal y como lo lista `GET /api/v1/documents`."""

    doc_id: str
    chunk_count: int
    page_count: int


class VectorStore(ABC):
    """Índice semántico: guarda fragmentos vectorizados y busca por similitud."""

    @abstractmethod
    def upsert(
        self,
        embedded: Sequence[EmbeddedChunk],
        *,
        metadata: Filters | None = None,
    ) -> int:
        """Inserta o actualiza fragmentos y devuelve cuántos se escribieron.

        Es idempotente: como el `chunk_id` es determinista, reingerir el mismo
        documento sobrescribe en lugar de duplicar.

        Args:
            embedded: fragmentos con su vector.
            metadata: etiquetas comunes del documento (departamento, categoría),
                las que acompañan al archivo en `POST /documents/ingest`.
        """

    @abstractmethod
    def search(
        self,
        embedding: Sequence[float],
        *,
        top_k: int,
        filters: Filters | None = None,
    ) -> list[SearchResult]:
        """Devuelve los `top_k` fragmentos más parecidos, de mayor a menor score."""

    @abstractmethod
    def delete_document(self, doc_id: str) -> int:
        """Borra todos los fragmentos de un documento; devuelve cuántos eliminó."""

    @abstractmethod
    def list_documents(self) -> list[DocumentSummary]:
        """Lista los documentos indexados."""

    @abstractmethod
    def count(self) -> int:
        """Número total de fragmentos almacenados."""

    @abstractmethod
    def reset(self) -> None:
        """Vacía la colección. Pensado para tests y para reindexar desde cero."""

    @staticmethod
    def build_chunk(chunk_id: str, text: str, metadata: dict) -> Chunk:
        """Reconstruye un `Chunk` a partir de lo almacenado en el índice.

        El `chunk_id` es la clave primaria del registro, no un metadato: llega
        aparte porque el motor lo devuelve en su propio campo.
        """
        return Chunk(
            chunk_id=chunk_id,
            index=int(metadata["chunk_index"]),
            text=text,
            page_number=int(metadata["page"]),
            token_count=int(metadata.get("token_count", 0)),
            filename=str(metadata["doc_id"]),
        )
