"""Módulo de Almacenamiento (sección 3.2 del documento de diseño).

Gestiona el índice semántico: ID de documento, ID de fragmento, vector, texto
original y metadatos (página, etiquetas del documento).

Uso:
    from app.storage import get_vector_store

    store = get_vector_store()
    store.upsert(embedded_chunks, metadata={"categoria": "contratos_2026"})
    resultados = store.search(vector_consulta, top_k=5)

Ramas: feature/storage-vectordb-chroma, feature/storage-pgvector-backend (v1.0.0).
"""

from functools import lru_cache

from app.core.config import get_settings
from app.storage.base import (
    DocumentSummary,
    Filters,
    SearchResult,
    VectorStore,
)
from app.storage.chroma import ChromaVectorStore
from app.storage.errors import (
    DimensionMismatchError,
    VectorStoreConfigurationError,
    VectorStoreError,
)


def build_vector_store(name: str | None = None) -> VectorStore:
    """Crea el índice indicado (o el de la configuración)."""
    backend = name or get_settings().vector_store

    match backend:
        case "chroma":
            return ChromaVectorStore()
        case "pgvector":
            raise VectorStoreConfigurationError(
                "El backend pgvector todavía no está implementado "
                "(rama feature/storage-pgvector-backend). Usa VECTOR_STORE=chroma."
            )
        case _:
            raise VectorStoreConfigurationError(f"Backend vectorial desconocido: '{backend}'")


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Índice único de la aplicación (cacheado: abre el cliente una sola vez)."""
    return build_vector_store()


__all__ = [
    "ChromaVectorStore",
    "DimensionMismatchError",
    "DocumentSummary",
    "Filters",
    "SearchResult",
    "VectorStore",
    "VectorStoreConfigurationError",
    "VectorStoreError",
    "build_vector_store",
    "get_vector_store",
]
