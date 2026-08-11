"""Vectorización de fragmentos (paso 3 del módulo 3.1).

Uso:
    from app.ingestion.embeddings import get_embedding_provider

    provider = get_embedding_provider()
    embedded = provider.embed_chunks(chunks)

El proveedor se elige con `EMBEDDING_PROVIDER`. La misma instancia se usa para
vectorizar documentos (ingesta) y consultas (recuperación), que es lo que
garantiza que ambos vivan en el mismo espacio vectorial.
"""

from functools import lru_cache

from app.core.config import get_settings
from app.ingestion.embeddings.base import EmbeddingProvider
from app.ingestion.embeddings.deterministic import DeterministicEmbeddingProvider
from app.ingestion.embeddings.errors import (
    EmbeddingConfigurationError,
    EmbeddingError,
    EmbeddingProviderError,
)
from app.ingestion.embeddings.openai import OpenAIEmbeddingProvider


def build_embedding_provider(name: str | None = None) -> EmbeddingProvider:
    """Crea el proveedor indicado (o el de la configuración)."""
    settings = get_settings()
    provider = name or settings.embedding_provider

    match provider:
        case "openai":
            return OpenAIEmbeddingProvider()
        case "fake":
            return DeterministicEmbeddingProvider()
        case "local":
            raise EmbeddingConfigurationError(
                "El proveedor local todavía no está implementado "
                "(rama feature/ingestion-local-embeddings). "
                "Usa EMBEDDING_PROVIDER=openai o =fake."
            )
        case _:
            raise EmbeddingConfigurationError(f"Proveedor de embeddings desconocido: '{provider}'")


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """Proveedor único de la aplicación (cacheado: crea el cliente HTTP una vez)."""
    return build_embedding_provider()


__all__ = [
    "DeterministicEmbeddingProvider",
    "EmbeddingConfigurationError",
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingProviderError",
    "OpenAIEmbeddingProvider",
    "build_embedding_provider",
    "get_embedding_provider",
]
