"""Módulo de Recuperación (sección 3.3 del documento de diseño).

Motor de búsqueda semántica en tiempo real: vectoriza la pregunta con el mismo
modelo de la ingesta, busca por similitud del coseno y devuelve el Top-K.

Uso:
    from app.retrieval import get_retriever

    resultado = get_retriever().retrieve("¿cuál es el preaviso?", top_k=5)
    contexto = resultado.as_context()

Ramas: feature/retrieval-similarity-search, feature/retrieval-reranker.
"""

from functools import lru_cache

from app.retrieval.errors import EmptyQueryError, RetrievalError
from app.retrieval.retriever import Retriever
from app.schemas.retrieval import RetrievalResult


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Recuperador único de la aplicación (reutiliza proveedor e índice)."""
    return Retriever()


__all__ = [
    "EmptyQueryError",
    "RetrievalError",
    "RetrievalResult",
    "Retriever",
    "get_retriever",
]
