"""Motor de búsqueda semántica (módulo 3.3).

Orquesta el camino consulta → vector → Top-K:

1. Vectoriza la pregunta con **el mismo proveedor** que se usó en la ingesta.
2. Busca por similitud del coseno en el índice, aplicando los filtros de metadatos.
3. Descarta lo que no llegue al umbral de score, si hay umbral configurado.

Sobre el umbral: viene desactivado por defecto a propósito. Los vectores de
`text-embedding-3-small` no están centrados, así que dos textos sin ninguna
relación puntúan habitualmente por encima de 0,7; un umbral elegido a ojo o no
filtra nada o se lleva por delante respuestas buenas. El valor correcto depende
del modelo y del corpus, y se calibra con el golden set de `feature/eval-rag-harness`.
"""

from time import perf_counter

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.embeddings import get_embedding_provider
from app.ingestion.embeddings.base import EmbeddingProvider
from app.retrieval.errors import EmptyQueryError, RetrievalError
from app.schemas.retrieval import RetrievalResult
from app.storage import get_vector_store
from app.storage.base import Filters, VectorStore

logger = get_logger(__name__)


class Retriever:
    """Recupera los fragmentos más relevantes para una pregunta."""

    def __init__(
        self,
        *,
        provider: EmbeddingProvider | None = None,
        store: VectorStore | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> None:
        settings = get_settings()
        self._provider = provider if provider is not None else get_embedding_provider()
        self._store = store if store is not None else get_vector_store()
        self._top_k = top_k if top_k is not None else settings.top_k
        self._min_score = min_score if min_score is not None else settings.retrieval_min_score

        if self._top_k <= 0:
            raise RetrievalError(f"top_k debe ser positivo, recibido {self._top_k}")

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: Filters | None = None,
        min_score: float | None = None,
    ) -> RetrievalResult:
        """Devuelve los fragmentos más parecidos a la consulta.

        Args:
            query: pregunta en lenguaje natural.
            top_k: número de fragmentos a recuperar. Por defecto, el de configuración.
            filters: filtros por metadatos, tal y como llegan en `POST /chat/query`.
            min_score: similitud mínima para conservar un fragmento.

        Raises:
            EmptyQueryError: la consulta está vacía.
            RetrievalError: parámetros inválidos.
        """
        cleaned = query.strip()
        if not cleaned:
            raise EmptyQueryError()

        limit = top_k if top_k is not None else self._top_k
        if limit <= 0:
            raise RetrievalError(f"top_k debe ser positivo, recibido {limit}")
        threshold = min_score if min_score is not None else self._min_score

        started = perf_counter()
        embedding = self._provider.embed_query(cleaned)
        candidates = self._store.search(embedding, top_k=limit, filters=filters)
        results = self._apply_threshold(candidates, threshold)
        elapsed_ms = (perf_counter() - started) * 1000

        logger.debug(
            "Consulta recuperada en %.1f ms: %d candidatos, %d tras el umbral",
            elapsed_ms,
            len(candidates),
            len(results),
        )

        return RetrievalResult(
            query=cleaned,
            results=results,
            candidates=len(candidates),
            elapsed_ms=elapsed_ms,
        )

    @staticmethod
    def _apply_threshold(candidates: list, threshold: float | None) -> list:
        if threshold is None:
            return candidates
        return [item for item in candidates if item.score >= threshold]

    @property
    def top_k(self) -> int:
        return self._top_k

    @property
    def min_score(self) -> float | None:
        return self._min_score
