"""Proveedor de embeddings basado en la API de OpenAI.

Modelo por defecto `text-embedding-3-small` (1536 dimensiones), el del documento
de diseño.

Nota de privacidad: este proveedor envía el texto de los fragmentos a OpenAI. Su
API no entrena modelos con datos de clientes y admite retención cero por acuerdo,
pero el texto sale de la infraestructura propia. Para un despliegue que no pueda
permitirlo está previsto un proveedor local (rama feature/ingestion-local-embeddings)
detrás de esta misma interfaz.
"""

from collections.abc import Sequence
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.embeddings.base import EmbeddingProvider
from app.ingestion.embeddings.errors import (
    EmbeddingConfigurationError,
    EmbeddingProviderError,
)

logger = get_logger(__name__)

#: La API acepta hasta 2048 entradas por petición (límite reflejado en el `le` de
#: `embedding_batch_size`), pero lotes moderados dan mensajes de error más útiles
#: y reintentos más baratos cuando algo falla.
MAX_BATCH_SIZE = 2048


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Vectoriza por lotes contra la API de OpenAI."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        batch_size: int | None = None,
        client: Any | None = None,
    ) -> None:
        settings = get_settings()
        # Comparación contra None, no truthiness: un 0 explícito es un error de
        # configuración que hay que señalar, no un valor ausente que rellenar.
        self._model = model if model is not None else settings.embedding_model
        self._dimensions = dimensions if dimensions is not None else settings.embedding_dimensions
        self._batch_size = batch_size if batch_size is not None else settings.embedding_batch_size

        if self._batch_size <= 0:
            raise EmbeddingConfigurationError(
                f"embedding batch_size debe ser positivo, recibido {self._batch_size}"
            )
        if self._dimensions <= 0:
            raise EmbeddingConfigurationError(
                f"embedding dimensions debe ser positivo, recibido {self._dimensions}"
            )

        if client is not None:
            self._client = client
            return

        key = api_key or settings.openai_api_key
        if not key:
            raise EmbeddingConfigurationError(
                "Falta OPENAI_API_KEY. Configúrala en .env o cambia EMBEDDING_PROVIDER."
            )
        self._client = self._build_client(key)

    @staticmethod
    def _build_client(api_key: str) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependencia declarada
            raise EmbeddingConfigurationError(
                "El paquete 'openai' no está instalado: ejecuta `uv sync`."
            ) from exc
        return OpenAI(api_key=api_key)

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            vectors.extend(self._embed_batch(batch, offset=start))
        return vectors

    def _embed_batch(self, batch: list[str], *, offset: int) -> list[list[float]]:
        try:
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
                dimensions=self._dimensions,
            )
        except Exception as exc:  # el SDK ya reintenta; aquí solo tipamos el fallo
            raise EmbeddingProviderError(
                f"Fallo al vectorizar el lote que empieza en {offset}: {exc}"
            ) from exc

        return self._parse(response, expected=len(batch), offset=offset)

    def _parse(self, response: Any, *, expected: int, offset: int) -> list[list[float]]:
        items = list(getattr(response, "data", []) or [])
        if len(items) != expected:
            raise EmbeddingProviderError(
                f"Se pidieron {expected} vectores y llegaron {len(items)} "
                f"(lote que empieza en {offset})"
            )

        # La API puede devolver los elementos desordenados: el campo `index` es la
        # única garantía de correspondencia entre texto y vector. Alinearlos mal
        # asociaría cada fragmento al vector de otro, y el fallo sería silencioso.
        items.sort(key=lambda item: getattr(item, "index", 0))

        vectors: list[list[float]] = []
        for item in items:
            vector = list(getattr(item, "embedding", []) or [])
            if len(vector) != self._dimensions:
                raise EmbeddingProviderError(
                    f"El modelo '{self._model}' devolvió un vector de {len(vector)} "
                    f"dimensiones y se esperaban {self._dimensions}"
                )
            vectors.append(vector)
        return vectors
