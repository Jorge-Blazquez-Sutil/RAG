"""Proveedor de generación con Claude (módulo 3.4).

Cuatro detalles de la API que condicionan este código:

* **Solo se leen los bloques de tipo `text`.** Claude Opus 5 razona por defecto,
  así que la respuesta trae bloques `thinking` además del texto. Leer
  `content[0]` a ciegas devolvería el bloque equivocado.
* **Nada de `temperature` ni `top_p`.** Los modelos actuales los rechazan con un
  400; el estilo se guía desde el prompt. El control de profundidad es `effort`.
* **`stop_reason` antes que el contenido.** Los clasificadores de seguridad
  pueden declinar una petición: llega un 200 con `stop_reason: "refusal"` y
  `content` vacío. Acceder al texto sin mirar antes revienta.
* **Fallback de rechazo activado.** Con `fallbacks: "default"` la propia API
  reintenta en otro modelo cuando el rechazo es de política, en la misma llamada.
"""

from time import perf_counter
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.generation.llm.base import LLMProvider, LLMResponse
from app.generation.llm.errors import LLMConfigurationError, LLMProviderError
from app.schemas.chat import Prompt

logger = get_logger(__name__)

#: Activa el reintento en otro modelo cuando el rechazo es de política.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

#: Texto que se devuelve si toda la cadena de modelos declina. No procede de la
#: documentación, así que se marca `refused=True` para que la UI lo distinga.
REFUSAL_MESSAGE = (
    "No puedo responder a esta consulta. Si crees que es un error, "
    "reformúlala o contacta con el administrador."
)


class AnthropicLLMProvider(LLMProvider):
    """Genera respuestas con la API de Claude."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        effort: str | None = None,
        client: Any | None = None,
    ) -> None:
        settings = get_settings()
        self._model = model if model is not None else settings.llm_model
        self._max_tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens
        self._effort = effort if effort is not None else settings.llm_effort

        if self._max_tokens <= 0:
            raise LLMConfigurationError(
                f"llm_max_tokens debe ser positivo, recibido {self._max_tokens}"
            )

        if client is not None:
            self._client = client
            return

        key = api_key or settings.anthropic_api_key
        if not key:
            raise LLMConfigurationError(
                "Falta ANTHROPIC_API_KEY. Configúrala en .env o cambia LLM_PROVIDER."
            )
        self._client = self._build_client(key)

    @staticmethod
    def _build_client(api_key: str) -> Any:
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - dependencia declarada
            raise LLMConfigurationError(
                "El paquete 'anthropic' no está instalado: ejecuta `uv sync`."
            ) from exc
        return Anthropic(api_key=api_key)

    @property
    def model(self) -> str:
        return self._model

    def generate(self, prompt: Prompt) -> LLMResponse:
        started = perf_counter()
        try:
            response = self._client.beta.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=prompt.system,
                messages=[{"role": m.role, "content": m.content} for m in prompt.messages],
                output_config={"effort": self._effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except Exception as exc:  # el SDK ya reintenta los fallos transitorios
            raise LLMProviderError(f"Fallo al generar la respuesta: {exc}") from exc

        elapsed_ms = (perf_counter() - started) * 1000
        return self._parse(response, elapsed_ms)

    def _parse(self, response: Any, elapsed_ms: float) -> LLMResponse:
        stop_reason = getattr(response, "stop_reason", None)
        usage = getattr(response, "usage", None)
        model = str(getattr(response, "model", self._model))

        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            logger.warning(
                "El modelo declinó la consulta (categoría=%s)",
                getattr(details, "category", None),
            )
            return LLMResponse(
                text=REFUSAL_MESSAGE,
                model=model,
                stop_reason=stop_reason,
                input_tokens=_tokens(usage, "input_tokens"),
                output_tokens=_tokens(usage, "output_tokens"),
                elapsed_ms=elapsed_ms,
                refused=True,
            )

        text = self._extract_text(response)
        if not text:
            raise LLMProviderError(
                f"El modelo no devolvió texto (stop_reason={stop_reason!r}). "
                "Si es 'max_tokens', sube LLM_MAX_TOKENS."
            )

        return LLMResponse(
            text=text,
            model=model,
            stop_reason=stop_reason,
            input_tokens=_tokens(usage, "input_tokens"),
            output_tokens=_tokens(usage, "output_tokens"),
            elapsed_ms=elapsed_ms,
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Concatena los bloques de texto, ignorando los de razonamiento."""
        blocks = getattr(response, "content", None) or []
        parts = [
            getattr(block, "text", "") for block in blocks if getattr(block, "type", None) == "text"
        ]
        return "".join(parts).strip()


def _tokens(usage: Any, field: str) -> int:
    value = getattr(usage, field, 0) if usage is not None else 0
    return int(value or 0)
