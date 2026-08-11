"""Proveedores de LLM (módulo 3.4).

Uso:
    from app.generation.llm import get_llm_provider

    respuesta = get_llm_provider().generate(prompt)

El proveedor se elige con `LLM_PROVIDER`.
"""

from functools import lru_cache

from app.core.config import get_settings
from app.generation.llm.anthropic_provider import AnthropicLLMProvider
from app.generation.llm.base import LLMProvider, LLMResponse
from app.generation.llm.echo import EchoLLMProvider
from app.generation.llm.errors import (
    LLMConfigurationError,
    LLMError,
    LLMProviderError,
)


def build_llm_provider(name: str | None = None) -> LLMProvider:
    """Crea el proveedor indicado (o el de la configuración)."""
    provider = name or get_settings().llm_provider

    match provider:
        case "anthropic":
            return AnthropicLLMProvider()
        case "fake":
            return EchoLLMProvider()
        case "ollama":
            raise LLMConfigurationError(
                "El proveedor Ollama todavía no está implementado "
                "(rama feature/generation-ollama-provider). "
                "Usa LLM_PROVIDER=anthropic o =fake."
            )
        case _:
            raise LLMConfigurationError(f"Proveedor de LLM desconocido: '{provider}'")


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    """Proveedor único de la aplicación (cacheado: crea el cliente una vez)."""
    return build_llm_provider()


__all__ = [
    "AnthropicLLMProvider",
    "EchoLLMProvider",
    "LLMConfigurationError",
    "LLMError",
    "LLMProvider",
    "LLMProviderError",
    "LLMResponse",
    "build_llm_provider",
    "get_llm_provider",
]
