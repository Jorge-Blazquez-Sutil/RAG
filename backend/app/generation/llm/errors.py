"""Errores de la generación."""


class LLMError(Exception):
    """Error base del módulo de generación."""


class LLMConfigurationError(LLMError):
    """Falta configuración para usar el proveedor (clave de API, modelo, etc.)."""


class LLMProviderError(LLMError):
    """El modelo falló o devolvió algo que no se puede interpretar."""
