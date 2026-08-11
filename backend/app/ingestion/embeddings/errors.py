"""Errores de la vectorización."""

from app.ingestion.errors import IngestionError


class EmbeddingError(IngestionError):
    """Error base de la vectorización."""


class EmbeddingConfigurationError(EmbeddingError):
    """Falta configuración para usar el proveedor (clave de API, modelo, etc.)."""


class EmbeddingProviderError(EmbeddingError):
    """El proveedor falló o devolvió una respuesta que no cuadra con lo esperado."""
