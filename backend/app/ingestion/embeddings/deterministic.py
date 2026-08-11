"""Proveedor de embeddings determinista, sin red ni claves de API.

Sirve para tests, para el CI y para levantar el pipeline completo en local sin
credenciales. **No tiene ninguna propiedad semántica**: dos textos parecidos
producen vectores sin relación entre sí, así que no vale para medir calidad de
recuperación — solo para comprobar que las piezas encajan.
"""

import hashlib
import math
import random
from collections.abc import Sequence

from app.core.config import get_settings
from app.ingestion.embeddings.base import EmbeddingProvider

MODEL_NAME = "deterministic-fake"


class DeterministicEmbeddingProvider(EmbeddingProvider):
    """Genera vectores reproducibles a partir del hash del texto."""

    def __init__(self, *, dimensions: int | None = None) -> None:
        self._dimensions = dimensions or get_settings().embedding_dimensions

    @property
    def model(self) -> str:
        return MODEL_NAME

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        rng = random.Random(seed)
        values = [rng.uniform(-1.0, 1.0) for _ in range(self._dimensions)]
        # Se normaliza a norma 1 para que la similitud del coseno se comporte
        # igual que con los vectores de un modelo real.
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]
