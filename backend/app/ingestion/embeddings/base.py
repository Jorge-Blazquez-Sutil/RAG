"""Contrato de los proveedores de embeddings.

La consulta del usuario y los fragmentos del documento **deben** vectorizarse con
el mismo modelo: si no, los vectores viven en espacios distintos y la similitud
del coseno del módulo 3.3 devuelve ruido. Por eso ambas operaciones cuelgan del
mismo objeto proveedor en lugar de ser funciones sueltas.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.schemas.chunk import Chunk, EmbeddedChunk


class EmbeddingProvider(ABC):
    """Convierte texto en vectores de alta dimensionalidad."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Identificador del modelo, para dejar traza en los metadatos del índice."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Dimensión de los vectores que produce este proveedor."""

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Vectoriza una lista de textos conservando el orden de entrada.

        Raises:
            EmbeddingProviderError: fallo al llamar al modelo o respuesta inválida.
        """

    def embed_query(self, text: str) -> list[float]:
        """Vectoriza la consulta del usuario (módulo 3.3).

        Comparte implementación con la ingesta a propósito: garantiza que pregunta
        y documentos acaben en el mismo espacio vectorial.
        """
        return self.embed_texts([text])[0]

    def embed_chunks(self, chunks: Sequence[Chunk]) -> list[EmbeddedChunk]:
        """Vectoriza fragmentos y los empareja con su vector."""
        if not chunks:
            return []
        vectors = self.embed_texts([chunk.text for chunk in chunks])
        return [
            EmbeddedChunk(chunk=chunk, embedding=vector, model=self.model)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
