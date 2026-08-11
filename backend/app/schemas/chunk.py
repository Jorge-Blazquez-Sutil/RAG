"""Modelo del fragmento de texto (chunk).

Es la unidad que se vectoriza y se almacena en la BD vectorial (sección 3.2):
un chunk lleva su texto, su posición en el documento y la página de origen,
que es lo que después alimenta el `sources[]` de la respuesta.
"""

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Fragmento de un documento listo para vectorizar."""

    chunk_id: str = Field(description="Identificador determinista y estable del fragmento")
    index: int = Field(ge=0, description="Posición del fragmento dentro del documento")
    text: str
    page_number: int = Field(ge=1, description="Página del documento original")
    token_count: int = Field(ge=0)
    filename: str = Field(description="Documento de origen")

    @property
    def snippet(self) -> str:
        """Extracto corto para mostrar en el panel de fuentes de la UI."""
        collapsed = " ".join(self.text.split())
        return collapsed if len(collapsed) <= 200 else f"{collapsed[:197]}..."


class EmbeddedChunk(BaseModel):
    """Fragmento con su vector, listo para insertar en la BD vectorial."""

    chunk: Chunk
    embedding: list[float]
    model: str = Field(description="Modelo que generó el vector, para trazar el índice")

    @property
    def dimensions(self) -> int:
        return len(self.embedding)
