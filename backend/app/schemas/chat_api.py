"""Contrato HTTP de `POST /api/v1/chat/query` (sección 4 del documento de diseño).

Los nombres de campo son los del documento —`query`, `chat_history_id`, `filters`,
`answer`, `sources[].doc_id/page/text_snippet`— porque el frontend se escribe
contra esa especificación. Los campos añadidos (`score`, `model`, tiempos) son
extras opcionales que no rompen a quien lea solo los del contrato.
"""

from uuid import UUID

from pydantic import BaseModel, Field

from app.storage.base import MetadataValue


class ChatQueryRequest(BaseModel):
    """Consulta del usuario."""

    query: str = Field(min_length=1, max_length=4000, description="Pregunta en lenguaje natural")
    chat_history_id: UUID | None = Field(
        default=None, description="Identificador de la conversación; se genera si no se envía"
    )
    filters: dict[str, MetadataValue] | None = Field(
        default=None, description='Filtros por metadatos, p. ej. {"categoria": "contratos_2026"}'
    )


class Source(BaseModel):
    """Fragmento citado en la respuesta."""

    doc_id: str
    page: int = Field(ge=1)
    text_snippet: str
    score: float = Field(description="Similitud del coseno con la consulta")


class ChatQueryResponse(BaseModel):
    """Respuesta generada junto con las fuentes que la sustentan."""

    answer: str
    sources: list[Source] = Field(default_factory=list)
    chat_history_id: UUID
    #: Modelo que generó la respuesta. Nulo si no se llegó a consultar (sin contexto).
    model: str | None = None
    retrieval_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)
    #: True si el modelo declinó por política; la respuesta no sale de la documentación.
    refused: bool = False
