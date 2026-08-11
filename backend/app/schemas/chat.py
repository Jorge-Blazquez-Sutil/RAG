"""Modelos de la conversación y del prompt final (módulo 3.4)."""

from typing import Literal

from pydantic import BaseModel, Field

from app.storage.base import SearchResult


class ChatMessage(BaseModel):
    """Turno de la conversación."""

    role: Literal["user", "assistant"]
    content: str


class Prompt(BaseModel):
    """Prompt listo para enviar al LLM, con la traza de lo que entró en él."""

    system: str
    messages: list[ChatMessage]
    #: Fragmentos que realmente entraron en el CONTEXTO. Es lo que debe viajar al
    #: `sources[]` de la respuesta: citar algo que se quedó fuera por presupuesto
    #: de tokens sería una cita falsa.
    sources: list[SearchResult] = Field(default_factory=list)
    context_tokens: int = Field(default=0, ge=0)
    #: Fragmentos recuperados que no cupieron en el presupuesto.
    dropped: int = Field(default=0, ge=0)

    @property
    def has_context(self) -> bool:
        return bool(self.sources)
