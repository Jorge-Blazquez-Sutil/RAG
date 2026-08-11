"""Contrato de los proveedores de LLM (módulo 3.4).

Aísla la generación del modelo concreto: Claude vía API hoy, Llama 3 local vía
Ollama mañana, sin que el resto del pipeline se entere.
"""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from app.schemas.chat import Prompt


class LLMResponse(BaseModel):
    """Respuesta del modelo con la traza necesaria para coste y analíticas."""

    text: str
    model: str
    stop_reason: str | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    elapsed_ms: float = Field(default=0.0, ge=0)
    #: True si el modelo declinó por política. La respuesta sigue siendo
    #: presentable al usuario, pero no procede de la documentación.
    refused: bool = False


class LLMProvider(ABC):
    """Genera la respuesta final a partir del prompt ensamblado."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Identificador del modelo, para trazar coste y respuestas."""

    @abstractmethod
    def generate(self, prompt: Prompt) -> LLMResponse:
        """Envía el prompt al modelo y devuelve su respuesta.

        Raises:
            LLMProviderError: fallo de la API o respuesta inesperada.
        """
