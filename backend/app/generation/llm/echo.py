"""Proveedor de generación sin modelo, para tests y arranque local.

No razona: construye la respuesta a partir del contexto que se le pasó. Sirve
para probar el circuito completo —API, UI, citas— sin clave ni coste, pero sus
respuestas no valen para juzgar calidad.
"""

from app.generation.llm.base import LLMProvider, LLMResponse
from app.generation.prompts import NO_ANSWER
from app.schemas.chat import Prompt

MODEL_NAME = "echo-fake"


class EchoLLMProvider(LLMProvider):
    """Devuelve una respuesta previsible derivada del prompt."""

    @property
    def model(self) -> str:
        return MODEL_NAME

    def generate(self, prompt: Prompt) -> LLMResponse:
        if not prompt.has_context:
            # Misma salida que daría un modelo real sin contexto: así el circuito
            # de "preguntas no respondidas" se puede probar sin clave de API.
            return LLMResponse(text=NO_ANSWER, model=MODEL_NAME, stop_reason="end_turn")

        citas = ", ".join(
            f"({source.chunk.filename}, p. {source.chunk.page_number})" for source in prompt.sources
        )
        fragmento = prompt.sources[0].chunk.snippet
        return LLMResponse(
            text=f"{fragmento} {citas}",
            model=MODEL_NAME,
            stop_reason="end_turn",
            output_tokens=len(fragmento.split()),
        )
