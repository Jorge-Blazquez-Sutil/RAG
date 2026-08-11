"""Módulo de Generación (sección 3.4 del documento de diseño).

Ensambla el prompt final (plantilla de sistema + CONTEXTO Top-K + PREGUNTA) y,
en la rama siguiente, llama al LLM.

Uso:
    from app.generation import build_prompt

    prompt = build_prompt(resultado_recuperacion, history=turnos_previos)

Ramas: feature/generation-prompt-assembly, feature/generation-llm-provider,
feature/generation-citations.
"""

from app.generation.prompt_builder import build_prompt
from app.generation.prompts import (
    CONTEXT_TEMPLATE,
    NO_ANSWER,
    NO_CONTEXT_TEMPLATE,
    SYSTEM_PROMPT,
)
from app.schemas.chat import ChatMessage, Prompt

__all__ = [
    "CONTEXT_TEMPLATE",
    "NO_ANSWER",
    "NO_CONTEXT_TEMPLATE",
    "SYSTEM_PROMPT",
    "ChatMessage",
    "Prompt",
    "build_prompt",
]
