"""Ensamblaje del prompt final (módulo 3.4).

Toma lo que devolvió la recuperación y lo convierte en el prompt que recibirá el
LLM, respetando un presupuesto de tokens para el CONTEXTO.

El presupuesto importa por dos motivos: el contexto se paga en cada consulta, y
un contexto enorme diluye la señal —el modelo rinde peor con veinte fragmentos
mediocres que con cinco buenos—. Cuando no cabe todo, se descartan los de menor
score, que es el orden en el que la recuperación ya los dejó.
"""

from collections.abc import Sequence

from app.core.config import get_settings
from app.core.logging import get_logger
from app.generation.prompts import (
    CONTEXT_TEMPLATE,
    NO_CONTEXT_TEMPLATE,
    SYSTEM_PROMPT,
)
from app.schemas.chat import ChatMessage, Prompt
from app.schemas.retrieval import RetrievalResult
from app.storage.base import SearchResult

logger = get_logger(__name__)


def build_prompt(
    retrieval: RetrievalResult,
    *,
    history: Sequence[ChatMessage] | None = None,
    max_context_tokens: int | None = None,
    max_history_turns: int | None = None,
) -> Prompt:
    """Monta el prompt para una consulta ya recuperada.

    Args:
        retrieval: resultado del módulo 3.3.
        history: turnos anteriores de la conversación, del más antiguo al más reciente.
        max_context_tokens: presupuesto de tokens para el CONTEXTO.
        max_history_turns: número máximo de turnos previos a conservar.
    """
    settings = get_settings()
    budget = max_context_tokens if max_context_tokens is not None else settings.max_context_tokens
    history_limit = (
        max_history_turns if max_history_turns is not None else settings.max_history_turns
    )

    included, dropped, context_tokens = _fit_to_budget(retrieval.results, budget)

    if included:
        question = CONTEXT_TEMPLATE.format(
            context=_render_context(included),
            question=retrieval.query,
        )
    else:
        question = NO_CONTEXT_TEMPLATE.format(question=retrieval.query)

    messages = [*_trim_history(history, history_limit), ChatMessage(role="user", content=question)]

    if retrieval.results and not included:
        # No es que no hubiera material: es que el presupuesto no da ni para un
        # fragmento. El sistema responderá "no tengo información suficiente", que
        # sería engañoso sin dejar constancia de la causa real.
        logger.warning(
            "Ningún fragmento cabe en el presupuesto de %d tokens (%d recuperados). "
            "Sube MAX_CONTEXT_TOKENS o baja CHUNK_SIZE.",
            budget,
            len(retrieval.results),
        )
    elif dropped:
        logger.info(
            "Contexto recortado al presupuesto de %d tokens: %d fragmentos dentro, %d fuera",
            budget,
            len(included),
            dropped,
        )

    return Prompt(
        system=SYSTEM_PROMPT,
        messages=messages,
        sources=included,
        context_tokens=context_tokens,
        dropped=dropped,
    )


def _fit_to_budget(
    results: Sequence[SearchResult], budget: int
) -> tuple[list[SearchResult], int, int]:
    """Selecciona los fragmentos que caben, de mayor a menor score.

    No corta un fragmento por la mitad: media cláusula puede cambiar el sentido
    de la respuesta, y además invalidaría la cita.
    """
    included: list[SearchResult] = []
    used = 0

    for item in results:
        cost = _tokens_for(item)
        if used + cost > budget:
            continue  # se salta este, pero uno más corto todavía puede caber
        included.append(item)
        used += cost

    return included, len(results) - len(included), used


def _tokens_for(item: SearchResult) -> int:
    """Coste del fragmento incluyendo la etiqueta de origen que lo acompaña."""
    from app.ingestion.chunker import count_tokens

    return count_tokens(_render_one(item))


def _render_one(item: SearchResult) -> str:
    chunk = item.chunk
    return f"[{chunk.filename}, página {chunk.page_number}]\n{chunk.text}"


def _render_context(items: Sequence[SearchResult]) -> str:
    return "\n\n---\n\n".join(_render_one(item) for item in items)


def _trim_history(history: Sequence[ChatMessage] | None, max_turns: int) -> list[ChatMessage]:
    """Conserva los últimos turnos de la conversación.

    Empezar por un turno del asistente confundiría al modelo (parecería que
    respondió sin que nadie preguntara), así que en ese caso se descarta.
    """
    if not history or max_turns <= 0:
        return []

    trimmed = list(history)[-max_turns:]
    while trimmed and trimmed[0].role == "assistant":
        trimmed.pop(0)
    return trimmed
