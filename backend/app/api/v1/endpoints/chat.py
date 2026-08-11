"""Endpoint de consulta (sección 4 del documento de diseño).

POST /api/v1/chat/query

Encadena los módulos 3.3 y 3.4: recuperar → ensamblar el prompt → generar, y
devuelve la respuesta con las fuentes que la sustentan.
"""

from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from app.api.deps import LLMProviderDep, RetrieverDep, SettingsDep
from app.core.logging import get_logger
from app.generation import NO_ANSWER, build_prompt
from app.generation.llm.errors import LLMConfigurationError, LLMProviderError
from app.ingestion.embeddings.errors import (
    EmbeddingConfigurationError,
    EmbeddingProviderError,
)
from app.retrieval.errors import EmptyQueryError, RetrievalError
from app.schemas.chat import Prompt
from app.schemas.chat_api import ChatQueryRequest, ChatQueryResponse, Source
from app.storage.errors import VectorStoreError

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "/query",
    response_model=ChatQueryResponse,
    summary="Consultar la documentación indexada",
)
async def query(
    request: ChatQueryRequest,
    settings: SettingsDep,
    retriever: RetrieverDep,
    llm: LLMProviderDep,
) -> ChatQueryResponse:
    started = perf_counter()
    history_id = request.chat_history_id or uuid4()

    # El historial se acepta y se devuelve, pero todavía no se carga: la
    # persistencia de la conversación es feature/api-chat-history. El prompt ya
    # sabe recibirlo, así que enchufarlo será pasar la lista aquí.
    retrieval = await _retrieve(retriever, request)
    prompt = build_prompt(retrieval)

    if not prompt.has_context:
        # Sin contexto, el prompt obliga al modelo a responder exactamente
        # NO_ANSWER: llamarlo sería pagar una ida y vuelta por una constante, y
        # además su paráfrasis rompería el recuento de preguntas sin respuesta.
        logger.info("Consulta sin contexto: '%s'", retrieval.query)
        return ChatQueryResponse(
            answer=NO_ANSWER,
            sources=[],
            chat_history_id=history_id,
            model=None,
            retrieval_ms=retrieval.elapsed_ms,
            total_ms=(perf_counter() - started) * 1000,
        )

    response = await _generate(llm, prompt)

    return ChatQueryResponse(
        answer=response.text,
        sources=_sources(prompt),
        chat_history_id=history_id,
        model=response.model,
        retrieval_ms=retrieval.elapsed_ms,
        total_ms=(perf_counter() - started) * 1000,
        refused=response.refused,
    )


async def _retrieve(retriever: RetrieverDep, request: ChatQueryRequest):
    try:
        return await run_in_threadpool(retriever.retrieve, request.query, filters=request.filters)
    except EmptyQueryError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except RetrievalError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except EmbeddingConfigurationError as exc:
        logger.error("Consulta abortada por configuración: %s", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except EmbeddingProviderError as exc:
        logger.error("Fallo al vectorizar la consulta: %s", exc)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    except VectorStoreError as exc:
        logger.error("Fallo del índice vectorial: %s", exc)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc)) from exc


async def _generate(llm: LLMProviderDep, prompt: Prompt):
    try:
        return await run_in_threadpool(llm.generate, prompt)
    except LLMConfigurationError as exc:
        logger.error("Generación abortada por configuración: %s", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except LLMProviderError as exc:
        logger.error("Fallo del modelo: %s", exc)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc


def _sources(prompt: Prompt) -> list[Source]:
    """Convierte al contrato del documento los fragmentos que entraron en el prompt.

    Se usan `prompt.sources` y no los resultados de la recuperación: si el
    presupuesto de tokens dejó fuera un fragmento, el modelo no lo vio y citarlo
    sería atribuirle una fuente que no consultó.
    """
    return [
        Source(
            doc_id=item.chunk.filename,
            page=item.chunk.page_number,
            text_snippet=item.chunk.snippet,
            score=item.score,
        )
        for item in prompt.sources
    ]
