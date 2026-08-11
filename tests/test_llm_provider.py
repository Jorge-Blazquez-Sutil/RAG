from dataclasses import dataclass, field
from typing import Any

import pytest

from app.generation import NO_ANSWER, build_prompt
from app.generation.llm import (
    AnthropicLLMProvider,
    EchoLLMProvider,
    LLMConfigurationError,
    LLMProviderError,
    build_llm_provider,
)
from app.generation.llm.anthropic_provider import FALLBACK_BETA
from app.schemas.chunk import Chunk
from app.schemas.retrieval import RetrievalResult
from app.storage.base import SearchResult

# ---------- Doble de prueba del cliente de Anthropic ----------


@dataclass
class Block:
    type: str
    text: str = ""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class StopDetails:
    category: str | None = None


@dataclass
class FakeResponse:
    content: list[Block]
    model: str = "claude-opus-5"
    stop_reason: str | None = "end_turn"
    usage: Usage = field(default_factory=Usage)
    stop_details: StopDetails | None = None


@dataclass
class FakeMessages:
    parent: "FakeAnthropicClient"

    def create(self, **kwargs: Any) -> FakeResponse:
        self.parent.calls.append(kwargs)
        if self.parent.error is not None:
            raise self.parent.error
        return self.parent.response


@dataclass
class FakeBeta:
    parent: "FakeAnthropicClient"

    @property
    def messages(self) -> FakeMessages:
        return FakeMessages(parent=self.parent)


@dataclass
class FakeAnthropicClient:
    response: FakeResponse = field(
        default_factory=lambda: FakeResponse(content=[Block(type="text", text="respuesta")])
    )
    error: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def beta(self) -> FakeBeta:
        return FakeBeta(parent=self)


def make_prompt(*, with_context: bool = True):
    results = []
    if with_context:
        results.append(
            SearchResult(
                chunk=Chunk(
                    chunk_id="1",
                    index=0,
                    text="La rescisión requiere un preaviso de 30 días",
                    page_number=12,
                    token_count=8,
                    filename="contrato.pdf",
                ),
                score=0.9,
            )
        )
    retrieval = RetrievalResult(
        query="¿cuál es el preaviso?",
        results=results,
        candidates=len(results),
        elapsed_ms=1.0,
    )
    return build_prompt(retrieval)


def build_provider(client: FakeAnthropicClient, **kwargs: Any) -> AnthropicLLMProvider:
    return AnthropicLLMProvider(client=client, **kwargs)


# ---------- Lectura de la respuesta ----------


def test_returns_the_text_of_the_answer() -> None:
    client = FakeAnthropicClient()

    respuesta = build_provider(client).generate(make_prompt())

    assert respuesta.text == "respuesta"
    assert respuesta.refused is False
    assert respuesta.stop_reason == "end_turn"


def test_ignores_thinking_blocks() -> None:
    """Opus 5 razona por defecto: leer content[0] devolvería el bloque equivocado."""
    client = FakeAnthropicClient(
        response=FakeResponse(
            content=[
                Block(type="thinking", text=""),
                Block(type="text", text="El preaviso es de 30 días."),
            ]
        )
    )

    assert build_provider(client).generate(make_prompt()).text == "El preaviso es de 30 días."


def test_concatenates_several_text_blocks() -> None:
    client = FakeAnthropicClient(
        response=FakeResponse(
            content=[
                Block(type="text", text="Primera parte. "),
                Block(type="text", text="Segunda."),
            ]
        )
    )

    assert build_provider(client).generate(make_prompt()).text == "Primera parte. Segunda."


def test_records_token_usage_and_latency() -> None:
    client = FakeAnthropicClient(
        response=FakeResponse(
            content=[Block(type="text", text="ok")],
            usage=Usage(input_tokens=1200, output_tokens=80),
        )
    )

    respuesta = build_provider(client).generate(make_prompt())

    assert (respuesta.input_tokens, respuesta.output_tokens) == (1200, 80)
    assert respuesta.elapsed_ms > 0


def test_an_empty_answer_is_an_error_not_an_empty_string() -> None:
    client = FakeAnthropicClient(
        response=FakeResponse(content=[Block(type="thinking")], stop_reason="max_tokens")
    )

    with pytest.raises(LLMProviderError, match="LLM_MAX_TOKENS"):
        build_provider(client).generate(make_prompt())


# ---------- Parámetros de la petición ----------


def test_sends_system_prompt_and_messages_separately() -> None:
    client = FakeAnthropicClient()
    prompt = make_prompt()

    build_provider(client).generate(prompt)

    call = client.calls[0]
    assert call["system"] == prompt.system
    assert call["messages"] == [{"role": "user", "content": prompt.messages[0].content}]


def test_sends_model_max_tokens_and_effort() -> None:
    client = FakeAnthropicClient()

    build_provider(client, model="claude-opus-5", max_tokens=4096, effort="high").generate(
        make_prompt()
    )

    call = client.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["max_tokens"] == 4096
    assert call["output_config"] == {"effort": "high"}


def test_never_sends_sampling_parameters() -> None:
    """temperature/top_p/top_k devuelven 400 en los modelos actuales."""
    client = FakeAnthropicClient()

    build_provider(client).generate(make_prompt())

    assert {"temperature", "top_p", "top_k"}.isdisjoint(client.calls[0])


def test_enables_the_refusal_fallback() -> None:
    client = FakeAnthropicClient()

    build_provider(client).generate(make_prompt())

    call = client.calls[0]
    assert call["fallbacks"] == "default"
    assert call["betas"] == [FALLBACK_BETA]


# ---------- Rechazos y fallos ----------


def test_a_refusal_is_reported_not_crashed(caplog) -> None:
    """Un rechazo llega como 200 con content vacío: leer el texto a ciegas rompería."""
    client = FakeAnthropicClient(
        response=FakeResponse(
            content=[],
            stop_reason="refusal",
            stop_details=StopDetails(category="cyber"),
        )
    )

    with caplog.at_level("WARNING"):
        respuesta = build_provider(client).generate(make_prompt())

    assert respuesta.refused is True
    assert respuesta.text
    assert respuesta.stop_reason == "refusal"
    assert "cyber" in caplog.text


def test_api_failures_are_wrapped() -> None:
    client = FakeAnthropicClient(error=RuntimeError("529 overloaded"))

    with pytest.raises(LLMProviderError, match="529"):
        build_provider(client).generate(make_prompt())


def test_missing_api_key_is_a_configuration_error(monkeypatch) -> None:
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "anthropic_api_key", None)

    with pytest.raises(LLMConfigurationError, match="ANTHROPIC_API_KEY"):
        AnthropicLLMProvider()


def test_invalid_max_tokens_is_a_configuration_error() -> None:
    with pytest.raises(LLMConfigurationError, match="llm_max_tokens"):
        build_provider(FakeAnthropicClient(), max_tokens=0)


# ---------- Proveedor sin modelo ----------


def test_echo_provider_answers_from_the_context() -> None:
    respuesta = EchoLLMProvider().generate(make_prompt())

    assert "preaviso de 30 días" in respuesta.text
    assert "(contrato.pdf, p. 12)" in respuesta.text


def test_echo_provider_surrenders_without_context() -> None:
    assert EchoLLMProvider().generate(make_prompt(with_context=False)).text == NO_ANSWER


def test_echo_provider_needs_no_credentials() -> None:
    assert build_llm_provider("fake").model == "echo-fake"


# ---------- Fábrica ----------


def test_ollama_provider_is_not_implemented_yet() -> None:
    with pytest.raises(LLMConfigurationError, match="Ollama"):
        build_llm_provider("ollama")


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(LLMConfigurationError, match="desconocido"):
        build_llm_provider("mistral")
