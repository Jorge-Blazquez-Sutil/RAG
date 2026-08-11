from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.deps import llm_provider_dep, retriever_dep, settings_dep
from app.core.config import Settings, get_settings
from app.generation import NO_ANSWER
from app.generation.llm import EchoLLMProvider
from app.generation.llm.base import LLMProvider, LLMResponse
from app.generation.llm.errors import LLMConfigurationError, LLMProviderError
from app.ingestion.embeddings import DeterministicEmbeddingProvider
from app.main import create_app
from app.retrieval import Retriever
from app.schemas.chat import Prompt
from app.schemas.chunk import Chunk, EmbeddedChunk
from app.storage import ChromaVectorStore

QUERY = "/api/v1/chat/query"
DIMENSIONS = 32

TEXTOS = [
    ("La rescisión anticipada requiere un preaviso de 30 días", 12),
    ("El arrendatario abonará la renta en los cinco primeros días", 4),
]


@pytest.fixture
def provider() -> DeterministicEmbeddingProvider:
    return DeterministicEmbeddingProvider(dimensions=DIMENSIONS)


@pytest.fixture
def store(tmp_path: Path, provider: DeterministicEmbeddingProvider) -> ChromaVectorStore:
    store = ChromaVectorStore(path=tmp_path / "chroma", collection_name="apichat")
    store.upsert(
        [
            EmbeddedChunk(
                chunk=Chunk(
                    chunk_id=f"c{i}",
                    index=i,
                    text=texto,
                    page_number=page,
                    token_count=len(texto.split()),
                    filename="contrato_finca_x.pdf",
                ),
                embedding=provider.embed_query(texto),
                model=provider.model,
            )
            for i, (texto, page) in enumerate(TEXTOS)
        ],
        metadata={"categoria": "contratos_2026"},
    )
    return store


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    base = get_settings().model_copy(deep=True)
    base.upload_dir = tmp_path / "uploads"
    return base


def build_client(settings: Settings, retriever: Retriever, llm: LLMProvider) -> TestClient:
    app = create_app()
    app.dependency_overrides[settings_dep] = lambda: settings
    app.dependency_overrides[retriever_dep] = lambda: retriever
    app.dependency_overrides[llm_provider_dep] = lambda: llm
    return TestClient(app)


@pytest.fixture
def client(
    settings: Settings, store: ChromaVectorStore, provider: DeterministicEmbeddingProvider
) -> Iterator[TestClient]:
    retriever = Retriever(provider=provider, store=store, top_k=2)
    with build_client(settings, retriever, EchoLLMProvider()) as test_client:
        yield test_client


# ---------- Contrato de la respuesta ----------


def test_query_answers_with_sources(client: TestClient) -> None:
    response = client.post(QUERY, json={"query": TEXTOS[0][0]})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["sources"]
    assert body["model"] == "echo-fake"


def test_sources_follow_the_documented_contract(client: TestClient) -> None:
    body = client.post(QUERY, json={"query": TEXTOS[0][0]}).json()

    primero = body["sources"][0]
    assert set(primero) >= {"doc_id", "page", "text_snippet"}
    assert primero["doc_id"] == "contrato_finca_x.pdf"
    assert primero["page"] in {12, 4}
    assert primero["text_snippet"]


def test_the_best_source_is_the_matching_page(client: TestClient) -> None:
    body = client.post(QUERY, json={"query": TEXTOS[0][0]}).json()

    assert body["sources"][0]["page"] == 12
    assert body["sources"][0]["score"] == pytest.approx(1.0, abs=1e-5)


def test_latency_is_reported_for_the_non_functional_requirement(client: TestClient) -> None:
    body = client.post(QUERY, json={"query": TEXTOS[0][0]}).json()

    assert body["retrieval_ms"] > 0
    assert body["total_ms"] >= body["retrieval_ms"]


# ---------- Conversación ----------


def test_a_chat_history_id_is_generated_when_absent(client: TestClient) -> None:
    body = client.post(QUERY, json={"query": TEXTOS[0][0]}).json()

    assert body["chat_history_id"]


def test_the_provided_chat_history_id_is_echoed_back(client: TestClient) -> None:
    conversacion = str(uuid4())

    body = client.post(QUERY, json={"query": TEXTOS[0][0], "chat_history_id": conversacion}).json()

    assert body["chat_history_id"] == conversacion


def test_an_invalid_chat_history_id_is_rejected(client: TestClient) -> None:
    response = client.post(QUERY, json={"query": "hola", "chat_history_id": "no-es-uuid"})

    assert response.status_code == 422


# ---------- Filtros ----------


def test_filters_narrow_the_search(client: TestClient) -> None:
    body = client.post(
        QUERY,
        json={"query": TEXTOS[0][0], "filters": {"categoria": "contratos_2026"}},
    ).json()

    assert body["sources"]


def test_a_filter_that_matches_nothing_surrenders(client: TestClient) -> None:
    body = client.post(
        QUERY, json={"query": TEXTOS[0][0], "filters": {"categoria": "facturas"}}
    ).json()

    assert body["answer"] == NO_ANSWER
    assert body["sources"] == []


# ---------- Sin contexto ----------


def test_without_context_the_model_is_not_called(
    settings: Settings, store: ChromaVectorStore, provider: DeterministicEmbeddingProvider
) -> None:
    """Sin contexto la respuesta está predeterminada: llamar al modelo sería tirar dinero."""

    class ExplosiveLLM(LLMProvider):
        @property
        def model(self) -> str:
            return "no-debería-usarse"

        def generate(self, prompt: Prompt) -> LLMResponse:
            raise AssertionError("no se debe llamar al modelo sin contexto")

    retriever = Retriever(provider=provider, store=store, top_k=2, min_score=0.99)
    with build_client(settings, retriever, ExplosiveLLM()) as client:
        body = client.post(QUERY, json={"query": "pregunta totalmente ajena"}).json()

    assert body["answer"] == NO_ANSWER
    assert body["sources"] == []
    assert body["model"] is None


def test_the_empty_index_case_also_surrenders(
    settings: Settings, tmp_path: Path, provider: DeterministicEmbeddingProvider
) -> None:
    vacio = ChromaVectorStore(path=tmp_path / "otro", collection_name="vacio")
    retriever = Retriever(provider=provider, store=vacio, top_k=2)

    with build_client(settings, retriever, EchoLLMProvider()) as client:
        body = client.post(QUERY, json={"query": "¿cuál es el preaviso?"}).json()

    assert body["answer"] == NO_ANSWER


# ---------- Validación ----------


@pytest.mark.parametrize("payload", [{}, {"query": ""}, {"query": "   " * 0}])
def test_an_empty_query_is_rejected(client: TestClient, payload: dict) -> None:
    assert client.post(QUERY, json=payload).status_code == 422


def test_an_oversized_query_is_rejected(client: TestClient) -> None:
    assert client.post(QUERY, json={"query": "x" * 4001}).status_code == 422


def test_nested_filters_are_rejected(client: TestClient) -> None:
    response = client.post(QUERY, json={"query": "hola", "filters": {"a": {"b": 1}}})

    assert response.status_code == 422


# ---------- Fallos del modelo ----------


def test_a_model_failure_becomes_a_502(
    settings: Settings, store: ChromaVectorStore, provider: DeterministicEmbeddingProvider
) -> None:
    class FailingLLM(LLMProvider):
        @property
        def model(self) -> str:
            return "roto"

        def generate(self, prompt: Prompt) -> LLMResponse:
            raise LLMProviderError("529 overloaded")

    retriever = Retriever(provider=provider, store=store, top_k=2)
    with build_client(settings, retriever, FailingLLM()) as client:
        response = client.post(QUERY, json={"query": TEXTOS[0][0]})

    assert response.status_code == 502


def test_a_missing_api_key_becomes_a_503(
    settings: Settings, store: ChromaVectorStore, provider: DeterministicEmbeddingProvider
) -> None:
    class UnconfiguredLLM(LLMProvider):
        @property
        def model(self) -> str:
            return "sin-configurar"

        def generate(self, prompt: Prompt) -> LLMResponse:
            raise LLMConfigurationError("Falta ANTHROPIC_API_KEY")

    retriever = Retriever(provider=provider, store=store, top_k=2)
    with build_client(settings, retriever, UnconfiguredLLM()) as client:
        response = client.post(QUERY, json={"query": TEXTOS[0][0]})

    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def test_a_refusal_is_flagged_in_the_response(
    settings: Settings, store: ChromaVectorStore, provider: DeterministicEmbeddingProvider
) -> None:
    class RefusingLLM(LLMProvider):
        @property
        def model(self) -> str:
            return "claude-opus-5"

        def generate(self, prompt: Prompt) -> LLMResponse:
            return LLMResponse(
                text="No puedo responder a esta consulta.",
                model="claude-opus-5",
                stop_reason="refusal",
                refused=True,
            )

    retriever = Retriever(provider=provider, store=store, top_k=2)
    with build_client(settings, retriever, RefusingLLM()) as client:
        body = client.post(QUERY, json={"query": TEXTOS[0][0]}).json()

    assert body["refused"] is True
    assert body["answer"]


# ---------- Documentación ----------


def test_the_endpoint_is_published_in_the_openapi_schema(client: TestClient) -> None:
    assert "/api/v1/chat/query" in client.get("/openapi.json").json()["paths"]
