from dataclasses import dataclass, field
from typing import Any

import pytest

from app.ingestion.embeddings import (
    DeterministicEmbeddingProvider,
    EmbeddingConfigurationError,
    EmbeddingProviderError,
    OpenAIEmbeddingProvider,
    build_embedding_provider,
)
from app.schemas.chunk import Chunk

DIMENSIONS = 8


# ---------- Doble de prueba del cliente de OpenAI ----------


@dataclass
class FakeItem:
    embedding: list[float]
    index: int


@dataclass
class FakeResponse:
    data: list[FakeItem]


@dataclass
class FakeEmbeddings:
    parent: "FakeOpenAIClient"

    def create(self, *, model: str, input: list[str], dimensions: int) -> FakeResponse:
        self.parent.calls.append({"model": model, "input": input, "dimensions": dimensions})
        if self.parent.error is not None:
            raise self.parent.error
        items = [
            FakeItem(embedding=[float(i)] * self.parent.returned_dimensions, index=i)
            for i in range(len(input))
        ]
        if self.parent.shuffle:
            items.reverse()
        if self.parent.drop_one and items:
            items.pop()
        return FakeResponse(data=items)


@dataclass
class FakeOpenAIClient:
    returned_dimensions: int = DIMENSIONS
    shuffle: bool = False
    drop_one: bool = False
    error: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def embeddings(self) -> FakeEmbeddings:
        return FakeEmbeddings(parent=self)


def build_provider(client: FakeOpenAIClient, **kwargs: Any) -> OpenAIEmbeddingProvider:
    return OpenAIEmbeddingProvider(client=client, dimensions=DIMENSIONS, **kwargs)


def build_chunk(text: str, *, index: int = 0, page: int = 1) -> Chunk:
    return Chunk(
        chunk_id=f"id-{index}",
        index=index,
        text=text,
        page_number=page,
        token_count=len(text.split()),
        filename="contrato.pdf",
    )


# ---------- Proveedor OpenAI ----------


def test_embeds_texts_preserving_order() -> None:
    client = FakeOpenAIClient()
    provider = build_provider(client)

    vectors = provider.embed_texts(["uno", "dos", "tres"])

    assert [vector[0] for vector in vectors] == [0.0, 1.0, 2.0]
    assert len(client.calls) == 1


def test_reorders_by_index_when_the_api_answers_out_of_order() -> None:
    """La API puede devolver los elementos desordenados; el campo `index` manda."""
    client = FakeOpenAIClient(shuffle=True)
    provider = build_provider(client)

    vectors = provider.embed_texts(["uno", "dos", "tres"])

    assert [vector[0] for vector in vectors] == [0.0, 1.0, 2.0]


def test_splits_the_input_into_batches() -> None:
    client = FakeOpenAIClient()
    provider = build_provider(client, batch_size=2)

    vectors = provider.embed_texts([f"texto {i}" for i in range(5)])

    assert len(vectors) == 5
    assert [len(call["input"]) for call in client.calls] == [2, 2, 1]


def test_empty_input_does_not_call_the_api() -> None:
    client = FakeOpenAIClient()
    provider = build_provider(client)

    assert provider.embed_texts([]) == []
    assert provider.embed_chunks([]) == []
    assert client.calls == []


def test_embed_query_uses_the_same_model_as_ingestion() -> None:
    client = FakeOpenAIClient()
    provider = build_provider(client, model="text-embedding-3-small")

    provider.embed_chunks([build_chunk("fragmento")])
    provider.embed_query("¿cuál es el preaviso?")

    assert {call["model"] for call in client.calls} == {"text-embedding-3-small"}


def test_embed_chunks_pairs_each_chunk_with_its_vector() -> None:
    client = FakeOpenAIClient()
    provider = build_provider(client, model="text-embedding-3-small")
    chunks = [build_chunk("uno", index=0), build_chunk("dos", index=1, page=4)]

    embedded = provider.embed_chunks(chunks)

    assert [item.chunk.chunk_id for item in embedded] == ["id-0", "id-1"]
    assert embedded[1].chunk.page_number == 4
    assert embedded[0].dimensions == DIMENSIONS
    assert embedded[0].model == "text-embedding-3-small"


def test_unexpected_vector_size_raises() -> None:
    client = FakeOpenAIClient(returned_dimensions=DIMENSIONS + 1)
    provider = build_provider(client)

    with pytest.raises(EmbeddingProviderError, match="dimensiones"):
        provider.embed_texts(["uno"])


def test_missing_vector_in_the_response_raises() -> None:
    client = FakeOpenAIClient(drop_one=True)
    provider = build_provider(client)

    with pytest.raises(EmbeddingProviderError, match="vectores"):
        provider.embed_texts(["uno", "dos"])


def test_api_failure_is_wrapped_with_the_batch_position() -> None:
    client = FakeOpenAIClient(error=RuntimeError("503 service unavailable"))
    provider = build_provider(client, batch_size=2)

    with pytest.raises(EmbeddingProviderError, match="503"):
        provider.embed_texts(["uno", "dos", "tres"])


def test_missing_api_key_is_a_configuration_error(monkeypatch) -> None:
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "openai_api_key", None)

    with pytest.raises(EmbeddingConfigurationError, match="OPENAI_API_KEY"):
        OpenAIEmbeddingProvider()


def test_invalid_batch_size_is_a_configuration_error() -> None:
    with pytest.raises(EmbeddingConfigurationError, match="batch_size"):
        build_provider(FakeOpenAIClient(), batch_size=0)


# ---------- Proveedor determinista ----------


def test_deterministic_provider_repeats_the_same_vector() -> None:
    provider = DeterministicEmbeddingProvider(dimensions=DIMENSIONS)

    assert provider.embed_query("preaviso") == provider.embed_query("preaviso")


def test_deterministic_provider_separates_different_texts() -> None:
    provider = DeterministicEmbeddingProvider(dimensions=DIMENSIONS)

    assert provider.embed_query("preaviso") != provider.embed_query("rescisión")


def test_deterministic_vectors_are_normalized() -> None:
    provider = DeterministicEmbeddingProvider(dimensions=DIMENSIONS)

    vector = provider.embed_query("cláusula")

    assert len(vector) == DIMENSIONS
    assert sum(value * value for value in vector) == pytest.approx(1.0)


def test_deterministic_provider_works_without_network_or_key() -> None:
    provider = build_embedding_provider("fake")

    embedded = provider.embed_chunks([build_chunk("sin credenciales")])

    assert embedded[0].model == "deterministic-fake"


# ---------- Fábrica ----------


def test_local_provider_is_not_implemented_yet() -> None:
    with pytest.raises(EmbeddingConfigurationError, match="local"):
        build_embedding_provider("local")


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(EmbeddingConfigurationError, match="desconocido"):
        build_embedding_provider("cohere")
