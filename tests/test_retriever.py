from pathlib import Path

import pytest

from app.ingestion.embeddings import DeterministicEmbeddingProvider
from app.retrieval import EmptyQueryError, RetrievalError, Retriever
from app.schemas.chunk import Chunk, EmbeddedChunk
from app.storage import ChromaVectorStore

DIMENSIONS = 16


@pytest.fixture
def provider() -> DeterministicEmbeddingProvider:
    return DeterministicEmbeddingProvider(dimensions=DIMENSIONS)


@pytest.fixture
def store(tmp_path: Path) -> ChromaVectorStore:
    return ChromaVectorStore(path=tmp_path / "chroma", collection_name="retrieval")


@pytest.fixture
def retriever(provider, store) -> Retriever:
    return Retriever(provider=provider, store=store, top_k=3)


def index(
    store: ChromaVectorStore,
    provider: DeterministicEmbeddingProvider,
    textos: list[str],
    *,
    doc_id: str = "contrato.pdf",
    metadata: dict | None = None,
) -> None:
    embedded = [
        EmbeddedChunk(
            chunk=Chunk(
                chunk_id=f"{doc_id}-{i}",
                index=i,
                text=texto,
                page_number=i + 1,
                token_count=len(texto.split()),
                filename=doc_id,
            ),
            embedding=provider.embed_query(texto),
            model=provider.model,
        )
        for i, texto in enumerate(textos)
    ]
    store.upsert(embedded, metadata=metadata or {})


# ---------- Recuperación básica ----------


def test_retrieves_the_matching_chunk_first(retriever, store, provider) -> None:
    index(store, provider, ["preaviso de treinta días", "renta mensual", "obras de conservación"])

    resultado = retriever.retrieve("preaviso de treinta días")

    assert resultado.results[0].chunk.text == "preaviso de treinta días"
    assert resultado.best_score == pytest.approx(1.0, abs=1e-5)


def test_respects_top_k_from_the_constructor(retriever, store, provider) -> None:
    index(store, provider, [f"cláusula {i}" for i in range(10)])

    assert len(retriever.retrieve("cláusula 1").results) == 3


def test_top_k_can_be_overridden_per_query(retriever, store, provider) -> None:
    index(store, provider, [f"cláusula {i}" for i in range(10)])

    assert len(retriever.retrieve("cláusula 1", top_k=5).results) == 5


def test_empty_index_returns_no_results_without_failing(retriever) -> None:
    resultado = retriever.retrieve("¿cuál es el preaviso?")

    assert resultado.is_empty
    assert resultado.candidates == 0
    assert resultado.best_score is None


# ---------- Filtros ----------


def test_filters_restrict_the_search_to_matching_documents(retriever, store, provider) -> None:
    index(store, provider, ["preaviso de treinta días"], metadata={"categoria": "contratos_2026"})
    index(
        store,
        provider,
        ["preaviso de treinta días"],
        doc_id="manual.pdf",
        metadata={"categoria": "manuales"},
    )

    resultado = retriever.retrieve(
        "preaviso de treinta días", filters={"categoria": "contratos_2026"}
    )

    assert [r.chunk.filename for r in resultado.results] == ["contrato.pdf"]


def test_a_filter_that_matches_nothing_returns_empty(retriever, store, provider) -> None:
    index(store, provider, ["preaviso"], metadata={"categoria": "contratos_2026"})

    assert retriever.retrieve("preaviso", filters={"categoria": "facturas"}).is_empty


# ---------- Umbral de score ----------


def test_threshold_discards_weak_matches(retriever, store, provider) -> None:
    index(store, provider, ["preaviso de treinta días", "materia sin relación alguna"])

    resultado = retriever.retrieve("preaviso de treinta días", min_score=0.9)

    assert len(resultado.results) == 1
    assert resultado.candidates == 2  # el índice devolvió los dos; el umbral filtró


def test_no_threshold_by_default_keeps_every_candidate(retriever, store, provider) -> None:
    index(store, provider, ["preaviso", "algo sin relación"])

    resultado = retriever.retrieve("preaviso")

    assert retriever.min_score is None
    assert len(resultado.results) == resultado.candidates == 2


def test_an_aggressive_threshold_can_empty_the_context(retriever, store, provider) -> None:
    """Si nada supera el umbral, la generación debe poder decir 'no tengo información'."""
    index(store, provider, ["preaviso de treinta días"])

    resultado = retriever.retrieve("pregunta sobre otra cosa", min_score=0.99)

    assert resultado.is_empty
    assert resultado.candidates == 1


# ---------- Validación ----------


@pytest.mark.parametrize("query", ["", "   ", "\n\t "])
def test_empty_queries_are_rejected(retriever, query: str) -> None:
    with pytest.raises(EmptyQueryError):
        retriever.retrieve(query)


def test_query_is_trimmed_before_searching(retriever, store, provider) -> None:
    index(store, provider, ["preaviso de treinta días"])

    resultado = retriever.retrieve("  preaviso de treinta días \n")

    assert resultado.query == "preaviso de treinta días"
    assert resultado.best_score == pytest.approx(1.0, abs=1e-5)


def test_non_positive_top_k_is_rejected(provider, store) -> None:
    with pytest.raises(RetrievalError, match="top_k"):
        Retriever(provider=provider, store=store, top_k=0)

    with pytest.raises(RetrievalError, match="top_k"):
        Retriever(provider=provider, store=store).retrieve("pregunta", top_k=-1)


# ---------- Trazabilidad y contexto ----------


def test_results_carry_the_citation_data(retriever, store, provider) -> None:
    index(store, provider, ["el arrendatario deberá notificar con 30 días"], doc_id="finca_x.pdf")

    chunk = retriever.retrieve("notificar con 30 días").results[0].chunk

    assert chunk.filename == "finca_x.pdf"
    assert chunk.page_number == 1
    assert "30 días" in chunk.snippet


def test_as_context_labels_each_fragment_with_its_source(retriever, store, provider) -> None:
    index(store, provider, ["preaviso de treinta días", "renta mensual"])

    contexto = retriever.retrieve("preaviso de treinta días", top_k=2).as_context()

    assert "[contrato.pdf, página 1]" in contexto
    assert "preaviso de treinta días" in contexto
    assert contexto.count("---") == 1  # un único separador entre dos fragmentos


def test_as_context_is_empty_when_nothing_was_retrieved(retriever) -> None:
    assert retriever.retrieve("sin índice").as_context() == ""


# ---------- Latencia ----------


def test_elapsed_time_is_measured(retriever, store, provider) -> None:
    """El requisito no funcional fija el objetivo de latencia en 500 ms."""
    index(store, provider, ["preaviso de treinta días"])

    resultado = retriever.retrieve("preaviso")

    assert resultado.elapsed_ms > 0
    assert resultado.elapsed_ms < 5000  # margen amplio: aquí solo se comprueba que se mide
