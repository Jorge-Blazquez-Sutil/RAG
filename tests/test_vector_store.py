from pathlib import Path

import pytest

from app.schemas.chunk import Chunk, EmbeddedChunk
from app.storage import (
    ChromaVectorStore,
    DimensionMismatchError,
    VectorStoreConfigurationError,
    VectorStoreError,
    build_vector_store,
)

MODEL = "test-embeddings"


@pytest.fixture
def store(tmp_path: Path) -> ChromaVectorStore:
    """Índice aislado por test.

    Cada test recibe su propio directorio: Chroma reutiliza el cliente cuando la
    configuración coincide, así que un cliente efímero compartido arrastraría los
    datos de un test al siguiente.
    """
    return ChromaVectorStore(path=tmp_path / "chroma", collection_name="tests")


def build_embedded(
    *,
    chunk_id: str,
    embedding: list[float],
    text: str = "texto",
    doc_id: str = "contrato.pdf",
    page: int = 1,
    index: int = 0,
) -> EmbeddedChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        index=index,
        text=text,
        page_number=page,
        token_count=len(text.split()),
        filename=doc_id,
    )
    return EmbeddedChunk(chunk=chunk, embedding=embedding, model=MODEL)


# ---------- Escritura ----------


def test_upsert_stores_chunks(store: ChromaVectorStore) -> None:
    written = store.upsert(
        [
            build_embedded(chunk_id="a", embedding=[1.0, 0.0, 0.0, 0.0]),
            build_embedded(chunk_id="b", embedding=[0.0, 1.0, 0.0, 0.0], index=1),
        ]
    )

    assert written == 2
    assert store.count() == 2


def test_upsert_is_idempotent(store: ChromaVectorStore) -> None:
    """Reingerir el mismo documento sobrescribe: el chunk_id es determinista."""
    item = build_embedded(chunk_id="a", embedding=[1.0, 0.0, 0.0, 0.0], text="original")
    store.upsert([item])

    updated = build_embedded(chunk_id="a", embedding=[1.0, 0.0, 0.0, 0.0], text="corregido")
    store.upsert([updated])

    assert store.count() == 1
    assert store.search([1.0, 0.0, 0.0, 0.0], top_k=1)[0].chunk.text == "corregido"


def test_upsert_of_nothing_is_a_no_op(store: ChromaVectorStore) -> None:
    assert store.upsert([]) == 0
    assert store.count() == 0


def test_reserved_metadata_keys_are_rejected(store: ChromaVectorStore) -> None:
    item = build_embedded(chunk_id="a", embedding=[1.0, 0.0, 0.0, 0.0])

    with pytest.raises(VectorStoreError, match="reservados"):
        store.upsert([item], metadata={"doc_id": "otro.pdf"})


def test_dimension_change_is_detected_on_write(store: ChromaVectorStore) -> None:
    """Cambiar de modelo de embeddings sin reindexar debe fallar, no degradar."""
    store.upsert([build_embedded(chunk_id="a", embedding=[1.0, 0.0, 0.0, 0.0])])

    with pytest.raises(DimensionMismatchError, match="reindexar"):
        store.upsert([build_embedded(chunk_id="b", embedding=[1.0, 0.0])])


def test_mixed_dimensions_in_the_same_batch_are_rejected(store: ChromaVectorStore) -> None:
    with pytest.raises(DimensionMismatchError):
        store.upsert(
            [
                build_embedded(chunk_id="a", embedding=[1.0, 0.0, 0.0, 0.0]),
                build_embedded(chunk_id="b", embedding=[1.0, 0.0]),
            ]
        )


# ---------- Búsqueda ----------


def test_search_ranks_by_cosine_similarity(store: ChromaVectorStore) -> None:
    store.upsert(
        [
            build_embedded(chunk_id="igual", embedding=[1.0, 0.0, 0.0, 0.0], text="idéntico"),
            build_embedded(
                chunk_id="parecido",
                embedding=[0.8, 0.6, 0.0, 0.0],
                text="parecido",
                index=1,
            ),
            build_embedded(
                chunk_id="ortogonal",
                embedding=[0.0, 0.0, 0.0, 1.0],
                text="sin relación",
                index=2,
            ),
        ]
    )

    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=3)

    assert [r.chunk.text for r in results] == ["idéntico", "parecido", "sin relación"]
    assert results[0].score == pytest.approx(1.0, abs=1e-5)
    assert results[2].score == pytest.approx(0.0, abs=1e-5)


def test_search_respects_top_k(store: ChromaVectorStore) -> None:
    store.upsert(
        [
            build_embedded(chunk_id=f"c{i}", embedding=[1.0, float(i) / 10, 0.0, 0.0], index=i)
            for i in range(5)
        ]
    )

    assert len(store.search([1.0, 0.0, 0.0, 0.0], top_k=2)) == 2


def test_search_on_an_empty_index_returns_nothing(store: ChromaVectorStore) -> None:
    assert store.search([1.0, 0.0, 0.0, 0.0], top_k=5) == []


def test_search_rejects_non_positive_top_k(store: ChromaVectorStore) -> None:
    with pytest.raises(VectorStoreError, match="top_k"):
        store.search([1.0, 0.0, 0.0, 0.0], top_k=0)


def test_search_filters_by_document_tag(store: ChromaVectorStore) -> None:
    store.upsert(
        [build_embedded(chunk_id="a", embedding=[1.0, 0.0, 0.0, 0.0], doc_id="contrato.pdf")],
        metadata={"categoria": "contratos_2026"},
    )
    store.upsert(
        [build_embedded(chunk_id="b", embedding=[1.0, 0.0, 0.0, 0.0], doc_id="manual.pdf")],
        metadata={"categoria": "manuales"},
    )

    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=5, filters={"categoria": "contratos_2026"})

    assert [r.chunk.filename for r in results] == ["contrato.pdf"]


def test_search_combines_several_filters(store: ChromaVectorStore) -> None:
    store.upsert(
        [build_embedded(chunk_id="a", embedding=[1.0, 0.0, 0.0, 0.0])],
        metadata={"categoria": "contratos_2026", "departamento": "legal"},
    )
    store.upsert(
        [build_embedded(chunk_id="b", embedding=[1.0, 0.0, 0.0, 0.0], doc_id="otro.pdf")],
        metadata={"categoria": "contratos_2026", "departamento": "compras"},
    )

    results = store.search(
        [1.0, 0.0, 0.0, 0.0],
        top_k=5,
        filters={"categoria": "contratos_2026", "departamento": "legal"},
    )

    assert [r.chunk.filename for r in results] == ["contrato.pdf"]


@pytest.mark.parametrize(
    ("distance", "expected"),
    [
        (-1e-6, 1.0),  # vector idéntico: el ruido de coma flotante saca el score de rango
        (0.0, 1.0),
        (1.0, 0.0),
        (2.0000001, -1.0),  # extremo opuesto, también fuera de rango por redondeo
    ],
)
def test_similarity_stays_within_the_valid_range(distance: float, expected: float) -> None:
    assert ChromaVectorStore.to_similarity(distance) == expected


def test_identical_vector_scores_one_without_overflowing(store: ChromaVectorStore) -> None:
    """Score > 1 rompería la validación de SearchResult; el acotado lo impide."""
    embedding = [0.37, -0.82, 0.11, 0.44]
    store.upsert([build_embedded(chunk_id="a", embedding=embedding)])

    assert store.search(embedding, top_k=1)[0].score == 1.0


# ---------- Trazabilidad ----------


def test_page_and_position_survive_the_round_trip(store: ChromaVectorStore) -> None:
    """La cita del sources[] se reconstruye desde lo almacenado en el índice."""
    store.upsert(
        [
            build_embedded(
                chunk_id="a",
                embedding=[1.0, 0.0, 0.0, 0.0],
                text="el arrendatario deberá notificar con 30 días",
                doc_id="contrato_finca_x.pdf",
                page=12,
                index=7,
            )
        ]
    )

    result = store.search([1.0, 0.0, 0.0, 0.0], top_k=1)[0]

    assert result.chunk.filename == "contrato_finca_x.pdf"
    assert result.chunk.page_number == 12
    assert result.chunk.index == 7
    assert "30 días" in result.chunk.snippet


# ---------- Inventario y borrado ----------


def test_list_documents_aggregates_chunks_and_pages(store: ChromaVectorStore) -> None:
    store.upsert(
        [
            build_embedded(chunk_id="a", embedding=[1.0, 0.0, 0.0, 0.0], page=1, index=0),
            build_embedded(chunk_id="b", embedding=[0.0, 1.0, 0.0, 0.0], page=1, index=1),
            build_embedded(chunk_id="c", embedding=[0.0, 0.0, 1.0, 0.0], page=2, index=2),
            build_embedded(
                chunk_id="d",
                embedding=[0.0, 0.0, 0.0, 1.0],
                doc_id="manual.pdf",
                page=5,
            ),
        ]
    )

    documents = store.list_documents()

    assert [d.doc_id for d in documents] == ["contrato.pdf", "manual.pdf"]
    assert documents[0].chunk_count == 3
    assert documents[0].page_count == 2
    assert documents[1].chunk_count == 1


def test_list_documents_is_empty_when_nothing_is_indexed(store: ChromaVectorStore) -> None:
    assert store.list_documents() == []


def test_delete_document_removes_only_its_chunks(store: ChromaVectorStore) -> None:
    store.upsert(
        [
            build_embedded(chunk_id="a", embedding=[1.0, 0.0, 0.0, 0.0]),
            build_embedded(chunk_id="b", embedding=[0.0, 1.0, 0.0, 0.0], index=1),
            build_embedded(chunk_id="c", embedding=[0.0, 0.0, 1.0, 0.0], doc_id="manual.pdf"),
        ]
    )

    removed = store.delete_document("contrato.pdf")

    assert removed == 2
    assert store.count() == 1
    assert [d.doc_id for d in store.list_documents()] == ["manual.pdf"]


def test_deleting_an_unknown_document_is_harmless(store: ChromaVectorStore) -> None:
    assert store.delete_document("no-existe.pdf") == 0


def test_reset_empties_the_collection(store: ChromaVectorStore) -> None:
    store.upsert([build_embedded(chunk_id="a", embedding=[1.0, 0.0, 0.0, 0.0])])

    store.reset()

    assert store.count() == 0


# ---------- Fábrica ----------


def test_pgvector_backend_is_not_implemented_yet() -> None:
    with pytest.raises(VectorStoreConfigurationError, match="pgvector"):
        build_vector_store("pgvector")


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(VectorStoreConfigurationError, match="desconocido"):
        build_vector_store("pinecone")
