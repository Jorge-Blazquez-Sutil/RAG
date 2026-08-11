import pytest

from app.ingestion import chunker
from app.ingestion.chunker import (
    ChunkingError,
    chunk_document,
    count_tokens,
    estimate_tokens,
    is_exact_counting_available,
)
from app.schemas.document import DocumentMetadata, ExtractedDocument, PageContent


def build_document(pages: list[str], *, filename: str = "contrato.pdf") -> ExtractedDocument:
    """Construye un ExtractedDocument sintético, sin pasar por la extracción."""
    contents = [PageContent(page_number=n, text=t) for n, t in enumerate(pages, start=1)]
    return ExtractedDocument(
        metadata=DocumentMetadata(
            filename=filename,
            extension=".pdf",
            page_count=len(contents),
            char_count=sum(len(t) for t in pages),
        ),
        pages=contents,
    )


def words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(count))


# ---------- Tamaño y solapamiento ----------


def test_short_page_produces_a_single_chunk() -> None:
    document = build_document(["La rescisión requiere un preaviso de 30 días."])

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].page_number == 1
    assert chunks[0].text == "La rescisión requiere un preaviso de 30 días."


def test_no_chunk_exceeds_the_configured_size() -> None:
    document = build_document([words("palabra", 2000)])

    chunks = chunk_document(document, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(chunk.token_count <= 100 for chunk in chunks)


def test_consecutive_chunks_share_content() -> None:
    """El solapamiento evita perder el contexto en la frontera entre fragmentos."""
    document = build_document([words("token", 600)])

    chunks = chunk_document(document, chunk_size=100, chunk_overlap=30)

    shared = set(chunks[0].text.split()) & set(chunks[1].text.split())
    assert shared, "los fragmentos consecutivos deberían solaparse"


def test_zero_overlap_produces_disjoint_chunks() -> None:
    document = build_document([words("token", 600)])

    chunks = chunk_document(document, chunk_size=100, chunk_overlap=0)

    assert not (set(chunks[0].text.split()) & set(chunks[1].text.split()))


# ---------- Trazabilidad por página ----------


def test_chunks_never_span_two_pages() -> None:
    document = build_document([words("alfa", 400), words("beta", 400)])

    chunks = chunk_document(document, chunk_size=100, chunk_overlap=20)

    for chunk in chunks:
        tokens = chunk.text.split()
        expected = "alfa" if chunk.page_number == 1 else "beta"
        assert all(token.startswith(expected) for token in tokens)


def test_page_numbers_are_preserved_across_pages() -> None:
    document = build_document(["Primera página", "Segunda página", "Tercera página"])

    chunks = chunk_document(document)

    assert [chunk.page_number for chunk in chunks] == [1, 2, 3]


def test_empty_pages_are_skipped_without_breaking_numbering() -> None:
    document = build_document(["Con texto", "   ", "También con texto"])

    chunks = chunk_document(document)

    assert [chunk.page_number for chunk in chunks] == [1, 3]


def test_indices_are_sequential_across_the_whole_document() -> None:
    document = build_document([words("alfa", 300), words("beta", 300)])

    chunks = chunk_document(document, chunk_size=100, chunk_overlap=10)

    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


# ---------- Identificadores ----------


def test_chunk_ids_are_deterministic_between_runs() -> None:
    document = build_document([words("palabra", 300)])

    first = chunk_document(document, chunk_size=100, chunk_overlap=20)
    second = chunk_document(document, chunk_size=100, chunk_overlap=20)

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_chunk_ids_are_unique_within_a_document() -> None:
    document = build_document([words("palabra", 500)])

    chunks = chunk_document(document, chunk_size=100, chunk_overlap=20)

    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_chunk_id_changes_when_the_content_changes() -> None:
    original = chunk_document(build_document(["Preaviso de 30 días"]))
    edited = chunk_document(build_document(["Preaviso de 60 días"]))

    assert original[0].chunk_id != edited[0].chunk_id


def test_chunk_id_changes_with_the_source_document() -> None:
    same_text = ["Cláusula idéntica"]
    a = chunk_document(build_document(same_text, filename="a.pdf"))
    b = chunk_document(build_document(same_text, filename="b.pdf"))

    assert a[0].chunk_id != b[0].chunk_id


# ---------- Validación de parámetros ----------


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(100, 100), (100, 150), (0, 0), (100, -1)],
)
def test_invalid_parameters_raise_chunking_error(size: int, overlap: int) -> None:
    document = build_document(["Texto"])

    with pytest.raises(ChunkingError):
        chunk_document(document, chunk_size=size, chunk_overlap=overlap)


def test_defaults_come_from_settings() -> None:
    from app.core.config import get_settings

    settings = get_settings()
    document = build_document([words("palabra", 5000)])

    chunks = chunk_document(document)

    assert all(chunk.token_count <= settings.chunk_size for chunk in chunks)


# ---------- Conteo de tokens ----------


def test_count_tokens_reflects_text_length() -> None:
    assert count_tokens("") == 0
    assert count_tokens("una frase corta") < count_tokens(words("palabra", 100))


def test_estimation_is_used_when_the_exact_tokenizer_is_unavailable(monkeypatch) -> None:
    """Sin red (o detrás de un proxy TLS) la segmentación debe seguir funcionando."""
    monkeypatch.setattr(chunker, "_load_encoding", lambda: None)

    assert not is_exact_counting_available()
    assert count_tokens("una frase") == estimate_tokens("una frase")

    chunks = chunk_document(
        build_document([words("palabra", 400)]), chunk_size=100, chunk_overlap=20
    )
    assert len(chunks) > 1
    assert all(chunk.token_count <= 100 for chunk in chunks)


def test_estimation_never_underestimates_short_spanish_text(monkeypatch) -> None:
    """La estimación sobrestima a propósito: así el chunk cabe en el modelo real."""
    text = "La rescisión anticipada requiere un preaviso de treinta días naturales."
    monkeypatch.setattr(chunker, "_load_encoding", lambda: None)

    assert estimate_tokens(text) >= len(text.split())


def test_tokenizer_mode_tiktoken_fails_loudly(monkeypatch) -> None:
    """Con TOKENIZER=tiktoken un despliegue sin el tokenizador debe romper, no degradar."""
    settings = chunker.get_settings()
    monkeypatch.setattr(settings, "tokenizer", "tiktoken")
    monkeypatch.setattr(chunker, "_load_encoding", chunker._load_encoding.__wrapped__)  # sin caché
    monkeypatch.setitem(__import__("sys").modules, "tiktoken", None)

    with pytest.raises(ChunkingError, match="tiktoken"):
        count_tokens("texto")


def test_snippet_collapses_whitespace_and_truncates() -> None:
    document = build_document(["  línea uno\n\n   línea dos  "])

    snippet = chunk_document(document)[0].snippet

    assert snippet == "línea uno línea dos"


def test_snippet_is_capped_at_200_characters() -> None:
    document = build_document([words("palabra", 200)])

    snippet = chunk_document(document, chunk_size=500, chunk_overlap=0)[0].snippet

    assert len(snippet) == 200
    assert snippet.endswith("...")
