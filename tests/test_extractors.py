from pathlib import Path

import pytest

from app.ingestion.errors import (
    CorruptDocumentError,
    EmptyTextLayerError,
    EncryptedDocumentError,
    UnsupportedFormatError,
)
from app.ingestion.extractors import (
    SUPPORTED_EXTENSIONS,
    DocxExtractor,
    PdfExtractor,
    TextExtractor,
    extract_document,
    get_extractor,
    normalize_text,
)

# ---------- PDF ----------


def test_pdf_keeps_one_entry_per_page_with_original_numbering(make_pdf) -> None:
    path = make_pdf(["Cláusula primera", "Cláusula segunda", "Cláusula tercera"])

    document = extract_document(path)

    assert [page.page_number for page in document.pages] == [1, 2, 3]
    assert "Cláusula segunda" in document.pages[1].text
    assert document.metadata.page_count == 3


def test_pdf_reads_title_and_author_metadata(make_pdf) -> None:
    path = make_pdf(["Texto"], title="Contrato de arrendamiento", author="Jorge")

    metadata = extract_document(path).metadata

    assert metadata.title == "Contrato de arrendamiento"
    assert metadata.author == "Jorge"
    assert metadata.filename == "documento.pdf"
    assert metadata.extension == ".pdf"
    assert metadata.char_count > 0


def test_pdf_without_text_layer_raises_empty_text_layer(make_pdf) -> None:
    """Un PDF escaneado se abre sin error pero no aporta texto: hay que distinguirlo."""
    path = make_pdf(["", ""], name="escaneado.pdf")

    with pytest.raises(EmptyTextLayerError) as excinfo:
        extract_document(path)

    assert "OCR" in str(excinfo.value)


def test_encrypted_pdf_raises_encrypted_document(make_encrypted_pdf) -> None:
    path = make_encrypted_pdf("secreto")

    with pytest.raises(EncryptedDocumentError):
        extract_document(path)


def test_corrupt_pdf_raises_corrupt_document(tmp_path: Path) -> None:
    path = tmp_path / "roto.pdf"
    path.write_bytes(b"%PDF-1.7 esto no es un PDF valido")

    with pytest.raises(CorruptDocumentError):
        extract_document(path)


# ---------- DOCX ----------


def test_docx_joins_paragraphs_into_a_single_page(make_docx) -> None:
    path = make_docx(["Primer párrafo", "Segundo párrafo"])

    document = extract_document(path)

    assert document.metadata.page_count == 1
    assert document.pages[0].page_number == 1
    assert document.pages[0].text == "Primer párrafo\nSegundo párrafo"


def test_docx_reads_core_properties(make_docx) -> None:
    path = make_docx(["Contenido"], title="Manual técnico", author="Jorge")

    metadata = extract_document(path).metadata

    assert metadata.title == "Manual técnico"
    assert metadata.author == "Jorge"


def test_empty_docx_raises_empty_text_layer(make_docx) -> None:
    path = make_docx(["", "   "])

    with pytest.raises(EmptyTextLayerError):
        extract_document(path)


# ---------- Texto plano ----------


def test_txt_is_extracted_as_one_page(tmp_path: Path) -> None:
    path = tmp_path / "notas.txt"
    path.write_text("Preaviso de 30 días.", encoding="utf-8")

    document = extract_document(path)

    assert document.pages[0].text == "Preaviso de 30 días."
    assert document.metadata.extension == ".txt"


def test_txt_falls_back_to_windows_encoding(tmp_path: Path) -> None:
    path = tmp_path / "legacy.txt"
    path.write_bytes("Rescisión anticipada".encode("cp1252"))

    document = extract_document(path)

    assert "Rescisión anticipada" in document.pages[0].text


def test_markdown_uses_the_text_extractor(tmp_path: Path) -> None:
    path = tmp_path / "readme.md"
    path.write_text("# Título\n\nCuerpo", encoding="utf-8")

    assert isinstance(get_extractor(path), TextExtractor)
    assert "Cuerpo" in extract_document(path).pages[0].text


# ---------- Registro y errores comunes ----------


@pytest.mark.parametrize(
    ("filename", "expected"),
    [("a.pdf", PdfExtractor), ("a.docx", DocxExtractor), ("a.txt", TextExtractor)],
)
def test_registry_dispatches_by_extension(filename: str, expected: type) -> None:
    assert isinstance(get_extractor(Path(filename)), expected)


def test_extension_matching_is_case_insensitive() -> None:
    assert isinstance(get_extractor(Path("CONTRATO.PDF")), PdfExtractor)


def test_unsupported_extension_lists_the_valid_ones(tmp_path: Path) -> None:
    path = tmp_path / "hoja.xlsx"
    path.write_bytes(b"contenido")

    with pytest.raises(UnsupportedFormatError) as excinfo:
        extract_document(path)

    assert excinfo.value.extension == ".xlsx"
    assert ".pdf" in str(excinfo.value)


def test_missing_file_raises_corrupt_document(tmp_path: Path) -> None:
    with pytest.raises(CorruptDocumentError):
        extract_document(tmp_path / "no-existe.pdf")


def test_supported_extensions_matches_the_functional_requirement() -> None:
    assert {".pdf", ".docx", ".txt"} <= SUPPORTED_EXTENSIONS


# ---------- Normalización ----------


def test_normalize_text_collapses_blank_lines_and_trailing_spaces() -> None:
    assert normalize_text("  hola   \r\n\n\n\n  mundo  \n") == "hola\n\n  mundo"


def test_normalize_text_preserves_paragraph_separation() -> None:
    assert normalize_text("párrafo uno\n\npárrafo dos") == "párrafo uno\n\npárrafo dos"


# ---------- Contrato del documento ----------


def test_full_text_concatenates_pages_in_order(make_pdf) -> None:
    path = make_pdf(["Uno", "Dos"])

    document = extract_document(path)

    assert document.full_text == "Uno\n\nDos"
    assert not document.is_empty
