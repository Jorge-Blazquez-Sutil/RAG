"""Fábricas de documentos de prueba.

Los archivos se generan en tiempo de test dentro de `tmp_path`: así no hay
binarios versionados en el repositorio y cada test declara exactamente el
contenido que necesita.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from docx import Document
from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

PdfFactory = Callable[..., Path]
DocxFactory = Callable[..., Path]


@pytest.fixture
def make_pdf(tmp_path: Path) -> PdfFactory:
    """Crea un PDF con una línea de texto por página."""

    def _make(
        pages: Sequence[str],
        *,
        name: str = "documento.pdf",
        title: str | None = None,
        author: str | None = None,
    ) -> Path:
        path = tmp_path / name
        pdf = canvas.Canvas(str(path), pagesize=A4)
        if title:
            pdf.setTitle(title)
        if author:
            pdf.setAuthor(author)
        for text in pages:
            if text:
                pdf.drawString(72, 750, text)
            pdf.showPage()
        pdf.save()
        return path

    return _make


@pytest.fixture
def make_encrypted_pdf(make_pdf: PdfFactory, tmp_path: Path) -> Callable[[str], Path]:
    """Crea un PDF protegido con contraseña de usuario."""

    def _make(password: str = "secreto") -> Path:
        source = make_pdf(["Contenido confidencial"], name="claro.pdf")
        writer = PdfWriter()
        for page in PdfReader(source).pages:
            writer.add_page(page)
        writer.encrypt(password)
        path = tmp_path / "cifrado.pdf"
        with path.open("wb") as handle:
            writer.write(handle)
        return path

    return _make


@pytest.fixture
def make_docx(tmp_path: Path) -> DocxFactory:
    """Crea un DOCX con los párrafos indicados."""

    def _make(
        paragraphs: Sequence[str],
        *,
        name: str = "documento.docx",
        title: str | None = None,
        author: str | None = None,
    ) -> Path:
        document = Document()
        for paragraph in paragraphs:
            document.add_paragraph(paragraph)
        if title:
            document.core_properties.title = title
        if author:
            document.core_properties.author = author
        path = tmp_path / name
        document.save(str(path))
        return path

    return _make
