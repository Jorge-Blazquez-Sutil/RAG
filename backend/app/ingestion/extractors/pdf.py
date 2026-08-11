"""Extractor de PDF basado en pypdf (licencia BSD)."""

from datetime import datetime
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.ingestion.errors import (
    CorruptDocumentError,
    EmptyTextLayerError,
    EncryptedDocumentError,
)
from app.ingestion.extractors.base import DocumentExtractor, normalize_text
from app.schemas.document import DocumentMetadata, ExtractedDocument, PageContent


class PdfExtractor(DocumentExtractor):
    """Extrae texto página a página conservando la numeración original."""

    extensions = frozenset({".pdf"})

    def extract(self, path: Path) -> ExtractedDocument:
        reader = self._open(path)
        pages = [
            PageContent(page_number=number, text=self._page_text(page, path, number))
            for number, page in enumerate(reader.pages, start=1)
        ]

        document = ExtractedDocument(
            metadata=self._metadata(reader, path, pages),
            pages=pages,
        )
        if document.is_empty:
            raise EmptyTextLayerError(path.name)
        return document

    def _open(self, path: Path) -> PdfReader:
        try:
            reader = PdfReader(path)
        except PdfReadError as exc:
            raise CorruptDocumentError(path.name, str(exc)) from exc

        if reader.is_encrypted:
            # Muchos PDFs "protegidos" solo restringen la impresión y se abren con
            # contraseña vacía; si eso falla, la contraseña es real y no la tenemos.
            try:
                opened = reader.decrypt("")
            except (PdfReadError, NotImplementedError) as exc:
                raise EncryptedDocumentError(path.name) from exc
            if not opened:
                raise EncryptedDocumentError(path.name)

        return reader

    def _page_text(self, page: object, path: Path, number: int) -> str:
        try:
            raw = page.extract_text() or ""  # type: ignore[attr-defined]
        except (PdfReadError, ValueError, KeyError) as exc:
            raise CorruptDocumentError(path.name, f"página {number}: {exc}") from exc
        return normalize_text(raw)

    def _metadata(
        self, reader: PdfReader, path: Path, pages: list[PageContent]
    ) -> DocumentMetadata:
        info = reader.metadata
        return DocumentMetadata(
            filename=path.name,
            extension=path.suffix.lower(),
            title=_clean(info.title) if info else None,
            author=_clean(info.author) if info else None,
            created_at=_creation_date(info),
            page_count=len(pages),
            char_count=sum(len(page.text) for page in pages),
        )


def _clean(value: str | None) -> str | None:
    """Descarta los metadatos vacíos que muchos generadores de PDF dejan puestos."""
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _creation_date(info: object | None) -> datetime | None:
    """La fecha del PDF es texto libre: si viene malformada, se ignora."""
    if info is None:
        return None
    try:
        return info.creation_date  # type: ignore[attr-defined]
    except (ValueError, TypeError, KeyError):
        return None
