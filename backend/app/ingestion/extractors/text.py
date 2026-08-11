"""Extractor de texto plano (.txt, .md)."""

from pathlib import Path

from app.ingestion.errors import CorruptDocumentError, EmptyTextLayerError
from app.ingestion.extractors.base import DocumentExtractor, normalize_text
from app.schemas.document import DocumentMetadata, ExtractedDocument, PageContent

#: Se intentan en orden. latin-1 nunca falla, así que actúa de red de seguridad
#: para los archivos exportados desde Windows sin declarar codificación.
_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


class TextExtractor(DocumentExtractor):
    """Lee el archivo completo como una sola página."""

    extensions = frozenset({".txt", ".md"})

    def extract(self, path: Path) -> ExtractedDocument:
        raw = self._read(path)
        text = normalize_text(raw)

        document = ExtractedDocument(
            metadata=DocumentMetadata(
                filename=path.name,
                extension=path.suffix.lower(),
                page_count=1,
                char_count=len(text),
            ),
            pages=[PageContent(page_number=1, text=text)],
        )
        if document.is_empty:
            raise EmptyTextLayerError(path.name)
        return document

    def _read(self, path: Path) -> str:
        for encoding in _ENCODINGS:
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
            except OSError as exc:
                raise CorruptDocumentError(path.name, str(exc)) from exc
        raise CorruptDocumentError(path.name, "no se pudo determinar la codificación")
