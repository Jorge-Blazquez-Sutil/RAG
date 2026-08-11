"""Registro de extractores y punto de entrada de la extracción.

Uso:
    from app.ingestion.extractors import extract_document
    document = extract_document(Path("contrato.pdf"))

Para añadir un formato: crea el extractor y añádelo a `_EXTRACTORS`.
"""

from pathlib import Path

from app.ingestion.errors import CorruptDocumentError, UnsupportedFormatError
from app.ingestion.extractors.base import DocumentExtractor, normalize_text
from app.ingestion.extractors.docx import DocxExtractor
from app.ingestion.extractors.pdf import PdfExtractor
from app.ingestion.extractors.text import TextExtractor
from app.schemas.document import ExtractedDocument

_EXTRACTORS: tuple[DocumentExtractor, ...] = (
    PdfExtractor(),
    DocxExtractor(),
    TextExtractor(),
)

_BY_EXTENSION: dict[str, DocumentExtractor] = {
    extension: extractor for extractor in _EXTRACTORS for extension in extractor.extensions
}

#: Extensiones aceptadas por el sistema, para validar en la capa de API.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(_BY_EXTENSION)


def get_extractor(path: Path) -> DocumentExtractor:
    """Devuelve el extractor correspondiente a la extensión del archivo.

    Raises:
        UnsupportedFormatError: no hay extractor para esa extensión.
    """
    extension = path.suffix.lower()
    try:
        return _BY_EXTENSION[extension]
    except KeyError as exc:
        raise UnsupportedFormatError(extension, set(SUPPORTED_EXTENSIONS)) from exc


def extract_document(path: Path) -> ExtractedDocument:
    """Extrae texto y metadatos del archivo indicado.

    Raises:
        UnsupportedFormatError: extensión sin extractor.
        CorruptDocumentError: el archivo no existe o no se puede leer.
        EncryptedDocumentError: el archivo está protegido por contraseña.
        EmptyTextLayerError: el archivo no contiene texto extraíble.
    """
    if not path.is_file():
        raise CorruptDocumentError(path.name, "el archivo no existe")
    return get_extractor(path).extract(path)


__all__ = [
    "SUPPORTED_EXTENSIONS",
    "DocumentExtractor",
    "DocxExtractor",
    "PdfExtractor",
    "TextExtractor",
    "extract_document",
    "get_extractor",
    "normalize_text",
]
