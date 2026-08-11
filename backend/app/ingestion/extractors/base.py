"""Contrato común de los extractores de texto."""

import re
from abc import ABC, abstractmethod
from pathlib import Path

from app.schemas.document import ExtractedDocument

_TRAILING_SPACES = re.compile(r"[ \t]+$", flags=re.MULTILINE)
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Limpieza mínima y no destructiva del texto extraído.

    Unifica saltos de línea, quita espacios al final de cada línea y colapsa
    las secuencias de líneas en blanco. No toca el contenido: el chunker
    necesita los párrafos intactos para respetar el contexto.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_SPACES.sub("", text)
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    return text.strip()


class DocumentExtractor(ABC):
    """Extrae texto paginado y metadatos de un formato concreto.

    Cada implementación declara las extensiones que atiende; el registro de
    `extractors/__init__.py` hace el dispatch.
    """

    #: Extensiones en minúsculas y con punto, p. ej. {".pdf"}
    extensions: frozenset[str] = frozenset()

    @abstractmethod
    def extract(self, path: Path) -> ExtractedDocument:
        """Devuelve el documento extraído.

        Raises:
            CorruptDocumentError: el archivo no se puede leer.
            EncryptedDocumentError: el archivo está protegido.
            EmptyTextLayerError: no hay texto extraíble.
        """
