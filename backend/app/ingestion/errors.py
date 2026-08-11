"""Errores del pipeline de ingesta.

Tipados para que la capa de API (rama feature/api-documents-endpoints) pueda
mapearlos a códigos HTTP sin inspeccionar mensajes de texto.
"""


class IngestionError(Exception):
    """Error base de la ingesta."""


class UnsupportedFormatError(IngestionError):
    """La extensión del archivo no tiene extractor registrado."""

    def __init__(self, extension: str, supported: set[str]) -> None:
        self.extension = extension
        self.supported = supported
        super().__init__(
            f"Formato no soportado: '{extension}'. Formatos válidos: {sorted(supported)}"
        )


class CorruptDocumentError(IngestionError):
    """El archivo existe pero no se puede leer (malformado o truncado)."""

    def __init__(self, filename: str, reason: str) -> None:
        self.filename = filename
        self.reason = reason
        super().__init__(f"No se pudo leer '{filename}': {reason}")


class EncryptedDocumentError(IngestionError):
    """El PDF está protegido por contraseña y no se puede abrir."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        super().__init__(f"'{filename}' está cifrado y requiere contraseña")


class EmptyTextLayerError(IngestionError):
    """El documento se abrió pero no contiene texto extraíble.

    Caso típico: un PDF escaneado sin OCR. Se trata aparte de un documento vacío
    porque la acción correctiva es distinta (pasar OCR, no volver a subirlo).
    """

    def __init__(self, filename: str) -> None:
        self.filename = filename
        super().__init__(
            f"'{filename}' no contiene texto extraíble. "
            "Si es un documento escaneado, necesita OCR antes de la ingesta."
        )
