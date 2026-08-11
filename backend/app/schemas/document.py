"""Modelo de dominio del documento extraído.

Es la frontera entre la extracción (paso 1 del módulo 3.1) y el chunking (paso 2).
La granularidad de página es deliberada: el número de página viaja con el texto
hasta el `sources[]` de la respuesta, que es lo que permite la trazabilidad exigida
en el requisito funcional de citas.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class PageContent(BaseModel):
    """Texto de una página del documento original."""

    page_number: int = Field(ge=1, description="Número de página, empezando en 1")
    text: str

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class DocumentMetadata(BaseModel):
    """Metadatos del documento de origen (sección 3.2: página, autor, fecha)."""

    filename: str
    extension: str
    title: str | None = None
    author: str | None = None
    created_at: datetime | None = None
    page_count: int = Field(ge=0)
    char_count: int = Field(ge=0)


class ExtractedDocument(BaseModel):
    """Resultado de la extracción: metadatos + texto paginado."""

    metadata: DocumentMetadata
    pages: list[PageContent]

    @property
    def full_text(self) -> str:
        """Texto completo con las páginas separadas por línea en blanco."""
        return "\n\n".join(page.text for page in self.pages if page.text)

    @property
    def is_empty(self) -> bool:
        """True si ninguna página aportó texto (p. ej. un PDF escaneado)."""
        return all(page.is_empty for page in self.pages)
