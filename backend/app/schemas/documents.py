"""Contratos HTTP de los endpoints de documentos (sección 4 del documento de diseño)."""

from pydantic import BaseModel, Field

from app.storage.base import DocumentSummary


class IngestResponse(BaseModel):
    """Respuesta de `POST /api/v1/documents/ingest`."""

    doc_id: str
    pages: int = Field(ge=0)
    chunks: int = Field(ge=0)
    tokens: int = Field(ge=0)
    embedding_model: str
    elapsed_ms: float = Field(ge=0)


class DocumentListResponse(BaseModel):
    """Respuesta de `GET /api/v1/documents`."""

    documents: list[DocumentSummary]
    total_documents: int = Field(ge=0)
    total_chunks: int = Field(ge=0)


class DeleteResponse(BaseModel):
    """Respuesta de `DELETE /api/v1/documents/{doc_id}`."""

    doc_id: str
    deleted_chunks: int = Field(ge=0)
