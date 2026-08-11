"""Pipeline de ingesta completo (módulo 3.1 → 3.2).

Encadena los cuatro pasos que ya existen —extraer, trocear, vectorizar, indexar—
en una sola operación, y devuelve el parte de lo ocurrido.

Vive aquí y no en la capa HTTP a propósito: la ingesta por lotes, la reindexación
del backoffice y los tests necesitan lo mismo sin levantar una petición.
"""

from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.ingestion.chunker import chunk_document
from app.ingestion.embeddings.base import EmbeddingProvider
from app.ingestion.extractors import extract_document
from app.storage.base import Filters, VectorStore

logger = get_logger(__name__)


class IngestionReport(BaseModel):
    """Resultado de ingerir un documento."""

    doc_id: str
    pages: int = Field(ge=0)
    chunks: int = Field(ge=0)
    tokens: int = Field(ge=0)
    model: str
    elapsed_ms: float = Field(ge=0)


def ingest_document(
    path: Path,
    *,
    provider: EmbeddingProvider,
    store: VectorStore,
    metadata: Filters | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> IngestionReport:
    """Extrae, trocea, vectoriza e indexa el documento.

    Raises:
        IngestionError: cualquier fallo de extracción o segmentación.
        EmbeddingError: fallo al vectorizar.
        VectorStoreError: fallo al escribir en el índice.
    """
    started = perf_counter()

    document = extract_document(path)
    chunks = chunk_document(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embedded = provider.embed_chunks(chunks)
    store.upsert(embedded, metadata=metadata)

    report = IngestionReport(
        doc_id=document.metadata.filename,
        pages=document.metadata.page_count,
        chunks=len(chunks),
        tokens=sum(chunk.token_count for chunk in chunks),
        model=provider.model,
        elapsed_ms=(perf_counter() - started) * 1000,
    )
    logger.info(
        "Ingerido '%s': %d páginas, %d fragmentos, %d tokens en %.0f ms",
        report.doc_id,
        report.pages,
        report.chunks,
        report.tokens,
        report.elapsed_ms,
    )
    return report
