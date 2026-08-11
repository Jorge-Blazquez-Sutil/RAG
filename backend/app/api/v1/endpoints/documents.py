"""Endpoints de gestión de documentos (sección 4 del documento de diseño).

POST   /api/v1/documents/ingest    sube, procesa e indexa un documento
GET    /api/v1/documents           lista lo indexado
DELETE /api/v1/documents/{doc_id}  elimina un documento del índice
"""

import json
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.api.deps import EmbeddingProviderDep, SettingsDep, VectorStoreDep
from app.core.config import Settings
from app.core.logging import get_logger
from app.ingestion.embeddings.errors import (
    EmbeddingConfigurationError,
    EmbeddingProviderError,
)
from app.ingestion.errors import (
    CorruptDocumentError,
    EmptyTextLayerError,
    EncryptedDocumentError,
    UnsupportedFormatError,
)
from app.ingestion.extractors import SUPPORTED_EXTENSIONS
from app.ingestion.pipeline import ingest_document
from app.schemas.documents import DeleteResponse, DocumentListResponse, IngestResponse
from app.storage.base import Filters
from app.storage.errors import VectorStoreError

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

#: Tamaño de lectura del archivo subido. Se lee a trozos para no cargar en
#: memoria un documento de decenas de megas mientras se comprueba su tamaño.
CHUNK_BYTES = 1024 * 1024


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Subir e indexar un documento",
)
async def ingest(
    settings: SettingsDep,
    provider: EmbeddingProviderDep,
    store: VectorStoreDep,
    file: Annotated[UploadFile, File(description="PDF, DOCX o TXT")],
    metadata: Annotated[
        str | None,
        Form(description='Etiquetas del documento en JSON: {"categoria": "contratos_2026"}'),
    ] = None,
) -> IngestResponse:
    filename = _safe_filename(file.filename)
    _check_extension(filename, settings)
    tags = _parse_metadata(metadata)

    destino = settings.upload_dir / filename
    await _save_upload(file, destino, max_mb=settings.max_upload_mb)

    try:
        report = await run_in_threadpool(
            ingest_document,
            destino,
            provider=provider,
            store=store,
            metadata=tags,
        )
    except (UnsupportedFormatError, EncryptedDocumentError, EmptyTextLayerError) as exc:
        # El archivo llegó bien pero no se puede aprovechar: es cosa del contenido,
        # no de la petición, así que 422 y el motivo tal cual para el usuario.
        destino.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except CorruptDocumentError as exc:
        destino.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except EmbeddingConfigurationError as exc:
        # Falta configuración del servidor: el cliente no puede hacer nada.
        logger.error("Ingesta abortada por configuración: %s", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except EmbeddingProviderError as exc:
        logger.error("Fallo del proveedor de embeddings: %s", exc)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    except VectorStoreError as exc:
        logger.error("Fallo del índice vectorial: %s", exc)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc)) from exc

    return IngestResponse(
        doc_id=report.doc_id,
        pages=report.pages,
        chunks=report.chunks,
        tokens=report.tokens,
        embedding_model=report.model,
        elapsed_ms=report.elapsed_ms,
    )


@router.get("", response_model=DocumentListResponse, summary="Listar documentos indexados")
async def list_documents(store: VectorStoreDep) -> DocumentListResponse:
    documents = await run_in_threadpool(store.list_documents)
    return DocumentListResponse(
        documents=documents,
        total_documents=len(documents),
        total_chunks=sum(doc.chunk_count for doc in documents),
    )


@router.delete(
    "/{doc_id}",
    response_model=DeleteResponse,
    summary="Eliminar un documento del índice",
)
async def delete_document(
    doc_id: str, settings: SettingsDep, store: VectorStoreDep
) -> DeleteResponse:
    safe = _safe_filename(doc_id)
    deleted = await run_in_threadpool(store.delete_document, safe)
    if deleted == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No hay ningún documento '{safe}' indexado")

    # El original se guarda para que el panel de fuentes pueda abrir el PDF:
    # al desindexar deja de tener sentido conservarlo.
    (settings.upload_dir / safe).unlink(missing_ok=True)
    return DeleteResponse(doc_id=safe, deleted_chunks=deleted)


# ---------- Utilidades ----------


def _safe_filename(name: str | None) -> str:
    """Devuelve el nombre sin ruta.

    `filename` lo controla quien sube el archivo: sin esto, un nombre como
    `../../.env` escribiría fuera del directorio de subidas.
    """
    candidate = Path(name or "").name.strip()
    if not candidate or candidate in {".", ".."}:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Nombre de archivo inválido")
    return candidate


def _check_extension(filename: str, settings: Settings) -> None:
    extension = Path(filename).suffix.lower()
    permitidas = settings.allowed_extensions & SUPPORTED_EXTENSIONS
    if extension not in permitidas:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Formato no soportado: '{extension}'. Formatos válidos: {sorted(permitidas)}",
        )


def _parse_metadata(raw: str | None) -> Filters:
    """Valida las etiquetas del documento.

    Solo escalares: los motores vectoriales no almacenan listas ni objetos
    anidados, y descubrirlo al escribir en el índice daría un error mucho más
    oscuro que este.
    """
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"El campo 'metadata' no es JSON válido: {exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "El campo 'metadata' debe ser un objeto JSON",
        )

    for key, value in parsed.items():
        if not isinstance(value, str | int | float | bool):
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"La etiqueta '{key}' debe ser texto, número o booleano",
            )
    return parsed


async def _save_upload(file: UploadFile, destino: Path, *, max_mb: int) -> None:
    """Guarda el archivo comprobando el tamaño mientras se escribe.

    El límite se vigila a medida que llegan los datos: confiar en la cabecera
    `content-length` dejaría pasar cualquier subida que mienta.
    """
    limite = max_mb * 1024 * 1024
    escrito = 0

    destino.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destino.open("wb") as salida:
            while trozo := await file.read(CHUNK_BYTES):
                escrito += len(trozo)
                if escrito > limite:
                    raise HTTPException(
                        status.HTTP_413_CONTENT_TOO_LARGE,
                        f"El archivo supera el límite de {max_mb} MB",
                    )
                salida.write(trozo)
    except HTTPException:
        destino.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    if escrito == 0:
        destino.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "El archivo está vacío")
