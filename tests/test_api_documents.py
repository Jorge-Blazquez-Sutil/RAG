import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from app.api.deps import embedding_provider_dep, settings_dep, vector_store_dep
from app.core.config import Settings, get_settings
from app.ingestion.embeddings import DeterministicEmbeddingProvider
from app.main import create_app
from app.storage import ChromaVectorStore

INGEST = "/api/v1/documents/ingest"
LIST = "/api/v1/documents"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Configuración aislada: subidas e índice viven bajo tmp_path."""
    base = get_settings().model_copy(deep=True)
    base.upload_dir = tmp_path / "uploads"
    base.chroma_path = tmp_path / "chroma"
    base.max_upload_mb = 1
    return base


@pytest.fixture
def store(settings: Settings) -> ChromaVectorStore:
    return ChromaVectorStore(path=settings.chroma_path, collection_name="apidocs")


@pytest.fixture
def client(settings: Settings, store: ChromaVectorStore) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[settings_dep] = lambda: settings
    app.dependency_overrides[vector_store_dep] = lambda: store
    app.dependency_overrides[embedding_provider_dep] = lambda: DeterministicEmbeddingProvider(
        dimensions=32
    )
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def pdf_bytes(tmp_path: Path) -> bytes:
    path = tmp_path / "fuente.pdf"
    pdf = canvas.Canvas(str(path))
    for texto in ["La rescision requiere preaviso de 30 dias", "La renta se abona por mes"]:
        pdf.drawString(72, 750, texto)
        pdf.showPage()
    pdf.save()
    return path.read_bytes()


def upload(client: TestClient, content: bytes, *, name: str = "contrato.pdf", metadata=None):
    data = {"metadata": json.dumps(metadata)} if metadata is not None else None
    return client.post(INGEST, files={"file": (name, content, "application/pdf")}, data=data)


# ---------- Ingesta ----------


def test_ingest_indexes_the_document(client: TestClient, pdf_bytes: bytes) -> None:
    response = upload(client, pdf_bytes)

    assert response.status_code == 201
    body = response.json()
    assert body["doc_id"] == "contrato.pdf"
    assert body["pages"] == 2
    assert body["chunks"] >= 2
    assert body["tokens"] > 0
    assert body["elapsed_ms"] > 0


def test_ingested_document_is_searchable(
    client: TestClient, pdf_bytes: bytes, store: ChromaVectorStore
) -> None:
    upload(client, pdf_bytes)

    assert store.count() >= 2


def test_the_original_file_is_kept_for_the_sources_panel(
    client: TestClient, pdf_bytes: bytes, settings: Settings
) -> None:
    upload(client, pdf_bytes)

    assert (settings.upload_dir / "contrato.pdf").exists()


def test_reingesting_the_same_document_does_not_duplicate(
    client: TestClient, pdf_bytes: bytes, store: ChromaVectorStore
) -> None:
    upload(client, pdf_bytes)
    primero = store.count()

    upload(client, pdf_bytes)

    assert store.count() == primero


def test_metadata_tags_are_stored_and_filterable(
    client: TestClient, pdf_bytes: bytes, store: ChromaVectorStore
) -> None:
    upload(client, pdf_bytes, metadata={"categoria": "contratos_2026"})

    vector = DeterministicEmbeddingProvider(dimensions=32).embed_query("preaviso")
    encontrados = store.search(vector, top_k=5, filters={"categoria": "contratos_2026"})

    assert encontrados


# ---------- Validación de la subida ----------


def test_path_traversal_in_the_filename_is_neutralised(
    client: TestClient, pdf_bytes: bytes, settings: Settings
) -> None:
    """El nombre lo controla quien sube: sin sanear escribiría fuera del directorio."""
    response = upload(client, pdf_bytes, name="../../secreto.pdf")

    assert response.status_code == 201
    assert response.json()["doc_id"] == "secreto.pdf"
    assert (settings.upload_dir / "secreto.pdf").exists()
    assert not (settings.upload_dir.parent.parent / "secreto.pdf").exists()


def test_unsupported_extension_is_rejected(client: TestClient) -> None:
    response = client.post(
        INGEST, files={"file": ("hoja.xlsx", b"contenido", "application/vnd.ms-excel")}
    )

    assert response.status_code == 415
    assert ".pdf" in response.json()["detail"]


def test_empty_file_is_rejected(client: TestClient) -> None:
    response = upload(client, b"")

    assert response.status_code == 400


def test_oversized_file_is_rejected(client: TestClient, settings: Settings) -> None:
    demasiado = b"x" * (settings.max_upload_mb * 1024 * 1024 + 1)

    response = upload(client, demasiado)

    assert response.status_code == 413
    assert not (settings.upload_dir / "contrato.pdf").exists()


def test_a_corrupt_pdf_is_rejected_and_not_kept(client: TestClient, settings: Settings) -> None:
    response = upload(client, b"%PDF-1.7 esto no es un PDF")

    assert response.status_code == 400
    assert not (settings.upload_dir / "contrato.pdf").exists()


def test_a_pdf_without_text_layer_explains_it_needs_ocr(client: TestClient, tmp_path: Path) -> None:
    vacio = tmp_path / "escaneado.pdf"
    pdf = canvas.Canvas(str(vacio))
    pdf.showPage()
    pdf.save()

    response = upload(client, vacio.read_bytes(), name="escaneado.pdf")

    assert response.status_code == 422
    assert "OCR" in response.json()["detail"]


@pytest.mark.parametrize(
    "metadata_raw",
    ["{no es json}", '["lista"]', '{"tags": ["a", "b"]}'],
)
def test_invalid_metadata_is_rejected(
    client: TestClient, pdf_bytes: bytes, metadata_raw: str
) -> None:
    response = client.post(
        INGEST,
        files={"file": ("contrato.pdf", pdf_bytes, "application/pdf")},
        data={"metadata": metadata_raw},
    )

    assert response.status_code == 422


# ---------- Listado ----------


def test_listing_is_empty_at_the_start(client: TestClient) -> None:
    response = client.get(LIST)

    assert response.status_code == 200
    assert response.json() == {"documents": [], "total_documents": 0, "total_chunks": 0}


def test_listing_aggregates_documents_and_chunks(client: TestClient, pdf_bytes: bytes) -> None:
    upload(client, pdf_bytes, name="contrato.pdf")
    upload(client, pdf_bytes, name="manual.pdf")

    body = client.get(LIST).json()

    assert body["total_documents"] == 2
    assert [d["doc_id"] for d in body["documents"]] == ["contrato.pdf", "manual.pdf"]
    assert body["documents"][0]["page_count"] == 2
    assert body["total_chunks"] == sum(d["chunk_count"] for d in body["documents"])


# ---------- Borrado ----------


def test_deleting_removes_chunks_and_the_stored_file(
    client: TestClient, pdf_bytes: bytes, settings: Settings, store: ChromaVectorStore
) -> None:
    upload(client, pdf_bytes)

    response = client.delete(f"{LIST}/contrato.pdf")

    assert response.status_code == 200
    assert response.json()["deleted_chunks"] > 0
    assert store.count() == 0
    assert not (settings.upload_dir / "contrato.pdf").exists()


def test_deleting_an_unknown_document_is_a_404(client: TestClient) -> None:
    assert client.delete(f"{LIST}/no-existe.pdf").status_code == 404


def test_deleting_only_affects_the_named_document(
    client: TestClient, pdf_bytes: bytes, store: ChromaVectorStore
) -> None:
    upload(client, pdf_bytes, name="contrato.pdf")
    upload(client, pdf_bytes, name="manual.pdf")

    client.delete(f"{LIST}/contrato.pdf")

    assert [d.doc_id for d in store.list_documents()] == ["manual.pdf"]


# ---------- Documentación ----------


def test_endpoints_are_published_in_the_openapi_schema(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    assert "/api/v1/documents/ingest" in paths
    assert "/api/v1/documents" in paths
    assert "/api/v1/documents/{doc_id}" in paths
