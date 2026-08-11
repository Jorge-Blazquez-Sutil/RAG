import json

import httpx
import pytest
from api_client import RagApiClient, RagApiError

BASE = "http://backend.test/api/v1"


def build_client(handler) -> RagApiClient:
    """Cliente apuntando a un transporte simulado: sin red, sin backend."""
    transport = httpx.MockTransport(handler)
    return RagApiClient(BASE, client=httpx.Client(transport=transport))


def ok(payload: dict) -> httpx.Response:
    return httpx.Response(200, json=payload)


# ---------- Consulta ----------


def test_query_sends_only_the_fields_that_have_a_value() -> None:
    capturado = {}

    def handler(request: httpx.Request) -> httpx.Response:
        capturado.update(json.loads(request.content))
        return ok({"answer": "sí", "sources": []})

    build_client(handler).query("¿preaviso?")

    assert capturado == {"query": "¿preaviso?"}


def test_query_includes_conversation_and_filters_when_given() -> None:
    capturado = {}

    def handler(request: httpx.Request) -> httpx.Response:
        capturado.update(json.loads(request.content))
        return ok({"answer": "sí", "sources": []})

    build_client(handler).query(
        "¿preaviso?",
        chat_history_id="uuid-1234",
        filters={"categoria": "contratos_2026"},
    )

    assert capturado["chat_history_id"] == "uuid-1234"
    assert capturado["filters"] == {"categoria": "contratos_2026"}


def test_query_hits_the_documented_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{BASE}/chat/query"
        assert request.method == "POST"
        return ok({"answer": "", "sources": []})

    build_client(handler).query("hola")


# ---------- Documentos ----------


def test_ingest_sends_the_file_as_multipart() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["content-type"].startswith("multipart/form-data")
        cuerpo = request.content.decode("latin-1")
        assert 'filename="contrato.pdf"' in cuerpo
        assert "contenido-pdf" in cuerpo
        return ok({"doc_id": "contrato.pdf"})

    build_client(handler).ingest("contrato.pdf", b"contenido-pdf", content_type="application/pdf")


def test_ingest_serialises_metadata_as_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert '{"categoria": "contratos_2026"}' in request.content.decode("latin-1")
        return ok({"doc_id": "contrato.pdf"})

    build_client(handler).ingest("contrato.pdf", b"x", metadata={"categoria": "contratos_2026"})


def test_ingest_omits_metadata_when_there_is_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "metadata" not in request.content.decode("latin-1")
        return ok({"doc_id": "contrato.pdf"})

    build_client(handler).ingest("contrato.pdf", b"x")


def test_delete_uses_the_document_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert str(request.url) == f"{BASE}/documents/contrato.pdf"
        return ok({"doc_id": "contrato.pdf", "deleted_chunks": 3})

    assert build_client(handler).delete_document("contrato.pdf")["deleted_chunks"] == 3


# ---------- Salud ----------


def test_is_available_is_true_when_the_backend_answers() -> None:
    assert build_client(lambda request: ok({"status": "ok"})).is_available()


def test_is_available_is_false_when_the_backend_is_down() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    assert not build_client(handler).is_available()


# ---------- Errores ----------


def test_a_connection_error_explains_the_backend_is_not_running() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(RagApiError, match=r"¿Está arrancado el backend\?"):
        build_client(handler).list_documents()


def test_a_timeout_is_reported_as_such() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    with pytest.raises(RagApiError, match="tardó demasiado"):
        build_client(handler).query("hola")


def test_the_api_error_message_reaches_the_user() -> None:
    """El backend explica por qué rechaza un documento: eso debe llegar a la UI."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "'x.pdf' no contiene texto extraíble"})

    with pytest.raises(RagApiError, match="no contiene texto extraíble") as excinfo:
        build_client(handler).ingest("x.pdf", b"x")

    assert excinfo.value.status_code == 422


def test_validation_errors_are_flattened_into_one_message() -> None:
    """FastAPI devuelve una lista en los errores de validación; la UI espera texto."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "detail": [
                    {"loc": ["body", "query"], "msg": "String should have at least 1 character"}
                ]
            },
        )

    with pytest.raises(RagApiError, match="query: String should have at least 1 character"):
        build_client(handler).query("hola")


def test_a_non_json_error_still_produces_a_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="<html>Internal Server Error</html>")

    with pytest.raises(RagApiError, match="Error 500"):
        build_client(handler).list_documents()


def test_the_base_url_tolerates_a_trailing_slash() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{BASE}/documents"
        return ok({"documents": []})

    cliente = RagApiClient(f"{BASE}/", client=httpx.Client(transport=httpx.MockTransport(handler)))
    cliente.list_documents()
