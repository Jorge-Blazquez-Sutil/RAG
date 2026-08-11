"""Cliente HTTP del backend RAG.

Separado de la interfaz a propósito: aquí vive lo que se puede probar sin
levantar Streamlit, y la UI queda como una capa fina de presentación.
"""

import json
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://localhost:8000/api/v1"
DEFAULT_TIMEOUT = 120.0


class RagApiError(Exception):
    """Fallo al hablar con el backend, con el mensaje ya listo para enseñar."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        self.status_code = status_code
        super().__init__(message)


class RagApiClient:
    """Envoltorio de los endpoints de la API v1."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        # La ingesta de un PDF grande puede tardar: el timeout por defecto de
        # httpx (5 s) cortaría la subida a mitad de proceso.
        self._client = client or httpx.Client(timeout=timeout)

    # ---------- Salud ----------

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def is_available(self) -> bool:
        """True si el backend responde. Pensado para el aviso de la barra lateral."""
        try:
            self.health()
        except RagApiError:
            return False
        return True

    # ---------- Documentos ----------

    def ingest(
        self,
        filename: str,
        content: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = {"metadata": json.dumps(metadata)} if metadata else None
        return self._request(
            "POST",
            "/documents/ingest",
            files={"file": (filename, content, content_type)},
            data=data,
        )

    def list_documents(self) -> dict[str, Any]:
        return self._request("GET", "/documents")

    def delete_document(self, doc_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/documents/{doc_id}")

    # ---------- Consulta ----------

    def query(
        self,
        question: str,
        *,
        chat_history_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"query": question}
        if chat_history_id:
            payload["chat_history_id"] = chat_history_id
        if filters:
            payload["filters"] = filters
        return self._request("POST", "/chat/query", json=payload)

    # ---------- Interno ----------

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._client.request(method, f"{self.base_url}{path}", **kwargs)
        except httpx.ConnectError as exc:
            raise RagApiError(
                f"No se puede conectar con la API en {self.base_url}. ¿Está arrancado el backend?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise RagApiError("La API tardó demasiado en responder.") from exc
        except httpx.HTTPError as exc:
            raise RagApiError(f"Error de red: {exc}") from exc

        if response.is_error:
            raise RagApiError(_detail(response), status_code=response.status_code)

        return response.json()


def _detail(response: httpx.Response) -> str:
    """Extrae el mensaje de error de la API.

    FastAPI usa `detail`, que es texto en los errores propios y una lista de
    problemas en los de validación; se normalizan aquí para que la UI no tenga
    que distinguir.
    """
    try:
        body = response.json()
    except ValueError:
        return f"Error {response.status_code}: {response.text[:200]}"

    detail = body.get("detail") if isinstance(body, dict) else None

    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        motivos = "; ".join(
            f"{'.'.join(str(p) for p in item.get('loc', [])[1:])}: {item.get('msg', '')}".strip(
                ": "
            )
            for item in detail
            if isinstance(item, dict)
        )
        return motivos or f"Error {response.status_code}"
    return f"Error {response.status_code}"
