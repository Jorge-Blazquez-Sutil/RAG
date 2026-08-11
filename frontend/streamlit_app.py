"""Prototipo de chat sobre la API RAG (sección 5.1 del documento de diseño).

Arranque:
    uv run streamlit run frontend/streamlit_app.py

La UI definitiva será React/Next.js (rama feature/ui-chat-nextjs). Esto existe
para validar el pipeline completo con las manos: subir documentos, preguntar y
comprobar que las citas apuntan donde deben.
"""

import os
from typing import Any

import streamlit as st
from api_client import DEFAULT_BASE_URL, RagApiClient, RagApiError

API_URL = os.getenv("RAG_API_URL", DEFAULT_BASE_URL)
UPLOAD_TYPES = ["pdf", "docx", "txt", "md"]


@st.cache_resource
def get_client() -> RagApiClient:
    """Un único cliente por sesión de servidor (reutiliza la conexión HTTP)."""
    return RagApiClient(API_URL)


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("chat_history_id", None)
    st.session_state.setdefault("category", "")


def render_sidebar(client: RagApiClient) -> None:
    """Gestión de documentos: subida, inventario y borrado."""
    with st.sidebar:
        st.header("Documentación")

        if not client.is_available():
            st.error(f"API no disponible en {API_URL}")
            st.caption("Arranca el backend con `uv run uvicorn app.main:app --app-dir backend`")
            return

        subida = st.file_uploader("Subir documento", type=UPLOAD_TYPES)
        categoria = st.text_input(
            "Categoría (opcional)",
            help="Etiqueta el documento para poder filtrar las consultas por ella",
        )

        if subida is not None and st.button("Indexar", use_container_width=True):
            _ingest(client, subida, categoria)

        st.divider()
        _render_documents(client)

        st.divider()
        st.session_state.category = st.text_input(
            "Filtrar consultas por categoría",
            value=st.session_state.category,
            help="Deja el campo vacío para buscar en toda la documentación",
        )


def _ingest(client: RagApiClient, subida: Any, categoria: str) -> None:
    metadata = {"categoria": categoria.strip()} if categoria.strip() else None
    with st.spinner(f"Procesando {subida.name}..."):
        try:
            report = client.ingest(
                subida.name,
                subida.getvalue(),
                content_type=subida.type or "application/octet-stream",
                metadata=metadata,
            )
        except RagApiError as exc:
            st.error(str(exc))
            return

    st.success(
        f"{report['doc_id']}: {report['pages']} páginas, {report['chunks']} fragmentos "
        f"en {report['elapsed_ms'] / 1000:.1f} s"
    )


def _render_documents(client: RagApiClient) -> None:
    try:
        listado = client.list_documents()
    except RagApiError as exc:
        st.error(str(exc))
        return

    documentos = listado["documents"]
    if not documentos:
        st.info("Todavía no hay documentos indexados.")
        return

    st.caption(f"{listado['total_documents']} documentos · {listado['total_chunks']} fragmentos")
    for documento in documentos:
        columna, boton = st.columns([4, 1])
        columna.write(f"**{documento['doc_id']}**")
        columna.caption(f"{documento['page_count']} págs · {documento['chunk_count']} fragmentos")
        if boton.button("🗑", key=f"del-{documento['doc_id']}", help="Eliminar del índice"):
            try:
                client.delete_document(documento["doc_id"])
            except RagApiError as exc:
                st.error(str(exc))
            else:
                st.rerun()


def render_history() -> None:
    for mensaje in st.session_state.messages:
        with st.chat_message(mensaje["role"]):
            st.markdown(mensaje["content"])
            if mensaje.get("sources"):
                render_sources(mensaje["sources"], mensaje.get("meta", ""))


def render_sources(sources: list[dict[str, Any]], meta: str = "") -> None:
    """Panel de fuentes: la parte que hace verificable la respuesta."""
    with st.expander(f"Fuentes ({len(sources)})", expanded=False):
        for fuente in sources:
            st.markdown(
                f"**{fuente['doc_id']}** · página {fuente['page']} "
                f"· similitud {fuente.get('score', 0):.3f}"
            )
            st.caption(fuente["text_snippet"])
    if meta:
        st.caption(meta)


def handle_question(client: RagApiClient, pregunta: str) -> None:
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    categoria = st.session_state.category.strip()
    filtros = {"categoria": categoria} if categoria else None

    with st.chat_message("assistant"), st.spinner("Consultando la documentación..."):
        try:
            respuesta = client.query(
                pregunta,
                chat_history_id=st.session_state.chat_history_id,
                filters=filtros,
            )
        except RagApiError as exc:
            st.error(str(exc))
            st.session_state.messages.pop()  # la pregunta no llegó a contestarse
            return

        st.session_state.chat_history_id = respuesta["chat_history_id"]
        meta = _meta(respuesta)

        st.markdown(respuesta["answer"])
        if respuesta["sources"]:
            render_sources(respuesta["sources"], meta)
        elif meta:
            st.caption(meta)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": respuesta["answer"],
            "sources": respuesta["sources"],
            "meta": meta,
        }
    )


def _meta(respuesta: dict[str, Any]) -> str:
    partes = [
        f"{respuesta['total_ms']:.0f} ms",
        f"recuperación {respuesta['retrieval_ms']:.0f} ms",
    ]
    if respuesta.get("model"):
        partes.append(respuesta["model"])
    if respuesta.get("refused"):
        partes.append("respuesta declinada por el modelo")
    return " · ".join(partes)


def main() -> None:
    st.set_page_config(page_title="Asistente RAG de Nicho", page_icon="📄", layout="wide")
    init_state()
    client = get_client()

    st.title("Asistente RAG de Nicho")
    st.caption("Responde solo con la documentación indexada, citando documento y página.")

    render_sidebar(client)
    render_history()

    if pregunta := st.chat_input("Pregunta sobre tus documentos"):
        handle_question(client, pregunta)


if __name__ == "__main__":
    main()
