"""Segmentación de documentos en fragmentos (paso 2 del módulo 3.1).

Usa `RecursiveCharacterTextSplitter` de LangChain midiendo en tokens, como indica
el documento de diseño: corta primero por párrafo, luego por línea, luego por
frase, de forma que el corte caiga en la frontera semántica más grande posible.

Conteo de tokens
----------------
El tamaño del fragmento se mide contra el modelo que lo va a vectorizar
(`text-embedding-3-small`), no contra el LLM que genera la respuesta. El conteo
exacto lo da `tiktoken` con el tokenizador `cl100k_base`, pero tiktoken descarga
ese tokenizador de internet la primera vez que se usa: en un despliegue aislado
—o detrás de un proxy que intercepta TLS— esa descarga falla.

Por eso el contador es sustituible: si el tokenizador exacto no está disponible,
se usa una estimación por caracteres que sobrestima ligeramente, de modo que los
fragmentos siguen cabiendo en el límite del modelo de embeddings. La segmentación
funciona sin red; solo pierde precisión en el recuento.
"""

import hashlib
import math
from collections.abc import Callable
from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.chunk import Chunk
from app.schemas.document import ExtractedDocument

logger = get_logger(__name__)

#: Tokenizador del modelo de embeddings de OpenAI (text-embedding-3-small).
ENCODING_NAME = "cl100k_base"

#: De mayor a menor prioridad: párrafo, línea, frase, palabra, carácter.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

#: Caracteres por token en la estimación de respaldo. El valor real ronda 4 en
#: inglés y 3,8 en castellano; 3,5 sobrestima a propósito para no pasarse del
#: límite del modelo de embeddings cuando el conteo exacto no está disponible.
CHARS_PER_TOKEN = 3.5


class ChunkingError(ValueError):
    """Parámetros de segmentación incoherentes."""


@lru_cache(maxsize=1)
def _load_encoding() -> object | None:
    """Carga el tokenizador exacto, o None si no está disponible.

    Se cachea el resultado —incluido el fallo— para no reintentar una descarga
    que ya sabemos que no funciona en cada llamada.
    """
    settings = get_settings()
    if settings.tokenizer == "heuristic":
        return None

    try:
        import tiktoken

        return tiktoken.get_encoding(ENCODING_NAME)
    except Exception as exc:  # descarga fallida, sin red, proxy TLS, etc.
        if settings.tokenizer == "tiktoken":
            raise ChunkingError(
                f"TOKENIZER=tiktoken pero no se pudo cargar '{ENCODING_NAME}': {exc}"
            ) from exc
        logger.warning(
            "Tokenizador exacto no disponible (%s). Se estima por caracteres; "
            "los recuentos serán aproximados.",
            exc,
        )
        return None


def estimate_tokens(text: str) -> int:
    """Estimación de tokens por longitud, sin dependencias externas."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def count_tokens(text: str) -> int:
    """Número de tokens del texto según el tokenizador del modelo de embeddings.

    Usa el conteo exacto si `tiktoken` está disponible; si no, la estimación.
    """
    if not text:
        return 0
    encoding = _load_encoding()
    if encoding is None:
        return estimate_tokens(text)
    return len(encoding.encode(text))  # type: ignore[attr-defined]


def is_exact_counting_available() -> bool:
    """True si el recuento de tokens es exacto (tiktoken cargado)."""
    return _load_encoding() is not None


@lru_cache(maxsize=8)
def _build_splitter(
    chunk_size: int,
    chunk_overlap: int,
    length_function: Callable[[str], int] = count_tokens,
) -> RecursiveCharacterTextSplitter:
    """Construye el splitter. Cacheado porque su inicialización no es gratis."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        length_function=length_function,
    )


def chunk_document(
    document: ExtractedDocument,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Divide el documento en fragmentos solapados.

    La segmentación es **por página**: un fragmento nunca cruza el límite de una
    página, de modo que su `page_number` es exacto y la cita del `sources[]`
    apunta siempre a un sitio concreto del documento original.

    Args:
        document: resultado de la extracción de texto.
        chunk_size: tokens por fragmento. Por defecto, el valor de configuración.
        chunk_overlap: tokens de solapamiento entre fragmentos consecutivos.

    Raises:
        ChunkingError: si los parámetros son incoherentes.
    """
    settings = get_settings()
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    if size <= 0:
        raise ChunkingError(f"chunk_size debe ser positivo, recibido {size}")
    if overlap < 0:
        raise ChunkingError(f"chunk_overlap no puede ser negativo, recibido {overlap}")
    if overlap >= size:
        raise ChunkingError(
            f"chunk_overlap ({overlap}) debe ser menor que chunk_size ({size}): "
            "con un solapamiento igual o mayor el troceado no avanza"
        )

    splitter = _build_splitter(size, overlap)
    filename = document.metadata.filename

    chunks: list[Chunk] = []
    for page in document.pages:
        if page.is_empty:
            continue
        for text in splitter.split_text(page.text):
            cleaned = text.strip()
            if not cleaned:
                continue
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=build_chunk_id(filename, page.page_number, index, cleaned),
                    index=index,
                    text=cleaned,
                    page_number=page.page_number,
                    token_count=count_tokens(cleaned),
                    filename=filename,
                )
            )

    return chunks


def build_chunk_id(filename: str, page_number: int, index: int, text: str) -> str:
    """ID determinista del fragmento.

    Depende del contenido, así que reingerir el mismo documento produce los mismos
    identificadores: la BD vectorial puede hacer upsert sin duplicar (y un documento
    editado genera IDs nuevos solo en los fragmentos que cambiaron).
    """
    digest = hashlib.sha256(f"{filename}|{page_number}|{index}|{text}".encode())
    return digest.hexdigest()[:16]
