"""Configuración central del sistema.

Todos los parámetros se leen de variables de entorno o del archivo `.env`.
El objetivo es que el backoffice (rama `feature/admin-runtime-params`) pueda
ajustar chunking y Top-K sin tocar código.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- Aplicación ----------
    app_name: str = "Asistente RAG de Nicho"
    environment: Literal["local", "dev", "prod"] = "local"
    debug: bool = True
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8501"]

    # ---------- Ingesta (módulo 3.1) ----------
    upload_dir: Path = REPO_ROOT / "data" / "uploads"
    max_upload_mb: int = 50
    allowed_extensions: set[str] = {".pdf", ".docx", ".txt"}
    chunk_size: int = Field(default=800, ge=100, le=4000, description="Tokens por chunk")
    chunk_overlap: int = Field(default=120, ge=0, description="Solapamiento en tokens (~15 %)")
    #: "auto" usa tiktoken si está disponible y estima por caracteres si no.
    #: "tiktoken" exige el conteo exacto y falla si no puede cargarlo.
    #: "heuristic" fuerza la estimación (despliegues sin salida a internet).
    tokenizer: Literal["auto", "tiktoken", "heuristic"] = "auto"

    # ---------- Embeddings ----------
    #: "fake" es determinista y funciona sin red ni clave: tests y arranque local.
    #: "local" queda reservado para el proveedor autoalojado (aún sin implementar).
    embedding_provider: Literal["openai", "local", "fake"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=1536, ge=1)
    embedding_batch_size: int = Field(default=64, ge=1, le=2048)
    openai_api_key: str | None = None

    # ---------- Almacenamiento (módulo 3.2) ----------
    vector_store: Literal["chroma", "pgvector"] = "chroma"
    chroma_path: Path = REPO_ROOT / "chroma_db"
    #: Chroma exige 3-512 caracteres de [a-zA-Z0-9._-], empezando y acabando en
    #: alfanumérico. Validarlo aquí convierte un fallo en tiempo de ejecución en
    #: un error de arranque con el motivo delante.
    collection_name: str = Field(
        default="documents",
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$",
    )

    # ---------- Recuperación (módulo 3.3) ----------
    top_k: int = Field(default=5, ge=1, le=50)
    #: Similitud mínima para conservar un fragmento. Desactivado por defecto: el
    #: valor útil depende del modelo de embeddings y del corpus, y se calibra con
    #: el golden set (rama feature/eval-rag-harness). Ver app/retrieval/retriever.py.
    retrieval_min_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    rerank_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ---------- Generación (módulo 3.4) ----------
    #: Presupuesto de tokens del CONTEXTO. Un contexto grande cuesta en cada
    #: consulta y diluye la señal: el modelo rinde mejor con pocos fragmentos
    #: buenos que con muchos mediocres.
    max_context_tokens: int = Field(default=6000, ge=500)
    max_history_turns: int = Field(default=6, ge=0)
    llm_provider: Literal["anthropic", "ollama"] = "anthropic"
    llm_model: str = "claude-opus-5"
    llm_max_tokens: int = 16000
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    anthropic_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    @model_validator(mode="after")
    def _check_chunking(self) -> "Settings":
        # Con solapamiento >= tamaño el troceado no avanzaría: cada fragmento
        # empezaría donde empezó el anterior.
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) debe ser menor que "
                f"CHUNK_SIZE ({self.chunk_size})"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Instancia única de configuración (cacheada para usar como dependencia FastAPI)."""
    return Settings()
