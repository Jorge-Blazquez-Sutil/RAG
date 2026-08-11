"""Configuración central del sistema.

Todos los parámetros se leen de variables de entorno o del archivo `.env`.
El objetivo es que el backoffice (rama `feature/admin-runtime-params`) pueda
ajustar chunking y Top-K sin tocar código.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
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

    # ---------- Embeddings ----------
    embedding_provider: Literal["openai", "local"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    openai_api_key: str | None = None

    # ---------- Almacenamiento (módulo 3.2) ----------
    vector_store: Literal["chroma", "pgvector"] = "chroma"
    chroma_path: Path = REPO_ROOT / "chroma_db"
    collection_name: str = "documents"

    # ---------- Recuperación (módulo 3.3) ----------
    top_k: int = Field(default=5, ge=1, le=50)
    rerank_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ---------- Generación (módulo 3.4) ----------
    llm_provider: Literal["anthropic", "ollama"] = "anthropic"
    llm_model: str = "claude-opus-5"
    llm_max_tokens: int = 16000
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    anthropic_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"


@lru_cache
def get_settings() -> Settings:
    """Instancia única de configuración (cacheada para usar como dependencia FastAPI)."""
    return Settings()
