"""Dependencias compartidas de la API.

Se declaran como funciones para que los tests puedan sustituirlas con
`app.dependency_overrides` sin tocar la configuración global.
"""

from typing import Annotated

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.generation.llm import get_llm_provider
from app.generation.llm.base import LLMProvider
from app.ingestion.embeddings import get_embedding_provider
from app.ingestion.embeddings.base import EmbeddingProvider
from app.retrieval import Retriever, get_retriever
from app.storage import get_vector_store
from app.storage.base import VectorStore


def settings_dep() -> Settings:
    return get_settings()


def embedding_provider_dep() -> EmbeddingProvider:
    return get_embedding_provider()


def vector_store_dep() -> VectorStore:
    return get_vector_store()


def retriever_dep() -> Retriever:
    return get_retriever()


def llm_provider_dep() -> LLMProvider:
    return get_llm_provider()


SettingsDep = Annotated[Settings, Depends(settings_dep)]
EmbeddingProviderDep = Annotated[EmbeddingProvider, Depends(embedding_provider_dep)]
VectorStoreDep = Annotated[VectorStore, Depends(vector_store_dep)]
RetrieverDep = Annotated[Retriever, Depends(retriever_dep)]
LLMProviderDep = Annotated[LLMProvider, Depends(llm_provider_dep)]
