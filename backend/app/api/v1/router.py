"""Router agregador de la API v1."""

from fastapi import APIRouter

from app.api.v1.endpoints import chat, documents, health

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(documents.router)
api_router.include_router(chat.router)
