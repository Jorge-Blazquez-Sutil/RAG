"""Router agregador de la API v1.

Cada rama de módulo registra aquí su router:
    feature/api-chat-query -> chat
"""

from fastapi import APIRouter

from app.api.v1.endpoints import documents, health

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(documents.router)
