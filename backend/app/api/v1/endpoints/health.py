"""Endpoint de salud — usado por Docker, el CI y el frontend para comprobar el arranque."""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.config import Settings, get_settings

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    app: str
    environment: str
    version: str


@router.get("/health", response_model=HealthResponse, summary="Estado del servicio")
async def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        environment=settings.environment,
        version="0.1.0",
    )
