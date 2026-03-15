"""
/api/v1/health — GET endpoint
"""

from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.services.model_service import ModelService
import torch

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    device = "unknown"
    if ModelService._device:
        device = str(ModelService._device)

    return HealthResponse(
        status="ok" if ModelService._loaded else "model_not_loaded",
        model_loaded=ModelService._loaded,
        device=device,
        version="2.1.0",
    )
