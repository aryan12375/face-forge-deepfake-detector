"""
/api/v1/analyze — POST endpoint
================================
Accepts an uploaded image, runs the full detection pipeline,
returns structured JSON with verdict, GradCAM URL, and metrics.
"""

import logging
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.models.schemas import AnalysisResponse
from app.services.model_service import ModelService

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Analyze an image for deepfake manipulation",
    response_description="Verdict, confidence, GradCAM overlay, and signal metrics",
)
async def analyze_image(
    file: UploadFile = File(..., description="Face image to analyze"),
    threshold: float = Form(
        default=0.5,
        ge=0.1,
        le=0.95,
        description="Confidence threshold for FAKE verdict (0.1–0.95)",
    ),
):
    """
    Pipeline:
    1. Validate file type and size
    2. Hand bytes to ModelService.analyze()
    3. Return structured AnalysisResponse
    """

    # ── Validation ──────────────────────────────────────────────────────────
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Accepted: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )

    image_bytes = await file.read()

    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(image_bytes) // 1024} KB). Max: 10 MB.",
        )

    if len(image_bytes) < 1024:
        raise HTTPException(status_code=400, detail="File appears to be empty or corrupted.")

    # ── Run analysis ─────────────────────────────────────────────────────────
    try:
        result = ModelService.analyze(image_bytes, threshold=threshold)
    except RuntimeError as e:
        logger.error(f"ModelService error: {e}")
        raise HTTPException(status_code=503, detail="Model not available. Try again shortly.")
    except Exception as e:
        logger.exception(f"Unexpected analysis error: {e}")
        raise HTTPException(status_code=500, detail="Internal analysis error.")

    return AnalysisResponse(**result)
