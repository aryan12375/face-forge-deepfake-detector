"""
Pydantic schemas for all API request/response models.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class Verdict(str, Enum):
    FAKE = "FAKE"
    REAL = "REAL"
    UNCERTAIN = "UNCERTAIN"


class RegionActivation(BaseModel):
    name: str = Field(..., description="Facial region label")
    score: float = Field(..., ge=0.0, le=1.0, description="GradCAM activation score")
    color: str = Field(..., description="Hex color for UI heat rendering")


class SignalMetrics(BaseModel):
    freq_artifacts: str
    freq_level: str        # "LOW" | "MODERATE" | "HIGH"
    skin_texture: str
    skin_level: str
    edge_coherence: str
    edge_level: str
    lighting_consistency: str
    lighting_level: str
    blending_score: float  # 0.0 (fake) → 1.0 (real)


class AnalysisResponse(BaseModel):
    verdict: Verdict
    confidence: float = Field(..., ge=0.0, le=100.0, description="Probability image is fake (%)")
    gradcam_url: str = Field(..., description="URL to GradCAM overlay image")
    face_url: str = Field(..., description="URL to cropped face image")
    fft_url: Optional[str] = Field(None, description="URL to FFT frequency spectrum image")
    regions: list[RegionActivation]
    metrics: SignalMetrics
    processing_time_ms: float
    model_version: str = "efficientnet-b4-v2.1"
    threshold_used: float = 0.5


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    version: str
