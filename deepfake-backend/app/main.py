"""
FaceForge Detection Lab — FastAPI Backend
=========================================
Main application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import os

from app.routers import analyze, health
from app.services.model_service import ModelService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    logger.info("Loading EfficientNet-B4 + GradCAM model...")
    ModelService.load()
    logger.info("Model ready.")
    yield
    logger.info("Shutting down — releasing model.")
    ModelService.unload()


app = FastAPI(
    title="FaceForge Detection Lab API",
    description="Deepfake detection via EfficientNet-B4 + GradCAM",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(analyze.router, prefix="/api/v1", tags=["analyze"])

# Serve GradCAM output images statically
os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
