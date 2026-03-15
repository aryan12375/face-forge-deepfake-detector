"""
tests/test_analyze.py
=====================
pytest suite — run with:  pytest tests/ -v

Tests the /analyze endpoint with synthetic images.
No real model weights needed for routing/validation tests.
"""

import io
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


def _make_jpeg_bytes(width=224, height=224) -> bytes:
    """Generate a random RGB JPEG in memory."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return buf.read()


FAKE_RESULT = {
    "verdict": "FAKE",
    "confidence": 91.5,
    "gradcam_url": "/outputs/gradcam_test.jpg",
    "face_url": "/outputs/face_test.jpg",
    "fft_url": "/outputs/fft_test.png",
    "regions": [
        {"name": "Left eye", "score": 0.88, "color": "#ff3030"},
        {"name": "Jaw / chin", "score": 0.72, "color": "#ff9040"},
    ],
    "metrics": {
        "freq_artifacts": "HIGH",
        "freq_level": "red",
        "skin_texture": "UNNATURAL",
        "skin_level": "red",
        "edge_coherence": "INCONSISTENT",
        "edge_level": "red",
        "lighting_consistency": "POOR",
        "lighting_level": "red",
        "blending_score": 0.085,
    },
    "processing_time_ms": 142.3,
    "threshold_used": 0.5,
}


@pytest.fixture
def client():
    with patch("app.services.model_service.ModelService.load"):
        with patch(
            "app.services.model_service.ModelService._loaded",
            new_callable=lambda: property(lambda self: True),
        ):
            from app.main import app
            with TestClient(app) as c:
                yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_schema(self, client):
        data = client.get("/api/v1/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
        assert "device" in data


class TestAnalyzeEndpoint:
    def test_rejects_non_image(self, client):
        resp = client.post(
            "/api/v1/analyze",
            files={"file": ("doc.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code == 415

    def test_rejects_empty_file(self, client):
        resp = client.post(
            "/api/v1/analyze",
            files={"file": ("empty.jpg", b"x" * 100, "image/jpeg")},
        )
        assert resp.status_code == 400

    def test_accepts_valid_jpeg(self, client):
        with patch(
            "app.services.model_service.ModelService.analyze",
            return_value=FAKE_RESULT,
        ):
            resp = client.post(
                "/api/v1/analyze",
                files={"file": ("face.jpg", _make_jpeg_bytes(), "image/jpeg")},
                data={"threshold": "0.5"},
            )
        assert resp.status_code == 200

    def test_response_schema(self, client):
        with patch(
            "app.services.model_service.ModelService.analyze",
            return_value=FAKE_RESULT,
        ):
            data = client.post(
                "/api/v1/analyze",
                files={"file": ("face.jpg", _make_jpeg_bytes(), "image/jpeg")},
            ).json()

        assert data["verdict"] in ("FAKE", "REAL", "UNCERTAIN")
        assert 0.0 <= data["confidence"] <= 100.0
        assert data["gradcam_url"].startswith("/outputs/")
        assert isinstance(data["regions"], list)
        assert "freq_artifacts" in data["metrics"]

    def test_threshold_bounds(self, client):
        with patch(
            "app.services.model_service.ModelService.analyze",
            return_value=FAKE_RESULT,
        ):
            # Too low
            resp = client.post(
                "/api/v1/analyze",
                files={"file": ("face.jpg", _make_jpeg_bytes(), "image/jpeg")},
                data={"threshold": "0.0"},
            )
            assert resp.status_code == 422

            # Too high
            resp = client.post(
                "/api/v1/analyze",
                files={"file": ("face.jpg", _make_jpeg_bytes(), "image/jpeg")},
                data={"threshold": "1.1"},
            )
            assert resp.status_code == 422

    def test_accepts_png(self, client):
        png_bytes = _make_jpeg_bytes()  # same random array, treated as PNG here
        with patch(
            "app.services.model_service.ModelService.analyze",
            return_value=FAKE_RESULT,
        ):
            resp = client.post(
                "/api/v1/analyze",
                files={"file": ("face.png", png_bytes, "image/png")},
            )
        assert resp.status_code == 200
