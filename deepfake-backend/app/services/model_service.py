"""
ModelService
============
Singleton that owns:
  - EfficientNet-B4 binary classifier (real/fake)
  - GradCAM hook on the last conv block
  - Face detection via MTCNN
  - FFT frequency-domain analysis
  - All image pre/post-processing

Everything here is CPU/GPU agnostic — runs on MPS, CUDA, or CPU.
"""

import io
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Normalisation identical to ImageNet pretrain ──────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

PREPROCESS = T.Compose([
    T.Resize((380, 380)),          # EfficientNet-B4 native resolution
    T.CenterCrop(380),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ── GradCAM target layer name inside EfficientNet-B4 ─────────────────────────
# features[-1] = MBConv block 8 — richest spatial features before pooling
GRADCAM_LAYER = "features.8"

# ── Facial region bounding boxes (relative to 380×380 crop) ──────────────────
# These are approximate — in production, use facial landmark detection
FACE_REGIONS = {
    "Left eye":       (60,  100, 160, 180),
    "Right eye":      (220, 100, 320, 180),
    "Nose bridge":    (150, 130, 230, 230),
    "Jaw / chin":     (80,  280, 300, 370),
    "Skin texture":   (100, 200, 280, 300),
    "Lip region":     (130, 250, 250, 310),
}

REGION_COLORS = {
    "Left eye":    "#ff3030",
    "Right eye":   "#ff3030",
    "Nose bridge": "#ff7030",
    "Jaw / chin":  "#ff9040",
    "Skin texture":"#ffb060",
    "Lip region":  "#ffd080",
}


class _GradCAMHook:
    """
    Forward + backward hooks to extract feature maps and gradients
    from a target layer, then produce a class-weighted heatmap.
    """

    def __init__(self):
        self._fmaps: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None
        self._handles: list = []

    def register(self, layer: nn.Module):
        self._handles.append(
            layer.register_forward_hook(self._save_fmaps)
        )
        self._handles.append(
            layer.register_full_backward_hook(self._save_grads)
        )

    def _save_fmaps(self, _module, _inp, output):
        self._fmaps = output.detach()

    def _save_grads(self, _module, _grad_in, grad_out):
        self._grads = grad_out[0].detach()

    def compute(self, target_size: tuple[int, int]) -> np.ndarray:
        """
        GAP the gradients → weights → weighted sum of feature maps →
        ReLU → normalise → resize to target_size.
        Returns float32 heatmap in [0, 1].
        """
        weights = self._grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self._fmaps).sum(dim=1).squeeze(0)    # (H, W)
        cam = torch.relu(cam).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, target_size, interpolation=cv2.INTER_CUBIC)
        return cam.astype(np.float32)

    def clear(self):
        self._fmaps = None
        self._grads = None

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


class ModelService:
    """
    Static singleton — call ModelService.load() once at startup.
    """

    _model: Optional[nn.Module] = None
    _hook: Optional[_GradCAMHook] = None
    _device: Optional[torch.device] = None
    _loaded: bool = False

    # ── Try MTCNN for face detection; fall back to Haar cascade ──────────────
    _mtcnn = None
    _haar: Optional[cv2.CascadeClassifier] = None

    @classmethod
    def load(cls):
        if cls._loaded:
            return

        # Device selection
        if torch.cuda.is_available():
            cls._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            cls._device = torch.device("mps")
        else:
            cls._device = torch.device("cpu")
        logger.info(f"Using device: {cls._device}")

        # ── Build EfficientNet-B4 ─────────────────────────────────────────────
        cls._model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)

        # Replace classifier head: 1792 → 512 → 2 (real / fake)
        in_features = cls._model.classifier[1].in_features
        cls._model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 2),
        )
        cls._model.to(cls._device)

        # ── Load fine-tuned weights if checkpoint exists ──────────────────────
        ckpt_path = Path(os.environ.get("MODEL_CHECKPOINT", "checkpoints/best_model.pth"))
        if ckpt_path.exists():
            logger.info(f"Loading fine-tuned weights from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=cls._device, weights_only=True)
            # Support both raw state_dict and wrapped checkpoint dicts
            state_dict = ckpt.get("model_state_dict", ckpt)
            cls._model.load_state_dict(state_dict)
            val_acc = ckpt.get("val_acc", "unknown")
            logger.info(f"Loaded checkpoint — val_acc={val_acc}")
        else:
            logger.warning(
                f"No checkpoint found at {ckpt_path}. "
                "Running with ImageNet-pretrained weights only. "
                "Predictions will be unreliable until the model is fine-tuned. "
                "Run scripts/train.py to produce a checkpoint."
            )

        cls._model.eval()

        # ── Register GradCAM hook ─────────────────────────────────────────────
        cls._hook = _GradCAMHook()
        target_layer = dict(cls._model.named_modules())[GRADCAM_LAYER]
        cls._hook.register(target_layer)

        # ── Face detector (MTCNN optional) ────────────────────────────────────
        try:
            from facenet_pytorch import MTCNN
            cls._mtcnn = MTCNN(
                image_size=380,
                margin=20,
                keep_all=False,
                device=cls._device,
            )
            logger.info("MTCNN face detector loaded.")
        except ImportError:
            logger.warning("facenet-pytorch not installed — using OpenCV Haar cascade.")
            cls._haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

        cls._loaded = True
        logger.info("ModelService ready.")

    @classmethod
    def unload(cls):
        if cls._hook:
            cls._hook.remove()
        cls._model = None
        cls._loaded = False

    # ── Public API ────────────────────────────────────────────────────────────

    @classmethod
    def analyze(
        cls,
        image_bytes: bytes,
        threshold: float = 0.5,
    ) -> dict:
        """
        Full pipeline:
          1. Decode image
          2. Detect & crop face
          3. EfficientNet forward pass
          4. GradCAM backward pass
          5. Build overlay image
          6. FFT frequency analysis
          7. Score facial regions
          8. Return structured result dict
        """
        if not cls._loaded:
            raise RuntimeError("Model not loaded. Call ModelService.load() first.")

        t0 = time.perf_counter()
        run_id = uuid.uuid4().hex[:10]

        # 1. Decode ─────────────────────────────────────────────────────────
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_np = np.array(pil_img)

        # 2. Face detection & crop ──────────────────────────────────────────
        face_pil, face_np = cls._detect_and_crop(pil_img, orig_np)

        # 3. Pre-process & forward pass ─────────────────────────────────────
        tensor = PREPROCESS(face_pil).unsqueeze(0).to(cls._device)
        tensor.requires_grad_(True)

        cls._hook.clear()
        logits = cls._model(tensor)                 # (1, 2)
        probs  = torch.softmax(logits, dim=1)
        fake_prob = float(probs[0, 0].item())       # index 1 = FAKE class

        # 4. GradCAM backward ───────────────────────────────────────────────
        cls._model.zero_grad()
        logits[0, 0].backward()                     # grad w.r.t. FAKE class
        cam = cls._hook.compute((380, 380))

        # 5. Build overlay ──────────────────────────────────────────────────
        face_url  = cls._save_face(face_np, run_id)
        grad_url  = cls._save_gradcam(face_np, cam, run_id)

        # 6. FFT analysis ───────────────────────────────────────────────────
        fft_url, fft_score = cls._fft_analysis(face_np, run_id)

        # 7. Region scoring ─────────────────────────────────────────────────
        regions = cls._score_regions(cam, fake_prob)

        # 8. Signal metrics ─────────────────────────────────────────────────
        metrics = cls._build_metrics(cam, fake_prob, fft_score, face_np)

        # 9. Verdict ────────────────────────────────────────────────────────
        confidence = round(fake_prob * 100, 2)
        if fake_prob >= threshold:
            verdict = "FAKE"
        elif fake_prob >= threshold - 0.15:
            verdict = "UNCERTAIN"
        else:
            verdict = "REAL"

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(f"[{run_id}] verdict={verdict} conf={confidence:.1f}% t={elapsed_ms}ms")

        return {
            "verdict": verdict,
            "confidence": confidence,
            "gradcam_url": grad_url,
            "face_url": face_url,
            "fft_url": fft_url,
            "regions": regions,
            "metrics": metrics,
            "processing_time_ms": elapsed_ms,
            "threshold_used": threshold,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    @classmethod
    def _detect_and_crop(
        cls,
        pil_img: Image.Image,
        orig_np: np.ndarray,
    ) -> tuple[Image.Image, np.ndarray]:
        """
        Try MTCNN first, fall back to Haar, fall back to centre crop.
        Returns (PIL face image resized to 380×380, np array).
        """
        W, H = pil_img.size

        # MTCNN path
        if cls._mtcnn is not None:
            try:
                box, _ = cls._mtcnn.detect(pil_img)
                if box is not None and len(box) > 0:
                    x1, y1, x2, y2 = [int(v) for v in box[0]]
                    margin = int((x2 - x1) * 0.2)
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(W, x2 + margin)
                    y2 = min(H, y2 + margin)
                    face_pil = pil_img.crop((x1, y1, x2, y2)).resize((380, 380))
                    face_np  = np.array(face_pil)
                    return face_pil, face_np
            except Exception as e:
                logger.warning(f"MTCNN failed: {e}")

        # Haar path
        if cls._haar is not None:
            gray = cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY)
            faces = cls._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                margin = int(w * 0.2)
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(W, x + w + margin)
                y2 = min(H, y + h + margin)
                face_pil = pil_img.crop((x1, y1, x2, y2)).resize((380, 380))
                face_np  = np.array(face_pil)
                return face_pil, face_np

        # Centre crop fallback
        logger.warning("No face detected — using centre crop.")
        short = min(W, H)
        left  = (W - short) // 2
        top   = (H - short) // 2
        face_pil = pil_img.crop((left, top, left + short, top + short)).resize((380, 380))
        face_np  = np.array(face_pil)
        return face_pil, face_np

    @classmethod
    def _save_face(cls, face_np: np.ndarray, run_id: str) -> str:
        path = OUTPUT_DIR / f"face_{run_id}.jpg"
        bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return f"/outputs/face_{run_id}.jpg"

    @classmethod
    def _save_gradcam(
        cls,
        face_np: np.ndarray,
        cam: np.ndarray,
        run_id: str,
    ) -> str:
        """
        Blend JET colourmap heatmap over the face with alpha=0.45.
        Save to disk, return URL path.
        """
        heatmap = np.uint8(cam * 255)
        heatmap_colour = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_colour = cv2.cvtColor(heatmap_colour, cv2.COLOR_BGR2RGB)

        face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
        heatmap_bgr = cv2.cvtColor(heatmap_colour, cv2.COLOR_RGB2BGR)

        overlay = cv2.addWeighted(face_bgr, 0.55, heatmap_bgr, 0.45, 0)

        path = OUTPUT_DIR / f"gradcam_{run_id}.jpg"
        cv2.imwrite(str(path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return f"/outputs/gradcam_{run_id}.jpg"

    @classmethod
    def _fft_analysis(
        cls,
        face_np: np.ndarray,
        run_id: str,
    ) -> tuple[str, float]:
        """
        Compute 2D FFT of the grayscale face.
        AI-generated images often show elevated high-frequency energy
        along certain axes (GAN grid artifacts, diffusion blocking).

        Returns (url_to_spectrum_png, anomaly_score_0_to_1).
        """
        gray = cv2.cvtColor(face_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
        fft  = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log1p(np.abs(fft_shift))

        # Normalise for display
        mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        spectrum = np.uint8(mag_norm)
        spectrum_colour = cv2.applyColorMap(spectrum, cv2.COLORMAP_INFERNO)

        path = OUTPUT_DIR / f"fft_{run_id}.png"
        cv2.imwrite(str(path), spectrum_colour)

        # Anomaly score: high-freq ring energy vs. DC lobe energy
        H, W = magnitude.shape
        cy, cx = H // 2, W // 2
        Y, X = np.ogrid[:H, :W]
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

        dc_mask   = r < 20
        hf_mask   = r > min(H, W) * 0.35
        dc_energy = magnitude[dc_mask].mean()
        hf_energy = magnitude[hf_mask].mean()

        # Higher ratio → more anomalous HF content
        ratio = float(np.clip(hf_energy / (dc_energy + 1e-8), 0, 1))
        return f"/outputs/fft_{run_id}.png", ratio

    @classmethod
    def _score_regions(
        cls,
        cam: np.ndarray,
        fake_prob: float,
    ) -> list[dict]:
        """
        Average GradCAM activation inside each anatomical bounding box.
        Scale by fake_prob so low-confidence runs don't show hot regions.
        """
        results = []
        for name, (x1, y1, x2, y2) in FACE_REGIONS.items():
            region_cam = cam[y1:y2, x1:x2]
            raw_score  = float(region_cam.mean()) if region_cam.size > 0 else 0.0
            score      = round(min(raw_score * fake_prob * 2.5, 1.0), 3)
            results.append({
                "name":  name,
                "score": score,
                "color": REGION_COLORS[name],
            })
        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    @classmethod
    def _build_metrics(
        cls,
        cam: np.ndarray,
        fake_prob: float,
        fft_score: float,
        face_np: np.ndarray,
    ) -> dict:
        """
        Derive human-readable signal metrics from raw analysis signals.
        """

        def _level(val: float, thresholds: tuple[float, float]) -> str:
            if val >= thresholds[1]:
                return "HIGH"
            elif val >= thresholds[0]:
                return "MODERATE"
            return "LOW"

        def _flag(level: str, invert: bool = False) -> str:
            if not invert:
                return {"HIGH": "red", "MODERATE": "amber", "LOW": "green"}[level]
            return {"HIGH": "green", "MODERATE": "amber", "LOW": "red"}[level]

        # Frequency artifact score
        freq_level = _level(fft_score, (0.35, 0.65))

        # Skin texture variance (low variance = oversmoothed = more fake)
        gray = cv2.cvtColor(face_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
        skin_variance = float(np.std(gray[80:300, 80:300]))
        skin_raw = 1.0 - np.clip(skin_variance / 60.0, 0, 1)
        skin_level = _level(skin_raw, (0.35, 0.65))

        # Edge coherence via Laplacian — low = blurry boundaries
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        edge_score = float(np.std(laplacian))
        edge_incoherence = 1.0 - np.clip(edge_score / 200.0, 0, 1)
        edge_level = _level(edge_incoherence, (0.35, 0.65))

        # Lighting consistency: channel std across RGB — imbalanced = fake
        ch_means = [float(face_np[:, :, c].mean()) for c in range(3)]
        light_std = float(np.std(ch_means))
        light_level = _level(light_std / 30.0, (0.3, 0.6))

        blending_score = round(1.0 - fake_prob, 3)

        labels = {
            "LOW":      {"freq": "CLEAN",       "skin": "NATURAL",     "edge": "COHERENT",     "light": "CONSISTENT"},
            "MODERATE": {"freq": "MODERATE",    "skin": "BORDERLINE",  "edge": "MINOR GAPS",   "light": "PARTIAL"},
            "HIGH":     {"freq": "HIGH",        "skin": "UNNATURAL",   "edge": "INCONSISTENT", "light": "POOR"},
        }

        return {
            "freq_artifacts":       labels[freq_level]["freq"],
            "freq_level":           _flag(freq_level),
            "skin_texture":         labels[skin_level]["skin"],
            "skin_level":           _flag(skin_level),
            "edge_coherence":       labels[edge_level]["edge"],
            "edge_level":           _flag(edge_level),
            "lighting_consistency": labels[light_level]["light"],
            "lighting_level":       _flag(light_level),
            "blending_score":       blending_score,
        }
