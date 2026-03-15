# FaceForge Detection Lab — Backend

**EfficientNet-B4 + GradCAM deepfake detection API**  
FastAPI · PyTorch · OpenCV · MTCNN · FFT Analysis

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         CLIENT (React UI)                         │
│  POST /api/v1/analyze  ──────────────────────────────────────►   │
└──────────────────────────────┬───────────────────────────────────┘
                               │ multipart/form-data
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                       FastAPI Application                         │
│                                                                   │
│  analyze.py router                                                │
│  ├── File type validation (JPEG / PNG / WEBP)                     │
│  ├── File size check (≤10 MB)                                     │
│  └── Calls ModelService.analyze(bytes, threshold)                 │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                        ModelService                               │
│                                                                   │
│  1. Face Detection                                                │
│     ├── MTCNN  (primary — accurate, learned)                      │
│     └── Haar   (fallback — fast, rule-based)                      │
│                                                                   │
│  2. EfficientNet-B4 Forward Pass                                  │
│     ├── Pre-trained on ImageNet                                   │
│     ├── Fine-tuned head: 1792 → 512 → 2                          │
│     └── Softmax → P(fake), P(real)                               │
│                                                                   │
│  3. GradCAM Backward Pass                                         │
│     ├── Hook on features.8 (last conv block)                      │
│     ├── GAP gradients → channel weights                           │
│     └── Weighted feature sum → ReLU → resize                     │
│                                                                   │
│  4. FFT Frequency Analysis                                        │
│     ├── 2D FFT on grayscale face                                  │
│     ├── HF ring energy / DC lobe energy ratio                     │
│     └── Detects GAN grid & diffusion blocking artifacts           │
│                                                                   │
│  5. Region Scoring                                                │
│     └── 6 anatomical regions × GradCAM mean activation           │
│                                                                   │
│  6. Signal Metrics                                                │
│     ├── Skin texture variance (oversmooth = fake)                 │
│     ├── Laplacian edge coherence                                  │
│     └── RGB channel lighting consistency                          │
│                                                                   │
│  7. Structured JSON response                                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
deepfake-backend/
├── app/
│   ├── main.py                  # FastAPI app, lifespan, CORS, routers
│   ├── routers/
│   │   ├── analyze.py           # POST /api/v1/analyze
│   │   └── health.py            # GET  /api/v1/health
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response schemas
│   ├── services/
│   │   └── model_service.py     # EfficientNet + GradCAM + FFT pipeline
│   └── utils/                   # (extend here: caching, rate-limiting, etc.)
├── scripts/
│   └── train.py                 # Fine-tuning script for your dataset
├── tests/
│   └── test_analyze.py          # pytest suite
├── outputs/                     # GradCAM/FFT images served statically
├── checkpoints/                 # Saved model weights (git-ignored)
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **CUDA users:** replace the `torch` line in requirements.txt with:
> ```
> torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
> torchvision==0.18.1+cu121
> ```

### 2. Run the API server

```bash
cd deepfake-backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Server starts at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

### 3. Test an image

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@path/to/face.jpg" \
  -F "threshold=0.5"
```

---

## Training Your Own Model

### Dataset Layout

```
data/
  train/
    real/   # ~50k authentic face crops
    fake/   # ~50k deepfake face crops
  val/
    real/
    fake/
```

**Recommended datasets:**
- [FaceForensics++](https://github.com/ondyari/FaceForensics) — manipulation types: Deepfakes, Face2Face, FaceShifter, NeuralTextures
- [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics)
- [DFDC](https://ai.facebook.com/datasets/dfdc/) — Meta's large-scale challenge set

### Run Training

```bash
python scripts/train.py \
    --data_dir ./data \
    --epochs 20 \
    --batch_size 16 \
    --lr 3e-4 \
    --output_dir ./checkpoints
```

**Tip:** start with `--freeze_backbone` for 5 epochs to warm up the head, then unfreeze and fine-tune end-to-end.

### Load Fine-tuned Weights

In `model_service.py`, after building the model, add:

```python
ckpt = torch.load("checkpoints/best_model.pth", map_location=cls._device)
cls._model.load_state_dict(ckpt["model_state_dict"])
```

---

## API Reference

### `POST /api/v1/analyze`

| Field | Type | Description |
|-------|------|-------------|
| `file` | File | Face image (JPEG / PNG / WEBP, ≤10 MB) |
| `threshold` | float | Fake detection threshold, 0.1–0.95 (default: 0.5) |

**Response `200 OK`:**

```json
{
  "verdict": "FAKE",
  "confidence": 91.5,
  "gradcam_url": "/outputs/gradcam_abc123.jpg",
  "face_url": "/outputs/face_abc123.jpg",
  "fft_url": "/outputs/fft_abc123.png",
  "regions": [
    { "name": "Left eye", "score": 0.88, "color": "#ff3030" },
    { "name": "Jaw / chin", "score": 0.72, "color": "#ff9040" }
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
    "blending_score": 0.085
  },
  "processing_time_ms": 142.3,
  "model_version": "efficientnet-b4-v2.1",
  "threshold_used": 0.5
}
```

### `GET /api/v1/health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda",
  "version": "2.1.0"
}
```

---

## Connecting to the React Frontend

In your React UI, replace the simulated analysis with a real fetch:

```javascript
const analyze = async (file, threshold) => {
  const form = new FormData();
  form.append("file", file);
  form.append("threshold", threshold);

  const res = await fetch("http://localhost:8000/api/v1/analyze", {
    method: "POST",
    body: form,
  });

  const data = await res.json();
  // data.gradcam_url → set as <img src={`http://localhost:8000${data.gradcam_url}`} />
  // data.verdict, data.confidence → verdict bar
  // data.regions → regions panel
  // data.metrics → signal metrics
  return data;
};
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Performance Notes

| Hardware | Inference time (per image) |
|----------|---------------------------|
| NVIDIA RTX 3090 | ~35 ms |
| Apple M2 Pro (MPS) | ~90 ms |
| CPU only | ~600–900 ms |

For production: enable `torch.compile()` and serve behind gunicorn with `--workers 2`.

---

## Roadmap

- [ ] Load fine-tuned weights from `checkpoints/best_model.pth`
- [ ] Add Redis caching for duplicate image requests (SHA256 hash key)
- [ ] Rate limiting middleware (slowapi)
- [ ] Video frame analysis endpoint (`/analyze/video`)
- [ ] Temporal inconsistency detection across frames
- [ ] Docker + docker-compose setup
- [ ] Prometheus metrics endpoint
