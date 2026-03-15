# FaceForge Detection Lab — React Frontend

**Vite + React · Zustand · CSS Modules · react-dropzone**

Connects directly to the FastAPI backend via Vite's dev proxy.

---

## Project Structure

```
deepfake-frontend/
├── index.html
├── vite.config.js              # Proxy: /api/* + /outputs/* → localhost:8000
├── package.json
└── src/
    ├── main.jsx                # React entry point
    ├── index.css               # Design tokens + global resets
    ├── api/
    │   └── client.js           # analyzeImage(), fetchHealth(), resolveOutputUrl()
    ├── hooks/
    │   ├── useStore.js         # Zustand global store (result, history, threshold)
    │   ├── useAnalyze.js       # Full analyze flow with progress simulation
    │   └── useHealth.js        # Polls /health every 30s
    ├── components/
    │   ├── layout/
    │   │   ├── Header.jsx      # Logo + live model status pill
    │   │   └── Header.module.css
    │   ├── panels/
    │   │   ├── UploadPanel.jsx         # Dropzone + preview + Run button
    │   │   ├── UploadPanel.module.css
    │   │   ├── ResultsPanel.jsx        # GradCAM images + verdict + FFT + threshold
    │   │   ├── ResultsPanel.module.css
    │   │   ├── Sidebar.jsx             # Tabbed: regions / metrics / history
    │   │   └── Sidebar.module.css
    │   └── shared/
    │       ├── ProgressBar.jsx         # Animated progress bar with glow
    │       ├── ProgressBar.module.css
    │       ├── ThresholdSlider.jsx     # Verdict-reactive confidence slider
    │       └── ThresholdSlider.module.css
    └── pages/
        ├── App.jsx             # Layout shell (header + center + sidebar)
        └── App.module.css
```

---

## Quick Start

### Prerequisites

- Node.js ≥ 18
- Backend server running at `localhost:8000` (see backend README)

### Install and run

```bash
cd deepfake-frontend
npm install
npm run dev
```

App opens at `http://localhost:5173`

The Vite proxy in `vite.config.js` forwards:
- `/api/*`     → `http://localhost:8000/api/*`
- `/outputs/*` → `http://localhost:8000/outputs/*`

So no CORS issues in development — the frontend and API feel like one origin.

---

## How the API Connection Works

### 1. Health polling (`useHealth.js`)

On mount, and every 30 seconds, the app calls `GET /api/v1/health`.
The response updates the header status pill (OFFLINE → LOADING... → MODEL ONLINE)
and shows the device (CPU / CUDA / MPS).

### 2. Image analysis flow (`useAnalyze.js`)

```
User drops image
      ↓
setFile(file)              // stores File + creates object URL for preview
      ↓
run()
  ├── simulateProgress(0 → 35, 600ms)   // "upload" phase feel
  ├── analyzeImage(file, threshold)      // POST /api/v1/analyze (FormData)
  ├── simulateProgress(35 → 85, 900ms)  // "inference" phase feel
  ├── await response
  └── simulateProgress(85 → 100, 300ms) // finish
      ↓
setResult(data)            // commits to Zustand store
addToHistory(file, data)   // prepends to history list (max 8)
```

### 3. API client (`api/client.js`)

```js
// Analyze an image
const result = await analyzeImage(file, threshold, abortSignal)

// result shape:
{
  verdict: "FAKE" | "REAL" | "UNCERTAIN",
  confidence: 91.5,           // % probability of being fake
  gradcam_url: "/outputs/gradcam_abc123.jpg",
  face_url:    "/outputs/face_abc123.jpg",
  fft_url:     "/outputs/fft_abc123.png",
  regions: [
    { name: "Left eye", score: 0.88, color: "#ff3030" },
    ...
  ],
  metrics: {
    freq_artifacts: "HIGH",   freq_level: "red",
    skin_texture:   "UNNATURAL", skin_level: "red",
    edge_coherence: "INCONSISTENT", edge_level: "red",
    lighting_consistency: "POOR", lighting_level: "red",
    blending_score: 0.085
  },
  processing_time_ms: 142.3,
  model_version: "efficientnet-b4-v2.1",
  threshold_used: 0.5
}
```

### 4. Threshold slider — no extra API call

The confidence score is stored in state once. When the user moves the threshold
slider, `getVerdict()` in the store re-derives the verdict client-side:

```js
getVerdict: () => {
  const prob = result.confidence / 100
  if (prob >= threshold)          return 'FAKE'
  if (prob >= threshold - 0.15)   return 'UNCERTAIN'
  return 'REAL'
}
```

The verdict bar color, tag, and description all react instantly — no re-fetch.

---

## Production Build

```bash
npm run build      # outputs to dist/
npm run preview    # preview the production build locally
```

For production, update the proxy target in `vite.config.js` to your deployed
API URL, or configure a reverse proxy (nginx/caddy) to route `/api` and
`/outputs` to the backend.

---

## Environment Variables (optional)

Create a `.env` file to override the API base:

```env
VITE_API_BASE=https://your-api.example.com
```

Then in `api/client.js`:
```js
const BASE = `${import.meta.env.VITE_API_BASE ?? ''}/api/v1`
```
