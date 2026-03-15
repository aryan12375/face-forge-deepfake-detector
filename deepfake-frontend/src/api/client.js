/**
 * api/client.js
 * =============
 * All communication with the FastAPI backend lives here.
 * Vite's dev proxy forwards /api/* and /outputs/* to localhost:8000.
 */

const BASE = '/api/v1'

/**
 * Analyze an image for deepfake manipulation.
 *
 * @param {File}   file       - Image file from input or dropzone
 * @param {number} threshold  - Fake verdict threshold (0.1 – 0.95)
 * @param {AbortSignal} signal - Optional AbortController signal
 * @returns {Promise<AnalysisResult>}
 */
export async function analyzeImage(file, threshold = 0.5, signal = null) {
  const form = new FormData()
  form.append('file', file)
  form.append('threshold', String(threshold))

  const res = await fetch(`${BASE}/analyze`, {
    method: 'POST',
    body: form,
    signal,
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }))
    throw new APIError(res.status, err.detail ?? 'Request failed')
  }

  return res.json()
}

/**
 * Health check — confirms model is loaded and server is up.
 * @returns {Promise<HealthResult>}
 */
export async function fetchHealth() {
  const res = await fetch(`${BASE}/health`)
  if (!res.ok) throw new APIError(res.status, 'Health check failed')
  return res.json()
}

/**
 * Resolve a backend output URL (gradcam_url, fft_url etc.)
 * to a fully-qualified src that <img> can use.
 */
export function resolveOutputUrl(path) {
  if (!path) return null
  // In dev: Vite proxy handles /outputs/* → backend
  // In prod: change BASE_URL to your deployed API origin
  return path.startsWith('http') ? path : path
}

/** Typed API error */
export class APIError extends Error {
  constructor(status, message) {
    super(message)
    this.name = 'APIError'
    this.status = status
  }
}
