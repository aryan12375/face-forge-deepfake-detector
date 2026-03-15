/**
 * store/useStore.js
 * =================
 * Zustand global state.
 * Keeps analysis result, scan history, threshold, and UI state.
 */

import { create } from 'zustand'

const MAX_HISTORY = 8

export const useStore = create((set, get) => ({
  // ── Server status ────────────────────────────────────────────────────────
  serverOnline: false,
  modelLoaded: false,
  device: '',
  setServerStatus: (s) => set({ serverOnline: s.model_loaded, modelLoaded: s.model_loaded, device: s.device }),

  // ── Upload / file ────────────────────────────────────────────────────────
  selectedFile: null,
  previewUrl: null,
  setFile: (file) => {
    const prev = get().previewUrl
    if (prev) URL.revokeObjectURL(prev)
    set({
      selectedFile: file,
      previewUrl: file ? URL.createObjectURL(file) : null,
      result: null,
      error: null,
    })
  },

  // ── Analysis state ───────────────────────────────────────────────────────
  loading: false,
  progress: 0,           // 0-100, simulated during upload + inference
  result: null,          // AnalysisResult from backend
  error: null,

  setLoading: (v) => set({ loading: v }),
  setProgress: (v) => set({ progress: v }),
  setResult: (r) => set({ result: r, loading: false, progress: 100, error: null }),
  setError: (e) => set({ error: e, loading: false, progress: 0 }),

  // ── Threshold ────────────────────────────────────────────────────────────
  threshold: 0.5,        // 0.1 – 0.95
  setThreshold: (v) => set({ threshold: v }),

  // Derived verdict from threshold (re-evaluated without re-running model)
  // The model returns raw confidence; UI verdict flips as user moves slider
  getVerdict: () => {
    const { result, threshold } = get()
    if (!result) return null
    const prob = result.confidence / 100
    if (prob >= threshold) return 'FAKE'
    if (prob >= threshold - 0.15) return 'UNCERTAIN'
    return 'REAL'
  },

  // ── Scan history ─────────────────────────────────────────────────────────
  history: [],
  addToHistory: (file, result) => {
    const entry = {
      id: Date.now(),
      filename: file.name,
      verdict: result.verdict,
      confidence: result.confidence,
      faceUrl: result.face_url,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    }
    set((s) => ({
      history: [entry, ...s.history].slice(0, MAX_HISTORY),
    }))
  },

  // ── Active panel tab (sidebar) ───────────────────────────────────────────
  activeTab: 'regions',   // 'regions' | 'metrics' | 'history'
  setActiveTab: (t) => set({ activeTab: t }),
}))
