/**
 * hooks/useAnalyze.js
 * ====================
 * Encapsulates the full analyze flow:
 *   1. Simulate upload progress
 *   2. Call backend
 *   3. Simulate inference progress
 *   4. Commit result to store
 */

import { useRef, useCallback } from 'react'
import { analyzeImage, APIError } from '@/api/client'
import { useStore } from '@/hooks/useStore'

export function useAnalyze() {
  const {
    selectedFile,
    threshold,
    setLoading,
    setProgress,
    setResult,
    setError,
    addToHistory,
  } = useStore()

  const abortRef = useRef(null)
  const progressTimerRef = useRef(null)

  const simulateProgress = useCallback((from, to, durationMs, onDone) => {
    clearInterval(progressTimerRef.current)
    const steps = 30
    const stepMs = durationMs / steps
    const stepSize = (to - from) / steps
    let current = from

    progressTimerRef.current = setInterval(() => {
      current = Math.min(current + stepSize + Math.random() * stepSize * 0.3, to)
      setProgress(Math.round(current))
      if (current >= to) {
        clearInterval(progressTimerRef.current)
        onDone?.()
      }
    }, stepMs)
  }, [setProgress])

  const run = useCallback(async () => {
    if (!selectedFile) return

    // Cancel any in-flight request
    abortRef.current?.abort()
    abortRef.current = new AbortController()

    setLoading(true)
    setProgress(0)
    useStore.getState().setError(null)

    try {
      // Phase 1: simulate upload (0 → 35%)
      await new Promise((resolve) => {
        simulateProgress(0, 35, 600, resolve)
      })

      // Phase 2: send to backend
      const resultPromise = analyzeImage(
        selectedFile,
        threshold,
        abortRef.current.signal,
      )

      // Phase 3: simulate inference while waiting (35 → 85%)
      simulateProgress(35, 85, 900, null)

      const result = await resultPromise

      // Phase 4: finish progress (85 → 100%)
      await new Promise((resolve) => {
        simulateProgress(85, 100, 300, resolve)
      })

      setResult(result)
      addToHistory(selectedFile, result)

    } catch (err) {
      clearInterval(progressTimerRef.current)
      if (err.name === 'AbortError') return  // user cancelled

      if (err instanceof APIError) {
        setError(
          err.status === 415 ? 'Unsupported file type. Use JPEG, PNG, or WEBP.' :
          err.status === 413 ? 'File too large. Max size is 10 MB.' :
          err.status === 503 ? 'Model is loading. Please wait a moment and retry.' :
          `Analysis failed: ${err.message}`
        )
      } else {
        setError('Could not connect to the backend. Is the server running?')
      }
    }
  }, [selectedFile, threshold, simulateProgress, setLoading, setProgress, setResult, setError, addToHistory])

  const cancel = useCallback(() => {
    abortRef.current?.abort()
    clearInterval(progressTimerRef.current)
    setLoading(false)
    setProgress(0)
  }, [setLoading, setProgress])

  return { run, cancel }
}
