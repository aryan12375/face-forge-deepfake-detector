import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { useStore } from '@/hooks/useStore'
import { useAnalyze } from '@/hooks/useAnalyze'
import ProgressBar from '@/components/shared/ProgressBar'
import styles from './UploadPanel.module.css'

const ACCEPTED = { 'image/jpeg': [], 'image/png': [], 'image/webp': [] }
const MAX_SIZE  = 10 * 1024 * 1024

export default function UploadPanel() {
  const { selectedFile, previewUrl, loading, progress, error, setFile } = useStore()
  const { run, cancel } = useAnalyze()

  const onDrop = useCallback((accepted) => {
    if (accepted[0]) setFile(accepted[0])
  }, [setFile])

  const clearFile = useCallback((e) => {
    e.stopPropagation()
    setFile(null)
    useStore.getState().setResult(null)
    useStore.getState().setError(null)
  }, [setFile])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED,
    maxSize: MAX_SIZE,
    multiple: false,
    disabled: loading,
  })

  return (
    <div className={styles.panel}>
      <p className="section-label">// INPUT — facial image</p>

      {/* ── Drop zone / preview ────────────────────────────────────────── */}
      <div
        {...getRootProps()}
        className={`${styles.dropzone} ${isDragActive ? styles.active : ''} ${loading ? styles.disabled : ''}`}
      >
        <input {...getInputProps()} />
        <div className={styles.scanLine} />

        {previewUrl ? (
          <div className={styles.preview}>
            <img src={previewUrl} alt="Selected face" className={styles.previewImg} />
            <div className={styles.previewOverlay}>
              <span className={styles.previewFileName}>{selectedFile?.name}</span>
              {!loading && (
                <span className={styles.previewHint}>click or drop to replace</span>
              )}
            </div>
            {/* ── Clear button ── */}
            {!loading && (
              <button
                className={styles.clearBtn}
                onClick={clearFile}
                type="button"
                title="Clear image"
              >
                ✕
              </button>
            )}
          </div>
        ) : (
          <div className={styles.placeholder}>
            <FaceIcon />
            <p className={styles.placeholderMain}>Drop a face image here</p>
            <p className={styles.placeholderSub}>PNG / JPG / WEBP — max 10 MB</p>
            <button className={styles.browseBtn} type="button">SELECT FILE</button>
          </div>
        )}
      </div>

      {/* ── Error message ─────────────────────────────────────────────── */}
      {error && (
        <div className={styles.error}>
          <span className={styles.errorIcon}>!</span>
          {error}
        </div>
      )}

      {/* ── Progress bar ──────────────────────────────────────────────── */}
      {loading && <ProgressBar value={progress} className={styles.progressBar} />}

      {/* ── Action button ─────────────────────────────────────────────── */}
      {selectedFile && (
        <button
          className={`${styles.runBtn} ${loading ? styles.runBtnLoading : ''}`}
          onClick={loading ? cancel : run}
          type="button"
        >
          {loading ? (
            <>
              <span className={styles.spinner} />
              SCANNING... {progress}%
            </>
          ) : (
            'RUN ANALYSIS'
          )}
        </button>
      )}
    </div>
  )
}

function FaceIcon() {
  return (
    <svg className={styles.icon} viewBox="0 0 48 48" fill="none">
      <circle cx="24" cy="24" r="22.5" stroke="#444" strokeWidth="1"/>
      <circle cx="24" cy="20" r="7" stroke="#444" strokeWidth="1"/>
      <path d="M10 38c0-7.7 6.3-14 14-14s14 6.3 14 14" stroke="#444" strokeWidth="1"/>
      <line x1="33" y1="16" x2="39" y2="16" stroke="var(--red)" strokeWidth="1.5"/>
      <line x1="36" y1="13" x2="36" y2="19" stroke="var(--red)" strokeWidth="1.5"/>
    </svg>
  )
}
