import { useStore } from '@/hooks/useStore'
import { resolveOutputUrl } from '@/api/client'
import ThresholdSlider from '@/components/shared/ThresholdSlider'
import styles from './ResultsPanel.module.css'

export default function ResultsPanel() {
  const { result, loading } = useStore()

  if (loading) return <LoadingSkeleton />
  if (!result)  return null

  return (
    <div className={`${styles.panel} fade-in`}>
      <p className="section-label">// OUTPUT — forensic analysis</p>

      {/* ── Image pair: face + GradCAM overlay ──────────────────────── */}
      <div className={styles.imageGrid}>
        <ImagePane
          src={resolveOutputUrl(result.face_url)}
          label="input face"
        />
        <ImagePane
          src={resolveOutputUrl(result.gradcam_url)}
          label="gradcam overlay"
          accent
        />
      </div>

      {/* ── FFT spectrum ─────────────────────────────────────────────── */}
      {result.fft_url && (
        <div className={styles.fftRow}>
          <div className={styles.fftPane}>
            <img
              src={resolveOutputUrl(result.fft_url)}
              alt="FFT frequency spectrum"
              className={styles.fftImg}
            />
            <span className={styles.paneLabel}>fft spectrum</span>
          </div>
          <div className={styles.fftInfo}>
            <p className={styles.fftTitle}>Frequency Domain Analysis</p>
            <p className={styles.fftDesc}>
              AI-generated faces leave distinctive high-frequency ring artifacts
              in the Fourier spectrum — GAN grid patterns or diffusion blocking.
              Elevated HF energy indicates manipulation.
            </p>
            <FFTScoreBar result={result} />
          </div>
        </div>
      )}

      {/* ── Verdict bar ───────────────────────────────────────────────── */}
      <VerdictBar result={result} />

      {/* ── Threshold slider ──────────────────────────────────────────── */}
      <ThresholdSlider />

      {/* ── Processing time ───────────────────────────────────────────── */}
      <div className={styles.meta}>
        <span>processed in {result.processing_time_ms} ms</span>
        <span>{result.model_version}</span>
      </div>
    </div>
  )
}

/* ── Sub-components ────────────────────────────────────────────────────────── */

function ImagePane({ src, label, accent }) {
  return (
    <div className={`${styles.imagePane} ${accent ? styles.imagePaneAccent : ''}`}>
      {src
        ? <img src={src} alt={label} className={styles.paneImg} />
        : <div className={styles.paneEmpty} />
      }
      <span className={styles.paneLabel}>{label}</span>
    </div>
  )
}

function FFTScoreBar({ result }) {
  // Derive a 0-1 HF anomaly score from blending_score (inverse)
  const anomaly = Math.round((1 - result.metrics.blending_score) * 100)
  const color = anomaly > 65 ? 'var(--red)' : anomaly > 35 ? 'var(--amber)' : 'var(--green)'
  return (
    <div className={styles.fftScore}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
        <span className={`mono ${styles.fftScoreLabel}`}>HF ANOMALY</span>
        <span className="mono" style={{ fontSize: 11, color }}>{anomaly}%</span>
      </div>
      <div className={styles.trackBg}>
        <div
          className={styles.trackFill}
          style={{ width: `${anomaly}%`, background: color, animation: 'bar-grow 0.8s ease' }}
        />
      </div>
    </div>
  )
}

function VerdictBar({ result }) {
  const { threshold, getVerdict } = useStore()
  const verdict  = getVerdict()
  const isFake   = verdict === 'FAKE'
  const isUncert = verdict === 'UNCERTAIN'

  const accentColor = isFake ? 'var(--red)' : isUncert ? 'var(--amber)' : 'var(--green)'
  const bgColor     = isFake ? 'var(--red-dim)' : isUncert ? 'var(--amber-dim)' : 'var(--green-dim)'
  const borderColor = isFake ? 'var(--red-border)' : isUncert ? 'var(--amber-dim)' : 'var(--green-border)'

  return (
    <div className={styles.verdict} style={{ background: bgColor, borderColor }}>
      <div className={styles.verdictTop}>
        <span className={styles.verdictLabel}>// VERDICT</span>
        <span className={styles.verdictTag} style={{ color: accentColor, borderColor: accentColor }}>
          {verdict}
        </span>
      </div>

      <div className={styles.verdictNum} style={{ color: 'var(--text-primary)' }}>
        {result.confidence.toFixed(1)}%
      </div>
      <div className={styles.verdictDesc}>
        {isFake
          ? `AI manipulation detected with ${result.confidence.toFixed(1)}% probability`
          : isUncert
          ? `Borderline result — ${result.confidence.toFixed(1)}% fake probability near threshold`
          : `Image appears authentic (${(100 - result.confidence).toFixed(1)}% real probability)`
        }
      </div>

      <div className={styles.confTrack}>
        <div
          className={styles.confFill}
          style={{
            width: `${result.confidence}%`,
            background: accentColor,
            animation: 'bar-grow 1.2s cubic-bezier(0.16, 1, 0.3, 1)',
          }}
        />
      </div>
    </div>
  )
}

function LoadingSkeleton() {
  return (
    <div className={styles.skeleton}>
      {[220, 40, 100, 60].map((h, i) => (
        <div key={i} className={styles.skeletonBlock} style={{ height: h }} />
      ))}
    </div>
  )
}
