import { useStore } from '@/hooks/useStore'
import styles from './ThresholdSlider.module.css'

export default function ThresholdSlider({ className }) {
  const { threshold, setThreshold, result, getVerdict } = useStore()
  const pct     = Math.round(threshold * 100)
  const verdict  = result ? getVerdict() : null
  const isFake   = verdict === 'FAKE'
  const isUncert = verdict === 'UNCERTAIN'

  const accentColor = !verdict ? 'var(--text-muted)'
    : isFake   ? 'var(--red)'
    : isUncert ? 'var(--amber)'
    : 'var(--green)'

  return (
    <div className={`${styles.wrapper} ${className || ''}`}>
      <div className={styles.header}>
        <span className={styles.label}>CONFIDENCE THRESHOLD</span>
        <div className={styles.right}>
          {verdict && (
            <span className={styles.verdictHint} style={{ color: accentColor }}>
              → {verdict}
            </span>
          )}
          <span className={styles.value}>{pct}%</span>
        </div>
      </div>

      <input
        type="range"
        min="10"
        max="95"
        step="1"
        value={pct}
        className={styles.slider}
        style={{ '--accent': accentColor }}
        onChange={(e) => setThreshold(Number(e.target.value) / 100)}
      />

      <div className={styles.annotations}>
        <span>lenient</span>
        <span>strict</span>
      </div>

      <p className={styles.hint}>
        Images above {pct}% fake probability are flagged as FAKE.
        Drag to adjust sensitivity.
      </p>
    </div>
  )
}
