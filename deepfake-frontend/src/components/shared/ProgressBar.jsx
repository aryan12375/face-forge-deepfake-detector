import styles from './ProgressBar.module.css'

export default function ProgressBar({ value = 0, className }) {
  return (
    <div className={`${styles.track} ${className || ''}`}>
      <div
        className={styles.fill}
        style={{ width: `${Math.min(value, 100)}%` }}
      />
      <div className={styles.glow} style={{ left: `${Math.min(value, 100)}%` }} />
    </div>
  )
}
