import styles from './Header.module.css'
import { useStore } from '@/hooks/useStore'

export default function Header() {
  const { serverOnline, modelLoaded, device } = useStore()

  return (
    <header className={styles.header}>
      <div className={styles.logo}>
        <div className={styles.logoIcon}>
          <div className={styles.logoDot} />
        </div>
        <span className={styles.logoText}>
          FACE<span className={styles.logoAccent}>FORGE</span>
          <span className={styles.logoSub}> — Detection Lab</span>
        </span>
      </div>

      <div className={styles.statusBar}>
        <div className={`${styles.statusPill} ${serverOnline ? styles.online : styles.offline}`}>
          <span className={styles.statusDot} />
          {modelLoaded ? 'MODEL ONLINE' : serverOnline ? 'LOADING...' : 'OFFLINE'}
        </div>
        <span className={styles.statusDetail}>EfficientNet-B4 + GradCAM</span>
        {device && device !== 'offline' && (
          <span className={styles.statusDevice}>{device.toUpperCase()}</span>
        )}
        <span className={styles.statusVersion}>v2.1.0</span>
      </div>
    </header>
  )
}
