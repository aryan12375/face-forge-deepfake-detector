import { useEffect } from 'react'
import Header from '@/components/layout/Header'
import UploadPanel from '@/components/panels/UploadPanel'
import ResultsPanel from '@/components/panels/ResultsPanel'
import Sidebar from '@/components/panels/Sidebar'
import { useHealth } from '@/hooks/useHealth'
import styles from './App.module.css'

export default function App() {
  useHealth()   // polls /health on mount + every 30s

  return (
    <div className={styles.root}>
      <Header />

      <div className={styles.body}>
        {/* ── Center column: upload + results ───────────────────────── */}
        <main className={styles.center}>
          <div className={styles.centerInner}>
            <UploadPanel />
            <ResultsPanel />
          </div>
        </main>

        {/* ── Right sidebar: regions / metrics / history ─────────────── */}
        <Sidebar />
      </div>
    </div>
  )
}
