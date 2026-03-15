import { useStore } from '@/hooks/useStore'
import styles from './Sidebar.module.css'

const TABS = ['regions', 'metrics', 'history']

export default function Sidebar() {
  const { activeTab, setActiveTab, result, history } = useStore()

  return (
    <aside className={styles.sidebar}>
      {/* ── Tab bar ──────────────────────────────────────────────────── */}
      <div className={styles.tabs}>
        {TABS.map((t) => (
          <button
            key={t}
            className={`${styles.tab} ${activeTab === t ? styles.tabActive : ''}`}
            onClick={() => setActiveTab(t)}
          >
            {t}
            {t === 'history' && history.length > 0 && (
              <span className={styles.badge}>{history.length}</span>
            )}
          </button>
        ))}
      </div>

      {/* ── Tab content ──────────────────────────────────────────────── */}
      <div className={styles.content}>
        {activeTab === 'regions'  && <RegionsTab result={result} />}
        {activeTab === 'metrics'  && <MetricsTab result={result} />}
        {activeTab === 'history'  && <HistoryTab />}
      </div>
    </aside>
  )
}

/* ── Regions ─────────────────────────────────────────────────────────────── */

function RegionsTab({ result }) {
  if (!result) return <Empty text="run analysis to see region activations" />

  const sorted = [...result.regions].sort((a, b) => b.score - a.score)

  return (
    <div className={styles.regionList}>
      <p className="section-label">gradcam activation by region</p>
      {sorted.map((r) => (
        <div key={r.name} className={styles.regionRow}>
          <div className={styles.regionSwatch} style={{ background: r.color }} />
          <div className={styles.regionInfo}>
            <span className={styles.regionName}>{r.name}</span>
            <div className={styles.regionTrack}>
              <div
                className={styles.regionFill}
                style={{
                  width: `${r.score * 100}%`,
                  background: r.color,
                  animation: 'bar-grow 0.7s ease',
                }}
              />
            </div>
          </div>
          <span className={styles.regionScore} style={{ color: r.color }}>
            {Math.round(r.score * 100)}%
          </span>
        </div>
      ))}

      <div className={styles.regionLegend}>
        <div className={styles.legendGrad} />
        <div className={styles.legendLabels}>
          <span>low activation</span>
          <span>high activation</span>
        </div>
      </div>
    </div>
  )
}

/* ── Metrics ─────────────────────────────────────────────────────────────── */

const METRIC_ROWS = [
  { key: 'freq_artifacts',        levelKey: 'freq_level',        label: 'FREQ ARTIFACTS' },
  { key: 'skin_texture',          levelKey: 'skin_level',        label: 'SKIN TEXTURE' },
  { key: 'edge_coherence',        levelKey: 'edge_level',        label: 'EDGE COHERENCE' },
  { key: 'lighting_consistency',  levelKey: 'lighting_level',    label: 'LIGHTING CONSIS.' },
]

const LEVEL_COLOR = {
  red:   'var(--red)',
  amber: 'var(--amber)',
  green: 'var(--green)',
}

function MetricsTab({ result }) {
  if (!result) return <Empty text="run analysis to see signal metrics" />

  const m = result.metrics
  const blendPct = Math.round(m.blending_score * 100)
  const blendColor = blendPct > 65 ? 'var(--green)' : blendPct > 35 ? 'var(--amber)' : 'var(--red)'

  return (
    <div>
      <p className="section-label">forensic signal breakdown</p>

      {METRIC_ROWS.map(({ key, levelKey, label }) => (
        <div key={key} className={styles.metricRow}>
          <span className={styles.metricKey}>{label}</span>
          <span
            className={styles.metricVal}
            style={{ color: LEVEL_COLOR[m[levelKey]] || 'var(--text-secondary)' }}
          >
            {m[key]}
          </span>
        </div>
      ))}

      {/* Blending score as a mini gauge */}
      <div className={styles.blendBlock}>
        <div className={styles.blendHeader}>
          <span className={styles.metricKey}>BLENDING SCORE</span>
          <span className={styles.metricVal} style={{ color: blendColor }}>
            {blendPct}%
          </span>
        </div>
        <div className={styles.blendTrack}>
          <div
            className={styles.blendFill}
            style={{
              width: `${blendPct}%`,
              background: blendColor,
              animation: 'bar-grow 0.8s ease',
            }}
          />
        </div>
        <div className={styles.blendAnnotations}>
          <span>FAKE</span>
          <span>REAL</span>
        </div>
      </div>
    </div>
  )
}

/* ── History ─────────────────────────────────────────────────────────────── */

function HistoryTab() {
  const { history, setFile } = useStore()

  if (history.length === 0) return <Empty text="scan history will appear here" />

  return (
    <div>
      <p className="section-label">{history.length} recent scans</p>
      {history.map((h) => {
        const isFake = h.verdict === 'FAKE'
        const color  = isFake ? 'var(--red)' : 'var(--green)'
        return (
          <div key={h.id} className={styles.historyItem}>
            <div
              className={styles.historyThumb}
              style={{ borderColor: isFake ? 'var(--red-border)' : 'var(--green-border)' }}
            >
              {h.faceUrl
                ? <img src={h.faceUrl} alt="" className={styles.historyImg} />
                : <FaceCircle color={color} />
              }
            </div>
            <div className={styles.historyInfo}>
              <span className={styles.historyName}>{h.filename}</span>
              <span className={styles.historyMeta}>
                {h.confidence.toFixed(1)}% · {h.timestamp}
              </span>
            </div>
            <span
              className={styles.historyBadge}
              style={{ color, borderColor: isFake ? 'var(--red-border)' : 'var(--green-border)' }}
            >
              {h.verdict}
            </span>
          </div>
        )
      })}
    </div>
  )
}

function FaceCircle({ color }) {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
      <circle cx="10" cy="10" r="9" stroke={color} strokeWidth="0.8" />
      <circle cx="10" cy="8.5" r="3" stroke={color} strokeWidth="0.8" />
      <path d="M4 17c0-3.3 2.7-6 6-6s6 2.7 6 6" stroke={color} strokeWidth="0.8" />
    </svg>
  )
}

function Empty({ text }) {
  return (
    <div className={styles.empty}>
      <span>{text}</span>
    </div>
  )
}
