import { useState } from 'react'
import { Scorecard } from './Scorecard'
import { NAMES, pct } from './pages'
import { day, dayTime } from './data'
import type { ExperimentTable, Lab, LiveHazard, SiteData } from './data'
import type { Hazard, Results } from './types'

const SPEC: Record<Hazard, { target: string; features: string }> = {
  cyclone: { target: 'Reaches 64 kt within 24 h; 24 h wind forecast', features: 'Wind, pressure, 6 to 24 h trends, position, season, distance to land' },
  flood: { target: 'Flood report surge within 3 days', features: 'Rain over 1, 3, 7 and 30 days and against local normal, river discharge, season, 3-day rain forecast' },
  earthquake: { target: 'M5.0+ in 1 degree cell within 30 days', features: 'Event counts, energy, depth, neighbouring activity, aftershock load' },
}

function training(r?: Results) {
  if (!r) return ''
  if (r.storms) return `${r.storms.train.toLocaleString()} storms, ${r.periods.train}`
  if (r.locations) return `${r.locations} locations, ${r.periods.train}`
  return `USGS catalogue, ${r.periods.train}`
}

function Tables({ items }: { items: ExperimentTable[] }) {
  return (
    <div className="exp">
      {items.map((t) => (
        <div key={t.title} className="exp-table">
          <h3>{t.title}</h3>
          <div className="scroll">
            <table className="events wide">
              <thead><tr>{t.columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
              <tbody>{t.rows.map((r, i) => <tr key={i}>{r.map((v, j) => <td key={j}>{v}</td>)}</tr>)}</tbody>
            </table>
          </div>
          {t.note && <p className="hint">{t.note}</p>}
        </div>
      ))}
    </div>
  )
}

function LiveCard({ h, s }: { h: Hazard; s: LiveHazard }) {
  return (
    <div className="exp-table">
      <h3>{NAMES[h]}</h3>
      <dl className="facts stack-3">
        <div><dt>Forecasts scored</dt><dd>{s.scored.toLocaleString()}</dd></div>
        {s.brier != null && <div><dt>Mean chance, then observed</dt><dd>{pct(s.mean_chance!)}, {pct(s.observed_rate!)}</dd></div>}
        {s.wind_mae_kt != null && <div><dt>24 h wind error</dt><dd>{s.wind_mae_kt.toFixed(1)} kt</dd></div>}
        {s.in_range != null && <div><dt>Inside 80% range</dt><dd>{pct(s.in_range)}</dd></div>}
        <div><dt>Waiting for window to close</dt><dd>{s.pending.toLocaleString()}</dd></div>
      </dl>
      <p className="hint">Outcome: {s.outcome_source}</p>
    </div>
  )
}

function ThresholdTool({ r }: { r: Results }) {
  const c = r.pr_curve ?? []
  const start = Math.max(0, c.findIndex((p) => p.threshold >= r.models[r.winner].test.threshold))
  const [i, setI] = useState(start)
  const p = c[Math.min(i, c.length - 1)]
  if (!p) return null
  return (
    <div className="tool">
      <label htmlFor="thr">Alert threshold <strong>{(p.threshold * 100).toFixed(1)}%</strong></label>
      <input id="thr" type="range" min="0" max={c.length - 1} value={i} onChange={(e) => setI(+e.target.value)} />
      <dl className="facts">
        <div><dt>Events caught</dt><dd>{p.caught.toLocaleString()}</dd></div>
        <div><dt>Events missed</dt><dd>{p.missed.toLocaleString()}</dd></div>
        <div><dt>False alarms</dt><dd>{p.false_alarms.toLocaleString()}</dd></div>
        <div><dt>Precision</dt><dd>{pct(p.precision)}</dd></div>
      </dl>
    </div>
  )
}

function Curve({ rows, x, label }: { rows: { x: string; train: number; val: number }[]; x: string; label: string }) {
  const w = 420, h = 200, top = Math.max(...rows.flatMap((r) => [r.train, r.val])) * 1.08
  const X = (i: number) => 40 + (i / (rows.length - 1)) * (w - 56)
  const Y = (v: number) => h - 34 - (v / top) * (h - 50)
  return (
    <svg className="chart" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={label}>
      <line x1="40" x2={w - 16} y1={Y(0)} y2={Y(0)} className="gridline" />
      <polyline points={rows.map((r, i) => `${X(i)},${Y(r.train)}`).join(' ')} className="obs" />
      <polyline points={rows.map((r, i) => `${X(i)},${Y(r.val)}`).join(' ')} className="fc" />
      {rows.map((r, i) => <text key={i} x={X(i)} y={h - 18} textAnchor="middle">{r.x}</text>)}
      <text x={w / 2} y={h - 3} textAnchor="middle">{x}</text>
    </svg>
  )
}

function Experiments({ lab }: { lab: Lab }) {
  const tree = lab.regularisation.tree.map((r) => ({ x: r.max_depth ? String(r.max_depth) : 'none', train: r.train_pr, val: r.val_pr }))
  const t = lab.default_vs_tuned
  return (
    <>
      <div className="two">
        <div>
          <h3>Decision tree depth: train (white) and validation (magenta) PR-AUC</h3>
          <Curve rows={tree} x="Tree depth" label="Decision tree score against depth" />
        </div>
        <div>
          <h3>XGBoost tuning, time-ordered folds</h3>
          <table className="events">
            <thead><tr><th>Settings</th><th>Validation</th><th>Test</th></tr></thead>
            <tbody>
              <tr><td>Defaults</td><td>{t.default_val.toFixed(3)}</td><td>{t.default_test.toFixed(3)}</td></tr>
              <tr><td>Tuned</td><td>{t.tuned_val.toFixed(3)}</td><td>{t.tuned_test.toFixed(3)}</td></tr>
            </tbody>
          </table>
        </div>
      </div>
      <div className="two">
        <div>
          <h3>Class imbalance</h3>
          <table className="events"><tbody>
            <tr><td>No correction</td><td>{lab.imbalance.no_correction.toFixed(3)}</td></tr>
            <tr><td>Class weights</td><td>{lab.imbalance.class_weights.toFixed(3)}</td></tr>
            <tr><td>SMOTE</td><td>{lab.imbalance.smote.toFixed(3)}</td></tr>
          </tbody></table>
        </div>
        <div>
          <h3>Logistic Regression regularisation</h3>
          <table className="events">
            <thead><tr><th>Penalty</th><th>Validation</th><th>Non-zero features</th></tr></thead>
            <tbody>{lab.regularisation.logistic.filter((r) => r.C === 0.01).map((r) => <tr key={r.penalty}><td>{r.penalty}</td><td>{r.val_pr.toFixed(3)}</td><td>{r.nonzero}</td></tr>)}</tbody>
          </table>
        </div>
      </div>
    </>
  )
}

export function ModelsPage({ d }: { d: SiteData }) {
  const [h, setH] = useState<Hazard>('cyclone')
  const res = d.results[h]
  const lab = d.labs[h]
  const live = (Object.entries(d.liveScore?.hazards ?? {}) as [Hazard, LiveHazard][])
  return (
    <section className="page">
      <h1>Models</h1>
      <div className="tabs gapped" role="tablist">
        {(Object.keys(NAMES) as Hazard[]).map((k) => <button key={k} role="tab" aria-selected={k === h} onClick={() => setH(k)}>{NAMES[k]}</button>)}
      </div>
      <dl className="ask">
        <div><dt>Target</dt><dd>{SPEC[h].target}</dd></div>
        <div><dt>Features</dt><dd>{SPEC[h].features}</dd></div>
        <div><dt>Training data</dt><dd>{training(res)}</dd></div>
        {res && <div><dt>Selected model</dt><dd>{res.winner}, trained {day(res.generated_at)}</dd></div>}
      </dl>

      <h2 className="gap">Threshold analysis, test period</h2>
      {res && <ThresholdTool key={h} r={res} />}

      <h2 className="gap">Live pipeline</h2>
      <ol className="steps">
        <li><strong>Ingest</strong><span>Every 5 minutes</span></li>
        <li><strong>Storm scoring</strong><span>Each refresh, from saved models</span></li>
        <li><strong>Flood and earthquake maps</strong><span>Daily, from saved models</span></li>
        <li><strong>Forecast log</strong><span>SQLite, with model date</span></li>
        <li><strong>Outcomes</strong><span>Written when each window closes</span></li>
      </ol>

      {live.length > 0 && (
        <>
          <h2 className="gap">Live scorecard</h2>
          {d.liveScore && <p className="hint">Updated {dayTime(d.liveScore.generated_at)}.</p>}
          <div className="exp">{live.map(([k, s]) => <LiveCard key={k} h={k} s={s} />)}</div>
        </>
      )}

      <h2 className="gap">Experiments</h2>
      {(d.experiments[h]?.length ?? 0) > 0 && <Tables items={d.experiments[h]!} />}
      {lab && <Experiments lab={lab} />}

      <Scorecard hazard={h} res={res ?? null} />
    </section>
  )
}
