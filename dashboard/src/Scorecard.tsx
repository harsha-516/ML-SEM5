import type { Hazard, Results } from './types'

const FEATURE_NAMES: Record<string, string> = {
  wind: 'Current wind', pressure: 'Central pressure', dwind_6: 'Wind change, 6 h', dwind_12: 'Wind change, 12 h',
  dp_6: 'Pressure change, 6 h', dp_12: 'Pressure change, 12 h', dp_24: 'Pressure change, 24 h', lat: 'Latitude', lon: 'Longitude',
  age_h: 'Storm age', grade: 'Storm grade', speed_kmh: 'Forward speed', max_wind_so_far: 'Strongest wind so far', month_sin: 'Season',
  dist_land: 'Distance to land', is_north_indian: 'North Indian basin', rain_1: 'Rain today', rain_3: 'Rain, 3 days', rain_7: 'Rain, 7 days', rain_30: 'Rain, 30 days', discharge: 'River discharge', discharge_anom: 'Discharge against normal', discharge_change_3: 'Discharge change, 3 days', etas_trigger: 'Aftershock load', next3_rain: 'Forecast rain, next 3 days', next3_ratio: 'Forecast rain against this place', rain_1_ratio: 'Rain today against this place',
  rain_3_ratio: 'Rain, 3 days against this place', rain_7_ratio: 'Rain, 7 days against this place', rain_30_ratio: 'Rain, 30 days against this place', wet_days_7: 'Wet days, last 7', wet_days_30: 'Wet days, last 30',
  discharge_rank: 'Discharge rank, last year', sst_c: 'Sea temperature', shear_kt: 'Wind shear', pi_proxy: 'Intensity potential', cold_frac: 'Cold cloud share', tb_mean: 'Mean cloud-top temperature', tb_min: 'Coldest cloud top', tb_asym: 'Cloud asymmetry',
  imerg_7: 'IMERG rain, 7 days', imerg_30: 'IMERG rain, 30 days', smap_sm: 'SMAP soil moisture', smap_anom: 'Soil moisture against normal', recent_reports_30: 'Recent flood reports',
  month_cos: 'Season', heading_sin: 'Heading', heading_cos: 'Heading', n_30: 'Events, last 30 days', n_90: 'Events, last 90 days',
  n_360: 'Events, last year', n_1800: 'Events, last 5 years', max_mag_90: 'Largest event, 90 days', days_since_m5: 'Days since last M5+',
  log_energy_360: 'Energy released, last year', mean_depth_360: 'Average depth', neighbour_n_90: 'Nearby events, 90 days',
  neighbour_n_360: 'Nearby events, last year', m5_rate_so_far: 'Long-run M5+ rate',
}

const pct = (x: number) => `${Math.round(x * 100)}%`

function Reliability({ res }: { res: Results }) {
  const pts = res.reliability
  const max = Math.max(0.02, ...pts.map((b) => Math.max(b.predicted, b.observed))) * 1.08
  const s = 240
  const X = (v: number) => 34 + (v / max) * (s - 44)
  const Y = (v: number) => s - 30 - (v / max) * (s - 44)
  return (
    <svg viewBox={`0 0 ${s} ${s}`} role="img" aria-label="Predicted chance against how often the event happened">
      <line x1={X(0)} y1={Y(0)} x2={X(max)} y2={Y(max)} className="ideal" />
      <polyline points={pts.map((b) => `${X(b.predicted)},${Y(b.observed)}`).join(' ')} className="curve" />
      {pts.map((b, i) => <circle key={i} cx={X(b.predicted)} cy={Y(b.observed)} r="3.5" className="dot" />)}
      <text x={s / 2 + 12} y={s - 8} textAnchor="middle">Predicted probability</text>
      <text transform={`translate(13 ${s / 2 - 12}) rotate(-90)`} textAnchor="middle">Observed frequency</text>
    </svg>
  )
}

export function Scorecard({ res }: { hazard?: Hazard; res: Results | null }) {
  if (!res) {
    return null
  }
  const test = res.models[res.winner].test
  const base = res.models.Baseline.test
  const names = Object.keys(res.models).filter((k) => k !== 'Baseline').sort((a, b) => res.models[b].test.pr_auc - res.models[a].test.pr_auc)
  const top = Math.max(...Object.values(res.models).map((m) => m.test.pr_auc)) * 1.1
  const c = test.confusion
  return (
    <section className="score embedded">
      <h2>Test results</h2>
      <p className="lede">Test period: {res.periods.test}. {res.winner}: recall {pct(test.recall)}, precision {pct(test.precision)}, PR-AUC {test.pr_auc.toFixed(3)} against baseline {base.pr_auc.toFixed(3)}.</p>

      <div className="grid">
        <div className="models">
          <h3>PR-AUC by model</h3>
          <p className="hint">Base rate {test.base_rate.toFixed(3)}.</p>
          {names.map((n) => (
            <div className="row" key={n}>
              <span>{n}</span>
              <div className="track">
                <b style={{ width: `${(res.models[n].test.pr_auc / top) * 100}%` }} className={n === res.winner ? 'win' : ''} />
                <i style={{ left: `${(base.pr_auc / top) * 100}%` }} />
              </div>
              <em>{res.models[n].test.pr_auc.toFixed(3)}</em>
            </div>
          ))}
          <p className="hint">Marker: {res.baseline_name}, {base.pr_auc.toFixed(3)}.</p>
        </div>

        <div className="rel">
          <h3>Calibration</h3>
          <Reliability res={res} />
        </div>

        <div>
          <h3>Confusion matrix</h3>
          <table>
            <tbody>
              <tr><th scope="row">True positives</th><td>{c.tp.toLocaleString()}</td></tr>
              <tr><th scope="row">False negatives</th><td>{c.fn.toLocaleString()}</td></tr>
              <tr><th scope="row">False positives</th><td>{c.fp.toLocaleString()}</td></tr>
              <tr><th scope="row">True negatives</th><td>{c.tn.toLocaleString()}</td></tr>
            </tbody>
          </table>
          <p className="hint">Unit: {res.unit}. Target: {res.label}.</p>
        </div>

        <div>
          <h3>Feature importance</h3>
          <ol className="drivers">
            {res.feature_importance.slice(0, 6).map((f) => (
              <li key={f.feature}>
                <span>{FEATURE_NAMES[f.feature] ?? f.feature}</span>
                <em>{pct(f.share)}</em>
              </li>
            ))}
          </ol>
        </div>
      </div>
      {res.note && <p className="note">{res.note}</p>}
    </section>
  )
}
