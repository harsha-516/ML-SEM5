import { useEffect, useState } from 'react'
import { ago, dayTime, getJson } from './data'
import type { FeedItem, SiteData, SourceFile } from './data'

const STATE: Record<string, string> = { live: 'Live', failed: 'Failing', archive: 'Archive', off: 'Offline' }

function useLive() {
  const [src, setSrc] = useState<SourceFile | null>(null)
  const [feed, setFeed] = useState<FeedItem[]>([])
  useEffect(() => {
    const load = () => {
      getJson<SourceFile>('sources').then(setSrc)
      getJson<FeedItem[]>('feed').then((f) => setFeed(f ?? []))
    }
    load()
    const id = setInterval(load, 60000)
    return () => clearInterval(id)
  }, [])
  return { src, feed }
}

export function Ticker({ feed }: { feed: FeedItem[] }) {
  if (!feed.length) return null
  return (
    <div className="ticker" tabIndex={0} aria-label="Latest items from live sources">
      <ul style={{ animationDuration: `${feed.length * 5}s` }}>
        {[...feed, ...feed].map((f, i) => (
          <li key={i} aria-hidden={i >= feed.length}>
            <b>{f.source}</b>
            {f.url ? <a href={f.url} target="_blank" rel="noreferrer">{f.text}</a> : <span>{f.text}</span>}
            <em>{ago(f.time)}</em>
          </li>
        ))}
      </ul>
    </div>
  )
}

export function LivePage({ d }: { d: SiteData }) {
  const { src, feed } = useLive()
  const rows = src?.sources ?? []
  const quake = d.results.earthquake
  const live = rows.filter((s) => s.state === 'live').length
  const checked = rows.filter((s) => s.state === 'live' || s.state === 'failed').length
  return (
    <section className="page">
      <h1>Live data</h1>
      {src && <p className="lede">{live} of {checked} sources live, checked {ago(src.refreshed_at)}.</p>}
      <h2>Arriving now</h2>
      <Ticker feed={feed} />
      <h2 className="gap">Every source</h2>
      <div className="wall">
        {rows.map((s) => (
          <div key={s.id} className={`src ${s.state}`}>
            <div className="head"><i aria-hidden="true" /><strong>{s.name}</strong><span>{STATE[s.state] ?? s.state}</span></div>
            <p>{s.latest}</p>
            <small>
              {s.use}, {s.kind}
              {s.latency_ms != null && `, ${s.latency_ms} ms`}
              {s.state === 'failed' && `, last worked ${ago(s.last_ok)}`}
            </small>
          </div>
        ))}
      </div>
      <h2 className="gap">Forecast log</h2>
      {src && <p>{src.forecasts_logged.toLocaleString()} forecasts stored.</p>}
      {d.storms && <p className="hint">Storm feed last read {dayTime(d.storms.fetched_at)}.</p>}
      <h2 className="gap">Known limits</h2>
      <ul className="limits">
        {quake && <li>Earthquake skill is modest: PR-AUC {quake.models[quake.winner].test.pr_auc.toFixed(3)} against {quake.models.Baseline.test.pr_auc.toFixed(3)} for {quake.baseline_name}. It estimates regional activity and cannot predict timing.</li>}
        <li>Flood labels come from news reports, so a surge means more reported flooding, not a measured water level. Places with less coverage are under-represented.</li>
        <li>Test scores use cleaned historical records. Live inputs are rougher, so live performance will likely be lower.</li>
        <li>This is a student project. Always follow your national weather and disaster agencies.</li>
      </ul>
    </section>
  )
}
