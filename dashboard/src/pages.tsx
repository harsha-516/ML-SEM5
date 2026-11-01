import { useMemo, useState } from 'react'
import { FlatMap } from './FlatMap'
import { Globe } from './Globe'
import { Scorecard } from './Scorecard'
import { RAMP } from './ramp'
import { day, dayTime } from './data'
import type { AlertItem, EventSummary, NewsItem, SiteData } from './data'
import type { Hazard, Storm } from './types'

export const NAMES: Record<Hazard, string> = { cyclone: 'Cyclones', flood: 'Floods', earthquake: 'Earthquakes' }
const GDACS: Record<Hazard, string> = { cyclone: 'TC', flood: 'FL', earthquake: 'EQ' }
const RANK: Record<string, number> = { Red: 0, Orange: 1, Green: 2 }
export const pct = (x: number) => `${Math.round(x * 100)}%`

export function News({ items }: { items?: NewsItem[] }) {
  if (!items?.length) return null
  return (
    <ul className="news">
      {items.map((n) => (
        <li key={n.url}>
          <a href={n.url} target="_blank" rel="noreferrer">{n.title}</a>
          <span>{n.source}, {day(n.published)}</span>
        </li>
      ))}
    </ul>
  )
}

function Alerts({ items, limit = 10 }: { items: AlertItem[]; limit?: number }) {
  if (!items.length) return null
  return (
    <table className="events">
      <thead><tr><th>Event</th><th>Last update</th><th>Alert level</th></tr></thead>
      <tbody>
        {items.slice(0, limit).map((e, i) => (
          <tr key={i}>
            <td>{e.url ? <a href={e.url} target="_blank" rel="noreferrer">{e.name}</a> : e.name}</td>
            <td>{day(e.to)}</td>
            <td><i className={`alert ${e.alert}`} />{e.alert}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export function EventList({ items }: { items: EventSummary[] }) {
  if (!items.length) return null
  return (
    <ul className="majors">
      {items.map((e) => (
        <li key={e.id}>
          <a href={`#/event/${e.id}`}>{e.title}</a>
          <span>{e.place}, {day(e.date)}</span>
          <em>{e.headline}</em>
        </li>
      ))}
    </ul>
  )
}

export function Home({ d }: { d: SiteData }) {
  const [hazard, setHazard] = useState<Hazard>('cyclone')
  const storm = d.storms?.storms[0]
  const ranked = [...new Map(d.alerts.map((a) => [a.name, a])).values()].sort((a, b) => (RANK[a.alert] ?? 3) - (RANK[b.alert] ?? 3) || b.to.localeCompare(a.to))
  const recent = [...d.majors].sort((a, b) => b.date.localeCompare(a.date)).slice(0, 6)
  const line = hazard === 'cyclone'
    ? storm ? `${storm.name} is at ${storm.wind_kt} knots and ${storm.pressure} hPa, near ${storm.lat.toFixed(1)}°N ${storm.lon.toFixed(1)}°E.` : 'No active storm in the JMA feed right now.'
    : hazard === 'earthquake' ? 'Chance of an M5+ earthquake in each area over the next 30 days.' : `Chance of a surge in flood reports over the next 3 days, at ${d.results.flood?.locations ?? 'tracked'} locations.`
  return (
    <>
      <section className="stage" aria-label="Live hazard map">
        <Globe hazard={hazard} results={d.results} storms={d.storms} />
        <div className="overlay">
          <div className="switch" role="tablist" aria-label="Choose a hazard">
            {(Object.keys(NAMES) as Hazard[]).map((h) => (
              <button key={h} role="tab" aria-selected={h === hazard} onClick={() => setHazard(h)}>{NAMES[h]}</button>
            ))}
          </div>
          <p className="line" aria-live="polite">{line}</p>
        </div>
        <div className="ramp">
          <span>Lower risk</span>
          <i style={{ background: `linear-gradient(90deg, ${RAMP.join(',')})` }} />
          <span>Higher risk</span>
          {d.earthDate && <span className="pic">Satellite picture from {day(d.earthDate)}</span>}
        </div>
      </section>
      <section className="page">
        <div className="two first">
          <div>
            <h2>Major events</h2>
            <EventList items={recent} />
            <p><a href="#/events">See all major events</a></p>
          </div>
          <div>
            <h2>Latest reports</h2>
            <News items={d.news.flood?.slice(0, 3).concat(d.news.cyclone?.slice(0, 3) ?? [])} />
          </div>
        </div>
        <h2 className="gap">Disaster alerts, last 60 days</h2>
        <p className="hint">From GDACS, most serious first.</p>
        <Alerts items={ranked} limit={8} />
      </section>
    </>
  )
}

export function EventsPage({ d }: { d: SiteData }) {
  return (
    <section className="page">
      <h1>Major events</h1>
      <p className="lede">Held-out event replays.</p>
      {(Object.keys(NAMES) as Hazard[]).map((h) => (
        <div key={h}>
          <h2 className="gap">{NAMES[h]}</h2>
          <EventList items={d.majors.filter((e) => e.kind === h).sort((a, b) => b.date.localeCompare(a.date))} />
        </div>
      ))}
    </section>
  )
}

function ActiveStorm({ s }: { s: Storm }) {
  const m = s.model
  const jma = (s.source ?? 'JMA') === 'JMA'
  return (
    <div className="active">
      <h2>{s.name}, active now</h2>
      <dl className="facts">
        <div><dt>{jma ? 'JMA wind, 10-minute' : 'IBTrACS wind, 1-minute'}</dt><dd>{s.wind_kt} kt</dd></div>
        {s.pressure != null && <div><dt>Pressure</dt><dd>{s.pressure} hPa</dd></div>}
        {m && <div><dt>Wind in 24 h</dt><dd>{m.wind_24.toFixed(0)} kt</dd></div>}
        {m && <div><dt>80% range</dt><dd>{m.low.toFixed(0)} to {m.high.toFixed(0)} kt</dd></div>}
        {m && <div><dt>{m.reach_64_chance != null ? 'Chance of 64 kt in 24 h' : 'Rapid intensification'}</dt><dd>{pct(m.reach_64_chance ?? m.ri_chance)}</dd></div>}
      </dl>
      {m && <p className="hint">Forecast from {dayTime(m.from)}, valid {dayTime(m.valid)}. {jma ? `Model winds are 1-minute: JMA ${s.wind_kt} kt converts to ${m.wind_now_1min.toFixed(0)} kt. ` : 'IBTrACS positions can lag by a day or more. '}Follow official warnings.</p>}
    </div>
  )
}

export function HazardPage({ hazard, d }: { hazard: Hazard; d: SiteData }) {
  const res = d.results[hazard]
  const dots = useMemo(() => (res?.current_risk ?? []).map((c) => ({ ...c, label: `${pct(c.p)} chance` })), [res])
  const alerts = useMemo(() => d.alerts.filter((e) => e.type === GDACS[hazard]), [d.alerts, hazard])
  const rings = useMemo(() => alerts.map((e) => ({ lat: e.lat, lon: e.lon, label: `${e.name}, ${e.alert}` })), [alerts])
  const top = [...dots].sort((a, b) => b.p - a.p).slice(0, 8)
  const majors = d.majors.filter((e) => e.kind === hazard).sort((a, b) => b.date.localeCompare(a.date))
  const active = hazard === 'cyclone' ? d.storms?.storms ?? [] : []
  const track = useMemo(() => {
    const s = active[0]
    if (!s) return undefined
    return [...s.past.map(([la, lo]) => [lo, la] as [number, number]), [s.lon, s.lat] as [number, number], ...s.forecast.map((f) => [f.lon, f.lat] as [number, number])]
  }, [active])
  return (
    <section className="page">
      <h1>{NAMES[hazard]}</h1>
      <p className="lede">
        {hazard === 'cyclone' && `Storm strength forecasts, trained on ${res?.storms?.train.toLocaleString() ?? 'historic'} North Indian and West Pacific storms.`}
        {hazard === 'flood' && 'Chance of a surge in flood reports over the next 3 days, from rain and river data.'}
        {hazard === 'earthquake' && 'Regional chance of an M5+ event over 30 days. This is never a prediction of when an earthquake will happen.'}
      </p>
      {hazard === 'cyclone' && active.map((s) => <ActiveStorm key={s.name} s={s} />)}
      <FlatMap dots={dots} rings={rings} line={track} marker={active[0] ? [active[0].lon, active[0].lat] : undefined} center={active[0] ? [active[0].lon, active[0].lat] : undefined} label={`${NAMES[hazard]} risk map`} />
      <p className="hint">{dots.length ? 'Coloured dots are our model. ' : ''}{track ? 'White line is the storm track and JMA forecast. ' : ''}White rings are GDACS alerts from the last 60 days. Select one for details.{res?.scored_at && hazard !== 'cyclone' ? ` Last scored ${dayTime(res.scored_at)}.` : ''}</p>
      <div className="two">
        <div><h2>Major events</h2><EventList items={majors} /></div>
        <div><h2>In the news</h2><News items={d.news[hazard]?.slice(0, 6)} /></div>
      </div>
      {top.length > 0 && (
        <>
          <h2 className="gap">Highest risk now</h2>
          <table className="events"><thead><tr><th>Location</th><th>Chance</th></tr></thead><tbody>
            {top.map((c, i) => <tr key={i}><td>{c.lat.toFixed(1)}°, {c.lon.toFixed(1)}°</td><td>{pct(c.p)}</td></tr>)}
          </tbody></table>
        </>
      )}
      <h2 className="gap">Disaster alerts</h2>
      <Alerts items={alerts} />
      <Scorecard hazard={hazard} res={res ?? null} />
    </section>
  )
}
