import { useEffect, useMemo, useState } from 'react'
import { FlatMap } from './FlatMap'
import { News, pct } from './pages'
import { day, dayTime, getJson } from './data'
import type { EventDetail, FloodDay, Fix, NewsItem, Shock } from './data'

function snapshot(date: string, lat: number, lon: number): string {
  const b = `${lat - 8},${lon - 10},${lat + 8},${lon + 10}`
  return `https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot&LAYERS=VIIRS_SNPP_CorrectedReflectance_TrueColor&CRS=EPSG:4326&TIME=${date}&BBOX=${b}&FORMAT=image/jpeg&WIDTH=800&HEIGHT=640`
}

function ForecastChart({ t, at }: { t: Fix[]; at: number }) {
  const w = 720, h = 280
  const top = Math.max(120, ...t.map((f) => Math.max(f.wind_kt ?? 0, f.forecast_high))) * 1.08
  const X = (i: number) => 46 + (i / (t.length + 3)) * (w - 66)
  const Y = (v: number) => h - 34 - (v / top) * (h - 54)
  const fut = (i: number) => X(i + 4)
  const band = t.map((f, i) => `${fut(i)},${Y(f.forecast_high)}`).join(' ') + ' ' + [...t].map((f, i) => `${fut(i)},${Y(f.forecast_low)}`).reverse().join(' ')
  return (
    <svg className="chart" viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Forecast wind against observed wind">
      {[0, 40, 80, 120, 160].filter((v) => v < top).map((v) => <g key={v}><line x1="46" x2={w - 16} y1={Y(v)} y2={Y(v)} className="gridline" /><text x="38" y={Y(v) + 4} textAnchor="end">{v}</text></g>)}
      <polygon points={band} className="band" />
      <polyline points={t.map((f, i) => `${fut(i)},${Y(f.forecast_wind_24)}`).join(' ')} className="fc" />
      <polyline points={t.flatMap((f, i) => (f.wind_kt == null ? [] : [`${X(i)},${Y(f.wind_kt)}`])).join(' ')} className="obs" />
      <line x1={X(at)} x2={X(at)} y1="14" y2={h - 34} className="now" />
      <text x="46" y={h - 8}>Wind in knots. White: observed. Magenta: forecast issued 24 hours earlier, with its 80% range.</text>
    </svg>
  )
}

function CyclonePanel({ e }: { e: EventDetail }) {
  const t = e.track!
  const [i, setI] = useState(0)
  const [play, setPlay] = useState(false)
  useEffect(() => {
    if (!play) return
    const id = setInterval(() => setI((n) => {
      if (n >= t.length - 1) { setPlay(false); return n }
      return n + 1
    }), 700)
    return () => clearInterval(id)
  }, [play, t.length])
  const line = useMemo(() => t.slice(0, i + 1).map((f) => [f.lon, f.lat] as [number, number]), [t, i])
  const f = t[i]
  const date = f.time.slice(0, 10)
  const mid = t[Math.floor(t.length / 2)]
  return (
    <>
      <div className="player">
        <button onClick={() => { if (i >= t.length - 1) setI(0); setPlay(!play) }}>{play ? 'Pause' : 'Play'}</button>
        <input type="range" min="0" max={t.length - 1} value={i} aria-label="Replay time" onChange={(x) => { setPlay(false); setI(+x.target.value) }} />
        <strong>{dayTime(f.time)}</strong>
      </div>
      <div className="two tight">
        <FlatMap dots={[]} line={line} marker={[f.lon, f.lat]} center={[mid.lon, mid.lat]} zoom={3.6} label={`${e.title} track`} />
        <figure className="sat">
          <img src={snapshot(date, f.lat, f.lon)} alt={`NASA VIIRS satellite picture around the storm on ${date}`} />
          <figcaption>NASA VIIRS picture centred on the storm, {day(date)}. One pass a day.</figcaption>
        </figure>
      </div>
      <dl className="facts">
        <div><dt>Wind now</dt><dd>{f.wind_kt ?? 'unknown'} kt</dd></div>
        <div><dt>Forecast in 24 h</dt><dd>{Math.round(f.forecast_wind_24)} kt</dd></div>
        <div><dt>What happened</dt><dd>{f.actual_wind_24 != null ? `${f.actual_wind_24} kt` : 'no record'}</dd></div>
        <div><dt>Rapid growth chance</dt><dd>{pct(f.rapid_intensification_chance)}</dd></div>
        <div><dt>Distance to land</dt><dd>{f.dist_land_km ?? 'unknown'} km</dd></div>
      </dl>
      <ForecastChart t={t} at={i} />
    </>
  )
}

function FloodChart({ days, date }: { days: FloodDay[]; date: string }) {
  const w = 720, h = 300
  const rain = Math.max(10, ...days.map((d) => d.rain_mm))
  const X = (i: number) => 46 + (i / (days.length - 1)) * (w - 66)
  const R = (v: number) => (v / rain) * 90
  const Y = (p: number) => 150 - p * 130
  const at = days.findIndex((d) => d.date === date)
  return (
    <svg className="chart" viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Model chance and rainfall by day around the event">
      {[0, 0.5, 1].map((v) => <g key={v}><line x1="46" x2={w - 16} y1={Y(v)} y2={Y(v)} className="gridline" /><text x="38" y={Y(v) + 4} textAnchor="end">{v * 100}%</text></g>)}
      {days.map((d, i) => <rect key={d.date} x={X(i) - 4} y={260 - R(d.rain_mm)} width="8" height={R(d.rain_mm)} className="rain" />)}
      <polyline points={days.map((d, i) => `${X(i)},${Y(d.chance)}`).join(' ')} className="fc" />
      <line x1={X(at)} x2={X(at)} y1="14" y2="262" className="now" />
      <text x={X(at) + 6} y="24">Event day</text>
      <text x="46" y="280">Magenta: model chance that day. Blue bars: daily rain, tallest {rain.toFixed(0)} mm.</text>
    </svg>
  )
}

function FloodPanel({ e }: { e: EventDetail }) {
  const days = e.days!
  const before = days.filter((d) => d.date < e.date).slice(-3)
  const rain = before.reduce((s, d) => s + d.rain_mm, 0)
  const top = Math.max(...before.map((d) => d.chance))
  const earlier = days.filter((d) => d.date < e.date).slice(0, -3).map((d) => d.chance).sort((a, b) => a - b)
  const usual = e.usual ?? earlier[Math.floor(earlier.length / 2)]
  const raised = e.warned ?? top >= usual * 1.5
  return (
    <>
      <div className="two tight">
        <FlatMap dots={[]} marker={[e.lon!, e.lat!]} center={[e.lon!, e.lat!]} zoom={4.6} label={`Location of ${e.title}`} />
        <figure className="sat">
          <img src={snapshot(e.date, e.lat!, e.lon!)} alt={`NASA VIIRS satellite picture around ${e.place} on ${e.date}`} />
          <figcaption>NASA VIIRS picture, {day(e.date)}.</figcaption>
        </figure>
      </div>
      <dl className="facts">
        <div><dt>Highest chance in 3 days before</dt><dd>{pct(top)}</dd></div>
        <div><dt>Usual chance before that</dt><dd>{pct(usual)}</dd></div>
        <div><dt>Rain in 3 days before</dt><dd>{rain.toFixed(0)} mm</dd></div>
        <div><dt>Did the model warn?</dt><dd>{raised ? 'Yes, risk rose' : 'No clear warning'}</dd></div>
      </dl>
      <FloodChart days={days} date={e.date} />
      <p className="note">Warning criterion: chance at least 1.5 times its 30-day median. Inputs are rain, river data and rain forecasts from the {e.forecast_source ?? 'archive'}.</p>
    </>
  )
}

function ShockChart({ seq }: { seq: Shock[] }) {
  const w = 720, h = 260
  const top = Math.max(7, ...seq.map((s) => s.mag))
  const X = (d: number) => 46 + ((d + 30) / 60) * (w - 66)
  const Y = (m: number) => h - 40 - ((m - 4.3) / (top + 0.2 - 4.3)) * (h - 60)
  return (
    <svg className="chart" viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Earthquakes nearby in the 30 days before and after">
      {[5, 6, 7].filter((m) => m <= top).map((m) => <g key={m}><line x1="46" x2={w - 16} y1={Y(m)} y2={Y(m)} className="gridline" /><text x="38" y={Y(m) + 4} textAnchor="end">M{m}</text></g>)}
      <line x1={X(0)} x2={X(0)} y1="14" y2={h - 40} className="now" />
      {seq.map((s, i) => <circle key={i} cx={X(s.days)} cy={Y(s.mag)} r={s.mag >= 6 ? 6 : 3} className="dot" />)}
      <text x="46" y={h - 12}>Events within about 200 km, 30 days before to 30 days after. Dashed line: main shock.</text>
    </svg>
  )
}

function QuakePanel({ e }: { e: EventDetail }) {
  const seq = e.sequence!
  const after = seq.filter((s) => s.days > 0).length
  const before = seq.filter((s) => s.days < 0).length
  return (
    <>
      <div className="two tight">
        <FlatMap dots={[]} marker={[e.lon!, e.lat!]} center={[e.lon!, e.lat!]} zoom={4.4} label={`Location of ${e.title}`} />
        <dl className="facts stack">
          {e.cell_chance != null && <div><dt>Model chance for this cell that window</dt><dd>{pct(e.cell_chance)}</dd></div>}
          {e.cell_rank != null && <div><dt>Higher than</dt><dd>{pct(e.cell_rank)} of cells</dd></div>}
          <div><dt>Nearby M4.5+ before and after</dt><dd>{before} before, {after} after</dd></div>
        </dl>
      </div>
      <ShockChart seq={seq} />
      <p className="note">Forecast window starts {e.window_start ? day(e.window_start) : 'unknown'}, 30 days.</p>
    </>
  )
}

export function EventPage({ id, news }: { id: string; news: Record<string, NewsItem[]> }) {
  const [e, set] = useState<EventDetail | null | undefined>(undefined)
  useEffect(() => { getJson<EventDetail>(`events/${id}`).then(set) }, [id])
  if (e === undefined) return <section className="page"><p className="lede">Loading event.</p></section>
  if (e === null) return <section className="page"><h1>Event not found</h1><p className="lede"><a href="#/events">Go to major events</a></p></section>
  return (
    <section className="page">
      <p><a href="#/events">Major events</a></p>
      <h1>{e.title}</h1>
      <p className="lede">{e.place}, {day(e.date)}. {e.summary ?? (e.kind === 'cyclone' ? 'This storm stayed out of training, so these are the forecasts the model would have made at each point.' : '')}</p>
      {e.kind === 'cyclone' && <CyclonePanel e={e} />}
      {e.kind === 'flood' && <FloodPanel e={e} />}
      {e.kind === 'earthquake' && <QuakePanel e={e} />}
      <h2 className="gap">News coverage</h2>
      <News items={news[id]} />
    </section>
  )
}
