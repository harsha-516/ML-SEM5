import { useEffect, useState } from 'react'
import type { Hazard, Results, StormFile } from './types'

export interface NewsItem { title: string; source: string; url: string; published: string }
export interface AlertItem { type: string; name: string; alert: string; from: string; to: string; country: string; lat: number; lon: number; url?: string }
export interface EventSummary { id: string; kind: Hazard; title: string; place: string; date: string; headline: string; severity: number }
export interface Fix {
  time: string; lat: number; lon: number; wind_kt: number | null; pressure: number | null; dist_land_km: number | null
  forecast_wind_24: number; forecast_low: number; forecast_high: number; actual_wind_24: number | null; rapid_intensification_chance: number
}
export interface FloodDay { date: string; rain_mm: number; discharge: number | null; chance: number; next3_mm?: number }
export interface Shock { days: number; mag: number; depth: number }
export interface EventDetail extends EventSummary {
  track?: Fix[]; days?: FloodDay[]; sequence?: Shock[]; summary?: string; lat?: number; lon?: number; mag?: number
  cell_chance?: number | null; cell_rank?: number | null; window_start?: string
  forecast_source?: string; warned?: boolean; top_3_days?: number; usual?: number
}
export interface Lab {
  hazard: string
  tuning_rows: number
  regularisation: {
    logistic: { penalty: string; C: number; train_pr: number; val_pr: number; nonzero: number }[]
    tree: { max_depth: number; train_pr: number; val_pr: number }[]
  }
  imbalance: Record<string, number>
  tuning: { best: Record<string, number> }
  default_vs_tuned: Record<string, number>
}
export interface ExperimentTable { title: string; columns: string[]; rows: string[][]; note?: string | null }
export interface LiveHazard { scored: number; pending: number; happened?: number; mean_chance?: number; observed_rate?: number; brier?: number; wind_mae_kt?: number; in_range?: number; outcome_source: string }
export interface LiveScore { generated_at: string; hazards: Partial<Record<Hazard, LiveHazard>> }
export interface Source { id: string; name: string; use: string; kind: string; state: string; latest: string; records?: number | null; latency_ms?: number; checked_at?: string; last_ok?: string | null }
export interface SourceFile { refreshed_at: string; forecasts_logged: number; sources: Source[] }
export interface FeedItem { source: string; text: string; time: string; url?: string }

export interface SiteData {
  results: Partial<Record<Hazard, Results>>
  labs: Partial<Record<Hazard, Lab>>
  storms: StormFile | null
  news: Record<string, NewsItem[]>
  eventNews: Record<string, NewsItem[]>
  alerts: AlertItem[]
  majors: EventSummary[]
  earthDate: string | null
  experiments: Partial<Record<Hazard, ExperimentTable[]>>
  liveScore: LiveScore | null
}

export async function getJson<T>(name: string): Promise<T | null> {
  try {
    const r = await fetch(`${import.meta.env.BASE_URL}data/${name}.json?t=${Date.now()}`)
    return r.ok ? ((await r.json()) as T) : null
  } catch {
    return null
  }
}

export function useSiteData(): SiteData {
  const [d, set] = useState<SiteData>({ results: {}, labs: {}, storms: null, news: {}, eventNews: {}, alerts: [], majors: [], earthDate: null, experiments: {}, liveScore: null })
  useEffect(() => {
    const hz: Hazard[] = ['cyclone', 'flood', 'earthquake']
    Promise.all([
      Promise.all(hz.map((h) => getJson<Results>(h))), Promise.all(hz.map((h) => getJson<Lab>(`lab_${h}`))), getJson<StormFile>('storms'),
      getJson<Record<string, NewsItem[]>>('news'), getJson<Record<string, NewsItem[]>>('event_news'), getJson<AlertItem[]>('events'),
      getJson<EventSummary[]>('event_index'), getJson<{ date: string }>('earth'), getJson<Partial<Record<Hazard, ExperimentTable[]>>>('experiments'), getJson<LiveScore>('live_score'),
    ]).then(([res, labs, storms, news, eventNews, alerts, majors, earth, experiments, liveScore]) => {
      const pick = <T,>(a: (T | null)[]) => Object.fromEntries(hz.flatMap((h, i) => (a[i] ? [[h, a[i]]] : []))) as Partial<Record<Hazard, T>>
      set({ results: pick(res), labs: pick(labs), storms, news: news ?? {}, eventNews: eventNews ?? {}, alerts: alerts ?? [], majors: majors ?? [], earthDate: earth?.date ?? null, experiments: experiments ?? {}, liveScore })
    })
  }, [])
  return d
}

export const day = (iso: string) => new Date(iso).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })
export const dayTime = (iso: string) => new Date(iso.includes('T') ? iso : `${iso.replace(' ', 'T')}Z`).toLocaleString('en-IN', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit', timeZone: 'UTC', hour12: false }) + ' UTC'
export function ago(iso?: string | null): string {
  if (!iso) return 'never'
  const m = Math.round((Date.now() - new Date(iso).getTime()) / 60000)
  if (m < 1) return 'just now'
  if (m < 60) return `${m} min ago`
  if (m < 1440) return `${Math.round(m / 60)} h ago`
  return `${Math.round(m / 1440)} days ago`
}
