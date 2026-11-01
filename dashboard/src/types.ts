export type Hazard = 'cyclone' | 'flood' | 'earthquake'

export interface Metrics {
  n: number
  base_rate: number
  pr_auc: number
  roc_auc: number
  brier: number
  recall: number
  precision: number
  f1: number
  threshold: number
  confusion: { tn: number; fp: number; fn: number; tp: number }
}

export interface Results {
  hazard: Hazard
  unit: string
  label: string
  baseline_name: string
  winner: string
  models: Record<string, { validation: Metrics; test: Metrics }>
  reliability: { predicted: number; observed: number; n: number }[]
  feature_importance: { feature: string; share: number }[]
  sets: Record<string, number>
  periods: Record<string, string>
  generated_at: string
  current_risk?: { lat: number; lon: number; p: number; name?: string }[]
  note?: string
  scored_at?: string
  storms?: Record<string, number>
  locations?: number
  catalog_events?: number
  pr_curve?: { threshold: number; recall: number; precision: number; caught: number; missed: number; false_alarms: number }[]
}

export interface Storm {
  name: string
  number: string
  category: string
  issued: string
  wind_kt: number
  pressure: number | null
  source?: string
  lat: number
  lon: number
  past: number[][]
  forecast: { lat: number; lon: number; time?: string }[]
  model?: { from: string; valid: string; wind_now_1min: number; wind_24: number; low: number; high: number; ri_chance: number; reach_64_chance: number | null; model_version: string }
}

export interface StormFile {
  status: string
  fetched_at: string
  storms: Storm[]
}

export interface Quake {
  lat: number
  lon: number
  mag: number
  place: string
  time: number
}
