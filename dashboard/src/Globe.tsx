import { useEffect, useRef, useState } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import { rampExpression } from './ramp'
import type { Hazard, Results, StormFile } from './types'

type Feature = { type: 'Feature'; properties: Record<string, unknown>; geometry: { type: string; coordinates: unknown } }
type FC = { type: 'FeatureCollection'; features: Feature[] }

const fc = (features: Feature[]): FC => ({ type: 'FeatureCollection', features })
const pt = (lon: number, lat: number, properties: Record<string, unknown>): Feature => ({
  type: 'Feature', properties, geometry: { type: 'Point', coordinates: [lon, lat] },
})
const line = (coordinates: number[][], properties: Record<string, unknown>): Feature => ({
  type: 'Feature', properties, geometry: { type: 'LineString', coordinates },
})


/** Largest zoom that keeps the whole globe inside the hero, whatever the window size. */
function fitZoom(box: HTMLElement): number {
  const wide = box.clientWidth > 820
  const room = wide ? box.clientHeight : box.clientHeight - 360
  const radius = Math.min(room, wide ? box.clientWidth * 0.56 : box.clientWidth) * 0.45
  return Math.log2(radius / (0.81 * (512 / (2 * Math.PI))))
}

const LAYERS: Record<Hazard, string[]> = {
  cyclone: ['track-past', 'track-forecast', 'storm-halo', 'storm-core', 'storm-label'],
  flood: ['flood-risk'],
  earthquake: ['eq-risk', 'eq-recent'],
}

export function Globe({ hazard, results, storms }: {
  hazard: Hazard
  results: Partial<Record<Hazard, Results>>
  storms: StormFile | null
}) {
  const el = useRef<HTMLDivElement>(null)
  const map = useRef<maplibregl.Map | null>(null)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    if (!el.current) return
    const m = new maplibregl.Map({
      container: el.current,
      center: [122, 18],
      zoom: fitZoom(el.current),
      attributionControl: { compact: true, customAttribution: 'NASA GIBS' },
      minZoom: 1,
      maxZoom: 5,
      style: {
        version: 8,
        projection: { type: 'globe' },
        glyphs: 'https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf',
        sources: {
          earth: {
            type: 'image', url: `${import.meta.env.BASE_URL}data/earth.webp`,
            coordinates: [[-180, 85.0511], [180, 85.0511], [180, -85.0511], [-180, -85.0511]],
          },
        },
        layers: [
          { id: 'space', type: 'background', paint: { 'background-color': '#0B1E3A' } },
          { id: 'earth', type: 'raster', source: 'earth', paint: { 'raster-fade-duration': 0, 'raster-saturation': -0.1, 'raster-contrast': 0.06 } },
        ],
        sky: { 'atmosphere-blend': 0.7 },
      } as maplibregl.StyleSpecification,
    })
    m.scrollZoom.disable()
    m.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right')
    m.on('load', () => {
      const color = rampExpression('p') as maplibregl.ExpressionSpecification
      for (const id of ['eq', 'recent', 'flood', 'past', 'forecast', 'storms']) m.addSource(id, { type: 'geojson', data: fc([]) as never })
      m.addLayer({ id: 'eq-risk', type: 'circle', source: 'eq', paint: { 'circle-color': color, 'circle-radius': ['interpolate', ['linear'], ['zoom'], 2, 2.6, 5, 11], 'circle-opacity': ['interpolate', ['linear'], ['get', 'p'], 0.01, 0.25, 0.1, 0.85], 'circle-blur': 0.3 } })
      m.addLayer({ id: 'eq-recent', type: 'circle', source: 'recent', paint: { 'circle-color': 'transparent', 'circle-stroke-color': '#FFFFFF', 'circle-stroke-width': 1.2, 'circle-radius': ['interpolate', ['linear'], ['get', 'mag'], 4.5, 3, 7, 12] } })
      m.addLayer({ id: 'flood-risk', type: 'circle', source: 'flood', paint: { 'circle-color': color, 'circle-radius': 7, 'circle-stroke-color': '#FFFFFF', 'circle-stroke-width': 1 } })
      m.addLayer({ id: 'track-past', type: 'line', source: 'past', paint: { 'line-color': '#FFFFFF', 'line-width': 2, 'line-opacity': 0.85 } })
      m.addLayer({ id: 'track-forecast', type: 'line', source: 'forecast', paint: { 'line-color': '#FFFFFF', 'line-width': 2, 'line-dasharray': [1.5, 2] } })
      m.addLayer({ id: 'storm-halo', type: 'circle', source: 'storms', paint: { 'circle-color': '#D24BE6', 'circle-radius': 24, 'circle-opacity': 0.22 } })
      m.addLayer({ id: 'storm-core', type: 'circle', source: 'storms', paint: { 'circle-color': '#D24BE6', 'circle-radius': 7, 'circle-stroke-color': '#FFFFFF', 'circle-stroke-width': 2 } })
      m.addLayer({ id: 'storm-label', type: 'symbol', source: 'storms', layout: { 'text-field': ['get', 'name'], 'text-font': ['Open Sans Semibold'], 'text-offset': [0, 1.8], 'text-size': 15 }, paint: { 'text-color': '#FFFFFF', 'text-halo-color': '#0B1E3A', 'text-halo-width': 1.5 } })
      setReady(true)
    })
    map.current = m
    return () => m.remove()
  }, [])

  useEffect(() => {
    const m = map.current
    if (!m || !ready) return
    const set = (id: string, data: FC) => (m.getSource(id) as maplibregl.GeoJSONSource).setData(data as never)
    set('eq', fc((results.earthquake?.current_risk ?? []).map((c) => pt(c.lon, c.lat, { p: c.p }))))
    set('flood', fc((results.flood?.current_risk ?? []).map((c) => pt(c.lon, c.lat, { p: c.p, name: c.name }))))
    const list = storms?.storms ?? []
    set('storms', fc(list.map((s) => pt(s.lon, s.lat, { name: s.name }))))
    set('past', fc(list.filter((s) => s.past.length > 1).map((s) => line(s.past.map(([la, lo]) => [lo, la]), {}))))
    set('forecast', fc(list.filter((s) => s.forecast.length).map((s) => line([[s.lon, s.lat], ...s.forecast.map((f) => [f.lon, f.lat])], {}))))
  }, [ready, results, storms])

  useEffect(() => {
    const m = map.current
    if (!m || !ready) return
    ;(Object.keys(LAYERS) as Hazard[]).forEach((h) => LAYERS[h].forEach((l) => m.setLayoutProperty(l, 'visibility', h === hazard ? 'visible' : 'none')))
    const s = storms?.storms[0]
    const center: [number, number] = hazard === 'cyclone' && s ? [s.lon - 6, s.lat - 2] : hazard === 'flood' ? [92, 22] : [105, 18]
    const still = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const wide = window.innerWidth > 820
    m.easeTo({ center, zoom: fitZoom(m.getContainer()), padding: wide ? { left: Math.min(window.innerWidth * 0.42, 620), top: 0, right: 0, bottom: 0 } : { left: 0, top: 0, right: 0, bottom: 360 }, duration: still ? 0 : 1400 })
  }, [hazard, ready, storms])

  return <div ref={el} className="globe" role="img" aria-label="Satellite globe of Asia with current hazard markers" />
}
