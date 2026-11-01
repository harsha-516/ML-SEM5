import { useEffect, useRef, useState } from 'react'
import maplibregl from 'maplibre-gl'
import { rampExpression } from './ramp'

export interface MapDot { lat: number; lon: number; p?: number; label?: string; ring?: boolean }

const collection = (dots: MapDot[]) => ({
  type: 'FeatureCollection',
  features: dots.map((d) => ({ type: 'Feature', properties: { p: d.p ?? 0, label: d.label ?? '' }, geometry: { type: 'Point', coordinates: [d.lon, d.lat] } })),
})

export function FlatMap({ dots, rings = [], line, marker, center = [100, 18], zoom = 2.3, overlay, label }: {
  dots: MapDot[]
  rings?: MapDot[]
  line?: [number, number][]
  marker?: [number, number]
  center?: [number, number]
  zoom?: number
  overlay?: string
  label: string
}) {
  const el = useRef<HTMLDivElement>(null)
  const map = useRef<maplibregl.Map | null>(null)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    if (!el.current) return
    const sources: Record<string, unknown> = {
      still: { type: 'image', url: `${import.meta.env.BASE_URL}data/earth.webp`, coordinates: [[-180, 85.0511], [180, 85.0511], [180, -85.0511], [-180, -85.0511]] },
      base: { type: 'raster', tileSize: 256, maxzoom: 8, attribution: 'NASA GIBS', tiles: ['https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_ShadedRelief_Bathymetry/default/2004-08-01/GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpeg'] },
    }
    const layers: unknown[] = [{ id: 'still', type: 'raster', source: 'still', paint: { 'raster-fade-duration': 0, 'raster-saturation': -0.35, 'raster-brightness-max': 0.72 } }, { id: 'base', type: 'raster', source: 'base', paint: { 'raster-saturation': -0.35, 'raster-brightness-max': 0.72, 'raster-fade-duration': 100 } }]
    if (overlay) {
      sources.over = { type: 'raster', tileSize: 256, maxzoom: 6, tiles: [overlay] }
      layers.push({ id: 'over', type: 'raster', source: 'over', paint: { 'raster-opacity': 0.75 } })
    }
    const m = new maplibregl.Map({ container: el.current, center, zoom, style: { version: 8, sources, layers } as maplibregl.StyleSpecification })
    m.scrollZoom.disable()
    m.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right')
    m.once('style.load', () => {
      for (const s of ['dots', 'rings', 'line', 'marker']) m.addSource(s, { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
      m.addLayer({ id: 'dots', type: 'circle', source: 'dots', paint: { 'circle-color': rampExpression('p') as maplibregl.ExpressionSpecification, 'circle-radius': ['interpolate', ['linear'], ['zoom'], 2, 3.2, 6, 12], 'circle-opacity': 0.9 } })
      m.addLayer({ id: 'rings', type: 'circle', source: 'rings', paint: { 'circle-color': 'transparent', 'circle-stroke-color': '#fff', 'circle-stroke-width': 1.2, 'circle-radius': 5 } })
      m.addLayer({ id: 'line', type: 'line', source: 'line', paint: { 'line-color': '#fff', 'line-width': 2.5 } })
      m.addLayer({ id: 'marker', type: 'circle', source: 'marker', paint: { 'circle-color': '#D24BE6', 'circle-radius': 9, 'circle-stroke-color': '#fff', 'circle-stroke-width': 2 } })
      setReady(true)
    })
    const pop = (e: maplibregl.MapLayerMouseEvent) => {
      const t = e.features?.[0]?.properties?.label
      if (t) new maplibregl.Popup({ closeButton: false }).setLngLat(e.lngLat).setText(String(t)).addTo(m)
    }
    m.on('click', 'dots', pop)
    m.on('click', 'rings', pop)
    map.current = m
    return () => m.remove()
    // the base map is created once; data updates arrive through the effect below
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [overlay])

  useEffect(() => {
    const m = map.current
    if (!m || !ready) return
    const set = (id: string, data: unknown) => (m.getSource(id) as maplibregl.GeoJSONSource).setData(data as never)
    set('dots', collection(dots))
    set('rings', collection(rings))
    set('line', { type: 'FeatureCollection', features: line && line.length > 1 ? [{ type: 'Feature', properties: {}, geometry: { type: 'LineString', coordinates: line } }] : [] })
    set('marker', collection(marker ? [{ lon: marker[0], lat: marker[1] }] : []))
    if (marker) m.jumpTo({ center: marker })  // storm data can arrive after the map is created
  }, [ready, dots, rings, line, marker])

  return <div ref={el} className="flat" role="img" aria-label={label} />
}
