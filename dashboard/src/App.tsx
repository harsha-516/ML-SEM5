import { useEffect, useState } from 'react'
import { useSiteData } from './data'
import { EventPage } from './EventPage'
import { LivePage } from './LivePage'
import { ModelsPage } from './ModelsPage'
import { EventsPage, HazardPage, Home } from './pages'

const LINKS = [['#/cyclones', 'Cyclones'], ['#/floods', 'Floods'], ['#/earthquakes', 'Earthquakes'], ['#/events', 'Major events'], ['#/models', 'Models'], ['#/live', 'Live data']]

function useRoute() {
  const [r, set] = useState(window.location.hash || '#/')
  useEffect(() => {
    const on = () => { set(window.location.hash || '#/'); window.scrollTo(0, 0) }
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  return r
}

function App() {
  const route = useRoute()
  const d = useSiteData()
  const event = route.startsWith('#/event/') ? route.slice(8) : null
  return (
    <>
      <nav className="top" aria-label="Main">
        <a className="brand" href="#/">Asia Hazard Watch</a>
        {LINKS.map(([h, t]) => <a key={h} href={h} aria-current={route === h ? 'page' : undefined}>{t}</a>)}
      </nav>
      <main>
        {event ? <EventPage key={event} id={event} news={d.eventNews} />
          : route === '#/cyclones' ? <HazardPage key="c" hazard="cyclone" d={d} />
            : route === '#/floods' ? <HazardPage key="f" hazard="flood" d={d} />
              : route === '#/earthquakes' ? <HazardPage key="e" hazard="earthquake" d={d} />
                : route === '#/events' ? <EventsPage d={d} />
                  : route === '#/models' ? <ModelsPage d={d} />
                    : route === '#/live' ? <LivePage d={d} /> : <Home d={d} />}
      </main>
      <footer>
        <p>A student machine learning project. Forecasts are model-based and do not replace official warnings; follow your national agency.</p>
      </footer>
    </>
  )
}

export default App
