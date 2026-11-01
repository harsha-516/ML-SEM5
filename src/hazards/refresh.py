"""Scheduled refresh: check every source, gather events, news and storms, and log forecasts for later scoring.

Run once with `python -m hazards.refresh`, or keep running with `python -m hazards.refresh --loop 5`.
"""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote_plus
from xml.etree.ElementTree import ParseError

import defusedxml.ElementTree as ET
import pandas as pd
import requests

from . import live
from .common import DATA, RESULTS, ROOT, log

HEADERS = {"User-Agent": "Mozilla/5.0 (student hazard research project)"}
TOPICS = {"cyclone": "cyclone OR typhoon India OR Asia", "flood": "flood India OR Asia", "earthquake": "earthquake magnitude tremors Asia"}
TITLE_MUST_MATCH = {"cyclone": r"cyclon|typhoon", "flood": r"\bfloods?\b|flooding|flooded", "earthquake": r"earthquake|tremor"}
NOT_DISASTER = r"voters|election|flood of|flood x|political"
HAZARD_WORDS = r"\b(cyclones?|cyclonic|typhoons?|floods?|flooding|flash flood|earthquakes?|tremors?|landslides?|tsunami)\b"
CHANNELS = ["DDNews", "ndtv", "IndiaToday", "WIONews", "NASAEarth", "NASAGoddard"]
GDACS = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"
JMA = "https://www.jma.go.jp/bosai/typhoon/data"
IBTRACS = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ACTIVE.list.v04r01.csv"
USGS = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_week.geojson"
CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
RSS = {"BBC Asia": "https://feeds.bbci.co.uk/news/world/asia/rss.xml", "The Hindu": "https://www.thehindu.com/news/national/feeder/default.rss",
       "Times of India": "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms"}
DB = ROOT / "data" / "forecasts.sqlite"
TARGETS = [ROOT / "dashboard" / "public" / "data", ROOT / "dashboard" / "dist" / "data"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_news(xml_text: str, limit: int = 12) -> list[dict]:
    """Newest-first items from an RSS feed, one per headline."""
    items, seen = [], set()
    for it in ET.fromstring(xml_text).iter("item"):
        title = (it.findtext("title") or "").strip()
        source = (it.findtext("source") or "").strip()
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3]
        date = it.findtext("pubDate")
        if not title or title in seen or not date:
            continue
        try:
            when = parsedate_to_datetime(date).astimezone(timezone.utc)
        except ValueError:  # some feeds publish non-standard dates
            continue
        seen.add(title)
        items.append({"title": title, "source": source, "url": it.findtext("link"), "published": when.isoformat(timespec="minutes")})
    return sorted(items, key=lambda x: x["published"], reverse=True)[:limit]


def google_news(query: str, match: str, limit: int = 12, days: int = 60) -> list[dict]:
    url = f"https://news.google.com/rss/search?q={quote_plus(f'{query} when:{days}d')}&hl=en-IN&gl=IN&ceid=IN:en"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return [n for n in parse_news(r.text, 60) if re.search(match, n["title"], re.I) and not re.search(NOT_DISASTER, n["title"], re.I)][:limit]


def news() -> dict:
    return {k: google_news(q, TITLE_MUST_MATCH[k], days=30) for k, q in TOPICS.items()}


def event_files() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((RESULTS / "events").glob("*.json"))] if (RESULTS / "events").exists() else []


def event_news() -> dict:
    """Coverage for each event. Older events may have little left in the news feed."""
    out = {}
    for e in event_files():
        try:
            out[e["id"]] = google_news(e["news_query"], e["news_match"], limit=8, days=1200 if e["date"] < "2026" else 120)
        except (requests.RequestException, ParseError):
            out[e["id"]] = []
        time.sleep(0.5)
    return out


def events(days: int = 60) -> list[dict]:
    today = datetime.now(timezone.utc).date()
    r = requests.get(GDACS, timeout=40, params={"eventlist": "TC;EQ;FL", "fromDate": str(today - timedelta(days=days)), "toDate": str(today), "alertlevel": "Green;Orange;Red"})
    r.raise_for_status()
    out = []
    for f in r.json().get("features", []):
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"][:2]
        out.append({"type": p.get("eventtype"), "name": p.get("name"), "alert": p.get("alertlevel"), "from": p.get("fromdate"), "to": p.get("todate"),
                    "country": p.get("country"), "lat": lat, "lon": lon, "url": (p.get("url") or {}).get("report")})
    return sorted(out, key=lambda e: e["to"] or "", reverse=True)


def storms() -> list[dict]:
    found = []
    for s in requests.get(f"{JMA}/targetTc.json", timeout=30).json():
        tc = s["tropicalCyclone"]
        spec = requests.get(f"{JMA}/{tc}/specifications.json", timeout=30).json()
        track = requests.get(f"{JMA}/{tc}/forecast.json", timeout=30).json()
        title, parts = spec[0], [p for p in spec[1:] if "position" in p]
        cur = parts[0]
        past, analysis = [], None
        for p in track[1:2]:
            t = p.get("track", {})
            past = t.get("preTyphoon", []) + t.get("typhoon", [])
            analysis = p.get("validtime", {}).get("UTC")
        found.append({"name": title["name"]["en"], "number": title["typhoonNumber"], "category": title["category"]["en"], "issued": title["issue"]["UTC"], "analysis_time": analysis, "source": "JMA",
                      "wind_kt": int(cur["maximumWind"]["sustained"]["kt"]), "pressure": int(cur["pressure"]), "lat": cur["position"]["deg"][0], "lon": cur["position"]["deg"][1],
                      "past": past, "forecast": [{"lat": p["position"]["deg"][0], "lon": p["position"]["deg"][1], "time": p.get("validtime", {}).get("UTC")} for p in parts[1:]]})
    return found


def _usgs(_):
    d = requests.get(USGS, timeout=30).json()["features"]
    top = max(d, key=lambda f: f["properties"]["time"])["properties"]
    items = [{"source": "USGS", "text": f"M{f['properties']['mag']:.1f} {f['properties']['place']}", "time": datetime.fromtimestamp(f["properties"]["time"] / 1000, timezone.utc).isoformat(timespec="minutes"), "url": f["properties"]["url"]}
             for f in sorted(d, key=lambda f: -f["properties"]["time"])[:6]]
    return len(d), f"M{top['mag']:.1f} {top['place']}", items


def _jma(_):
    s = storms()
    return len(s), f"{s[0]['name']}, {s[0]['wind_kt']} knots" if s else "No active storm", [
        {"source": "JMA", "text": f"{x['name']}: {x['wind_kt']} knots, {x['pressure']} hPa", "time": x["issued"], "url": "https://www.jma.go.jp/bosai/map.html#contents=typhoon"} for x in s]


def _ibtracs(_):
    r = requests.head(IBTRACS, timeout=30)
    r.raise_for_status()
    return None, f"File updated {r.headers.get('Last-Modified', 'unknown')}", []


def _gdacs(_):
    e = events()
    return len(e), e[0]["name"] if e else "No events", [{"source": "GDACS", "text": f"{x['name']}, {x['alert']} alert", "time": x["to"], "url": x["url"]} for x in e[:6]]


def _meteo(_):
    r = requests.get("https://flood-api.open-meteo.com/v1/flood", timeout=30, params={"latitude": 25.6, "longitude": 85.1, "daily": "river_discharge", "forecast_days": 1})
    r.raise_for_status()
    return 1, f"Patna river discharge {r.json()['daily']['river_discharge'][0]} m3/s", []


def _rain(_):
    r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=30, params={"latitude": 27.7, "longitude": 85.3, "daily": "precipitation_sum", "forecast_days": 1})
    r.raise_for_status()
    return 1, f"Kathmandu rain today {r.json()['daily']['precipitation_sum'][0]} mm", []


def _gibs(_):
    day = (datetime.now(timezone.utc).date() - timedelta(days=2)).isoformat()
    r = requests.get(f"https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_SNPP_CorrectedReflectance_TrueColor/default/{day}/GoogleMapsCompatible_Level9/2/1/2.jpg", timeout=30)
    r.raise_for_status()
    return None, f"Imagery tile for {day}", []


def _granule(short):
    def fn(_):
        r = requests.get(CMR, timeout=30, params={"short_name": short, "sort_key": "-start_date", "page_size": 1})
        r.raise_for_status()
        return None, f"Newest record {r.json()['feed']['entry'][0]['time_start'][:16].replace('T', ' ')} UTC", []
    return fn


def _himawari(_):
    d = datetime.now(timezone.utc) - timedelta(hours=2)
    r = requests.get("https://noaa-himawari9.s3.amazonaws.com/", timeout=30, params={"list-type": "2", "prefix": f"AHI-L2-FLDK-Clouds/{d:%Y/%m/%d}/", "max-keys": 5})
    r.raise_for_status()
    n = r.text.count("<Key>")
    if not n:
        raise ValueError("no files")
    return n, f"Files listed for {d:%d %b %Y}", []


def _rss(name):
    def fn(url):
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        items = parse_news(r.text, 80)
        hits = [n for n in items if re.search(HAZARD_WORDS, n["title"], re.I) and not re.search(NOT_DISASTER, n["title"], re.I)]
        return len(items), f"{len(hits)} hazard headlines" if hits else "No hazard headlines right now", [
            {"source": name, "text": n["title"], "time": n["published"], "url": n["url"]} for n in hits[:4]]
    return fn


def _google(_):
    n = google_news("cyclone OR flood OR earthquake Asia", HAZARD_WORDS, limit=8, days=3)
    return len(n), n[0]["title"] if n else "No items", [{"source": x["source"] or "Google News", "text": x["title"], "time": x["published"], "url": x["url"]} for x in n]


SOURCES = [
    ("usgs", "USGS earthquakes", "Earthquakes", "API", USGS, _usgs), ("jma", "JMA active storms", "Cyclones", "API", None, _jma),
    ("ibtracs", "IBTrACS storm tracks", "Cyclones", "File", None, _ibtracs), ("gdacs", "GDACS disaster alerts", "All hazards", "API", None, _gdacs),
    ("discharge", "GloFAS river discharge", "Floods", "API", None, _meteo), ("rain", "Open-Meteo rainfall", "Floods", "API", None, _rain),
    ("gibs", "NASA GIBS imagery", "Maps", "Satellite", None, _gibs), ("imerg", "NASA IMERG rainfall", "Floods", "Satellite", None, _granule("GPM_3IMERGHHE")),
    ("smap", "NASA SMAP soil moisture", "Floods", "Satellite", None, _granule("SPL3SMP_E")), ("himawari", "Himawari-9 clouds", "Cyclones", "Satellite", None, _himawari),
    ("bbc", "BBC Asia", "News", "RSS", RSS["BBC Asia"], _rss("BBC Asia")), ("hindu", "The Hindu", "News", "RSS", RSS["The Hindu"], _rss("The Hindu")),
    ("toi", "Times of India", "News", "RSS", RSS["Times of India"], _rss("Times of India")), ("gnews", "Google News", "News", "RSS", None, _google),
]
OFFLINE = [
    {"id": "groundsource", "name": "Groundsource flood reports", "use": "Floods", "kind": "Archive", "state": "archive", "latest": "2000 to February 2026"},
    {"id": "copernicus", "name": "Copernicus Sentinel-1", "use": "Floods", "kind": "Satellite", "state": "off", "latest": "Radar imagery"},
    {"id": "youtube", "name": "YouTube news video", "use": "News", "kind": "API", "state": "off", "latest": "News video"},
    {"id": "telegram", "name": "Telegram alerts", "use": "Alerts", "kind": "API", "state": "off", "latest": "Alert delivery"},
]


def check_sources(previous: dict) -> tuple[list[dict], list[dict]]:
    """Call every source for real and record what happened, so the page never shows a guess."""
    rows, feed = [], []
    for sid, name, use, kind, url, fn in SOURCES:
        start = time.perf_counter()
        row = {"id": sid, "name": name, "use": use, "kind": kind, "checked_at": now(), "last_ok": previous.get(sid, {}).get("last_ok")}
        try:
            count, latest, items = fn(url)
            row.update(state="live", records=count, latest=latest, last_ok=row["checked_at"])
            feed += items
        except Exception as exc:  # any source failure is shown on the page, never hidden
            row.update(state="failed", records=None, latest=f"Request failed: {type(exc).__name__}")
        row["latency_ms"] = round((time.perf_counter() - start) * 1000)
        rows.append(row)
    return rows + OFFLINE, sorted(feed, key=lambda x: x.get("time") or "", reverse=True)


def log_forecasts(db: Path = DB) -> int:
    """Store today's forecasts once, with the model date, so they can be scored after their window closes."""
    day = datetime.now(timezone.utc).date().isoformat()
    db.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(db)
    con.execute("create table if not exists forecasts (issued text, hazard text, lat real, lon real, chance real, window_days int, "
                "model_version text, outcome int, primary key (issued, hazard, lat, lon))")
    n = 0
    for hazard, days in (("earthquake", 30), ("flood", 3)):
        path = RESULTS / f"{hazard}.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        rows = [(day, hazard, c["lat"], c["lon"], c["p"], days, r["generated_at"]) for c in r.get("current_risk", [])]
        n += con.executemany("insert or ignore into forecasts (issued, hazard, lat, lon, chance, window_days, model_version) values (?,?,?,?,?,?,?)", rows).rowcount
    con.commit()
    total = con.execute("select count(*) from forecasts").fetchone()[0]
    con.close()
    return total


def publish(name: str, data) -> None:
    for t in TARGETS:
        if t.parent.exists():
            t.mkdir(parents=True, exist_ok=True)
            (t / name).write_text(json.dumps(data, indent=1))


def copy_results() -> None:
    for t in TARGETS:
        if not t.parent.exists():
            continue
        (t / "events").mkdir(parents=True, exist_ok=True)
        for f in RESULTS.glob("*.json"):
            shutil.copy(f, t / f.name)
        for f in (RESULTS / "events").glob("*.json"):
            shutil.copy(f, t / "events" / f.name)


def read_previous() -> dict:
    p = TARGETS[0] / "sources.json"
    try:
        return {s["id"]: s for s in json.loads(p.read_text())["sources"]}
    except (OSError, ValueError, KeyError):
        return {}


def _gdacs_floods() -> pd.DataFrame:
    f = [e for e in events() if e["type"] == "FL"]
    d = pd.DataFrame(f, columns=["lat", "lon", "from", "to"])
    return d.assign(start=pd.to_datetime(d["from"], utc=True), end=pd.to_datetime(d["to"], utc=True))[["lat", "lon", "start", "end"]]


def scored_today(hazard: str) -> bool:
    try:
        return json.loads((RESULTS / f"{hazard}.json").read_text()).get("scored_at", "")[:10] == now()[:10]
    except (OSError, ValueError):
        return False


def daily() -> bool:
    """Rescore the flood and earthquake maps once per UTC day. A failure keeps the previous map and retries next cycle."""
    changed = False
    for hazard, fn in (("earthquake", live.score_earthquake), ("flood", live.score_flood)):
        if scored_today(hazard):
            continue
        try:
            changed = bool(fn()) or changed
        except Exception as exc:
            log(f"Daily {hazard} rescoring failed: {type(exc).__name__}: {exc}")
    return changed


def score_closed() -> None:
    cat = pd.read_parquet(DATA / "usgs_catalog.parquet")
    n = live.score_outcomes(live.DB, cat, _gdacs_floods(), pd.Timestamp.now(tz="UTC"))
    k = live.score_storm_outcomes(live.DB, pd.Timestamp.now(tz="UTC"))
    if n or k:
        log(f"Outcomes written: {n} map forecasts, {k} storm forecasts")
    publish("live_score.json", live.scorecard())


def cycle(full: bool) -> None:
    rows, feed = check_sources(read_previous())
    publish("feed.json", feed[:40])
    try:
        active = storms()
        try:
            active += live.north_indian_storms(live.active_history(), pd.Timestamp.now(tz="UTC"))
            active = live.score_storms(active)
        except Exception as exc:  # the storm still shows without a model forecast
            log(f"Storm scoring failed: {type(exc).__name__}: {exc}")
        publish("storms.json", {"status": "ok", "fetched_at": now(), "storms": active})
    except Exception as exc:  # the page shows the failure instead of stale data
        publish("storms.json", {"status": f"failed: {type(exc).__name__}", "fetched_at": now(), "storms": []})
    full = daily() or full
    if full:
        try:
            score_closed()
        except Exception as exc:
            log(f"Outcome scoring failed: {type(exc).__name__}: {exc}")
        copy_results()
        publish("event_index.json", [{k: e.get(k) for k in ("id", "kind", "title", "place", "date", "headline", "severity")} for e in event_files()])
        for name, fn in (("news.json", news), ("events.json", events), ("event_news.json", event_news)):
            try:
                publish(name, fn())
            except Exception as exc:
                print(name, "failed:", type(exc).__name__)
    logged = log_forecasts()
    publish("sources.json", {"refreshed_at": now(), "forecasts_logged": logged, "sources": rows})
    print(now(), sum(r["state"] == "live" for r in rows), "live of", len(SOURCES), "| feed items", len(feed), flush=True)


def main() -> None:
    if "--loop" in sys.argv:
        minutes = int(sys.argv[sys.argv.index("--loop") + 1])
        n = 0
        while True:
            cycle(full=n % 12 == 0)
            n += 1
            time.sleep(minutes * 60)
    cycle(full=True)


if __name__ == "__main__":
    main()
