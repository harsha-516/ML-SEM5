"""Major event pages: replay what the models would have said for real events they never trained on."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import requests

from .common import RESULTS

OUT = RESULTS / "events"

CYCLONES = [
    ("murasu-2026", "MURASU", 2026, "Cyclone Murasu", "Bay of Bengal"),
    ("arnab-2026", "ARNAB", 2026, "Cyclone Arnab", "Bay of Bengal, landfall in Andhra Pradesh"),
    ("champi-2026", "CHAMPI", 2026, "Typhoon Champi", "Western North Pacific"),
    ("mocha-2023", "MOCHA", 2023, "Cyclone Mocha", "Bay of Bengal, landfall near Myanmar and Bangladesh"),
    ("biparjoy-2023", "BIPARJOY", 2023, "Cyclone Biparjoy", "Arabian Sea, landfall in Gujarat"),
    ("yagi-2024", "YAGI", 2024, "Typhoon Yagi", "South China Sea and Vietnam"),
]
FLOODS = [
    (
        "rasuwa-2026",
        "Bhote Koshi flood, Nepal",
        "2026-08-26",
        28.2,
        85.35,
        "Rasuwa district, Nepal",
        "Flash flood from the high Himalaya. The trigger is unconfirmed: reports point to a glacier collapse, a lake outburst or a landslide, and death tolls differ between sources.",
    ),
    (
        "budhi-gandaki-2026",
        "Budhi Gandaki flash flood, Nepal",
        "2026-10-11",
        28.3,
        84.85,
        "Gorkha district, Nepal",
        "ICIMOD reported that a landslide-dam breach may have triggered this flood.",
    ),
    (
        "nilambur-2026",
        "Nilambur flash flood, Kerala",
        "2026-09-20",
        11.3,
        76.2,
        "Malappuram district, Kerala",
        "Heavy rain caused a sudden rise on the Kottapuzha River.",
    ),
    (
        "assam-2026",
        "Assam monsoon floods",
        "2026-07-20",
        26.98,
        94.64,
        "Sivasagar district, Assam",
        "Sivasagar was reported as the worst-affected district in July. The date marks the seasonal peak period, not a single moment.",
    ),
]
ARCHIVE_START = pd.Timestamp("2024-01-20")  # first day the previous-runs archive has all 3 leads
EARTHQUAKES = [
    ("ende-2026", "M7.8 earthquake near Ende, Indonesia", "2026-08-14", -8.34, 121.34, 7.8),
    ("nagqu-2026", "M7.0 earthquake near Nagqu, China", "2026-10-22", 32.38, 93.78, 7.0),
]


NEWS = {
    "rasuwa-2026": ("Rasuwa Bhote Koshi flood Nepal", "rasuwa|bhote"),
    "budhi-gandaki-2026": ("Budhi Gandaki flash flood Gorkha", "budhi|gandaki|gorkha"),
    "nilambur-2026": ("Nilambur flash flood Kerala", "nilambur|kottapuzha|malappuram"),
    "assam-2026": ("Assam floods Sivasagar 2026", "assam|sivasagar"),
    "ende-2026": ("Indonesia earthquake Flores Ende magnitude", "ende|flores|indonesia"),
    "nagqu-2026": ("Nagqu Tibet earthquake magnitude", "nagqu|tibet|xizang"),
}


def _write(event: dict) -> None:
    """Save one event page file."""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{event['id']}.json").write_text(json.dumps(event, indent=1))


def cyclone_events(tracks: pd.DataFrame, replay, m: dict) -> None:
    """Write event pages for the listed storms."""
    for eid, name, year, title, place in CYCLONES:
        r = replay(tracks, m, name, year)
        if not r["track"]:  # a storm still forecast to form has no track yet
            continue
        peak = max((f["wind_kt"] or 0) for f in r["track"])
        _write(
            {
                "id": eid,
                "kind": "cyclone",
                "title": title,
                "place": place,
                "date": r["track"][0]["time"][:10],
                "severity": peak,
                "headline": f"Peak wind {peak:.0f} knots",
                "news_query": f'"{title.split()[-1]}" cyclone OR typhoon {year}',
                "news_match": name.lower(),
                "track": r["track"],
            }
        )


def flood_events(score_day, build, real_forecast, warned) -> list[dict]:
    """Score each day around the event, using forecasts as issued where the archive has them (2024 onward)."""
    out = []
    for eid, title, date, lat, lon, place, summary in FLOODS:
        d = pd.Timestamp(date)
        base = dict(
            latitude=lat,
            longitude=lon,
            start_date=str((d - pd.Timedelta(days=430)).date()),
            end_date=str((d + pd.Timedelta(days=10)).date()),
        )
        a = requests.get(
            "https://archive-api.open-meteo.com/v1/archive", params={**base, "daily": "precipitation_sum"}, timeout=90
        )
        b = requests.get(
            "https://flood-api.open-meteo.com/v1/flood", params={**base, "daily": "river_discharge"}, timeout=90
        )
        a.raise_for_status()
        b.raise_for_status()
        idx = pd.to_datetime(a.json()["daily"]["time"])
        w = pd.DataFrame(
            {
                "rain": pd.to_numeric(a.json()["daily"]["precipitation_sum"]),
                "discharge": pd.to_numeric(b.json()["daily"]["river_discharge"]),
            },
            index=idx,
        )
        w["rain"] = w["rain"].fillna(0.0)
        fc, source = None, "reanalysis"
        if d >= ARCHIVE_START:
            first = d - pd.Timedelta(days=35)
            c = requests.get(
                "https://previous-runs-api.open-meteo.com/v1/forecast",
                timeout=120,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "timezone": "GMT",
                    "start_date": str(first.date()),
                    "end_date": base["end_date"],
                    "hourly": "precipitation_previous_day1,precipitation_previous_day2,precipitation_previous_day3",
                },
            )
            c.raise_for_status()
            h = (
                pd.DataFrame(c.json()["hourly"])
                .assign(time=lambda x: pd.to_datetime(x.time))
                .set_index("time")
                .apply(pd.to_numeric)
            )
            daily = h.resample("1D").sum(min_count=18)
            daily.columns = ["lead1", "lead2", "lead3"]
            fc, source = real_forecast(daily), "archived forecast"
        f = build(w, fc, lat, lon).loc[d - pd.Timedelta(days=30) :]
        f = f[f.next3_rain.notna()]
        chance = score_day(f, lat, lon)
        days = [
            {
                "date": str(t.date()),
                "rain_mm": round(float(w.rain[t]), 1),
                "discharge": None if pd.isna(w.discharge[t]) else round(float(w.discharge[t]), 1),
                "next3_mm": round(float(n), 1),
                "chance": round(float(p), 4),
            }
            for t, p, n in zip(f.index, chance, f.next3_rain)
        ]
        check = warned(days, date)
        _write(
            {
                "id": eid,
                "kind": "flood",
                "title": title,
                "place": place,
                "date": date,
                "lat": lat,
                "lon": lon,
                "summary": summary,
                "severity": check["top_3_days"],
                "headline": f"Model chance before event: {check['top_3_days']:.0%}",
                "forecast_source": source,
                **check,
                "news_query": NEWS[eid][0],
                "news_match": NEWS[eid][1],
                "days": days,
            }
        )
        out.append({"id": eid, "forecast_source": source, **check})
    return out


def quake_events(table: pd.DataFrame, cat: pd.DataFrame, score) -> None:
    """Write event pages for the listed earthquakes."""
    for eid, title, date, lat, lon, mag in EARTHQUAKES:
        d = pd.Timestamp(date, tz="UTC")
        row = table[(table.start <= d) & (table.start + pd.Timedelta(days=30) > d)]
        window = row.start.iloc[0]
        p = score(row)
        cell = (row.lat == np.floor(lat) + 0.5) & (row.lon == np.floor(lon) + 0.5)
        here = float(p[cell.to_numpy()][0]) if cell.any() else None
        rank = float((p < here).mean()) if here is not None else None
        near = cat[
            (cat.latitude.sub(lat).abs() <= 2)
            & (cat.longitude.sub(lon).abs() <= 2)
            & (cat.time >= d - pd.Timedelta(days=30))
            & (cat.time <= d + pd.Timedelta(days=30))
        ]
        seq = [
            {"days": round((t - d).total_seconds() / 86400, 2), "mag": float(m), "depth": float(z)}
            for t, m, z in zip(near.time, near.mag, near.depth)
        ]
        _write(
            {
                "id": eid,
                "kind": "earthquake",
                "title": title,
                "place": title.split(" near ")[-1],
                "date": date,
                "lat": lat,
                "lon": lon,
                "mag": mag,
                "severity": mag,
                "headline": f"Magnitude {mag}" if here is None else f"Model chance for this cell, {here:.0%}",
                "cell_chance": here,
                "cell_rank": rank,
                "window_start": str(window.date()),
                "sequence": seq,
                "news_query": NEWS[eid][0],
                "news_match": NEWS[eid][1],
            }
        )
