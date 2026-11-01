"""Task 3: do news-based flood surges agree with independent sources? Writes results/label_check.json.

GDACS gives one point per listed flood, so a match means within 500 km and 2 days. MODIS flood maps run from 2021.
Control rows are the same cell and month with no surge within 7 days, so agreement can be compared with chance.
"""
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from PIL import Image

from hazards import flood
from hazards.common import DATA, RESULTS, log
from hazards.events import FLOODS
from hazards.refresh import GDACS
from hazards.typhoon import _dist_km

GIBS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
FLOOD_RGB, CLOUD_RGB = (250, 30, 36), (175, 175, 175)


def gdacs_floods() -> pd.DataFrame:
    path = DATA / "gdacs_floods.parquet"
    if path.exists():
        return pd.read_parquet(path)
    rows = []
    for a in pd.date_range("2016-01-01", "2026-10-01", freq="MS"):
        b = a + pd.offsets.MonthEnd(0)
        r = requests.get(GDACS, timeout=60, params={"eventlist": "FL", "fromDate": str(a.date()), "toDate": str(b.date()), "alertlevel": "Green;Orange;Red", "pagesize": 100})
        if r.status_code != 200 or not r.content:
            continue
        for f in r.json().get("features", []):
            p = f["properties"]
            lon, lat = f["geometry"]["coordinates"][:2]
            rows.append({"id": p.get("eventid"), "name": p.get("name"), "alert": p.get("alertlevel"), "lat": lat, "lon": lon,
                         "start": pd.to_datetime(p.get("fromdate")), "end": pd.to_datetime(p.get("todate"))})
        time.sleep(0.2)
    d = pd.DataFrame(rows).drop_duplicates("id")
    d.to_parquet(path)
    log(f"GDACS floods cached: {len(d)}")
    return d


def gdacs_match(g: pd.DataFrame, lat: float, lon: float, first: pd.Timestamp, last: pd.Timestamp) -> bool:
    h = g[(g.start <= last + pd.Timedelta(days=2)) & (g.end >= first - pd.Timedelta(days=2))]
    return bool(len(h)) and bool((_dist_km(lat, lon, h.lat.to_numpy(), h.lon.to_numpy()) <= 500).any())


def modis(lat: float, lon: float, when: pd.Timestamp) -> dict | None:
    """Share of a 1 degree box mapped as flood, and share with a clear view, from the 3-day composite."""
    box = f"{lat - 0.5},{lon - 0.5},{lat + 0.5},{lon + 0.5}"
    for attempt in range(3):
        try:
            r = requests.get(GIBS, timeout=60, params=dict(SERVICE="WMS", VERSION="1.3.0", REQUEST="GetMap", LAYERS="MODIS_Combined_Flood_3-Day", CRS="EPSG:4326",
                                                            BBOX=box, WIDTH=256, HEIGHT=256, FORMAT="image/png", TRANSPARENT="true", TIME=str(when.date()), STYLES=""))
            if r.status_code == 200 and "image" in r.headers.get("content-type", ""):
                a = np.array(Image.open(io.BytesIO(r.content)).convert("RGB"), dtype=int)
                alpha = np.array(Image.open(io.BytesIO(r.content)).convert("RGBA"))[..., 3]
                solid = alpha > 0
                near = lambda c: (np.abs(a - np.array(c)).sum(axis=2) < 40) & solid  # noqa: E731
                return {"flood": float(near(FLOOD_RGB).mean()), "clear": float(1 - near(CLOUD_RGB).mean())}
        except requests.RequestException:
            pass
        time.sleep(3)
    return None


def episodes(df: pd.DataFrame) -> pd.DataFrame:
    """First positive day of each run of surge days at a location."""
    d = df.sort_values(["lat", "lon", "date"])
    gap = d.groupby(["lat", "lon"]).date.diff().dt.days
    return d[(d.y == 1)][(gap[d.y == 1].isna() | (gap[d.y == 1] > 1) | (d.groupby(["lat", "lon"]).y.shift(1)[d.y == 1] != 1))]


def controls(df: pd.DataFrame, pos: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """For each surge, a quiet day at the same location and month in the same period."""
    rng = np.random.default_rng(seed)
    quiet = df.copy()
    quiet["busy"] = quiet.sort_values("date").groupby(["lat", "lon"]).y.transform(lambda y: y.rolling(15, center=True, min_periods=1).max())
    quiet = quiet[quiet.busy == 0]
    keys = {k: v for k, v in quiet.groupby(["lat", "lon", quiet.date.dt.month])}
    rows = []
    for r in pos.itertuples():
        g = keys.get((r.lat, r.lon, r.date.month))
        if g is not None:
            rows.append(g.iloc[rng.integers(len(g))])
    return pd.DataFrame(rows)


def rate(flags) -> float:
    return float(np.mean(flags)) if len(flags) else float("nan")


def region(lat: float, lon: float) -> str:
    return "South Asia" if lon <= 92 else "Mainland Southeast Asia" if lat >= 8 and lon <= 110 else "East and island Asia"


def main() -> None:
    log("Flood label check start")
    g = gdacs_floods()
    log(f"GDACS floods listed: {len(g)}, {g.start.min()} to {g.start.max()}")
    _, df, *_ = flood.load_sets()
    df = df[["lat", "lon", "date", "y"]]
    pos, ctl = episodes(df), None
    ctl = controls(df, pos)
    log(f"Surge episodes {len(pos)}, controls {len(ctl)}")

    def gd(d):
        return [gdacs_match(g, r.lat, r.lon, r.date + pd.Timedelta(days=1), r.date + pd.Timedelta(days=3)) for r in d.itertuples()]

    # GDACS history is thin before it listed floods regularly, so measure from its first busy year too.
    pf, cf = np.array(gd(pos)), np.array(gd(ctl))
    pos = pos.assign(gdacs=pf)
    per_year = {str(y): {"surges": int((pos.date.dt.year == y).sum()), "agree": rate(pos.gdacs[pos.date.dt.year == y]),
                         "gdacs_listed": int((g.start.dt.year == y).sum())} for y in sorted(pos.date.dt.year.unique())}
    reg = pos.assign(r=[region(a, b) for a, b in zip(pos.lat, pos.lon)])
    by_region = {k: {"surges": int(len(v)), "agree": rate(v.gdacs)} for k, v in reg.groupby("r")}
    counts = df[df.y == 1].groupby(["lat", "lon"]).size().sort_values(ascending=False)
    top10 = float(counts.head(10).sum() / counts.sum())

    recent = pos[pos.date >= "2021-03-08"].sample(200, random_state=42)
    quiet = controls(df[df.date >= "2021-03-08"], recent)
    log(f"MODIS check: {len(recent)} surges, {len(quiet)} controls")

    def fetch(d):
        jobs = [(r.lat, r.lon, r.date + pd.Timedelta(days=3)) for r in d.itertuples()]
        with ThreadPoolExecutor(4) as pool:
            return [x for x in pool.map(lambda j: modis(*j), jobs)]

    mp, mc = fetch(recent), fetch(quiet)
    log("MODIS maps fetched")

    def summary(items):
        ok = [x for x in items if x]
        seen = [x for x in ok if x["clear"] >= 0.3]
        return {"maps": len(ok), "with_clear_view": len(seen), "flood_seen": rate([x["flood"] >= 0.001 for x in seen]),
                "median_clear": float(np.median([x["clear"] for x in ok])) if ok else None}

    replays = []
    for eid, title, date, lat, lon, *_ in FLOODS:
        d = pd.Timestamp(date)
        m = modis(lat, lon, d + pd.Timedelta(days=2))
        replays.append({"id": eid, "gdacs": gdacs_match(g, lat, lon, d, d), "modis_flood_share": None if m is None else round(m["flood"], 4),
                        "modis_clear_share": None if m is None else round(m["clear"], 3)})
    out = {"method": "GDACS match within 500 km and 2 days. MODIS flood share at least 0.1% of a 1 degree box with at least 30% clear view, from the 3-day composite. Controls are quiet days at the same location and month.",
           "gdacs_floods_listed": int(len(g)), "surge_episodes": int(len(pos)),
           "gdacs_agree_surges": rate(pf), "gdacs_agree_controls": rate(cf), "gdacs_by_year": per_year, "gdacs_by_region": by_region,
           "modis_surges": summary(mp), "modis_controls": summary(mc), "replays": replays,
           "place_bias": {"locations": int(len(counts)), "top_10_share_of_surges": top10,
                          "most_surges": [{"lat": k[0], "lon": k[1], "surges": int(v)} for k, v in counts.head(5).items()],
                          "fewest_surges": [{"lat": k[0], "lon": k[1], "surges": int(v)} for k, v in counts.tail(5).items()]}}
    (RESULTS / "label_check.json").write_text(json.dumps(out, indent=1))
    log(f"Flood label check finish: GDACS {rate(pf):.3f} vs control {rate(cf):.3f}, MODIS {out['modis_surges']} vs {out['modis_controls']}")


if __name__ == "__main__":
    main()
