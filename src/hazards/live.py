"""Live scoring: saved models, daily rescoring, active storms and outcome scoring."""

from __future__ import annotations

import io
import json
import sqlite3
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

from .common import DATA, RESULTS, ROOT, dist_km, log

MODELS = ROOT / "data" / "models"
DB = ROOT / "data" / "forecasts.sqlite"
ONE_MINUTE_FACTOR = 0.88  # standard ratio of 10-minute to 1-minute sustained wind over the sea
FDSN = "https://earthquake.usgs.gov/fdsnws/event/1/query"
IBTRACS_ACTIVE = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ACTIVE.list.v04r01.csv"
FLOOD_KM = 500


def to_one_minute(wind_10min: float) -> float:
    """JMA reports 10-minute winds. The models use 1-minute winds."""
    return wind_10min / ONE_MINUTE_FACTOR


def save_model(name: str, bundle: dict) -> Path:
    """Save a model bundle and return its path."""
    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / f"{name}.joblib"
    joblib.dump({**bundle, "version": datetime.now(UTC).date().isoformat()}, path)
    return path


def load_model(name: str) -> dict | None:
    """Only reads files save_model wrote to data/models on this machine; never point this at downloaded files."""
    path = MODELS / f"{name}.joblib"
    return joblib.load(path) if path.exists() else None


def merge_catalog(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Add new earthquakes to the catalogue without duplicates."""
    cols = list(old.columns)
    return (
        pd.concat([old, new[cols]], ignore_index=True)
        .drop_duplicates(["time", "latitude", "longitude"], keep="last")
        .sort_values("time")
        .reset_index(drop=True)
    )


def update_catalog() -> pd.DataFrame:
    """Add events since the newest stored one, with a week of overlap for revised magnitudes."""
    from .earthquake import BOX

    path = DATA / "usgs_catalog.parquet"
    cat = pd.read_parquet(path)
    since = (cat.time.max() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    r = requests.get(
        FDSN,
        timeout=120,
        params={
            "format": "csv",
            "starttime": since,
            "minmagnitude": 4.5,
            "minlatitude": BOX["lat0"],
            "maxlatitude": BOX["lat1"],
            "minlongitude": BOX["lon0"],
            "maxlongitude": BOX["lon1"],
            "eventtype": "earthquake",
            "orderby": "time-asc",
        },
    )
    r.raise_for_status()
    new = pd.read_csv(io.StringIO(r.text), usecols=["time", "latitude", "longitude", "depth", "mag", "magType"])
    new["time"] = pd.to_datetime(new["time"], utc=True, format="ISO8601")
    merged = merge_catalog(cat, new)
    merged.to_parquet(path)
    log(f"USGS catalogue: {len(merged) - len(cat)} new events, newest {merged.time.max()}")
    return merged


def window_shift(today: pd.Timestamp, start: pd.Timestamp, days: int) -> pd.Timedelta:
    """Shift that puts today on a window boundary, so each day's forecast sees every event up to today."""
    return pd.Timedelta(days=(days - (today - start).days % days) % days)


def score_earthquake(cat: pd.DataFrame | None = None) -> int:
    """Score the current window for every cell and log it. Returns rows logged."""
    from . import earthquake as q
    from . import etas

    m = load_model("earthquake")
    if m is None:
        return 0
    cat = update_catalog() if cat is None else cat
    today = pd.Timestamp(datetime.now(UTC).date(), tz="UTC")
    shift = window_shift(today, q.START, q.WINDOW_DAYS)
    k = (today + shift - q.START).days // q.WINDOW_DAYS
    table = q.build_table(
        cat[cat.time < today].assign(time=lambda x: x.time + shift), n_windows=k + 1, train_end=q.TRAIN_END
    )
    row = table[table.window == k]
    shifted = cat[cat.time < today].assign(time=lambda x: x.time + shift)
    base = etas.table_forecast(m["etas"], shifted, row, q.START, q.WINDOW_DAYS)
    p = m["cal"](m["blend"](m["model"].predict_proba(row[m["features"]].to_numpy(np.float32))[:, 1], base))
    risk = [{"lat": float(a), "lon": float(b), "p": round(float(c), 4)} for a, b, c in zip(row.lat, row.lon, p)]
    _update_results("earthquake", current_risk=risk, current_window_start=str(today.date()), catalog_events=len(cat))
    log(f"Earthquake rescored: {len(risk)} cells, top chance {max(x['p'] for x in risk):.2f}")
    return len(risk)


def score_flood() -> int:
    """Score today for every location and log it. Returns rows logged."""
    from . import flood as f

    m = load_model("flood")
    if m is None:
        return 0
    cells = pd.DataFrame(m["cells"])
    risk = f.score_current(cells, m["model"], m["cal"])
    if len(risk) < len(cells) // 2:
        log(f"Flood rescoring kept the old map: only {len(risk)} of {len(cells)} locations answered")
        return 0
    _update_results("flood", current_risk=risk)
    log(f"Flood rescored: {len(risk)} locations, top chance {max(x['p'] for x in risk):.2f}")
    return len(risk)


def _update_results(hazard: str, **values) -> None:
    """Change named values in a results file."""
    path = RESULTS / f"{hazard}.json"
    r = json.loads(path.read_text())
    r.update(values, scored_at=datetime.now(UTC).isoformat(timespec="seconds"))
    path.write_text(json.dumps(r, indent=2))


@lru_cache(maxsize=1)
def land_lookup():
    """Distance to land for any position, from the nearest historic track point."""
    from .cyclone import load_tracks

    t = load_tracks([DATA / "ibtracs_NI.csv", DATA / "ibtracs_WP.csv"]).dropna(subset=["dist_land", "lat", "lon"])
    tree = cKDTree(t[["lat", "lon"]].to_numpy())
    known = t.dist_land.to_numpy()

    def find(lat, lon):
        return known[tree.query(np.column_stack([lat, lon]))[1]]

    return find


def storm_series(history: pd.DataFrame, fixes: pd.DataFrame, name: str, dist_land) -> pd.DataFrame:
    """One 6-hourly track on the 1-minute scale: IBTrACS history, then logged JMA fixes."""
    wind = fixes.wind_10min.map(to_one_minute)
    grade = np.select([wind >= 64, wind >= 48, wind >= 34], [5, 4, 3], default=2)
    own = pd.DataFrame(
        {
            "storm": name.upper(),
            "name": name.upper(),
            "time": fixes.time.to_numpy(),
            "grade": grade,
            "lat": fixes.lat.to_numpy(),
            "lon": fixes.lon.to_numpy(),
            "pressure": fixes.pressure.to_numpy(),
            "wind": wind.to_numpy(),
            "dist_land": dist_land(fixes.lat.to_numpy(), fixes.lon.to_numpy()),
            "is_north_indian": 0,
        }
    )
    old = history[history.name.str.upper() == name.upper()]
    if len(old):
        old = old[old.time < own.time.min()].assign(storm=name.upper()) if len(own) else old
    s = pd.concat([x for x in (old, own) if len(x)], ignore_index=True)
    return s.sort_values("time").reset_index(drop=True)


def _storm_tables(con: sqlite3.Connection) -> None:
    """Create the storm tables if needed."""
    con.execute(
        "create table if not exists storm_fixes (storm text, time text, lat real, lon real, wind_10min real, pressure real, primary key (storm, time))"
    )
    con.execute(
        "create table if not exists storm_forecasts (issued text, storm text, lat real, lon real, wind_now real, low real, mid real, high real, ri_chance real, "
        "valid text, outcome_wind real, primary key (issued, storm))"
    )


def active_history() -> pd.DataFrame:
    """IBTrACS active tracks, refreshed at most every 6 hours."""
    from .cyclone import load_tracks

    path = DATA / "ibtracs_ACTIVE.csv"
    age = datetime.now().timestamp() - path.stat().st_mtime if path.exists() else 1e9
    if age > 6 * 3600:
        r = requests.get(IBTRACS_ACTIVE, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    return load_tracks([path])


def north_indian_storms(history: pd.DataFrame, now: pd.Timestamp) -> list[dict]:
    """Active North Indian storms. JMA does not cover this basin, so these come from IBTrACS, which lags by a day or more."""
    out = []
    h = history[(history.is_north_indian == 1) & history.wind.notna()]
    for _, g in h.groupby("storm"):
        g = g.sort_values("time")
        last = g.iloc[-1]
        if now.tz_localize(None) - last.time > pd.Timedelta(hours=48):
            continue
        name = str(last["name"]).title()
        out.append(
            {
                "name": name,
                "number": "",
                "category": "IBTrACS",
                "issued": str(last.time),
                "analysis_time": None,
                "source": "IBTrACS",
                "wind_kt": int(last.wind),
                "pressure": None if pd.isna(last.pressure) else int(last.pressure),
                "lat": float(last.lat),
                "lon": float(last.lon),
                "past": g[["lat", "lon"]].to_numpy().tolist(),
                "forecast": [],
            }
        )
    return out


def _forecast(s: dict, series: pd.DataFrame, m: dict, con: sqlite3.Connection) -> dict:
    """Score one active storm and log the forecast."""
    from .typhoon import build_table

    row = build_table(series, first_year=2000, keep_strong=True).sort_values("time").iloc[[-1]]
    X = row[m["features"]].to_numpy(np.float32)
    q = m["quantiles"]
    low, mid, high = (
        float(q["low"].predict(X)[0] - m["margin"]),
        float(q["mid"].predict(X)[0]),
        float(q["high"].predict(X)[0] + m["margin"]),
    )
    ri = float(m["ri_cal"](m["ri"].predict_proba(X)[:, 1])[0])
    t = pd.Timestamp(row.time.iloc[0])
    reach = (
        round(float(m["reach_cal"](m["reach"].predict_proba(X)[:, 1])[0]), 4) if float(row.wind.iloc[0]) < 64 else None
    )
    forecast = {
        "reach_64_chance": reach,
        "from": str(t),
        "valid": str(t + pd.Timedelta(hours=24)),
        "wind_now_1min": round(float(row.wind.iloc[0]), 1),
        "wind_24": round(mid, 1),
        "low": round(low, 1),
        "high": round(high, 1),
        "ri_chance": round(ri, 4),
        "model_version": m["version"],
    }
    con.execute(
        "insert or ignore into storm_forecasts values (?,?,?,?,?,?,?,?,?,?,null)",
        (str(t), s["name"], s["lat"], s["lon"], forecast["wind_now_1min"], low, mid, high, ri, forecast["valid"]),
    )
    return {**s, "model": forecast}


def score_storms(storms: list[dict], db: Path = DB) -> list[dict]:
    """Attach 24-hour wind, range and rapid intensification chance to each active JMA storm, and log them."""
    m = load_model("cyclone")
    if m is None or not storms:
        return storms
    con = sqlite3.connect(db)
    _storm_tables(con)
    history = active_history()
    land = land_lookup()
    out = []
    for s in storms:
        when = pd.Timestamp(s["analysis_time"]).tz_localize(None) if s.get("analysis_time") else None
        if s.get("source") == "IBTrACS":  # already 1-minute winds
            g = history[history.name.str.upper() == s["name"].upper()].assign(storm=s["name"].upper())
            con.execute(
                "insert or replace into storm_fixes values (?,?,?,?,?,?)",
                (s["name"], s["issued"], s["lat"], s["lon"], s["wind_kt"] * ONE_MINUTE_FACTOR, s["pressure"]),
            )
            out.append(_forecast(s, g, m, con))
            continue
        if when is not None and when.hour % 6 == 0:
            con.execute(
                "insert or replace into storm_fixes values (?,?,?,?,?,?)",
                (s["name"], str(when), s["lat"], s["lon"], s["wind_kt"], s["pressure"]),
            )
        fixes = (
            pd.read_sql("select * from storm_fixes where storm = ?", con, params=(s["name"],))
            .assign(time=lambda x: pd.to_datetime(x.time))
            .sort_values("time")
        )
        if not len(fixes):
            out.append(s)
            continue
        out.append(_forecast(s, storm_series(history, fixes, s["name"], land), m, con))
    con.commit()
    con.close()
    return out


def score_outcomes(db: Path, cat: pd.DataFrame, gdacs: pd.DataFrame, now: pd.Timestamp) -> int:
    """Fill outcomes for forecasts whose window has closed. Earthquakes use USGS M5+, floods use GDACS."""
    con = sqlite3.connect(db)
    rows = con.execute(
        "select rowid, issued, hazard, lat, lon, window_days from forecasts where outcome is null"
    ).fetchall()
    done = 0
    m5 = cat[cat.mag >= 5] if len(cat) else cat
    for rowid, issued, hazard, lat, lon, days in rows:
        start = pd.Timestamp(issued, tz="UTC")
        end = start + pd.Timedelta(days=days)
        if end > now:
            continue
        if hazard == "earthquake":
            hit = (
                (m5.time >= start)
                & (m5.time < end)
                & (np.floor(m5.latitude) + 0.5 == lat)
                & (np.floor(m5.longitude) + 0.5 == lon)
            ).any()
        else:
            g = (
                gdacs[(gdacs.start <= end + pd.Timedelta(days=2)) & (gdacs.end >= start - pd.Timedelta(days=1))]
                if len(gdacs)
                else gdacs
            )
            hit = len(g) > 0 and (dist_km(lat, lon, g.lat.to_numpy(float), g.lon.to_numpy(float)) <= FLOOD_KM).any()
        con.execute("update forecasts set outcome = ? where rowid = ?", (int(hit), rowid))
        done += 1
    con.commit()
    con.close()
    return done


def score_storm_outcomes(db: Path, now: pd.Timestamp) -> int:
    """Compare storm forecasts with later fixes. Returns how many were scored."""
    con = sqlite3.connect(db)
    _storm_tables(con)
    n = 0
    for issued, storm, valid in con.execute(
        "select issued, storm, valid from storm_forecasts where outcome_wind is null"
    ).fetchall():
        if pd.Timestamp(valid, tz="UTC") > now:
            continue
        r = con.execute("select wind_10min from storm_fixes where storm = ? and time = ?", (storm, valid)).fetchone()
        if r:
            con.execute(
                "update storm_forecasts set outcome_wind = ? where issued = ? and storm = ?",
                (to_one_minute(r[0]), issued, storm),
            )
            n += 1
    con.commit()
    con.close()
    return n


SOURCES = {
    "earthquake": "USGS M5+ in the same 1 degree cell within 30 days",
    "flood": "GDACS flood events within 500 km in the 3-day window. Groundsource ended in February 2026, so this is not the training label.",
}


def scorecard(db: Path = DB) -> dict:
    """Live forecasts scored against what happened. Empty until the first window closes."""
    con = sqlite3.connect(db)
    _storm_tables(con)
    out = {}
    for hazard in ("earthquake", "flood"):
        d = pd.read_sql("select chance, outcome from forecasts where hazard = ?", con, params=(hazard,))
        s = d[d.outcome.notna()]
        if len(s):
            out[hazard] = {
                "scored": len(s),
                "pending": int(d.outcome.isna().sum()),
                "happened": int(s.outcome.sum()),
                "mean_chance": float(s.chance.mean()),
                "observed_rate": float(s.outcome.mean()),
                "brier": float(((s.chance - s.outcome) ** 2).mean()),
                "outcome_source": SOURCES[hazard],
            }
    c = pd.read_sql("select * from storm_forecasts", con)
    s = c[c.outcome_wind.notna()]
    if len(s):
        out["cyclone"] = {
            "scored": len(s),
            "pending": int(c.outcome_wind.isna().sum()),
            "wind_mae_kt": float((s.mid - s.outcome_wind).abs().mean()),
            "in_range": float(((s.outcome_wind >= s.low) & (s.outcome_wind <= s.high)).mean()),
            "outcome_source": "JMA analysis 24 hours later, converted to 1-minute winds",
        }
    con.close()
    return {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"), "hazards": out}
