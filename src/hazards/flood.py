"""Flood: will reported flooding at a location surge above its own recent normal in the next 3 days.

Groundsource events come from news, so raw counts grow with coverage and favour big cities.
Comparing each location with its own trailing year removes most of that drift.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import requests

from sklearn.metrics import average_precision_score

from .common import DATA, RESULTS, Platt, compare_models, log, print_summary, tuned, write_results
from . import satellite
from .events import FLOODS, flood_events
from .live import save_model

HORIZON = 3
FIRST, LAST = "2015-01-01", "2026-01-31"
CACHE = DATA / "flood_cells"
FORECASTS = DATA / "flood_forecasts"
FEATURES = ["rain_1", "rain_3", "rain_7", "rain_30", "discharge", "discharge_anom", "discharge_change_3",
            "month_sin", "month_cos", "recent_reports_30", "lat", "lon"]
LIVE_FEATURES = [f for f in FEATURES if f != "recent_reports_30"]  # Groundsource stops in February 2026
RATIO_FEATURES = ["rain_1_ratio", "rain_3_ratio", "rain_7_ratio", "rain_30_ratio", "wet_days_7", "wet_days_30", "discharge_rank"]
FORECAST_FEATURES = ["next3_rain", "next3_ratio"]
IMERG_FEATURES = ["imerg_3", "imerg_7", "imerg_30", "imerg_7_ratio"]
MODEL_FEATURES = LIVE_FEATURES + RATIO_FEATURES + FORECAST_FEATURES + IMERG_FEATURES
IMERG_LAG = 2  # days until a daily IMERG Late file is published


def choose_cells(events: pd.DataFrame, n: int = 60, min_events: int = 1000, spacing: int = 3) -> pd.DataFrame:
    """Busiest 1 degree cells, at most one per spacing-degree block, chosen from pre-2022 data only."""
    e = events[(events.start >= "2015-01-01") & (events.start < "2022-01-01")]
    c = e.assign(la=np.floor(e.lat).astype(int), lo=np.floor(e.lon).astype(int)).groupby(["la", "lo"]).size().rename("n").reset_index()
    c = c[c.n >= min_events // 4].sort_values("n", ascending=False)
    c["block"] = (c.la // spacing).astype(str) + "_" + (c.lo // spacing).astype(str)
    return c.drop_duplicates("block").head(n).reset_index(drop=True)


def daily_counts(events: pd.DataFrame, la: int, lo: int, days: pd.DatetimeIndex) -> pd.Series:
    e = events[(np.floor(events.lat) == la) & (np.floor(events.lon) == lo)]
    return e.groupby(e.start.dt.normalize()).size().reindex(days, fill_value=0).astype(float)


def surge_label(counts: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """1 when the next horizon days exceed the trailing-year 95th percentile of 3-day totals, and at least 3 reports."""
    total = counts.rolling(horizon).sum()
    limit = total.rolling(365, min_periods=180).quantile(0.95).shift(1)
    future = counts.rolling(horizon).sum().shift(-horizon)
    y = ((future > limit) & (future >= 3)).astype(float)
    y[limit.isna() | future.isna()] = np.nan
    return y


def rain_limit(rain: pd.Series, n: int) -> pd.Series:
    """Trailing-year 95th percentile of n-day rain, known before each day."""
    return rain.rolling(n, min_periods=1).sum().rolling(365, min_periods=180).quantile(0.95).shift(1).clip(lower=1)


def forecast_features(w: pd.DataFrame, forecast: pd.Series | None = None) -> pd.DataFrame:
    """Rain over the next 3 days. Without a forecast this is reanalysis, which is impossible live."""
    next3 = w["rain"].rolling(HORIZON).sum().shift(-HORIZON) if forecast is None else forecast.reindex(w.index)
    return pd.DataFrame({"next3_rain": next3, "next3_ratio": next3 / rain_limit(w["rain"], HORIZON)}, index=w.index)


def imerg_features(s: pd.Series, lag: int = IMERG_LAG) -> pd.DataFrame:
    """Satellite rain, reading only days already published."""
    r = s.asfreq("D").shift(lag)
    f = pd.DataFrame({f"imerg_{n}": r.rolling(n, min_periods=max(1, n - 2)).sum() for n in (3, 7, 30)})
    f["imerg_7_ratio"] = f.imerg_7 / f.imerg_7.rolling(365, min_periods=180).quantile(0.95).shift(1).clip(lower=1)
    return f


def real_forecast(d: pd.DataFrame) -> pd.Series:
    """Rain expected over the next 3 days, from forecasts issued 1, 2 and 3 days before each target day."""
    return d["lead1"].shift(-1) + d["lead2"].shift(-2) + d["lead3"].shift(-3)


def weather_features(w: pd.DataFrame) -> pd.DataFrame:
    """All features look backward from the end of each day."""
    f = pd.DataFrame(index=w.index)
    for n in (1, 3, 7, 30):
        f[f"rain_{n}"] = w["rain"].rolling(n, min_periods=1).sum()
    q = w["discharge"]
    f["discharge"] = q
    f["discharge_anom"] = q / q.rolling(365, min_periods=90).median().shift(1).clip(lower=0.01)
    f["discharge_change_3"] = q - q.shift(3)
    for n in (1, 3, 7, 30):
        f[f"rain_{n}_ratio"] = f[f"rain_{n}"] / rain_limit(w["rain"], n)
    wet = (w["rain"] > 5).astype(float)
    f["wet_days_7"], f["wet_days_30"] = wet.rolling(7, min_periods=1).sum(), wet.rolling(30, min_periods=1).sum()
    f["discharge_rank"] = q.rolling(365, min_periods=180).rank(pct=True)
    m = w.index.month
    f["month_sin"], f["month_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    return f


def fetch_cell(lat: float, lon: float, path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    base = dict(latitude=lat, longitude=lon, start_date=FIRST, end_date=LAST)
    for attempt in range(6):
        a = requests.get("https://archive-api.open-meteo.com/v1/archive", params={**base, "daily": "precipitation_sum"}, timeout=90)
        b = requests.get("https://flood-api.open-meteo.com/v1/flood", params={**base, "daily": "river_discharge"}, timeout=90)
        if a.status_code == 200 and b.status_code == 200:
            break
        if 429 in (a.status_code, b.status_code):
            raise RuntimeError("Open-Meteo daily limit reached")
        time.sleep(5 * (attempt + 1))
    a.raise_for_status()
    b.raise_for_status()
    w = pd.DataFrame({"rain": a.json()["daily"]["precipitation_sum"], "discharge": b.json()["daily"]["river_discharge"]},
                     index=pd.to_datetime(a.json()["daily"]["time"]))
    w["rain"] = pd.to_numeric(w["rain"]).fillna(0.0)
    w["discharge"] = pd.to_numeric(w["discharge"])
    w.to_parquet(path)
    return w


def load_imerg(cells: pd.DataFrame) -> pd.DataFrame:
    files = sorted((DATA / "imerg").glob("*.npy"))
    return pd.DataFrame(np.array([np.load(f) for f in files]), index=pd.to_datetime([f.stem for f in files]),
                        columns=[(c.la + 0.5, c.lo + 0.5) for c in cells.itertuples()]).sort_index()


def build_table(events: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    CACHE.mkdir(exist_ok=True)
    frames = []
    imerg = load_imerg(cells)
    for c in cells.itertuples():
        try:
            w = fetch_cell(c.la + 0.5, c.lo + 0.5, CACHE / f"{c.la}_{c.lo}.parquet")
        except RuntimeError:  # daily quota: train on what is cached, the rest fills in on a later run
            continue
        counts = daily_counts(events, c.la, c.lo, w.index)
        f = weather_features(w).join(forecast_features(w)).join(imerg_features(imerg[(c.la + 0.5, c.lo + 0.5)]))
        issued = FORECASTS / f"{c.la}_{c.lo}.parquet"
        if issued.exists():
            real = forecast_features(w, real_forecast(pd.read_parquet(issued)))
            f["real_next3"], f["real_ratio"] = real.next3_rain, real.next3_ratio
        f["recent_reports_30"] = counts.rolling(30).sum()
        f["y"] = surge_label(counts)
        f["lat"], f["lon"], f["date"] = c.la + 0.5, c.lo + 0.5, f.index
        frames.append(f.dropna(subset=["y"]))
        time.sleep(0.4)
    return pd.concat(frames, ignore_index=True)


def load_sets():
    events = pd.read_parquet(DATA / "groundsource_region.parquet")
    cells = choose_cells(events)
    df = build_table(events, cells)
    gap = pd.Timedelta(days=HORIZON + 30)
    tr = df[df.date < pd.Timestamp("2022-01-01") - gap]
    va = df[(df.date >= "2022-01-01") & (df.date < pd.Timestamp("2024-01-01") - gap)]
    te = df[df.date >= "2024-01-01"]
    return cells, df, tr, va, te


def with_real_forecasts(d: pd.DataFrame) -> pd.DataFrame:
    """Rows that have a forecast as issued at the time, with it replacing later actual rain."""
    return d[d.real_next3.notna()].assign(next3_rain=lambda x: x.real_next3, next3_ratio=lambda x: x.real_ratio)


def live_row(rain: pd.Series, discharge: pd.Series, today: pd.Timestamp) -> pd.Series:
    """Feature row at the end of today. Only the forecast features read rain after today."""
    w = pd.concat([rain.reindex(discharge.index).fillna(0.0).rename("rain"), discharge.rename("discharge")], axis=1).loc[:today]
    ahead = rain.loc[today + pd.Timedelta(days=1):today + pd.Timedelta(days=HORIZON)]
    if len(ahead) < HORIZON:
        raise ValueError("forecast does not reach 3 days ahead")
    f = weather_features(w).join(forecast_features(w, pd.Series({today: float(ahead.sum())})))
    return f.loc[today]


def live_features(lat: float, lon: float, imerg: pd.Series) -> pd.Series:
    """Newest feature row for one location: reanalysis history, recent forecast-model rain and a real 3-day forecast."""
    today = pd.Timestamp(pd.Timestamp.now(tz="UTC").date())
    start, end = str((today - pd.Timedelta(days=420)).date()), str(today.date())
    old = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=60, params=dict(
        latitude=lat, longitude=lon, daily="precipitation_sum", start_date=start, end_date=end))
    new = requests.get("https://api.open-meteo.com/v1/forecast", timeout=60, params=dict(
        latitude=lat, longitude=lon, daily="precipitation_sum", past_days=10, forecast_days=HORIZON + 1))
    q = requests.get("https://flood-api.open-meteo.com/v1/flood", timeout=60, params=dict(
        latitude=lat, longitude=lon, daily="river_discharge", start_date=start, end_date=end))
    for r in (old, new, q):
        r.raise_for_status()

    def series(r, key):
        return pd.Series(pd.to_numeric(pd.Series(r.json()["daily"][key])).to_numpy(), index=pd.to_datetime(r.json()["daily"]["time"]))

    row = live_row(series(new, "precipitation_sum").combine_first(series(old, "precipitation_sum")).fillna(0.0), series(q, "river_discharge"), today)
    sat = imerg_features(imerg.reindex(pd.date_range(imerg.index.min(), today)))
    return pd.concat([row, sat.loc[today]])


def warned(days: list[dict], event_date: str) -> dict:
    """Warning criterion: top chance in the 3 days before at least 1.5 times the median of the days before that."""
    before = [x["chance"] for x in days if x["date"] < event_date]
    top, usual = max(before[-3:]), float(np.median(before[:-3]))
    return {"top_3_days": top, "usual": usual, "warned": bool(top >= 1.5 * usual)}


def run() -> dict:
    cells, df, tr, va, te_all = load_sets()
    te = with_real_forecasts(te_all)  # the test uses forecasts as issued, never later actual rain
    rate = tr.assign(m=tr.date.dt.month).groupby(["lat", "lon", "m"]).y.mean().rename("r").reset_index()

    def base(d):
        return d.assign(m=d.date.dt.month).merge(rate, how="left", on=["lat", "lon", "m"]).r.fillna(tr.y.mean()).to_numpy()

    def X(d, cols):
        return d[cols].to_numpy(np.float32)

    ys = [d.y.astype(int).to_numpy() for d in (tr, va, te)]
    res = compare_models(X(tr, MODEL_FEATURES), ys[0], X(va, MODEL_FEATURES), ys[1], X(te, MODEL_FEATURES), ys[2], base(va), base(te), MODEL_FEATURES,
                         xgb_params=tuned("flood"))
    no_fc = LIVE_FEATURES + RATIO_FEATURES
    plain = compare_models(X(tr, no_fc), ys[0], X(va, no_fc), ys[1], X(te, no_fc), ys[2], base(va), base(te), no_fc, xgb_params=tuned("flood"))
    actual = te_all.loc[te.index]
    xgb = res["_fitted"]["XGBoost"]
    cal = Platt().fit(xgb.predict_proba(X(va, MODEL_FEATURES))[:, 1], ys[1])
    res["actual_rain_test_pr_auc"] = float(average_precision_score(actual.y, xgb.predict_proba(X(actual, MODEL_FEATURES))[:, 1]))
    res["no_forecast_test_pr_auc"] = plain["models"][plain["winner"]]["test"]["pr_auc"]

    def score_day(f, lat, lon):
        return cal(xgb.predict_proba(f.assign(lat=lat, lon=lon)[MODEL_FEATURES].to_numpy(np.float32))[:, 1])

    spots = pd.DataFrame([{"la": int(np.floor(e[3])), "lo": int(np.floor(e[4]))} for e in FLOODS])
    sat = satellite.sync("imerg_events", spots, "2025-04-01")

    def build(w, fc, lat, lon):
        return weather_features(w).join(forecast_features(w, fc)).join(imerg_features(sat[(np.floor(lat) + 0.5, np.floor(lon) + 0.5)]))

    res["replays"] = flood_events(score_day, build, real_forecast, warned)
    res["current_risk"] = score_current(cells, xgb, cal)
    res["scored_at"] = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    res.update(hazard="flood", unit="location-day", label="Flood reports surge in the next 3 days",
               baseline_name="Location and month historical rate", locations=int(df.groupby(["lat", "lon"]).ngroups),
               sets={"train": int(len(tr)), "validation": int(len(va)), "test": int(len(te))},
               periods={"train": "2016 to 2021", "validation": "2022 to 2023", "test": "2024 to January 2026"},
               forecast_note="Training and validation use reanalysis rain for the next 3 days. Test uses forecasts as issued.",
               note="Labels come from news reports, so a surge means more reported flooding, not a measured river level. "
                    "Quiet or poorly covered places are under-represented.")
    label = RESULTS / "label_check.json"
    if label.exists():
        res["label_check"] = json.loads(label.read_text())
    write_results("flood", res)
    save_model("flood", {"features": MODEL_FEATURES, "model": xgb, "cal": cal, "cells": cells.to_dict("list")})
    return res


def score_current(cells: pd.DataFrame, xgb, cal) -> list[dict]:
    current = []
    imerg = satellite.sync("imerg", cells, str((pd.Timestamp.now() - pd.Timedelta(days=30)).date()))
    for c in cells.itertuples():
        lat, lon = c.la + 0.5, c.lo + 0.5
        try:
            f = live_features(lat, lon, imerg[(lat, lon)])
        except (requests.RequestException, ValueError) as exc:
            log(f"  Flood live score skipped at {lat}, {lon}: {type(exc).__name__}")
            continue
        row = np.array([[lat if k == "lat" else lon if k == "lon" else f[k] for k in MODEL_FEATURES]], dtype=np.float32)
        current.append({"lat": lat, "lon": lon, "p": round(float(cal(xgb.predict_proba(row)[:, 1])[0]), 4),
                        "rain_7_mm": round(float(f["rain_7"]), 1), "next3_mm": round(float(f["next3_rain"]), 1)})
    return current


if __name__ == "__main__":
    print_summary(run())
