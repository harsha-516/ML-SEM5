"""Earthquake: chance of M5+ in a 1 degree cell over the next 30 days."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from . import etas
from .common import (
    DATA,
    Platt,
    best_threshold,
    compare_models,
    evaluate,
    log,
    pr_points,
    print_summary,
    reliability,
    risk_cuts,
    tuned,
    write_results,
)
from .events import quake_events
from .live import save_model
from .recal import Blend, rolling_platt

BOX = dict(lat0=-12, lat1=50, lon0=60, lon1=150)
START = pd.Timestamp("1973-01-01", tz="UTC")
WINDOW_DAYS = 30
MIN_ROW_WINDOW = 90  # about 1980: rows start after feature history exists
FEATURES = [
    "n_30",
    "n_90",
    "n_360",
    "n_1800",
    "max_mag_90",
    "days_since_m5",
    "log_energy_360",
    "mean_depth_360",
    "neighbour_n_90",
    "neighbour_n_360",
    "m5_rate_so_far",
    "etas_trigger",
    "lat",
    "lon",
]
TRAIN_END = pd.Timestamp("2016-01-01", tz="UTC")
RECAL_WINDOWS = 48  # about 4 years, chosen on late validation in scripts/experiments/recal_experiments.py
ETAS_ALPHA, ETAS_P, ETAS_C_DAYS, ETAS_LAGS = 1.0, 1.1, 0.01, 60


def fetch_catalog() -> pd.DataFrame:
    """Download M4.5+ year by year (the API allows 20,000 events a query) and cache it."""
    path = DATA / "usgs_catalog.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frames = []
    for year in range(1973, pd.Timestamp.now().year + 1):
        r = requests.get(
            "https://earthquake.usgs.gov/fdsnws/event/1/query",
            timeout=120,
            params={
                "format": "csv",
                "starttime": f"{year}-01-01",
                "endtime": f"{year + 1}-01-01",
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
        frames.append(
            pd.read_csv(io.StringIO(r.text), usecols=["time", "latitude", "longitude", "depth", "mag", "magType"])
        )
    cat = pd.concat(frames, ignore_index=True)
    cat["time"] = pd.to_datetime(cat["time"], utc=True, format="ISO8601")
    cat = cat.drop_duplicates().sort_values("time").reset_index(drop=True)
    cat.to_parquet(path)
    return cat


def _roll(a: np.ndarray, n: int) -> np.ndarray:
    """Sum of the n windows strictly before each window."""
    c = np.cumsum(np.concatenate([np.zeros((1, *a.shape[1:]), a.dtype), a], axis=0), axis=0)
    out = np.zeros_like(c[:-1])
    for k in range(a.shape[0]):
        out[k] = c[k] - c[max(k - n, 0)]
    return out


def _neigh(a: np.ndarray) -> np.ndarray:
    """Sum each grid cell with its neighbours."""
    out = np.zeros_like(a)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                out += np.roll(np.roll(a, dy, axis=1), dx, axis=2)
    return out


def omori_weights(lags: int = ETAS_LAGS) -> np.ndarray:
    """Share of aftershock rate falling in the target window, for an event `lag` windows earlier."""
    e = 1 - ETAS_P
    t0 = (np.arange(1, lags + 1) - 1) * WINDOW_DAYS + WINDOW_DAYS / 2 + ETAS_C_DAYS
    return ((t0 + WINDOW_DAYS) ** e - t0**e) / e


def etas_trigger(productivity: np.ndarray) -> np.ndarray:
    """ETAS-style expected aftershock load: past events, scaled by magnitude, decayed by Omori, spread to neighbours."""
    w = omori_weights()
    out = np.zeros_like(productivity)
    for lag, weight in enumerate(w, start=1):
        out[lag:] += productivity[:-lag] * weight
    return out + 0.25 * _neigh(out)


def build_table(
    cat: pd.DataFrame,
    n_windows: int | None = None,
    min_row: int = MIN_ROW_WINDOW,
    train_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """One row per active cell and window. Features use only earlier windows; quiet windows stay in."""
    H, W = BOX["lat1"] - BOX["lat0"], BOX["lon1"] - BOX["lon0"]
    w = ((cat["time"] - START).dt.total_seconds() // (WINDOW_DAYS * 86400)).astype(int).to_numpy()
    K = n_windows or int(w.max()) + 1
    keep = (w >= 0) & (w < K)
    w = w[keep]
    y = np.clip((cat["latitude"].to_numpy()[keep] - BOX["lat0"]).astype(int), 0, H - 1)
    x = np.clip((cat["longitude"].to_numpy()[keep] - BOX["lon0"]).astype(int), 0, W - 1)
    mag, depth = cat["mag"].to_numpy()[keep], np.nan_to_num(cat["depth"].to_numpy()[keep])

    n = np.zeros((K, H, W), np.float32)
    m5 = np.zeros_like(n)
    energy = np.zeros((K, H, W), np.float64)
    depthsum = np.zeros_like(n)
    maxmag = np.zeros_like(n)
    np.add.at(n, (w, y, x), 1)
    np.add.at(m5, (w, y, x), (mag >= 5).astype(np.float32))
    np.add.at(energy, (w, y, x), 10 ** (1.5 * (mag - 4.5)))
    productivity = np.zeros_like(n)
    np.add.at(productivity, (w, y, x), (10 ** (ETAS_ALPHA * (mag - 4.5))).astype(np.float32))
    np.add.at(depthsum, (w, y, x), depth)
    np.maximum.at(maxmag, (w, y, x), mag)

    n90, n360 = _roll(n, 3), _roll(n, 12)
    max90 = np.zeros_like(maxmag)
    for k in range(1, K):
        max90[k] = maxmag[max(k - 3, 0) : k].max(axis=0)
    since = np.full((H, W), 1e6, np.float32)
    since_all = np.zeros_like(n)
    for k in range(K):
        since_all[k] = since
        since = np.where(m5[k] > 0, 0, since + 1)
    rate = _roll((m5 > 0).astype(np.float32), K) / np.maximum(np.arange(K), 1).reshape(-1, 1, 1)
    table = dict(
        n_30=_roll(n, 1),
        n_90=n90,
        n_360=n360,
        n_1800=_roll(n, 60),
        max_mag_90=max90,
        days_since_m5=np.minimum(since_all * WINDOW_DAYS, 20000),
        log_energy_360=np.log1p(_roll(energy, 12)),
        mean_depth_360=_roll(depthsum, 12) / np.maximum(n360, 1),
        neighbour_n_90=_neigh(n90),
        neighbour_n_360=_neigh(n360),
        m5_rate_so_far=rate,
        etas_trigger=etas_trigger(productivity),
    )
    label = m5 > 0

    # The cell list comes from training-period events only, so test events cannot choose cells.
    limit = K if train_end is None else int((train_end - START).days // WINDOW_DAYS)
    cells = np.argwhere(n[:limit].sum(axis=0) > 0)
    rows = []
    for k in range(min_row, K):
        d = {f: v[k, cells[:, 0], cells[:, 1]] for f, v in table.items()}
        d.update(
            lat=cells[:, 0] + BOX["lat0"] + 0.5,
            lon=cells[:, 1] + BOX["lon0"] + 0.5,
            window=k,
            y=label[k, cells[:, 0], cells[:, 1]].astype(np.int8),
        )
        rows.append(pd.DataFrame(d))
    df = pd.concat(rows, ignore_index=True)
    df["start"] = START + pd.to_timedelta(df["window"] * WINDOW_DAYS, unit="D")
    return df


def load_sets():
    """Train, validation, test and the current window. Used by run() and the experiments."""
    cat = fetch_catalog()
    now = pd.Timestamp.now(tz="UTC")
    complete = int((now - START).days // WINDOW_DAYS)
    train_end = TRAIN_END
    val_end = pd.Timestamp("2020-01-01", tz="UTC")
    df = build_table(cat, n_windows=complete + 1, train_end=train_end)
    current = df[df.window == complete]
    df = df[df.window < complete]
    gap = pd.Timedelta(days=WINDOW_DAYS)
    tr = df[df.start < train_end - gap]
    va = df[(df.start >= train_end) & (df.start < val_end - gap)]
    te = df[df.start >= val_end]
    return cat, tr, va, te, current


def _rule_scores(tr, va, te):
    """Older fixed-setting rule: long-run rate plus trigger load, combined by a two-weight logistic fit."""

    def Z(d):
        return np.column_stack([np.log(d.m5_rate_so_far + 1e-4), np.log1p(d.etas_trigger)])

    m = LogisticRegression(max_iter=1000).fit(Z(tr), tr.y)
    return m.predict_proba(Z(va))[:, 1], m.predict_proba(Z(te))[:, 1]


def run() -> dict:
    """Train, compare and save the earthquake models."""
    cat, tr, va, te, current = load_sets()

    def X(d):
        return d[FEATURES].to_numpy(np.float32)

    fitted = etas.fit(cat, START, START + pd.Timedelta(days=MIN_ROW_WINDOW * WINDOW_DAYS), TRAIN_END)
    base_va = etas.table_forecast(fitted, cat, va, START, WINDOW_DAYS)
    base_te = etas.table_forecast(fitted, cat, te, START, WINDOW_DAYS)
    res = compare_models(
        X(tr),
        tr.y.to_numpy(),
        X(va),
        va.y.to_numpy(),
        X(te),
        te.y.to_numpy(),
        base_va,
        base_te,
        FEATURES,
        xgb_params=tuned("earthquake"),
    )
    res["rate_only_test_pr_auc"] = float(average_precision_score(te.y, te.m5_rate_so_far))
    res["rule_test_pr_auc"] = float(average_precision_score(te.y, _rule_scores(tr, va, te)[1]))
    xgb = res["_fitted"]["XGBoost"]
    y_va, y_te = va.y.to_numpy(), te.y.to_numpy()
    ml_va, ml_te = xgb.predict_proba(X(va))[:, 1], xgb.predict_proba(X(te))[:, 1]
    blend = Blend().fit(ml_va, base_va, y_va)
    mix_va, mix_te = blend(ml_va, base_va), blend(ml_te, base_te)
    t = np.r_[va.window.to_numpy(), te.window.to_numpy()]
    y_all, mix_all = np.r_[y_va, y_te], np.r_[mix_va, mix_te]
    final_te = rolling_platt(mix_all, y_all, t, RECAL_WINDOWS)[len(va) :]
    recent = t > t.max() - RECAL_WINDOWS
    cal = Platt().fit(mix_all[recent], y_all[recent])  # the newest closed windows correct live scores
    name = "Blend with ETAS"
    thr = best_threshold(y_va, mix_va)
    res["models"][name] = {"validation": evaluate(y_va, mix_va, thr), "test": evaluate(y_te, final_te, thr)}
    res["risk_levels"] = risk_cuts(y_va, mix_va)
    res["winner"], res["pr_curve"], res["reliability"] = name, pr_points(y_te, final_te), reliability(y_te, final_te)
    cell = (te.lat.to_numpy() * 1000 + te.lon.to_numpy()).round(1)
    log("  Running pyCSEP tests")
    res["etas"] = {
        **{k: v for k, v in fitted.items() if not k.startswith("_")},
        "validation_pr_auc": float(average_precision_score(y_va, base_va)),
        "test_pr_auc": float(average_precision_score(y_te, base_te)),
        "ml_gain_nats_per_row": etas.binary_gain(y_te, final_te, base_te),
        "csep_ml": etas.consistency(y_te, final_te, cell),
        "csep_etas": etas.consistency(y_te, base_te, cell),
        "count_ml": etas.exact_count(y_te, final_te),
        "count_etas": etas.exact_count(y_te, base_te),
    }
    shown = pd.concat([te, current])
    shown_etas = pd.Series(etas.table_forecast(fitted, cat, shown, START, WINDOW_DAYS), index=shown.index)

    def score(d):
        return cal(blend(xgb.predict_proba(X(d))[:, 1], shown_etas.loc[d.index].to_numpy()))

    p = score(current)
    quake_events(shown, cat, score)
    res.update(
        hazard="earthquake",
        unit="1 degree cell, 30-day window",
        label="M5.0+ in next 30 days",
        baseline_name="ETAS fitted by maximum likelihood",
        catalog_events=len(cat),
        cells=len(current),
        sets={"train": len(tr), "validation": len(va), "test": len(te)},
        periods={"train": "1980 to 2015", "validation": "2016 to 2019", "test": "2020 onward"},
        current_window_start=str(current.start.iloc[0].date()),
        scored_at=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        current_risk=[
            {"lat": float(a), "lon": float(b), "p": round(float(c), 4)} for a, b, c in zip(current.lat, current.lon, p)
        ],
    )
    write_results("earthquake", res)
    save_model(
        "earthquake",
        {
            "features": FEATURES,
            "model": xgb,
            "cal": cal,
            "blend": blend,
            "etas": {"params": fitted["params"], "_u": fitted["_u"]},
        },
    )
    return res


if __name__ == "__main__":
    print_summary(run())
