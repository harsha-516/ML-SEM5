"""Does calibration that follows time fix the earthquake over-forecast and the cyclone range? Writes results/exp_recal.json.

Choice uses late validation only. Each period is corrected with periods that closed before it.
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as ap

from hazards import cyclone, etas
from hazards import earthquake as q
from hazards.common import RESULTS, Platt, conformal_margin, log
from hazards.etas import exact_count
from hazards.live import load_model
from hazards.recal import rolling_margin, rolling_platt


def logit(p):
    """Log-odds of a chance."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def loss(y, p):
    """Log loss per row."""
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def n_test(y, p):
    """Expected and observed event counts."""
    total, n = float((-np.log1p(-np.clip(p, 0, 1 - 1e-9))).sum()), float(y.sum())
    return total, n, min(1 - poisson.cdf(n - 1, total), poisson.cdf(n, total)) >= 0.025


def earthquake() -> dict:
    """Earthquake recalibration experiment."""
    cat, tr, va, te, _ = q.load_sets()
    m = load_model("earthquake")
    fitted = etas.fit(cat, q.START, q.START + pd.Timedelta(days=q.MIN_ROW_WINDOW * q.WINDOW_DAYS), q.TRAIN_END)
    rows = pd.concat([va, te])
    y, t = rows.y.to_numpy(), rows.window.to_numpy()
    raw = m["model"].predict_proba(rows[m["features"]].to_numpy(np.float32))[:, 1]
    base = etas.table_forecast(fitted, cat, rows, q.START, q.WINDOW_DAYS)
    is_va = np.arange(len(rows)) < len(va)
    w = np.unique(t[is_va])
    cut = w[len(w) // 2]
    early, late = is_va & (t < cut), is_va & (t >= cut)
    is_te = ~is_va
    Z = np.column_stack([logit(raw), logit(base)])

    def versions(fit_on):
        """Each version, with static parts fitted only on the rows given."""
        fixed = Platt().fit(raw[fit_on], y[fit_on])(raw)
        blend = LogisticRegression(C=1e6, max_iter=2000).fit(Z[fit_on], y[fit_on]).predict_proba(Z)[:, 1]
        return {
            "Fixed calibration": fixed,
            "Rolling, 2 years": rolling_platt(raw, y, t, 24),
            "Rolling, 4 years": rolling_platt(raw, y, t, 48),
            "Blend with ETAS": blend,
            "Blend, rolling 2 years": rolling_platt(blend, y, t, 24),
            "Blend, rolling 4 years": rolling_platt(blend, y, t, 48),
        }

    pick, final = versions(early), versions(is_va)
    per = pd.DataFrame({"t": t[late], "y": y[late]})
    table, ref = [], None
    for name in pick:
        per["l"] = loss(y[late], pick[name][late])
        by = per.groupby("t").l.mean()
        ref = by if ref is None else ref
        gain = (
            (ref - by).mean() / max((ref - by).std() / np.sqrt(len(by)), 1e-12) if name != "Fixed calibration" else 0.0
        )
        exp, obs, ok = n_test(y[is_te], final[name][is_te])
        table.append(
            {
                "name": name,
                "late_validation_loss": float(by.mean()),
                "late_validation_pr_auc": float(ap(y[late], pick[name][late])),
                "gain_z": float(gain),
                "test_pr_auc": float(ap(y[is_te], final[name][is_te])),
                "expected": exp,
                "number_test": "Pass" if ok else "Fail",
                **exact_count(y[is_te], final[name][is_te]),
            }
        )
        log(f"  {table[-1]}")
    best = max(table, key=lambda r: r["gain_z"])
    kept = best["name"] if best["gain_z"] > 2 else "Fixed calibration"
    exp, obs, ok = n_test(y[is_te], base[is_te])
    return {
        "table": table,
        "kept": kept,
        "etas": {
            "test_pr_auc": float(ap(y[is_te], base[is_te])),
            "expected": exp,
            "number_test": "Pass" if ok else "Fail",
            **exact_count(y[is_te], base[is_te]),
        },
    }


def cyclones() -> dict:
    """Cyclone range experiment."""
    tracks = cyclone.load_tracks([cyclone.DATA / "ibtracs_NI.csv", cyclone.DATA / "ibtracs_WP.csv"])
    d = cyclone.storm_table(tracks, first_year=1980, keep_strong=True)
    d = d[d.wind.notna() & d.wind_24.notna() & (d.storm_year >= 2017)]
    m = load_model("cyclone")
    X = d[m["features"]].to_numpy(np.float32)
    low, high, y, year = (
        m["quantiles"]["low"].predict(X),
        m["quantiles"]["high"].predict(X),
        d.wind_24.to_numpy(),
        d.storm_year.to_numpy(),
    )
    te, late, early = year >= 2022, (year >= 2020) & (year <= 2021), year <= 2019

    def cover(mask, margin):
        return float(((y[mask] >= low[mask] - margin[mask]) & (y[mask] <= high[mask] + margin[mask])).mean())

    def width(mask, margin):
        return float((high[mask] - low[mask] + 2 * margin[mask]).mean())

    fixed_pick = np.full(y.shape, conformal_margin(y[early], low[early], high[early]))
    fixed_final = np.full(y.shape, conformal_margin(y[~te], low[~te], high[~te]))
    table = [
        {
            "name": "Fixed margin",
            "late_validation": cover(late, fixed_pick),
            "test": cover(te, fixed_final),
            "width": width(te, fixed_final),
        }
    ]
    for span in (2, 3):
        r = rolling_margin(y, low, high, year, span)
        table.append(
            {
                "name": f"Rolling, {span} years",
                "late_validation": cover(late, r),
                "test": cover(te, r),
                "width": width(te, r),
            }
        )
    for r in table:
        log(f"  {r}")
    best = min(table, key=lambda r: abs(r["late_validation"] - 0.8))
    latest = (year > year.max() - 3) & (year < year.max())
    return {
        "table": table,
        "kept": best["name"],
        "margin_latest_3_years": conformal_margin(y[latest], low[latest], high[latest]),
    }


if __name__ == "__main__":
    log("Calibration experiment start")
    out = {"earthquake": earthquake(), "cyclone": cyclones()}
    (RESULTS / "exp_recal.json").write_text(json.dumps(out, indent=1))
    log("Calibration experiment finish")
