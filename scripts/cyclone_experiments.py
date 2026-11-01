"""Tasks 4 and 5: with and without ERA5 or Himawari on the same rows, for 3 targets.

Usage: python scripts/cyclone_experiments.py era5|himawari
Only years with data are usable, so the split is built from those years, oldest first.
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap
from xgboost import XGBClassifier, XGBRegressor

from hazards import cyclone, era5
from hazards import himawari as hw
from hazards.common import DATA, RESULTS, log
from hazards.typhoon import build_table


NOTES = {"era5": " ERA5 arrives 5 or more days late, so this measures physical value and could not run live.",
         "himawari": " Cloud is read at best-track positions, which are corrected after the storm; live positions would be rougher."}


def split_years(years: list[int]) -> tuple[list[int], list[int], list[int]]:
    """Time-ordered split. Test stays 2022 onward when enough earlier years exist."""
    years = sorted(years)
    early = [y for y in years if y < 2022]
    if len(early) >= 4:
        cut = max(1, round(len(early) * 0.3))
        return early[:-cut], early[-cut:], [y for y in years if y >= 2022]
    n = len(years)
    a = min(max(1, round(n * 0.6)), n - 2)
    b = a + max(1, round(n * 0.2))
    return years[:a], years[a:b], years[b:]


def fit_score(tr, va, te, cols, target, seed):
    common = dict(n_estimators=800, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=40, tree_method="hist", n_jobs=-1, random_state=seed)
    X = lambda d: d[cols].to_numpy(np.float32)  # noqa: E731
    if target == "wind":
        m = XGBRegressor(objective="reg:absoluteerror", **common).fit(X(tr), tr.wind_24, eval_set=[(X(va), va.wind_24)], verbose=False)
        return -float(np.abs(m.predict(X(va)) - va.wind_24).mean()), -float(np.abs(m.predict(X(te)) - te.wind_24).mean())
    spw = float((tr[target] == 0).sum() / max(tr[target].sum(), 1))
    m = XGBClassifier(eval_metric="aucpr", scale_pos_weight=spw, **common).fit(X(tr), tr[target], eval_set=[(X(va), va[target])], verbose=False)
    return float(ap(va[target], m.predict_proba(X(va))[:, 1])), float(ap(te[target], m.predict_proba(X(te))[:, 1]))


def compare(tr, va, te, extra, target):
    def many(cols):
        r = np.array([fit_score(tr, va, te, cols, target, s) for s in range(3)])
        return r[:, 0].mean(), r[:, 0].std(), r[:, 1].mean()

    a, b = many(cyclone.FEATURES), many(cyclone.FEATURES + extra)
    return a, b, b[0] > a[0] + 2 * max(a[1], b[1])


def main(kind: str) -> None:
    log(f"Cyclone experiment start: {kind}")
    tracks = cyclone.load_tracks([DATA / "ibtracs_NI.csv", DATA / "ibtracs_WP.csv"])
    d = build_table(tracks, first_year=1980, keep_strong=True)
    d = d[d.wind.notna() & d.wind_24.notna()].assign(ri=lambda x: (x.wind_24 - x.wind >= cyclone.RI_KT).astype(int))
    if kind == "era5":
        extra = era5.NAMES
        d = era5.add_features(d[d.time.dt.year.isin(era5.years_ready())])
    else:
        extra = hw.NAMES
        s = pd.read_parquet(DATA / "himawari_features.parquet")
        s["time"] = pd.to_datetime(s.time)
        d = d.merge(s, on=["storm", "time"])
    d = d[d[extra].notna().all(axis=1)]
    years = sorted(d.storm_year.unique().tolist())
    a, b, c = split_years(years)
    log(f"  Rows {len(d)}, train years {a}, validation {b}, test {c}")
    rows = []
    for target, name, text in (("y", "Reach 64 kt", "PR-AUC"), ("wind", "24 h wind", "MAE, kt"), ("ri", "Rapid intensification", "PR-AUC")):
        part = d[d.wind < 64] if target == "y" else d
        tr, va, te = part[part.storm_year.isin(a)], part[part.storm_year.isin(b)], part[part.storm_year.isin(c)]
        if target != "wind" and (va[target].sum() < 5 or te[target].sum() < 5 or tr[target].sum() < 5):
            log(f"  {name}: too few positives")
            continue
        p, q, win = compare(tr, va, te, extra, target)
        show = (lambda v: f"{-v:.2f} kt") if target == "wind" else (lambda v: f"{v:.3f}")
        rows.append({"target": name, "rows": len(tr) + len(va) + len(te), "without_val": show(p[0]), "with_val": show(q[0]), "without_test": show(p[2]), "with_test": show(q[2]), "kept": "Yes" if win else "No"})
        log(f"  {rows[-1]}")
    out = {"table": rows, "years": {"train": a, "validation": b, "test": c}, "note": f"Same rows with and without. Train {a[0]} to {a[-1]}, validation {b[0]} to {b[-1]}, test {c[0]} to {c[-1]}. Kept only if validation improves by more than 2 standard deviations over 3 seeds." + NOTES[kind]}
    (RESULTS / f"exp_{kind}.json").write_text(json.dumps(out, indent=1))
    log(f"Cyclone experiment finish: {kind}")


if __name__ == "__main__":
    main(sys.argv[1])
