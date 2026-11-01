"""Task 2: do IMERG Late rain or SMAP soil moisture beat the Task 1 flood model? Writes results/exp_satellite.json.

Each feature only reads days that would have been published by then: IMERG Late daily lags 2 days, SMAP 3.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap
from xgboost import XGBClassifier

from hazards import flood
from hazards.flood import IMERG_FEATURES as NAMES_I, imerg_features
from hazards.common import DATA, RESULTS, log, tuned

SMAP_LAG = 3
NAMES_S = ["smap_sm", "smap_anom"]


def load(folder: str, cells: pd.DataFrame) -> pd.DataFrame:
    """Days by locations, in choose_cells order."""
    files = sorted((DATA / folder).glob("*.npy"))
    idx = pd.to_datetime([f.stem for f in files])
    return pd.DataFrame(np.array([np.load(f) for f in files]), index=idx, columns=[(c.la + 0.5, c.lo + 0.5) for c in cells.itertuples()]).sort_index()


def smap_features(s: pd.Series, lag: int = SMAP_LAG) -> pd.DataFrame:
    r = s.asfreq("D").shift(lag).ffill(limit=6)  # swaths leave gaps, so keep the newest pass from the last week
    return pd.DataFrame({"smap_sm": r, "smap_anom": r - r.rolling(365, min_periods=180).median().shift(1)})


def main() -> None:
    log("Satellite experiments start")
    cells, df, tr, va, te_all = flood.load_sets()
    im, sm = load("imerg", cells), load("smap", cells)
    log(f"IMERG days {len(im)}, {im.index.min().date()} to {im.index.max().date()}; SMAP days {len(sm)}, {sm.index.min().date()} to {sm.index.max().date()}")
    corr, parts = [], []
    for c in cells.itertuples():
        key = (c.la + 0.5, c.lo + 0.5)
        w = pd.read_parquet(flood.CACHE / f"{c.la}_{c.lo}.parquet")
        corr.append(w.rain.corr(im[key].reindex(w.index)))
        f = imerg_features(im[key]).join(smap_features(sm[key]), how="outer")
        parts.append(f.assign(lat=key[0], lon=key[1], date=f.index))
    sat = pd.concat(parts, ignore_index=True)
    corr_median = float(np.median(corr))
    log(f"IMERG against Open-Meteo rain, daily correlation: median {corr_median:.3f}, lowest {min(corr):.3f}")

    def add(d):
        return d.drop(columns=[c for c in NAMES_I if c in d]).merge(sat, how="left", on=["lat", "lon", "date"]).set_index(d.index)

    tr2, va2, te2 = add(tr), add(va), add(flood.with_real_forecasts(te_all))
    params = tuned("flood")
    spw = float((tr2.y == 0).sum() / (tr2.y == 1).sum())

    def score(cols):
        v, t = [], []
        for seed in range(5):
            m = XGBClassifier(n_estimators=300, tree_method="hist", n_jobs=-1, random_state=seed, scale_pos_weight=spw, **params).fit(tr2[cols].to_numpy(np.float32), tr2.y.astype(int))
            v.append(ap(va2.y, m.predict_proba(va2[cols].to_numpy(np.float32))[:, 1]))
            t.append(ap(te2.y, m.predict_proba(te2[cols].to_numpy(np.float32))[:, 1]))
        return float(np.mean(v)), float(np.std(v)), float(np.mean(t))

    base = [c for c in flood.MODEL_FEATURES if c not in NAMES_I]
    b = score(base)
    log(f"  Task 1 model: {b}")
    rows = [{"name": "Task 1 model", "validation": b[0], "test": b[2], "kept": "Baseline"}]
    keep = list(base)
    for name, cols in (("Add IMERG", NAMES_I), ("Add SMAP", NAMES_S), ("Add both", NAMES_I + NAMES_S)):
        r = score(base + cols)
        win = r[0] > b[0] + 2 * max(r[1], b[1])
        rows.append({"name": name, "validation": r[0], "test": r[2], "kept": "Yes" if win else "No"})
        log(f"  {name}: {r}, beats Task 1: {win}")
    out = {"table": rows, "imerg_rain_correlation": corr_median, "imerg_days": int(len(im)), "smap_days": int(len(sm)), "rows": {"train": len(tr2), "validation": len(va2), "test": len(te2)},
           "note": f"IMERG Late, 2 days behind, correlates {corr_median:.2f} with reanalysis daily rain at the median location. SMAP is 3 days behind. Kept only if validation beats the Task 1 model by more than 2 standard deviations over 5 seeds."}
    (RESULTS / "exp_satellite.json").write_text(json.dumps(out, indent=1))
    log("Satellite experiments finish")


if __name__ == "__main__":
    main()
