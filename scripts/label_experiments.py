"""Step 4: do GDACS flood events improve the news-report label? Writes results/exp_labels.json.

GDACS is a separate record, so it also gives a yardstick that the news label did not write.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap
from xgboost import XGBClassifier

from hazards import flood
from hazards.common import DATA, RESULTS, log, tuned
from hazards.typhoon import _dist_km

KM = 300


def gdacs_label(rows: pd.DataFrame, g: pd.DataFrame, horizon: int = flood.HORIZON, km: float = KM) -> np.ndarray:
    """1 when a GDACS flood starts within km of the location during the next horizon days. Rows need lat, lon, date."""
    out = np.zeros(len(rows), dtype=int)
    starts = g.start.dt.tz_localize(None).dt.normalize()
    for (lat, lon), part in rows.groupby(["lat", "lon"]):
        near = _dist_km(lat, lon, g.lat.to_numpy(float), g.lon.to_numpy(float)) <= km
        days = pd.DatetimeIndex(starts[near].unique())
        hit = pd.Series(0, index=pd.date_range(part.date.min(), part.date.max() + pd.Timedelta(days=horizon)))
        hit.loc[hit.index.intersection(days)] = 1
        ahead = hit[::-1].rolling(horizon).sum()[::-1].shift(-1)
        out[rows.index.get_indexer(part.index)] = (ahead.reindex(part.date).to_numpy() > 0).astype(int)
    return out


def main() -> None:
    log("Label experiment start")
    cells, df, tr, va, te_all = flood.load_sets()
    te = flood.with_real_forecasts(te_all)
    g = pd.read_parquet(DATA / "gdacs_floods.parquet")
    log(f"GDACS events {len(g)}, {g.start.min()} to {g.start.max()}")
    tr, va, te = [x.reset_index(drop=True) for x in (tr, va, te)]
    for x in (tr, va, te):
        x["gdacs"] = gdacs_label(x, g)
    params = tuned("flood")
    cols = flood.MODEL_FEATURES

    def X(d):
        return d[cols].to_numpy(np.float32)

    def score(ytr):
        rows = []
        for seed in range(3):
            spw = float((ytr == 0).sum() / (ytr == 1).sum())
            m = XGBClassifier(n_estimators=300, tree_method="hist", n_jobs=-1, random_state=seed, scale_pos_weight=spw, **params).fit(X(tr), ytr)
            pv, pt = m.predict_proba(X(va))[:, 1], m.predict_proba(X(te))[:, 1]
            rows.append((ap(va.y, pv), ap(va.gdacs, pv), ap(te.y, pt), ap(te.gdacs, pt)))
        r = np.array(rows)
        return r.mean(axis=0), r.std(axis=0)

    labels = {"News surge": tr.y.astype(int), "GDACS only": tr.gdacs, "News or GDACS": ((tr.y == 1) | (tr.gdacs == 1)).astype(int)}
    table, base = [], None
    for name, ytr in labels.items():
        mean, sd = score(ytr.to_numpy())
        base = (mean, sd) if base is None else base
        win = bool(mean[1] > base[0][1] + 2 * max(sd[1], base[1][1]) and mean[0] >= base[0][0] - 2 * max(sd[0], base[1][0]))
        table.append({"name": name, "positives_train": float(ytr.mean()), "val_news": float(mean[0]), "val_gdacs": float(mean[1]), "test_news": float(mean[2]), "test_gdacs": float(mean[3]), "kept": "Baseline" if name == "News surge" else "Yes" if win else "No"})
        log(f"  {table[-1]}")
    out = {"table": table, "base_rates": {"val_news": float(va.y.mean()), "val_gdacs": float(va.gdacs.mean()), "test_news": float(te.y.mean()), "test_gdacs": float(te.gdacs.mean())},
           "km": KM, "note": f"Kept only if PR-AUC against GDACS on validation rises by more than 2 standard deviations over 3 seeds, without losing the news label. A GDACS match is a new event within {KM} km in the next {flood.HORIZON} days."}
    (RESULTS / "exp_labels.json").write_text(json.dumps(out, indent=1))
    log("Label experiment finish")


if __name__ == "__main__":
    main()
