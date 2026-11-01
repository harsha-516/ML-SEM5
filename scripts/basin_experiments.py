"""Step 3 and 4 together: catchment rain and the wider label, same rows. Writes results/exp_basins.json."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap
from xgboost import XGBClassifier

from hazards import basins, flood
from hazards.common import DATA, RESULTS, log, tuned
from hazards.flood import gdacs_label


def main() -> None:
    log("Basin experiment start")
    cells, df, tr, va, te_all = flood.load_sets()
    te = flood.with_real_forecasts(te_all)
    path = DATA / "basin_rain.parquet"
    rain = basins.basin_rain(cells)
    rain.columns = [f"{a}_{b}" for a, b in rain.columns]
    rain.to_parquet(path)
    log(f"Basin rain: {len(rain)} days, {rain.index.min().date()} to {rain.index.max().date()}")
    parts = []
    for c in cells.itertuples():
        lat, lon = c.la + 0.5, c.lo + 0.5
        f = basins.features(rain[f"{lat}_{lon}"])
        parts.append(f.assign(lat=lat, lon=lon, date=f.index))
    sat = pd.concat(parts, ignore_index=True)
    g = pd.read_parquet(DATA / "gdacs_floods.parquet")

    def add(d):
        d = d.reset_index(drop=True).merge(sat, how="left", on=["lat", "lon", "date"])
        d["gdacs"] = gdacs_label(d, g)
        return d

    tr2, va2, te2 = add(tr), add(va), add(te)
    params = tuned("flood")

    def score(cols, wide):
        ytr = ((tr2.y == 1) | (tr2.gdacs == 1)).astype(int) if wide else tr2.y.astype(int)
        spw = float((ytr == 0).sum() / (ytr == 1).sum())
        rows = []
        for seed in range(5):
            m = XGBClassifier(n_estimators=300, tree_method="hist", n_jobs=-1, random_state=seed, scale_pos_weight=spw, **params).fit(tr2[cols].to_numpy(np.float32), ytr)
            pv, pt = m.predict_proba(va2[cols].to_numpy(np.float32))[:, 1], m.predict_proba(te2[cols].to_numpy(np.float32))[:, 1]
            rows.append((ap(va2.y, pv), ap(va2.gdacs, pv), ap(te2.y, pt), ap(te2.gdacs, pt)))
        r = np.array(rows)
        return r.mean(axis=0), r.std(axis=0)

    base = flood.MODEL_FEATURES
    table, ref = [], None
    for name, cols, wide in (("Current model", base, False), ("Add catchment rain", base + basins.NAMES, False), ("Wider label", base, True), ("Both", base + basins.NAMES, True)):
        mean, sd = score(cols, wide)
        ref = (mean, sd) if ref is None else ref
        win = bool(mean[0] > ref[0][0] + 2 * max(sd[0], ref[1][0]) or (mean[1] > ref[0][1] + 2 * max(sd[1], ref[1][1]) and mean[0] >= ref[0][0] - 2 * max(sd[0], ref[1][0])))
        table.append({"name": name, "val_news": float(mean[0]), "val_gdacs": float(mean[1]), "test_news": float(mean[2]), "test_gdacs": float(mean[3]), "kept": "Baseline" if ref[0] is mean else "Yes" if win else "No"})
        log(f"  {table[-1]}")
    (RESULTS / "exp_basins.json").write_text(json.dumps({"table": table, "note": "Same rows, 5 seeds. Kept if validation against news rises by more than 2 standard deviations, or against GDACS without losing news."}, indent=1))
    log("Basin experiment finish")


if __name__ == "__main__":
    main()
