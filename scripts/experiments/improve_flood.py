"""Which new flood features earn their place? Choices use validation; test is printed for the record."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap
from xgboost import XGBClassifier

from hazards import flood
from hazards.common import DATA, RESULTS

events = pd.read_parquet(DATA / "groundsource_region.parquet")
cells = flood.choose_cells(events)
best = json.load(open(RESULTS / "lab_flood.json"))["tuning"]["best"]
rows = []
for c in cells.itertuples():
    w = flood.fetch_cell(c.la + 0.5, c.lo + 0.5, flood.CACHE / f"{c.la}_{c.lo}.parquet")
    n = flood.daily_counts(events, c.la, c.lo, w.index)
    f = flood.weather_features(w)
    for k in (1, 3, 7, 30):
        s = w.rain.rolling(k, min_periods=1).sum()
        f[f"rain_{k}_ratio"] = s / s.rolling(365, min_periods=180).quantile(0.95).shift(1).clip(lower=1)
    f["wet_days_7"] = (w.rain > 5).rolling(7).sum()
    f["wet_days_30"] = (w.rain > 5).rolling(30).sum()
    q = w.discharge
    f["discharge_rank"] = q.rolling(365, min_periods=180).rank(pct=True)
    next3 = w.rain.rolling(3).sum().shift(-3)
    f["next3_rain"] = next3
    f["next3_ratio"] = next3 / w.rain.rolling(3).sum().rolling(365, min_periods=180).quantile(0.95).shift(1).clip(
        lower=1
    )
    f["y"] = flood.surge_label(n)
    f["lat"], f["lon"], f["date"] = c.la + 0.5, c.lo + 0.5, f.index
    rows.append(f.dropna(subset=["y"]))
df = pd.concat(rows, ignore_index=True)
gap = pd.Timedelta(days=33)
tr = df[df.date < pd.Timestamp("2022-01-01") - gap]
va = df[(df.date >= "2022-01-01") & (df.date < pd.Timestamp("2024-01-01") - gap)]
te = df[df.date >= "2024-01-01"]
base = flood.LIVE_FEATURES
ratio = ["rain_1_ratio", "rain_3_ratio", "rain_7_ratio", "rain_30_ratio", "wet_days_7", "wet_days_30", "discharge_rank"]
fc = ["next3_rain", "next3_ratio"]


def run(name, cols):
    """Train with these columns and print the scores."""
    m = XGBClassifier(
        n_estimators=300,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        scale_pos_weight=(tr.y == 0).sum() / (tr.y == 1).sum(),
        **best,
    )
    m.fit(tr[cols].to_numpy(np.float32), tr.y.astype(int))
    v, t = m.predict_proba(va[cols].to_numpy(np.float32))[:, 1], m.predict_proba(te[cols].to_numpy(np.float32))[:, 1]
    print(f"{name:30s} val {ap(va.y, v):.3f} test {ap(te.y, t):.3f}", flush=True)
    return m


run("current", base)
run("+ unusual for this place", base + ratio)
run("+ rain in next 3 days", base + fc)
m = run("+ both", base + ratio + fc)
cols = base + ratio + fc
print(sorted(zip(m.feature_importances_.round(3), cols), reverse=True)[:8])
print(
    "share of surges by month, test:",
    te[te.y == 1].date.dt.month.value_counts(normalize=True).sort_index().round(2).to_dict(),
)

# Honest test: replace actual later rain with rain forecasts as they were issued.
parts = []
for f in (DATA / "flood_forecasts").glob("*.parquet"):
    la, lo = map(int, f.stem.split("_"))
    d = pd.read_parquet(f)
    real = d.lead1.shift(-1) + d.lead2.shift(-2) + d.lead3.shift(-3)
    parts.append(pd.DataFrame({"lat": la + 0.5, "lon": lo + 0.5, "date": d.index, "real": real.to_numpy()}))
rf = pd.concat(parts, ignore_index=True)
t2 = te.merge(rf, on=["lat", "lon", "date"], how="inner").dropna(subset=["real"])
print("rows", len(te), "matched", len(t2), "locations", t2.groupby(["lat", "lon"]).ngroups)
scale = (t2.next3_rain / t2.next3_ratio).replace([np.inf, -np.inf], np.nan)
t3 = t2.assign(next3_rain=t2.real, next3_ratio=t2.real / scale)
cols2 = base + ratio
m2 = XGBClassifier(
    n_estimators=300,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    scale_pos_weight=(tr.y == 0).sum() / (tr.y == 1).sum(),
    **best,
).fit(tr[cols2].to_numpy(np.float32), tr.y.astype(int))
print("same rows, no forecast    ", round(ap(t2.y, m2.predict_proba(t2[cols2].to_numpy(np.float32))[:, 1]), 3))
print("same rows, actual rain    ", round(ap(t2.y, m.predict_proba(t2[cols].to_numpy(np.float32))[:, 1]), 3))
print("same rows, real forecasts ", round(ap(t3.y, m.predict_proba(t3[cols].to_numpy(np.float32))[:, 1]), 3))
print("forecast vs actual correlation", round(t2.real.corr(t2.next3_rain), 3), "base rate", round(t2.y.mean(), 3))
