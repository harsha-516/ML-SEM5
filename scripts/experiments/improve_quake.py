"""Test new earthquake features: activity change, magnitude mix and longer memory. Choices use validation."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap
from xgboost import XGBClassifier

from hazards import earthquake as q
from hazards.common import RESULTS

cat, tr, va, te, _ = q.load_sets()
best = json.load(open(RESULTS / "lab_earthquake.json"))["tuning"]["best"]
H, W = q.BOX["lat1"] - q.BOX["lat0"], q.BOX["lon1"] - q.BOX["lon0"]
w = ((cat.time - q.START).dt.total_seconds() // (30 * 86400)).astype(int).to_numpy()
K = w.max() + 1
y = np.clip((cat.latitude.to_numpy() - q.BOX["lat0"]).astype(int), 0, H - 1)
x = np.clip((cat.longitude.to_numpy() - q.BOX["lon0"]).astype(int), 0, W - 1)
ok = (cat.latitude >= q.BOX["lat0"]) & (cat.latitude < q.BOX["lat1"])
n, big, shallow = np.zeros((K, H, W), np.float32), np.zeros((K, H, W), np.float32), np.zeros((K, H, W), np.float32)
np.add.at(n, (w, y, x), 1)
np.add.at(big, (w, y, x), (cat.mag >= 5).astype(np.float32).to_numpy())
np.add.at(shallow, (w, y, x), (cat.depth < 70).astype(np.float32).to_numpy())
extra = {}
n60, b60, s60 = q._roll(n, 60), q._roll(big, 60), q._roll(shallow, 60)
n120, b120 = q._roll(n, 120), q._roll(big, 120)
extra["large_share_5y"] = b60 / np.maximum(n60, 1)
extra["shallow_share_5y"] = s60 / np.maximum(n60, 1)
extra["rate_change"] = (q._roll(n, 12) + 0.25) / (n60 / 5 + 0.25)
extra["n_3600"] = n120
extra["m5_10y"] = b120
extra["neighbour_m5_5y"] = q._neigh(b60)
two = np.zeros_like(n60)
for dy in range(-2, 3):
    for dx in range(-2, 3):
        two += np.roll(np.roll(b60, dy, axis=1), dx, axis=2)
extra["region_m5_5y"] = two


def add(d):
    """Add the new candidate features."""
    d = d.copy()
    a, b = (d.lat - q.BOX["lat0"] - 0.5).astype(int).to_numpy(), (d.lon - q.BOX["lon0"] - 0.5).astype(int).to_numpy()
    k = d.window.to_numpy()
    for name, v in extra.items():
        d[name] = v[k, a, b]
    return d


tr, va, te = add(tr), add(va), add(te)
new = list(extra)
tr2 = tr[tr.start >= pd.Timestamp("1990-01-01", tz="UTC")]


def run(name, cols, data):
    """Train with these columns and print the scores."""
    m = XGBClassifier(
        n_estimators=300,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        scale_pos_weight=(data.y == 0).sum() / (data.y == 1).sum(),
        **best,
    )
    m.fit(data[cols].to_numpy(np.float32), data.y)
    v, t = m.predict_proba(va[cols].to_numpy(np.float32))[:, 1], m.predict_proba(te[cols].to_numpy(np.float32))[:, 1]
    print(f"{name:32s} val {ap(va.y, v):.3f} test {ap(te.y, t):.3f}", flush=True)
    return m


run("current", q.FEATURES, tr)
m = run("+ new features", q.FEATURES + new, tr)
run("+ new, training from 1990", q.FEATURES + new, tr2)
print(sorted(zip(m.feature_importances_.round(3), q.FEATURES + new), reverse=True)[:8])
