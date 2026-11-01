"""Task 1 evidence: which flood features help. Decisions use validation; test is reported once."""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as ap
from xgboost import XGBClassifier

from hazards import flood
from hazards.common import RESULTS, log

log("Flood experiments start")
cells, df, tr, va, te = flood.load_sets()
real = flood.with_real_forecasts(te)
actual = te.loc[real.index]
best = json.load(open(RESULTS / "lab_flood.json"))["tuning"]["best"]
base, ratio, fc = flood.LIVE_FEATURES, flood.RATIO_FEATURES, flood.FORECAST_FEATURES
log(f"Rows train {len(tr):,} validation {len(va):,} test {len(te):,}, with real forecasts {len(real):,}, base rate {real.y.mean():.3f}")


def fit(cols):
    spw = float((tr.y == 0).sum() / (tr.y == 1).sum())
    return XGBClassifier(n_estimators=300, tree_method="hist", n_jobs=-1, random_state=42, scale_pos_weight=spw, **best).fit(
        tr[cols].to_numpy(np.float32), tr.y.astype(int))


def p(m, d, cols):
    return m.predict_proba(d[cols].to_numpy(np.float32))[:, 1]


rows, models = [], {}
for name, cols in (("Current", base), ("Add ratios", base + ratio), ("Add forecast", base + fc), ("Add both", base + ratio + fc)):
    m = models[name] = fit(cols)
    r = {"name": name, "validation": float(ap(va.y, p(m, va, cols))), "test_real_forecast": float(ap(real.y, p(m, real, cols)))}
    if set(fc) & set(cols):
        r["test_actual_rain"] = float(ap(actual.y, p(m, actual, cols)))
    rows.append(r)
    log(f"  {r}")

# Final stage on 2024 forecast rows: decided inside 2024, then scored on 2025 onward.
no_fc, both = models["Add ratios"], models["Add both"]
cols_nf, cols_b = base + ratio, base + ratio + fc


def stage_x(d):
    a = np.clip(p(no_fc, d, cols_nf), 1e-6, 1 - 1e-6)
    return np.column_stack([np.log(a / (1 - a)), np.log1p(d.next3_rain), np.log1p(d.next3_ratio)])


early, late = real[real.date < "2024-09-01"], real[(real.date >= "2024-09-04") & (real.date < "2025-01-01")]
stage = LogisticRegression(C=1.0, max_iter=2000).fit(stage_x(early), early.y.astype(int))
y24 = float(ap(late.y, stage.predict_proba(stage_x(late))[:, 1]))
d24 = float(ap(late.y, p(both, late, cols_b)))
log(f"  Stage check on Sep to Dec 2024: stage {y24:.3f}, reanalysis-trained {d24:.3f}")
hold = real[real.date >= "2025-01-01"]
full = real[real.date < "2024-12-29"]
stage = LogisticRegression(C=1.0, max_iter=2000).fit(stage_x(full), full.y.astype(int))
out = {"table": rows, "rows": {"test": len(te), "test_with_real_forecast": len(real)}, "base_rate_real": float(real.y.mean()),
       "stage_2024": {"check_stage": y24, "check_reanalysis_trained": d24, "kept": y24 > d24,
                      "test_2025_stage": float(ap(hold.y, stage.predict_proba(stage_x(hold))[:, 1])),
                      "test_2025_reanalysis_trained": float(ap(hold.y, p(both, hold, cols_b))), "test_rows_2025": len(hold)},
       "forecast_actual_correlation": float(actual.next3_rain.corr(real.next3_rain))}
(RESULTS / "exp_flood.json").write_text(json.dumps(out, indent=1))
log(f"Flood experiments finish: {out['stage_2024']}, correlation {out['forecast_actual_correlation']:.3f}")
