"""Fit ETAS on training events and save it, then compare with the current ML model on the same rows."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from hazards import earthquake as q
from hazards import etas
from hazards.common import RESULTS, log

log("ETAS task start")
cat, tr, va, te, current = q.load_sets()
first = q.START + pd.Timedelta(days=q.MIN_ROW_WINDOW * q.WINDOW_DAYS)
result = etas.fit(cat, q.START, first, q.TRAIN_END)
np.save(RESULTS.parent / "data" / "raw" / "etas_background.npy", result["_u"])
pv, pt = etas.table_forecast(result, cat, va, q.START, q.WINDOW_DAYS), etas.table_forecast(result, cat, te, q.START, q.WINDOW_DAYS)
out = {k: v for k, v in result.items() if not k.startswith("_")}
out.update(validation_pr_auc=float(ap(va.y, pv)), test_pr_auc=float(ap(te.y, pt)), rows_validation=len(va), rows_test=len(te),
           expected_test=float((-np.log1p(-np.minimum(pt, 0.999999))).sum()), cells_active_test=int(te.y.sum()))
(RESULTS / "etas.json").write_text(json.dumps(out, indent=1))
log(f"ETAS task finish: validation {out['validation_pr_auc']:.3f}, test {out['test_pr_auc']:.3f}, params {out['params']}")
