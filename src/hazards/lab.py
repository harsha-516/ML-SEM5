"""Tuning, regularisation and imbalance experiments. Writes results/lab_<hazard>.json."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .common import RESULTS, log

MAX_ROWS = 300_000
SPACE = {"max_depth": [3, 4, 5, 6, 8], "learning_rate": [0.01, 0.02, 0.05, 0.1], "subsample": [0.6, 0.8, 1.0],
         "colsample_bytree": [0.5, 0.7, 1.0], "reg_lambda": [0.1, 1, 5, 20], "reg_alpha": [0, 0.1, 1, 5],
         "min_child_weight": [1, 5, 20, 50]}


def time_folds(when: pd.Series, n: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding folds: every validation row is later than every training row of that fold."""
    cuts = np.quantile(when.to_numpy(), np.linspace(0.4, 1, n + 1))
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        upper = (when <= b) if b == cuts[-1] else (when < b)
        out.append((np.where(when < a)[0], np.where((when >= a) & upper)[0]))
    return out


def _pr(y, p):
    return float(average_precision_score(y, p))


def _xgb(y, **kw):
    spw = float((len(y) - y.sum()) / max(y.sum(), 1))
    params = dict(n_estimators=300, tree_method="hist", n_jobs=-1, random_state=42, eval_metric="aucpr", scale_pos_weight=spw)
    params.update(kw)
    return XGBClassifier(**params)


def regularisation(Xt, yt, Xv, yv) -> dict:
    n = min(len(yt), 60_000)
    a, b = Xt[-n:], yt[-n:]
    lr = []
    for penalty, ratio in (("l2", 0.0), ("l1", 1.0), ("elasticnet", 0.5)):
        for C in (0.001, 0.01, 0.1, 1, 10):
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=C, l1_ratio=ratio, solver="saga", max_iter=300, class_weight="balanced"))
            m.fit(a, b)
            lr.append({"penalty": penalty, "C": C, "train_pr": _pr(b, m.predict_proba(a)[:, 1]),
                       "val_pr": _pr(yv, m.predict_proba(Xv)[:, 1]), "nonzero": int((np.abs(m[-1].coef_[0]) > 1e-6).sum())})
    tree = []
    for depth in (2, 4, 6, 8, 12, 16, None):
        m = make_pipeline(SimpleImputer(strategy="median"), DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=42))
        m.fit(a, b)
        tree.append({"max_depth": depth or 0, "train_pr": _pr(b, m.predict_proba(a)[:, 1]), "val_pr": _pr(yv, m.predict_proba(Xv)[:, 1])})
    return {"logistic": lr, "tree": tree}


def tune(X, y, folds) -> dict:
    def score(p):
        return float(np.mean([_pr(y[v], _xgb(y[t], **p).fit(X[t], y[t]).predict_proba(X[v])[:, 1]) for t, v in folds]))

    rows = []
    for n, p in enumerate(ParameterSampler(SPACE, n_iter=20, random_state=42), start=1):
        rows.append({"params": p, "cv_pr": score(p)})
        log(f"  Random search {n}/20: CV PR-AUC {rows[-1]['cv_pr']:.4f}, best {max(r['cv_pr'] for r in rows):.4f}")
    rows.sort(key=lambda r: -r["cv_pr"])
    best = rows[0]["params"]
    grid = []
    for lr in (best["learning_rate"] / 2, best["learning_rate"], best["learning_rate"] * 1.5):
        for depth in (best["max_depth"] - 1, best["max_depth"], best["max_depth"] + 1):
            p = {**best, "learning_rate": lr, "max_depth": max(depth, 2)}
            grid.append({"params": p, "cv_pr": score(p)})
            log(f"  Grid search {len(grid)}/9: CV PR-AUC {grid[-1]['cv_pr']:.4f}")
    grid.sort(key=lambda r: -r["cv_pr"])
    return {"random": rows, "grid": grid, "best": grid[0]["params"] if grid[0]["cv_pr"] >= rows[0]["cv_pr"] else best}


def imbalance(Xt, yt, Xv, yv) -> dict:
    n = min(len(yt), 150_000)
    imp = SimpleImputer(strategy="median").fit(Xt[-n:])
    A, b, V = imp.transform(Xt[-n:]), yt[-n:], imp.transform(Xv)
    Sa, Sb = SMOTE(random_state=42).fit_resample(A, b)  # training rows only
    return {"no_correction": _pr(yv, _xgb(b, scale_pos_weight=1).fit(A, b).predict_proba(V)[:, 1]),
            "class_weights": _pr(yv, _xgb(b).fit(A, b).predict_proba(V)[:, 1]),
            "smote": _pr(yv, _xgb(Sb, scale_pos_weight=1).fit(Sa, Sb).predict_proba(V)[:, 1])}


def run_lab(hazard, tr, va, te, cols, when) -> dict:
    tr = tr.assign(_w=when(tr)).sort_values("_w")
    Xt, yt = tr[cols].to_numpy(np.float32), tr.y.astype(int).to_numpy()
    Xv, yv = va[cols].to_numpy(np.float32), va.y.astype(int).to_numpy()
    Xe, ye = te[cols].to_numpy(np.float32), te.y.astype(int).to_numpy()
    Xc, yc, wc = Xt[-MAX_ROWS:], yt[-MAX_ROWS:], tr["_w"].iloc[-MAX_ROWS:].reset_index(drop=True)
    res = {"hazard": hazard, "tuning_rows": int(len(yc)), "regularisation": regularisation(Xt, yt, Xv, yv),
           "imbalance": imbalance(Xt, yt, Xv, yv), "tuning": tune(Xc, yc, time_folds(wc))}
    default = _xgb(yt).fit(Xt, yt)
    tuned = _xgb(yt, **res["tuning"]["best"]).fit(Xt, yt)
    res["default_vs_tuned"] = {"default_val": _pr(yv, default.predict_proba(Xv)[:, 1]), "tuned_val": _pr(yv, tuned.predict_proba(Xv)[:, 1]),
                               "default_test": _pr(ye, default.predict_proba(Xe)[:, 1]), "tuned_test": _pr(ye, tuned.predict_proba(Xe)[:, 1])}
    (RESULTS / f"lab_{hazard}.json").write_text(json.dumps(res, indent=1, default=float))
    print(hazard, res["imbalance"], res["default_vs_tuned"], flush=True)
    return res


def main(which: list[str]) -> None:
    if "cyclone" in which:
        from . import cyclone
        _, _, tr, va, te = cyclone.load_sets()
        run_lab("cyclone", tr, va, te, cyclone.FEATURES, lambda d: d.storm_year)
    if "typhoon" in which:
        from . import typhoon
        _, tr, va, te = typhoon.load_sets()
        run_lab("typhoon", tr, va, te, typhoon.FEATURES, lambda d: d.storm_year)
    if "flood" in which:
        from . import flood
        _, _, tr, va, te = flood.load_sets()
        te = flood.with_real_forecasts(te)
        run_lab("flood", tr, va, te, flood.MODEL_FEATURES, lambda d: d.date.astype("int64"))
    if "earthquake" in which:
        from . import earthquake
        _, tr, va, te, _ = earthquake.load_sets()
        run_lab("earthquake", tr, va, te, earthquake.FEATURES, lambda d: d.window)


if __name__ == "__main__":
    main(sys.argv[1:] or ["typhoon", "flood", "earthquake"])
