"""Shared evaluation and model comparison for all hazards."""

from __future__ import annotations

import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xgboost as xgb_core
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
LOG = ROOT / "logs" / "training.log"


def log(message: str) -> None:
    """Print progress and append it to logs/training.log, so it can be watched from another terminal."""
    line = f"{datetime.now():%H:%M:%S}  {message}"
    print(line, flush=True)
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf8") as f:
        f.write(line + "\n")


class RoundLog(xgb_core.callback.TrainingCallback):
    """Log the validation score every 25 boosting rounds."""

    def after_iteration(self, model, epoch, evals_log):
        if epoch % 25 == 0:
            scores = next(iter(next(iter(evals_log.values())).values()))
            log(f"    XGBoost round {epoch}: validation PR-AUC {scores[-1]:.4f}")
        return False


def evaluate(y_true, proba, threshold: float = 0.5) -> dict:
    """Score probabilities. PR-AUC is the headline; accuracy is left out on purpose."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "n": len(y_true),
        "base_rate": float(y_true.mean()),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, np.clip(proba, 0, 1))),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def best_threshold(y_true, proba, beta: float = 2.0) -> float:
    """Threshold maximising F-beta on validation data. Beta 2 favours recall."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f = (1 + beta**2) * prec * rec / np.maximum(beta**2 * prec + rec, 1e-12)
    return float(thr[int(np.argmax(f[:-1]))])


def risk_cuts(y_true, proba) -> dict:
    """Low, Medium and High boundaries from the validation precision-recall curve. Medium favours recall, High favours precision."""
    medium, high = best_threshold(y_true, proba, 2.0), best_threshold(y_true, proba, 0.5)
    return {"medium": min(medium, high), "high": max(medium, high)}


def reliability(y_true, proba, bins: int = 10) -> list[dict]:
    """Observed rate per equal-count probability bin."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    order = np.argsort(proba)
    return [
        {"predicted": float(proba[c].mean()), "observed": float(y_true[c].mean()), "n": len(c)}
        for c in np.array_split(order, bins)
        if len(c)
    ]


def conformal_margin(y, low, high, level: float = 0.8) -> float:
    """Amount to widen a quantile range so it covers `level` of validation outcomes (conformalized quantile regression)."""
    y, low, high = map(np.asarray, (y, low, high))
    score = np.maximum(low - y, y - high)
    n = len(score)
    return float(np.quantile(score, min(1.0, level * (n + 1) / n), method="higher"))


def _logit(p):
    """Log-odds of a chance, as one column for the stacking model."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class Platt:
    """One-variable logistic recalibration fitted on validation data."""

    def fit(self, p, y):
        self.m = LogisticRegression(C=1e6).fit(_logit(p), y)
        return self

    def __call__(self, p):
        return self.m.predict_proba(_logit(p))[:, 1]


def tuned(hazard: str) -> dict:
    """Best XGBoost settings from the tuning lab, or nothing if it has not run."""
    path = RESULTS / f"lab_{hazard}.json"
    return json.loads(path.read_text())["tuning"]["best"] if path.exists() else {}


def pr_points(y, p) -> list[dict]:
    """Thin the precision-recall curve to a small list for the website."""
    prec, rec, thr = precision_recall_curve(y, p)
    pos = int(np.sum(y))
    picks = np.unique(np.linspace(0, len(thr) - 1, 60).astype(int))
    return [
        {
            "threshold": float(thr[i]),
            "recall": float(rec[i]),
            "precision": float(prec[i]),
            "caught": int(round(rec[i] * pos)),
            "missed": pos - int(round(rec[i] * pos)),
            "false_alarms": int(round(rec[i] * pos * (1 / max(prec[i], 1e-9) - 1))),
        }
        for i in picks
    ]


def compare_models(
    X_tr, y_tr, X_va, y_va, X_te, y_te, base_va, base_te, feature_names, seed: int = 42, xgb_params: dict | None = None
) -> dict:
    """Fit all model families, stack on validation, pick the winner on validation only."""
    spw = float((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1))
    models = {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"),
        ),
        "Decision Tree": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, class_weight="balanced", random_state=seed),
        ),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=20,
                max_samples=min(0.5, 200000 / len(y_tr) + 0.05),
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            ),
        ),
    }
    settings = dict(
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.1,
    )
    settings.update(xgb_params or {})
    xgb = XGBClassifier(
        n_estimators=1500,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        early_stopping_rounds=40,
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
        callbacks=[RoundLog()],
        **settings,
    )
    out, va_p, te_p, fitted = {}, {}, {}, {}
    log(f"Training on {len(y_tr):,} rows, {len(feature_names)} features")
    for name, m in models.items():
        log(f"  Fitting {name}")
        m.fit(X_tr, y_tr)
        va_p[name], te_p[name] = m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]
        fitted[name] = m
    log("  Fitting XGBoost")
    xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    log(f"  XGBoost stopped at round {xgb.best_iteration}")
    va_p["XGBoost"], te_p["XGBoost"] = xgb.predict_proba(X_va)[:, 1], xgb.predict_proba(X_te)[:, 1]
    fitted["XGBoost"] = xgb
    train_pr = float(average_precision_score(y_tr, xgb.predict_proba(X_tr)[:, 1]))

    # Base models never saw validation, so stacking on it cannot learn from training fit.
    keys = ["Logistic Regression", "Random Forest", "XGBoost"]
    S_va = np.hstack([_logit(va_p[k]) for k in keys])
    S_te = np.hstack([_logit(te_p[k]) for k in keys])
    meta = LogisticRegression(C=1.0, max_iter=2000).fit(S_va, y_va)
    va_p["Stacking"], te_p["Stacking"] = meta.predict_proba(S_va)[:, 1], meta.predict_proba(S_te)[:, 1]

    va_p["Baseline"], te_p["Baseline"] = np.asarray(base_va, float), np.asarray(base_te, float)
    for name in list(va_p):
        if name == "Baseline":
            pv, pt = va_p[name], te_p[name]  # raw score: ranking metrics only need order
            lo, hi = pv.min(), max(pv.max(), pv.min() + 1e-9)
            pv, pt = np.clip((pv - lo) / (hi - lo), 0, 1), np.clip((pt - lo) / (hi - lo), 0, 1)
        else:
            pv, pt = va_p[name], te_p[name]
        cal = Platt().fit(pv, y_va)
        pv, pt = cal(pv), cal(pt)
        thr = best_threshold(y_va, pv)
        out[name] = {"validation": evaluate(y_va, pv, thr), "test": evaluate(y_te, pt, thr)}
        va_p[name], te_p[name] = pv, pt

    contenders = [k for k in out if k != "Baseline"]
    winner = max(contenders, key=lambda k: out[k]["validation"]["pr_auc"])
    for k in out:
        log(f"  {k}: validation PR-AUC {out[k]['validation']['pr_auc']:.3f}, test {out[k]['test']['pr_auc']:.3f}")
    log(f"  Selected on validation: {winner}")
    curve = pr_points(y_te, te_p[winner])
    gain = xgb.get_booster().get_score(importance_type="gain")
    imp = sorted(((feature_names[int(k[1:])], float(v)) for k, v in gain.items()), key=lambda x: -x[1])
    total = sum(v for _, v in imp) or 1.0
    return {
        "models": out,
        "winner": winner,
        "pr_curve": curve,
        "xgb_train_pr_auc": train_pr,
        "xgb_best_iteration": int(xgb.best_iteration),
        "reliability": reliability(y_te, te_p[winner]),
        "feature_importance": [{"feature": k, "share": v / total} for k, v in imp[:12]],
        "_fitted": fitted,
        "_test_proba": te_p,
        "_meta": meta,
        "_meta_keys": keys,
    }


def write_results(hazard: str, payload: dict) -> Path:
    """Save a results file, leaving out keys that start with an underscore."""
    RESULTS.mkdir(exist_ok=True)
    clean = {k: v for k, v in payload.items() if not k.startswith("_")}
    clean["generated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    path = RESULTS / f"{hazard}.json"
    path.write_text(json.dumps(clean, indent=2))
    return path


def print_summary(r: dict) -> None:
    """Print the winner and each model score."""
    print("winner:", r["winner"], "| sets:", r["sets"])
    for k, v in r["models"].items():
        t = v["test"]
        print(
            f"{k:22s} PR {t['pr_auc']:.3f} ROC {t['roc_auc']:.3f} rec {t['recall']:.2f} "
            f"prec {t['precision']:.2f} brier {t['brier']:.4f} base {t['base_rate']:.4f}"
        )


def dist_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres between two points. Accepts numbers or arrays."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))
