"""Tests: Shared evaluation helpers."""

import numpy as np
import pandas as pd

from hazards.common import best_threshold, evaluate, reliability
from hazards.lab import time_folds


def test_constant_score_pr_auc_equals_base_rate():
    y = np.array([0, 0, 0, 1])
    assert evaluate(y, np.full(4, 0.3))["pr_auc"] == 0.25


def test_confusion_counts():
    c = evaluate([0, 1, 1, 0], [0.1, 0.9, 0.2, 0.8], 0.5)["confusion"]
    assert c == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}


def test_threshold_catches_positive():
    assert best_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) <= 0.8


def test_time_folds_never_train_on_later_rows():
    when = pd.Series(np.arange(1000))
    folds = time_folds(when)
    assert len(folds) == 3
    for t, v in folds:
        assert when.iloc[t].max() < when.iloc[v].min()


def test_reliability_counts_all_rows():
    assert sum(b["n"] for b in reliability(np.zeros(50), np.linspace(0, 1, 50))) == 50
