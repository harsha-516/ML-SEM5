"""Calibration that follows time: each period is corrected using only periods that closed before it."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .common import Platt, conformal_margin


def rolling_platt(p: np.ndarray, y: np.ndarray, t: np.ndarray, span: int) -> np.ndarray:
    """Recalibrate each period t on the `span` periods before it. Periods with no history stay NaN."""
    p, y, t = np.asarray(p, float), np.asarray(y), np.asarray(t)
    out = np.full(p.shape, np.nan)
    for now in np.unique(t):
        past = (t < now) & (t >= now - span)
        if not past.any() or len(np.unique(y[past])) < 2:
            continue
        here = t == now
        out[here] = Platt().fit(p[past], y[past])(p[here])
    return out


def rolling_margin(
    y: np.ndarray, low: np.ndarray, high: np.ndarray, year: np.ndarray, span: int, level: float = 0.8
) -> np.ndarray:
    """Conformal margin for each year from the `span` years before it. Years with no history stay NaN."""
    y, low, high, year = map(np.asarray, (y, low, high, year))
    out = np.full(y.shape, np.nan)
    for now in np.unique(year):
        past = (year < now) & (year >= now - span)
        if past.any():
            out[year == now] = conformal_margin(y[past], low[past], high[past], level)
    return out


class Blend:
    """Log-odds mix of the ML chance and the ETAS chance."""

    @staticmethod
    def _z(ml, base):
        a, b = np.clip(ml, 1e-6, 1 - 1e-6), np.clip(base, 1e-6, 1 - 1e-6)
        return np.column_stack([np.log(a / (1 - a)), np.log(b / (1 - b))])

    def fit(self, ml, base, y):
        self.m = LogisticRegression(C=1e6, max_iter=2000).fit(self._z(ml, base), y)
        return self

    def __call__(self, ml, base):
        return self.m.predict_proba(self._z(ml, base))[:, 1]
