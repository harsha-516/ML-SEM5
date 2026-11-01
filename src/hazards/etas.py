"""Space-time ETAS for M5+: fitted by maximum likelihood, then turned into 30-day grid forecasts.

Rate at time t and place x:
    mu * u(x) + sum over past events K * exp(alpha * (m - 5)) * (t - ti + c)^-p * f(r; D * 10^(0.5 * (m - 5)))
f is a power-law spatial kernel with fixed tail q. u(x) is a background density, refined by stochastic declustering.
Forecasts use known events only, so aftershock cascades inside a window are not simulated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .common import log

M0 = 5.0
Q = 1.7
MAX_DAYS = 1825.0
MAX_KM = 500.0
EARTH = 6371.0
LAT0, LAT1, LON0, LON1 = -12, 50, 60, 150


@dataclass
class Params:
    mu: float  # background events per day, whole region
    k: float
    alpha: float
    c: float
    p: float
    d: float  # km, kernel scale for an M5


def excess(m: np.ndarray) -> np.ndarray:
    return m - M0


def omori_integral(dt: np.ndarray, c: float, p: float) -> np.ndarray:
    """Integral of (s + c)^-p for s from 0 to dt."""
    if abs(p - 1) < 1e-6:
        return np.log1p(dt / c)
    return (c ** (1 - p) - (dt + c) ** (1 - p)) / (p - 1)


def kernel(r2: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """Spatial density per square km."""
    return (Q - 1) / (np.pi * d2) * (1 + r2 / d2) ** -Q


def within(r: float, d2: np.ndarray) -> np.ndarray:
    """Share of the kernel inside radius r."""
    return 1 - (1 + r**2 / d2) ** (1 - Q)


def distance2(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Squared km on a local flat approximation, accurate at the distances that matter here."""
    y = np.radians(a - c) * EARTH
    x = np.radians(b - d) * EARTH * np.cos(np.radians((a + c) / 2))
    return x * x + y * y


def make_pairs(t: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Child, parent, days and squared km for each parent within the time and distance limits."""
    child, parent, gap, r2 = [], [], [], []
    start = 0
    for j in range(1, len(t)):
        while t[j] - t[start] > MAX_DAYS:
            start += 1
        i = np.arange(start, j)
        d = distance2(lat[j], lon[j], lat[i], lon[i])
        keep = (d <= MAX_KM**2) & (t[j] > t[i])
        child.append(np.full(keep.sum(), j, np.int32))
        parent.append(i[keep].astype(np.int32))
        gap.append((t[j] - t[i])[keep].astype(np.float32))
        r2.append(d[keep].astype(np.float32))
    return tuple(np.concatenate(x) for x in (child, parent, gap, r2))


def cell_area(lat: np.ndarray) -> np.ndarray:
    return (np.pi * EARTH / 180) ** 2 * np.cos(np.radians(lat))


def density(u: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Background per square km at each point. u sums to 1 over the region grid."""
    y = np.clip((lat - LAT0).astype(int), 0, u.shape[0] - 1)
    x = np.clip((lon - LON0).astype(int), 0, u.shape[1] - 1)
    return u[y, x] / cell_area(y + LAT0 + 0.5)


def _rates(par: Params, u, t, lat, lon, m, pairs) -> tuple[np.ndarray, np.ndarray]:
    child, parent, gap, r2 = pairs
    e = excess(m)
    d2 = (par.d * 10 ** (0.5 * e)) ** 2
    trig = par.k * np.exp(par.alpha * e[parent]) * (gap + par.c) ** -par.p * kernel(r2, d2[parent])
    triggered = np.bincount(child, weights=trig, minlength=len(t))
    return par.mu * density(u, lat, lon), triggered


def log_likelihood(par: Params, u, t, lat, lon, m, pairs, t0: float, t1: float) -> float:
    back, trig = _rates(par, u, t, lat, lon, m, pairs)
    use = t >= t0
    e = excess(m)
    d2 = (par.d * 10 ** (0.5 * e)) ** 2
    span = np.clip(t1 - t, 0, MAX_DAYS)
    lost = np.where(t < t0, omori_integral(np.clip(t0 - t, 0, MAX_DAYS), par.c, par.p), 0.0)
    total = par.mu * (t1 - t0) + (par.k * np.exp(par.alpha * e) * (omori_integral(span, par.c, par.p) - lost) * within(MAX_KM, d2)).sum()
    return float(np.log(back[use] + trig[use] + 1e-300).sum() - total)


def smooth(counts: np.ndarray) -> np.ndarray:
    """Light 3 by 3 smoothing with a small floor, so quiet cells keep a nonzero background."""
    k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], float) / 16
    out = np.zeros_like(counts, float)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out += k[dy + 1, dx + 1] * np.roll(np.roll(counts, dy, 0), dx, 1)
    out = out + 0.02
    return out / out.sum()


def background(lat, lon, weight, shape) -> np.ndarray:
    y = np.clip((lat - LAT0).astype(int), 0, shape[0] - 1)
    x = np.clip((lon - LON0).astype(int), 0, shape[1] - 1)
    c = np.zeros(shape)
    np.add.at(c, (y, x), weight)
    return smooth(c)


def fit(cat: pd.DataFrame, origin: pd.Timestamp, first: pd.Timestamp, last: pd.Timestamp, rounds: int = 3) -> dict:
    """Maximum likelihood on M5+ events before `last`. Events before `first` act only as parents."""
    e = cat[(cat.mag >= M0) & (cat.time < last)].sort_values("time")
    t = ((e.time - origin).dt.total_seconds() / 86400).to_numpy()
    lat, lon, m = e.latitude.to_numpy(), e.longitude.to_numpy(), e.mag.to_numpy()
    t0, t1 = (first - origin).days, (last - origin).days
    log(f"ETAS fit start: {len(t):,} events, finding parent pairs")
    pairs = make_pairs(t, lat, lon)
    log(f"ETAS pairs: {len(pairs[0]):,}")
    shape = (LAT1 - LAT0, LON1 - LON0)
    use = t >= t0
    u = background(lat[use], lon[use], np.ones(use.sum()), shape)
    x0 = np.log([use.sum() / (t1 - t0) / 2, 0.02, 1.5, 0.05, 1.15, 8.0])

    def to_par(z):
        return Params(*np.exp(z))

    steps = [0]

    def loss(z):
        v = -log_likelihood(to_par(z), u, t, lat, lon, m, pairs, t0, t1)
        steps[0] += 1
        if steps[0] % 100 == 0:
            log(f"  ETAS likelihood call {steps[0]}: {-v:,.1f}")
        return v if np.isfinite(v) else 1e12

    bounds = [(np.log(1e-3), np.log(50)), (np.log(1e-5), np.log(1)), (np.log(0.2), np.log(3)), (np.log(1e-3), np.log(2)),
              (np.log(0.8), np.log(2)), (np.log(0.5), np.log(100))]
    for n in range(rounds):
        r = minimize(loss, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 60, "maxfun": 600})
        x0 = r.x
        par = to_par(r.x)
        back, trig = _rates(par, u, t, lat, lon, m, pairs)
        share = back / (back + trig + 1e-300)
        u = background(lat[use], lon[use], share[use], shape)
        log(f"  ETAS round {n + 1}: {asdict(par)}, log-likelihood {-r.fun:,.1f}, background share {share[use].mean():.3f}")
    return {"params": asdict(par), "log_likelihood": float(-r.fun), "events": int(use.sum()), "background_share": float(share[use].mean()), "_u": u}


def grid_forecast(par: Params, u, t, lat, lon, m, start: float, days: float, cell_lat, cell_lon) -> np.ndarray:
    """Expected M5+ count per cell over [start, start + days), from events strictly before start."""
    keep = (t < start) & (start - t <= MAX_DAYS)
    t, lat, lon, m = t[keep], lat[keep], lon[keep], m[keep]
    area = cell_area(cell_lat)
    out = par.mu * density(u, cell_lat, cell_lon) * area * days
    if not len(t):
        return out
    e = excess(m)
    d2 = (par.d * 10 ** (0.5 * e)) ** 2
    time_part = par.k * np.exp(par.alpha * e) * (omori_integral(start - t + days, par.c, par.p) - omori_integral(start - t, par.c, par.p))
    radius = np.sqrt(area / np.pi)
    for i in range(len(t)):
        r2 = distance2(lat[i], lon[i], cell_lat, cell_lon)
        mass = kernel(r2, d2[i]) * area
        own = np.floor(cell_lat) == np.floor(lat[i])
        own &= np.floor(cell_lon) == np.floor(lon[i])
        mass = np.where(own, within(radius, d2[i]), np.minimum(mass, 1.0))  # a small kernel sits inside its own cell
        out = out + time_part[i] * mass
    return out


def table_forecast(result: dict, cat: pd.DataFrame, table: pd.DataFrame, origin: pd.Timestamp, window_days: int) -> np.ndarray:
    """Chance of at least one M5+ for every row of an earthquake table, aligned with its rows."""
    par, u = Params(**result["params"]), result["_u"]
    e = cat[cat.mag >= M0].sort_values("time")
    t = ((e.time - origin).dt.total_seconds() / 86400).to_numpy()
    lat, lon, m = e.latitude.to_numpy(), e.longitude.to_numpy(), e.mag.to_numpy()
    out = np.zeros(len(table))
    w = table.window.to_numpy()
    la, lo = table.lat.to_numpy(), table.lon.to_numpy()
    ids = np.unique(w)
    for n, k in enumerate(ids, start=1):
        row = w == k
        out[row] = 1 - np.exp(-grid_forecast(par, u, t, lat, lon, m, float(k * window_days), window_days, la[row], lo[row]))
        if n % 20 == 0:
            log(f"  ETAS forecast window {n}/{len(ids)}")
    return out


def exact_count(y: np.ndarray, p: np.ndarray) -> dict:
    """Active windows are yes or no, so the expected count is the plain sum of chances, with binomial spread."""
    mean, sd = float(p.sum()), float(np.sqrt((p * (1 - p)).sum()))
    return {"expected_active": mean, "observed": int(y.sum()), "z": (float(y.sum()) - mean) / sd}


def binary_gain(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Mean log-likelihood gain per row of forecast a over forecast b, for a yes or no outcome."""
    a, b = np.clip(a, 1e-9, 1 - 1e-9), np.clip(b, 1e-9, 1 - 1e-9)
    return float(np.mean(y * np.log(a / b) + (1 - y) * np.log((1 - a) / (1 - b))))


def consistency(y: np.ndarray, proba: np.ndarray, cell: np.ndarray, sims: int = 2000) -> dict:
    """pyCSEP number, likelihood and spatial tests on active cell-windows summed over the period.

    Cell totals are used because both forecasts predict whether a cell-window is active, not event counts.
    """
    from csep.core.poisson_evaluations import _poisson_likelihood_test
    from scipy.stats import poisson

    ids, where = np.unique(cell, return_inverse=True)
    rate = np.bincount(where, weights=-np.log1p(-np.clip(proba, 0, 1 - 1e-9)), minlength=len(ids)).reshape(-1, 1)
    seen = np.bincount(where, weights=y.astype(float), minlength=len(ids)).reshape(-1, 1)
    total, n = float(rate.sum()), float(seen.sum())
    delta1, delta2 = 1 - poisson.cdf(n - 1, total), poisson.cdf(n, total)
    like = _poisson_likelihood_test(rate, seen, num_simulations=sims, seed=42, use_observed_counts=False, verbose=False)
    spatial = _poisson_likelihood_test(rate, seen, num_simulations=sims, seed=42, use_observed_counts=True, normalize_likelihood=True, verbose=False)
    return {"expected": total, "observed": n, "n_test_low": float(delta2), "n_test_high": float(delta1),
            "l_test_quantile": float(like[0]), "l_test_observed": float(like[1]),
            "s_test_quantile": float(spatial[0]), "s_test_observed": float(spatial[1])}
