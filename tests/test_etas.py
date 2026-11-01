import numpy as np
import pandas as pd

from hazards.etas import Params, binary_gain, consistency, excess, grid_forecast, log_likelihood, make_pairs, omori_integral


def test_omori_integral_matches_numeric_sum():
    t = np.linspace(0, 100, 2_000_001)
    c, p = 0.05, 1.2
    y = (t + c) ** -p
    numeric = float(((y[:-1] + y[1:]) / 2 * np.diff(t)).sum())
    assert abs(omori_integral(np.array([100.0]), c, p)[0] - numeric) < 1e-3


def test_omori_integral_handles_p_equal_one():
    a = omori_integral(np.array([10.0]), 0.1, 1.0)[0]
    assert abs(a - np.log(101)) < 1e-9


def events(n=400, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 5000, n))
    return t, rng.uniform(10, 12, n), rng.uniform(80, 82, n), 5 + rng.exponential(0.4, n)


def field():
    return np.full((62, 90), 1 / (62 * 90))


def test_forecast_ignores_events_after_window_start():
    t, la, lo, m = events()
    p = Params(mu=0.01, k=0.01, alpha=1.5, c=0.05, p=1.1, d=10.0)
    cells_lat, cells_lon = np.array([10.5, 11.5]), np.array([80.5, 81.5])
    a = grid_forecast(p, field(), t, la, lo, m, 3000.0, 30, cells_lat, cells_lon)
    t2, m2 = t.copy(), m.copy()
    m2[t2 >= 3000] = 8.0
    b = grid_forecast(p, field(), t2, la, lo, m2, 3000.0, 30, cells_lat, cells_lon)
    np.testing.assert_allclose(a, b)


def test_recent_big_event_raises_only_nearby_cells():
    t, la, lo, m = np.array([990.0]), np.array([10.5]), np.array([80.5]), np.array([7.5])
    p = Params(mu=0.01, k=0.05, alpha=1.5, c=0.05, p=1.1, d=10.0)
    r = grid_forecast(p, field(), t, la, lo, m, 1000.0, 30, np.array([10.5, 40.5]), np.array([80.5, 140.5]))
    quiet = grid_forecast(p, field(), t[:0], la[:0], lo[:0], m[:0], 1000.0, 30, np.array([10.5, 40.5]), np.array([80.5, 140.5]))
    assert r[0] > 10 * quiet[0]
    assert abs(r[1] - quiet[1]) < 1e-6


def test_true_clustering_scores_better_than_no_clustering():
    rng = np.random.default_rng(3)
    main = np.sort(rng.uniform(0, 4000, 150))
    after = np.concatenate([x + rng.exponential(3, 6) for x in main])
    t = np.sort(np.concatenate([main, after]))
    la, lo = 10.5 + rng.normal(0, 0.05, len(t)), 80.5 + rng.normal(0, 0.05, len(t))
    m = 5 + rng.exponential(0.4, len(t))
    u = field()
    pairs = make_pairs(t, la, lo)
    with_c = log_likelihood(Params(0.04, 0.08, 1.0, 0.1, 1.1, 10.0), u, t, la, lo, m, pairs, 0.0, 4100.0)
    none = log_likelihood(Params(0.26, 1e-9, 1.0, 0.1, 1.1, 10.0), u, t, la, lo, m, pairs, 0.0, 4100.0)
    assert with_c > none


def test_excess_is_zero_at_reference_magnitude():
    assert excess(np.array([5.0]))[0] == 0


def test_binary_gain_prefers_the_forecast_that_was_right():
    y = np.array([1, 0, 0, 0])
    assert binary_gain(y, np.array([0.9, 0.1, 0.1, 0.1]), np.array([0.25] * 4)) > 0


def test_consistency_reports_expected_and_observed_counts():
    rng = np.random.default_rng(0)
    p = np.tile(np.linspace(0.01, 0.3, 50), 20)
    y = (rng.uniform(size=p.size) < p).astype(int)
    cell = np.tile(np.arange(50), 20)
    r = consistency(y, p, cell, sims=200)
    assert r["observed"] == y.sum()
    assert 0.001 < r["n_test_low"] < 0.999 or 0.001 < r["n_test_high"] < 0.999
    assert 0 <= r["l_test_quantile"] <= 1
