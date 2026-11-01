import numpy as np

from hazards.etas import exact_count
from hazards.recal import Blend, rolling_margin, rolling_platt


def test_rolling_platt_never_reads_the_period_it_scores():
    rng = np.random.default_rng(0)
    t = np.repeat(np.arange(8), 300)
    p = rng.uniform(0.01, 0.5, t.size)
    y = (rng.uniform(size=t.size) < p).astype(int)
    a = rolling_platt(p, y, t, span=3)
    flipped = y.copy()
    flipped[t >= 5] = 1 - flipped[t >= 5]
    b = rolling_platt(p, flipped, t, span=3)
    assert np.array_equal(a[t <= 5], b[t <= 5], equal_nan=True)
    assert not np.array_equal(a[t == 6], b[t == 6])


def test_rolling_platt_corrects_a_steady_over_forecast():
    rng = np.random.default_rng(1)
    t = np.repeat(np.arange(10), 2000)
    p = rng.uniform(0.05, 0.4, t.size)
    y = (rng.uniform(size=t.size) < p / 2).astype(int)
    out = rolling_platt(p, y, t, span=4)
    late = t >= 5
    assert abs(out[late].sum() - y[late].sum()) < 0.1 * y[late].sum()


def test_rolling_platt_is_empty_before_enough_history():
    t = np.repeat(np.arange(4), 50)
    y = np.tile([0, 1], 100)
    out = rolling_platt(np.full(200, 0.3), y, t, span=2)
    assert np.isnan(out[t == 0]).all() and np.isfinite(out[t == 1]).all()


def test_rolling_margin_uses_only_earlier_years():
    year = np.repeat([2020, 2021, 2022], 100)
    y = np.zeros(300)
    low, high = np.full(300, -1.0), np.full(300, 1.0)
    y[year == 2022] = 50.0
    m = rolling_margin(y, low, high, year, span=2)
    assert np.isnan(m[year == 2020]).all()
    assert (m[year == 2021] <= 0).all() and (m[year == 2022] <= 0).all()


def test_rolling_margin_widens_after_a_wide_year():
    year = np.repeat([2020, 2021], 100)
    y = np.where(year == 2020, 5.0, 0.0)
    m = rolling_margin(y, np.full(200, -1.0), np.full(200, 1.0), year, span=1)
    assert (m[year == 2021] >= 4).all()


def test_blend_and_final_chance_stay_probabilities():
    rng = np.random.default_rng(2)
    ml, base = rng.uniform(0.01, 0.4, 4000), rng.uniform(0.01, 0.4, 4000)
    y = (rng.uniform(size=4000) < (ml + base) / 2).astype(int)
    p = Blend().fit(ml, base, y)(ml, base)
    assert ((p > 0) & (p < 1)).all() and abs(p.sum() - y.sum()) < 0.01 * y.sum()


def test_exact_count_uses_plain_sum_of_chances():
    y = np.array([1, 0, 0, 0])
    c = exact_count(y, np.full(4, 0.25))
    assert c["expected_active"] == 1.0 and c["z"] == 0.0
