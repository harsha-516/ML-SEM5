"""Tests: ERA5 features and year splits."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hazards.common import conformal_margin
from hazards.era5 import field_index, near_mean


def test_fix_uses_newest_field_at_or_before_its_time_never_later():
    fields = pd.date_range("2023-01-01", periods=4, freq="12h").to_numpy()
    fixes = pd.to_datetime(["2023-01-01 06:00", "2023-01-01 12:00", "2023-01-01 18:00", "2022-12-31 18:00"]).to_numpy()
    i = field_index(fields, fixes)
    assert list(i) == [0, 1, 1, -1]
    assert all(fields[k] <= t for k, t in zip(i[:3], fixes[:3]))


def test_near_mean_averages_only_inside_radius_and_skips_land():
    lon, lat = np.meshgrid(np.arange(80.0, 90.0, 2), np.arange(10.0, 20.0, 2))
    f = np.ones(lon.shape)
    f[0, 0] = np.nan
    f[-1, -1] = 100.0
    assert near_mean(lat, lon, f, 10.0, 80.0, 300) == 1.0


def test_conformal_margin_lifts_coverage_to_target():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 10, 4000)
    low, high = np.full(4000, -5.0), np.full(4000, 5.0)
    m = conformal_margin(y[:2000], low[:2000], high[:2000], 0.8)
    assert m > 0
    inside = (y[2000:] >= low[2000:] - m) & (y[2000:] <= high[2000:] + m)
    assert abs(inside.mean() - 0.8) < 0.03


def test_split_years_is_time_ordered_and_keeps_test_from_2022():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "experiments"))
    from cyclone_experiments import split_years

    a, b, c = split_years(list(range(2012, 2027)))
    assert max(a) < min(b) and max(b) < min(c) and min(c) == 2022
    a, b, c = split_years([2023, 2024, 2025, 2026])
    assert a and b and c and max(a) < min(b) and max(b) < min(c)


def test_split_years_never_leaves_a_set_empty():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "experiments"))
    from cyclone_experiments import split_years

    assert split_years([2024, 2025, 2026]) == ([2024], [2025], [2026])
    assert split_years([2023, 2024, 2025, 2026]) == ([2023, 2024], [2025], [2026])
