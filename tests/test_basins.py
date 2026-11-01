import numpy as np
import pandas as pd

from hazards.basins import features, grid_points, upstream
from hazards.satellite import LAT1, LON1


def test_upstream_follows_rivers_and_ignores_side_basins():
    down = {1: 0, 2: 1, 3: 2, 4: 1, 5: 0, 6: 5}
    assert upstream({1}, down) == {1, 2, 3, 4}
    assert upstream({2}, down) == {2, 3}


def test_grid_points_stay_inside_the_box():
    x, y = grid_points()
    assert x.max() < LON1 and y.max() < LAT1


def test_basin_features_never_read_unpublished_rain():
    days = pd.date_range("2024-01-01", periods=30)
    s = pd.Series(1.0, index=days)
    a = features(s)
    s2 = s.copy()
    s2.iloc[-2:] = 500.0
    b = features(s2)
    pd.testing.assert_frame_equal(a, b)


def test_gdacs_label_looks_only_at_the_next_days_near_the_place():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    from label_experiments import gdacs_label
    days = pd.date_range("2024-03-01", periods=10)
    rows = pd.DataFrame({"lat": 10.5, "lon": 80.5, "date": days})
    g = pd.DataFrame({"lat": [10.6, 40.0], "lon": [80.6, 120.0], "start": pd.to_datetime(["2024-03-05", "2024-03-02"], utc=True)})
    y = gdacs_label(rows, g, horizon=3, km=300)
    assert list(y) == [0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
