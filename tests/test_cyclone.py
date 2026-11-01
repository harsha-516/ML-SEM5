"""Tests: Cyclone table."""

import pandas as pd

from hazards.typhoon import build_table


def tracks():
    t = pd.date_range("2020-09-01", periods=6, freq="6h")
    return pd.DataFrame(
        {
            "storm": "A",
            "name": "A",
            "time": t,
            "grade": [3, 3, 4, 5, 5, 5],
            "lat": 15.0,
            "lon": 85.0,
            "pressure": 1000.0,
            "wind": [40.0, 45, 55, 70, 80, 90],
        }
    )


def test_keep_strong_retains_typhoon_rows_and_future_wind():
    d = build_table(tracks(), keep_strong=True)
    assert len(d) == 6
    assert d.wind_24.iloc[0] == 80 and pd.isna(d.wind_24.iloc[-1])


def test_default_table_still_drops_typhoon_rows():
    assert len(build_table(tracks())) == 3
