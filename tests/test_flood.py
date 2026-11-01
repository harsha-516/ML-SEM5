"""Tests: Flood label, features and leakage."""

import numpy as np
import pandas as pd

from hazards.flood import (
    choose_cells,
    forecast_features,
    live_row,
    real_forecast,
    surge_label,
    warned,
    weather_features,
)


def test_label_looks_forward_and_ignores_today():
    days = pd.date_range("2020-01-01", periods=400)
    c = pd.Series(0.0, index=days)
    c.iloc[300] = 10
    y = surge_label(c)
    assert y.iloc[297] == 1 and y.iloc[299] == 1
    assert y.iloc[300] == 0


def test_label_is_missing_at_the_end_and_before_history():
    c = pd.Series(1.0, index=pd.date_range("2020-01-01", periods=400))
    y = surge_label(c)
    assert y.iloc[:100].isna().all() and y.iloc[-3:].isna().all()


def test_features_do_not_change_when_the_future_changes():
    days = pd.date_range("2020-01-01", periods=200)
    a = pd.DataFrame({"rain": np.arange(200.0), "discharge": np.arange(200.0) + 1}, index=days)
    b = a.copy()
    b.iloc[150:] = 999.0
    pd.testing.assert_frame_equal(weather_features(a).iloc[:150], weather_features(b).iloc[:150])


def test_ratio_features_exist_and_ignore_the_future():
    days = pd.date_range("2020-01-01", periods=500)
    rng = np.random.default_rng(1)
    a = pd.DataFrame({"rain": rng.gamma(1, 5, 500), "discharge": rng.gamma(2, 50, 500)}, index=days)
    b = a.copy()
    b.iloc[450:] = 999.0
    fa, fb = weather_features(a), weather_features(b)
    for c in (
        "rain_1_ratio",
        "rain_3_ratio",
        "rain_7_ratio",
        "rain_30_ratio",
        "wet_days_7",
        "wet_days_30",
        "discharge_rank",
    ):
        assert fa[c].iloc[300:].notna().all()
        pd.testing.assert_series_equal(fa[c].iloc[:450], fb[c].iloc[:450])


def test_reanalysis_next3_sums_the_three_following_days_only():
    days = pd.date_range("2020-01-01", periods=400)
    w = pd.DataFrame({"rain": 0.0, "discharge": 1.0}, index=days)
    w.iloc[300, 0] = 30.0
    w.iloc[301, 0] = 12.0
    f = forecast_features(w)
    assert f.next3_rain.iloc[299] == 42 and f.next3_rain.iloc[297] == 30
    assert f.next3_rain.iloc[300] == 12
    assert f.next3_rain.iloc[-3:].isna().all()


def test_real_forecast_uses_lead1_tomorrow_lead2_and_lead3_after():
    days = pd.date_range("2024-01-01", periods=6)
    d = pd.DataFrame(
        {"lead1": [0, 1, 0, 0, 0, 0.0], "lead2": [0, 0, 10, 0, 0, 0.0], "lead3": [0, 0, 0, 100, 0, 0.0]}, index=days
    )
    r = real_forecast(d)
    assert r.iloc[0] == 111
    assert r.iloc[-3:].isna().all()


def test_live_row_takes_forecast_from_after_today_only():
    days = pd.date_range("2025-01-01", periods=403)
    rain = pd.Series(1.0, index=days)
    rain.iloc[-3:] = [10.0, 20.0, 30.0]
    today = days[-4]
    q = pd.Series(5.0, index=days[:-3])
    r = live_row(rain, q, today)
    assert r.next3_rain == 60
    assert r.rain_1 == 1 and r.rain_3 == 3


def test_warning_needs_one_and_a_half_times_the_usual_chance():
    days = [{"date": f"2026-01-{d:02d}", "chance": 0.1} for d in range(1, 11)]
    assert not warned(days, "2026-01-11")["warned"]
    days[-1]["chance"] = 0.16
    assert warned(days, "2026-01-11")["warned"]


def test_cells_are_spaced_and_chosen_before_2022():
    e = pd.DataFrame(
        {
            "lat": [10.5] * 300 + [10.6] * 300 + [40.5] * 300,
            "lon": [80.5] * 600 + [100.5] * 300,
            "start": pd.to_datetime(["2018-06-01"] * 600 + ["2023-06-01"] * 300),
        }
    )
    c = choose_cells(e)
    assert len(c) == 1 and c.la[0] == 10


def test_satellite_features_respect_publication_lag():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "experiments"))
    from satellite_experiments import smap_features

    from hazards.flood import imerg_features

    days = pd.date_range("2023-01-01", periods=400)
    a = pd.Series(1.0, index=days)
    b = a.copy()
    b.iloc[300:] = 500.0
    pd.testing.assert_frame_equal(imerg_features(a).iloc[:302], imerg_features(b).iloc[:302])
    pd.testing.assert_frame_equal(smap_features(a).iloc[:303], smap_features(b).iloc[:303])
    assert imerg_features(b).imerg_3.iloc[302] != imerg_features(a).imerg_3.iloc[302]
