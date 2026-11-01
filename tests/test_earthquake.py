"""Tests: Earthquake table and leakage."""

import pandas as pd

from hazards.earthquake import START, build_table


def cat(rows):
    return pd.DataFrame(
        [
            {"time": START + pd.Timedelta(days=d), "latitude": la, "longitude": lo, "depth": 10.0, "mag": m}
            for d, la, lo, m in rows
        ]
    )


def test_quiet_windows_are_negative_rows():
    t = build_table(cat([(5, 20.5, 90.5, 5.5)]), n_windows=4, min_row=0)
    assert len(t) == 4 and t.y.sum() == 1


def test_features_use_only_earlier_windows():
    t = build_table(cat([(35, 20.5, 90.5, 5.5)]), n_windows=3, min_row=0).set_index("window")
    assert t.loc[1, "n_30"] == 0 and t.loc[1, "y"] == 1
    assert t.loc[2, "n_30"] == 1 and t.loc[2, "y"] == 0


def test_etas_trigger_only_follows_events_and_decays():
    t = build_table(cat([(35, 20.5, 90.5, 6.0)]), n_windows=5, min_row=0).set_index("window")
    assert t.loc[1, "etas_trigger"] == 0
    assert t.loc[2, "etas_trigger"] > t.loc[3, "etas_trigger"] > t.loc[4, "etas_trigger"] > 0


def test_m45_is_not_a_positive_label():
    assert build_table(cat([(5, 20.5, 90.5, 4.6)]), n_windows=2, min_row=0).y.sum() == 0
