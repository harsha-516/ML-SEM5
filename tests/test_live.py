import sqlite3

import numpy as np
import pandas as pd

from hazards import live


def utc(s):
    return pd.Timestamp(s, tz="UTC")


def test_ten_minute_wind_converts_to_one_minute():
    assert abs(live.to_one_minute(88.0) - 100.0) < 1e-9


def test_merge_catalog_adds_new_events_once():
    a = pd.DataFrame({"time": [utc("2026-01-01")], "latitude": [1.0], "longitude": [2.0], "depth": [10.0], "mag": [5.0], "magType": ["mb"]})
    b = pd.DataFrame({"time": [utc("2026-01-01"), utc("2026-01-02")], "latitude": [1.0, 3.0], "longitude": [2.0, 4.0], "depth": [10.0, 5.0], "mag": [5.0, 6.0], "magType": ["mb", "mww"]})
    m = live.merge_catalog(a, b)
    assert len(m) == 2 and m.time.is_monotonic_increasing


def test_shift_puts_today_on_a_window_boundary():
    start = utc("1973-01-01")
    for day in ("2026-11-01", "2026-11-02", "2026-11-30"):
        today = utc(day)
        s = live.window_shift(today, start, 30)
        assert ((today + s - start).days % 30) == 0


def forecasts(path, rows):
    c = sqlite3.connect(path)
    c.execute("create table forecasts (issued text, hazard text, lat real, lon real, chance real, window_days int, model_version text, outcome int, primary key (issued, hazard, lat, lon))")
    c.executemany("insert into forecasts values (?,?,?,?,?,?,?,null)", rows)
    c.commit()
    c.close()


def test_earthquake_outcome_needs_closed_window_and_the_same_cell(tmp_path):
    db = tmp_path / "f.sqlite"
    forecasts(db, [("2026-09-01", "earthquake", 10.5, 80.5, 0.2, 30, "v"), ("2026-09-01", "earthquake", 20.5, 90.5, 0.1, 30, "v"), ("2026-10-25", "earthquake", 10.5, 80.5, 0.2, 30, "v")])
    cat = pd.DataFrame({"time": [utc("2026-09-10"), utc("2026-08-30"), utc("2026-10-26")], "latitude": [10.2, 20.2, 10.2], "longitude": [80.9, 90.9, 80.9], "mag": [5.2, 6.0, 7.0]})
    n = live.score_outcomes(db, cat, pd.DataFrame(columns=["lat", "lon", "start", "end"]), utc("2026-11-01"))
    assert n == 2
    c = sqlite3.connect(db)
    got = dict(((a, b), o) for a, b, o in c.execute("select issued, lat, outcome from forecasts"))
    assert got[("2026-09-01", 10.5)] == 1 and got[("2026-09-01", 20.5)] == 0 and got[("2026-10-25", 10.5)] is None


def test_flood_outcome_uses_gdacs_within_500_km_and_window(tmp_path):
    db = tmp_path / "f.sqlite"
    forecasts(db, [("2026-10-20", "flood", 25.5, 85.5, 0.3, 3, "v"), ("2026-10-20", "flood", 5.5, 100.5, 0.3, 3, "v")])
    g = pd.DataFrame({"lat": [26.0], "lon": [86.0], "start": [utc("2026-10-21")], "end": [utc("2026-10-22")]})
    live.score_outcomes(db, pd.DataFrame(columns=["time", "latitude", "longitude", "mag"]), g, utc("2026-11-01"))
    c = sqlite3.connect(db)
    assert dict(c.execute("select lat, outcome from forecasts")) == {25.5: 1, 5.5: 0}


def test_scorecard_is_empty_until_outcomes_exist_and_labels_flood_source(tmp_path):
    db = tmp_path / "f.sqlite"
    forecasts(db, [("2026-10-20", "flood", 25.5, 85.5, 0.3, 3, "v")])
    assert live.scorecard(db)["hazards"] == {}
    c = sqlite3.connect(db)
    c.execute("update forecasts set outcome = 1")
    c.commit()
    c.close()
    s = live.scorecard(db)["hazards"]["flood"]
    assert s["scored"] == 1 and abs(s["brier"] - 0.49) < 1e-9 and "GDACS" in s["outcome_source"]


def test_storm_series_joins_history_and_new_fix_on_one_scale():
    hist = pd.DataFrame({"storm": "S", "name": "IN-FA", "time": pd.date_range("2026-10-30", periods=4, freq="6h"), "grade": 5, "lat": 12.0, "lon": 160.0,
                         "pressure": 960.0, "wind": 100.0, "dist_land": 500.0, "is_north_indian": 0})
    fixes = pd.DataFrame({"time": [pd.Timestamp("2026-10-31 00:00")], "lat": [13.0], "lon": [159.0], "wind_10min": [88.0], "pressure": [950.0]})
    s = live.storm_series(hist, fixes, "In-fa", lambda la, lo: np.full(len(la), 400.0))
    assert len(s) == 5
    assert s.wind.iloc[-1] == 100.0 and s.dist_land.iloc[-1] == 400.0 and s.storm.nunique() == 1


def test_storm_series_from_fixes_only_when_no_history():
    fixes = pd.DataFrame({"time": pd.date_range("2026-10-31", periods=2, freq="6h"), "lat": [13.0, 14.0], "lon": [159.0, 158.0], "wind_10min": [44.0, 66.0], "pressure": [990.0, 980.0]})
    s = live.storm_series(pd.DataFrame(columns=["storm", "name", "time", "grade", "lat", "lon", "pressure", "wind", "dist_land", "is_north_indian"]), fixes, "New", lambda la, lo: np.full(len(la), 100.0))
    assert list(s.wind) == [50.0, 75.0] and list(s.grade) == [4, 5]


def test_north_indian_storm_needs_a_recent_fix():
    t = pd.DataFrame({"storm": ["A", "B"], "name": ["MURASU", "OLD"], "time": pd.to_datetime(["2026-11-02 00:00", "2026-10-01 00:00"]), "lat": [12.0, 10.0], "lon": [85.0, 80.0],
                      "wind": [55.0, 40.0], "pressure": [985.0, np.nan], "is_north_indian": [1, 1]})
    s = live.north_indian_storms(t, utc("2026-11-02 12:00"))
    assert [x["name"] for x in s] == ["Murasu"] and s[0]["source"] == "IBTrACS" and s[0]["wind_kt"] == 55
