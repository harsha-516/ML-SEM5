"""Tests: Telegram bot text, alert rules and commands."""

from datetime import UTC, datetime, timedelta

import numpy as np

from hazards import bot
from hazards.common import risk_cuts

CUTS = {"medium": 0.1, "high": 0.3}
NOW = datetime(2026, 11, 1, tzinfo=UTC)
MAP = {
    "risk_levels": CUTS,
    "scored_at": "2026-11-01T00:00:00",
    "current_risk": [
        {"lat": 25.5, "lon": 85.5, "p": 0.4},
        {"lat": 12.5, "lon": 80.5, "p": 0.05},
        {"lat": 35.5, "lon": 139.5, "p": 0.9},
    ],
}


def test_level_uses_both_cuts():
    assert [bot.level(p, CUTS) for p in (0.05, 0.1, 0.3)] == ["Low", "Medium", "High"]


def test_risk_cuts_keep_high_at_or_above_medium():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 3000)
    y = (rng.uniform(size=3000) < p**2).astype(int)
    c = risk_cuts(y, p)
    assert 0 < c["medium"] <= c["high"] < 1


def test_map_status_takes_the_highest_inside_the_region():
    s = bot.map_status("flood", "india", MAP)
    assert s["level"] == "High" and s["p"] == 0.4
    assert bot.map_status("flood", "pakistan", MAP) is None


def test_map_status_keeps_points_inside_the_country():
    r = {**MAP, "current_risk": [{"lat": 22.5, "lon": 88.5, "p": 0.9}, {"lat": 23.7, "lon": 90.4, "p": 0.2}]}
    assert bot.map_status("flood", "bangladesh", r)["p"] == 0.2
    assert bot.map_status("flood", "india", r)["p"] == 0.9
    assert bot.map_status("flood", "bayofbengal", r)["p"] == 0.9


def test_storm_regions_stay_as_boxes_so_a_sea_storm_is_not_lost():
    m = {"wind_now_1min": 40.0, "reach_64_chance": 0.05, "from": "t", "low": 30.0, "high": 50.0}
    assert bot.storm_status("india", [{"lat": 13.0, "lon": 90.0, "name": "A", "model": m}], CUTS)["name"] == "A"


def test_storm_at_typhoon_force_is_high_and_weak_storm_uses_chance():
    m = {"wind_now_1min": 40.0, "reach_64_chance": 0.05, "from": "t", "low": 30.0, "high": 50.0}
    s = {"lat": 15.0, "lon": 130.0, "name": "A", "model": m}
    assert bot.storm_status("westpacific", [s], CUTS)["level"] == "Low"
    strong = {**s, "model": {**m, "reach_64_chance": None}}
    assert bot.storm_status("westpacific", [strong], CUTS)["level"] == "High"
    assert bot.storm_status("india", [s], CUTS) is None


def test_place_names_town_state_and_country():
    assert bot.place(25.6, 85.1) == "near Patna, Bihar, India"


def test_place_far_from_town_gives_distance_and_direction():
    t = bot.place(13.0, 130.0)
    assert " km east of " in t and "Philippines" in t


def test_place_falls_back_to_coordinates_without_the_table(monkeypatch):
    monkeypatch.setattr(bot, "PLACES", bot.ROOT / "missing.csv")
    bot.load_places.cache_clear()
    assert bot.place(25.5, 85.5) == "25.5, 85.5"
    bot.load_places.cache_clear()


def test_messages_name_the_area():
    flood = bot.text_for("flood", "india", {"level": "High", "p": 0.4, "lat": 25.5, "lon": 85.5, "issued": "x"})
    assert "Bihar" in flood and "25.5, 85.5" not in flood
    m = {"wind_now_1min": 40.0, "reach_64_chance": 0.05, "from": "t", "low": 30.0, "high": 50.0}
    storm = bot.storm_status("westpacific", [{"lat": 13.0, "lon": 130.0, "name": "A", "model": m}], CUTS)
    assert "Philippines" in bot.text_for("cyclone", "westpacific", storm)


def test_every_alert_says_model_based_and_official_instructions():
    s = {"level": "High", "p": 0.4, "lat": 25.5, "lon": 85.5, "issued": "2026-11-01"}
    for hazard in ("flood", "earthquake"):
        t = bot.text_for(hazard, "india", s)
        assert "model-based" in t and "official" in t and "Prediction time" in t and "High" in t


def test_earthquake_text_never_gives_a_time():
    t = bot.text_for(
        "earthquake", "india", {"level": "High", "p": 0.4, "lat": 25.5, "lon": 85.5, "issued": "2026-11-01"}
    )
    assert "never a predicted time" in t and "will occur" not in t


def test_should_send_on_entry_new_event_and_after_a_week():
    day = (NOW - timedelta(days=2)).isoformat()
    week = (NOW - timedelta(days=8)).isoformat()
    assert bot.should_send(None, None, "High", "", NOW) and not bot.should_send(None, None, "Medium", "", NOW)
    assert not bot.should_send(("High", "A"), day, "High", "A", NOW)
    assert bot.should_send(("High", "A"), day, "High", "B", NOW)
    assert bot.should_send(("Low", ""), day, "High", "", NOW)
    assert bot.should_send(("High", "A"), week, "High", "A", NOW)


def test_new_high_after_a_quiet_spell_alerts_at_once(tmp_path):
    con = bot.connect(tmp_path / "b.sqlite")
    con.execute("insert into subscribers values (1, 'flood', 'india')")
    states = iter(["High", "Low", "High"])
    look = lambda h, r: {"level": next(states), "p": 0.4, "lat": 25.5, "lon": 85.5, "issued": "x"}
    assert [bot.queue_alerts(con, NOW, look) for _ in range(3)] == [1, 0, 1]


def test_unavailable_never_reads_as_quiet_and_keeps_state(tmp_path):
    assert "does not mean there is no risk" in bot.text_for("cyclone", "india", bot.UNAVAILABLE)
    con = bot.connect(tmp_path / "b.sqlite")
    con.execute("insert into subscribers values (1, 'flood', 'india')")
    high = lambda h, r: {"level": "High", "p": 0.4, "lat": 25.5, "lon": 85.5, "issued": "x"}
    bot.queue_alerts(con, NOW, high)
    assert bot.queue_alerts(con, NOW, lambda h, r: bot.UNAVAILABLE) == 0
    assert con.execute("select level from alert_state").fetchone() == ("High",)


def test_failed_or_old_storm_feed_is_unavailable(tmp_path, monkeypatch):
    f = tmp_path / "storms.json"
    monkeypatch.setattr(bot, "STORMS", f)
    monkeypatch.setattr(bot, "read_results", lambda h: {"risk_levels": CUTS})
    f.write_text('{"status": "failed: Timeout", "fetched_at": "2026-11-01T00:00:00+00:00", "storms": []}')
    assert bot.status("cyclone", "india", NOW)["unavailable"]
    f.write_text('{"status": "ok", "fetched_at": "2026-10-30T00:00:00+00:00", "storms": []}')
    assert bot.status("cyclone", "india", NOW)["unavailable"]
    f.write_text('{"status": "ok", "north_indian": "ok", "fetched_at": "2026-11-01T00:00:00+00:00", "storms": []}')
    assert bot.status("cyclone", "india", NOW) is None


def test_north_indian_failure_is_unknown_only_for_regions_it_covers(tmp_path, monkeypatch):
    f = tmp_path / "storms.json"
    monkeypatch.setattr(bot, "STORMS", f)
    monkeypatch.setattr(bot, "read_results", lambda h: {"risk_levels": CUTS})
    f.write_text(
        '{"status": "ok", "north_indian": "failed: Timeout", "fetched_at": "2026-11-01T00:00:00+00:00", "storms": []}'
    )
    assert bot.status("cyclone", "bayofbengal", NOW)["unavailable"]
    assert bot.status("cyclone", "japan", NOW) is None


def test_old_map_is_unavailable(monkeypatch):
    monkeypatch.setattr(bot, "read_results", lambda h: {**MAP, "scored_at": "2026-10-01T00:00:00+00:00"})
    assert bot.status("flood", "india", NOW)["unavailable"]


def test_unscored_storm_at_typhoon_force_is_high():
    s = bot.storm_status("westpacific", [{"lat": 15.0, "lon": 130.0, "name": "A", "wind_kt": 90, "issued": "t"}], CUTS)
    assert s["level"] == "High" and "could not score" in bot.text_for("cyclone", "westpacific", s)


def test_subscribe_alert_is_logged_once(tmp_path):
    con = bot.connect(tmp_path / "b.sqlite")
    look = lambda h, r: {"level": "High", "p": 0.4, "lat": 25.5, "lon": 85.5, "issued": "x"}
    assert "Subscribed" in bot.handle(con, 1, "/subscribe flood india", look)
    assert bot.queue_alerts(con, NOW, look) == 1 and bot.queue_alerts(con, NOW, look) == 0
    assert con.execute("select count(*), sum(sent) from alerts").fetchone() == (1, 0)
    assert "Unsubscribed" in bot.handle(con, 1, "/unsubscribe flood india", look)
    assert con.execute("select count(*) from subscribers").fetchone() == (0,)


def test_bad_commands_get_guidance(tmp_path):
    con = bot.connect(tmp_path / "b.sqlite")
    assert "Send /help" in bot.handle(con, 1, "/risk flood mars")
    assert bot.handle(con, 1, "hello") == bot.WELCOME
    assert "bay" in bot.handle(con, 1, "/help@tarang_ML_bot").lower()
    assert "no active storm" in bot.handle(con, 1, "/risk cyclone bay of bengal", lambda h, r: None)


def test_connect_creates_the_data_folder_on_a_new_computer(tmp_path):
    assert bot.connect(tmp_path / "new" / "b.sqlite").execute("select count(*) from subscribers").fetchone() == (0,)


def test_buttons_walk_through_subscribe_and_unsubscribe(tmp_path):
    con = bot.connect(tmp_path / "b.sqlite")
    text, markup = bot.press(con, 1, "m|sub")
    assert "hazard" in text and markup["inline_keyboard"]
    text, markup = bot.press(con, 1, "h|sub|flood")
    assert "region" in text
    assert all(len(b["callback_data"]) <= 64 for row in markup["inline_keyboard"] for b in row)
    text, _ = bot.press(con, 1, "r|sub|flood|india")
    assert "Subscribed" in text
    assert "flood" in bot.press(con, 1, "mine")[0]
    text, markup = bot.press(con, 1, "m|unsub")
    assert markup["inline_keyboard"][0][0]["callback_data"] == "r|unsub|flood|india"
    assert "Unsubscribed" in bot.press(con, 1, "r|unsub|flood|india")[0]
    assert bot.press(con, 1, "garbage")[0] == bot.WELCOME
