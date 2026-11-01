import json

from hazards import refresh

RSS = """<rss><channel>
<item><title>Old story - The Hindu</title><link>http://a</link><pubDate>Mon, 26 Oct 2026 10:00:00 GMT</pubDate><source url="x">The Hindu</source></item>
<item><title>New story - BBC</title><link>http://b</link><pubDate>Wed, 28 Oct 2026 10:00:00 GMT</pubDate><source url="x">BBC</source></item>
<item><title>New story - BBC</title><link>http://c</link><pubDate>Wed, 28 Oct 2026 09:00:00 GMT</pubDate><source url="x">BBC</source></item>
</channel></rss>"""


def test_news_is_newest_first_without_duplicates_or_source_suffix():
    items = refresh.parse_news(RSS)
    assert [i["title"] for i in items] == ["New story", "Old story"]
    assert items[0]["source"] == "BBC"


def test_empty_feed_gives_no_items():
    assert refresh.parse_news("<rss><channel></channel></rss>") == []


def test_forecast_log_is_not_duplicated_on_a_second_run(tmp_path, monkeypatch):
    (tmp_path / "flood.json").write_text(json.dumps({"generated_at": "v1", "current_risk": [{"lat": 1.5, "lon": 2.5, "p": 0.2}]}))
    monkeypatch.setattr(refresh, "RESULTS", tmp_path)
    db = tmp_path / "f.sqlite"
    assert refresh.log_forecasts(db) == 1
    assert refresh.log_forecasts(db) == 1
