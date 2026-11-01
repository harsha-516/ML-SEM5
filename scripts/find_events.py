"""Candidates for event pages: strongest held-out storms, recent floods in the news, large recent earthquakes."""
import re
from urllib.parse import quote_plus

import defusedxml.ElementTree as ET
import pandas as pd
import requests

from hazards.cyclone import load_tracks
from hazards.common import DATA

t = load_tracks([DATA / "ibtracs_NI.csv", DATA / "ibtracs_WP.csv"])
s = t[t.time >= "2022-01-01"].groupby("storm").agg(name=("name", "first"), start=("time", "min"), peak=("wind", "max"), ni=("is_north_indian", "max"))
print(s[s.ni == 1].sort_values("peak", ascending=False).head(8).to_string())
print(s[s.ni == 0].sort_values("peak", ascending=False).head(8).to_string())

for q in ["Nepal flash flood October 2026", "Nepal flood death toll", "flash flood Asia October 2026", "Kerala OR Assam OR Pakistan flood 2026"]:
    r = requests.get(f"https://news.google.com/rss/search?q={quote_plus(q + ' when:60d')}&hl=en-IN&gl=IN&ceid=IN:en", timeout=30)
    print("---", q)
    for it in list(ET.fromstring(r.text).iter("item"))[:6]:
        print(it.findtext("pubDate")[5:16], it.findtext("title")[:100])

q = requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query", timeout=60, params={
    "format": "geojson", "starttime": "2026-08-01", "minmagnitude": 6.3, "minlatitude": -12, "maxlatitude": 50, "minlongitude": 60,
    "maxlongitude": 150, "orderby": "magnitude"}).json()
for f in q["features"][:8]:
    p = f["properties"]
    print(p["mag"], pd.to_datetime(p["time"], unit="ms").date(), p["place"], f["geometry"]["coordinates"], p.get("alert"))
