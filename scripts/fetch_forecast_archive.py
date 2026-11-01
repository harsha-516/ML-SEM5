"""Download rain forecasts as issued 1, 2 and 3 days ahead, for an honest flood test."""
import time

import pandas as pd
import requests

from hazards import flood
from hazards.common import DATA

OUT = DATA / "flood_forecasts"
OUT.mkdir(exist_ok=True)
cells = flood.choose_cells(pd.read_parquet(DATA / "groundsource_region.parquet"))
done = 0
for c in cells.itertuples():
    path = OUT / f"{c.la}_{c.lo}.parquet"
    if path.exists():
        done += 1
        continue
    h = None
    for attempt in range(8):
        try:
            r = requests.get("https://previous-runs-api.open-meteo.com/v1/forecast", timeout=120, params={
                "latitude": c.la + 0.5, "longitude": c.lo + 0.5, "timezone": "GMT", "start_date": "2024-01-01", "end_date": "2026-01-31",
                "hourly": "precipitation_previous_day1,precipitation_previous_day2,precipitation_previous_day3"})
            if r.status_code == 200:
                h = r.json()["hourly"]
                break
            print("status", r.status_code, r.text[:100], flush=True)
        except (requests.RequestException, ValueError) as exc:
            print("retry", type(exc).__name__, flush=True)
        time.sleep(60 * (attempt + 1))
    if h is None:
        print("giving up at", c.la, c.lo, flush=True)
        break
    d = pd.DataFrame(h).assign(time=lambda x: pd.to_datetime(x.time)).set_index("time").apply(pd.to_numeric)
    daily = d.resample("1D").sum(min_count=18)
    daily.columns = ["lead1", "lead2", "lead3"]
    daily.to_parquet(path)
    done += 1
    time.sleep(1)
print("cells done", done, "of", len(cells))
