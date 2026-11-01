"""Himawari features for storm fixes, newest first, cached so a restart resumes.

Usage: python scripts/fetch_himawari.py [hours] [first_date]. Defaults: 3 hours, back to 2016.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from hazards import cyclone, himawari as hw
from hazards.common import DATA, log
from hazards.typhoon import build_table

OUT = DATA / "himawari_features.parquet"
CROPS = DATA / "himawari_crops"
HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
FIRST = sys.argv[2] if len(sys.argv) > 2 else "2016-01-01"


def main() -> None:
    tracks = cyclone.load_tracks([DATA / "ibtracs_NI.csv", DATA / "ibtracs_WP.csv"])
    d = build_table(tracks, first_year=1980, keep_strong=True)
    d = d[d.wind.notna() & (d.time >= FIRST) & ((d.lon - 140.7).abs() <= hw.MAX_DLON)][["storm", "time", "lat", "lon"]]
    done = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=["storm", "time", *hw.NAMES])
    CROPS.mkdir(exist_ok=True)
    have = {p.stem for p in CROPS.glob("*.npy")}
    todo = d[[hw.crop_name(s, t) not in have for s, t in zip(d.storm, d.time)]]
    again = set(zip(todo.storm, todo.time))
    done = done[[k not in again for k in zip(done.storm, done.time)]]
    times = sorted(todo.time.unique(), reverse=True)
    log(f"Himawari fetch start: {len(todo)} fixes at {len(times)} times")
    start, rows = time.time(), []
    session = requests.Session()

    def one(t):
        group = todo[todo.time == t]
        segs = sorted({s for r in group.itertuples() for s in hw.segments_for(r.lat, r.lon)})
        parts = hw.load_image(session, pd.Timestamp(t), segs)
        out = []
        for r in group.itertuples():
            f = hw.features(r.lat, r.lon, parts) if parts else dict.fromkeys(hw.NAMES)
            if parts:
                np.save(CROPS / f"{hw.crop_name(r.storm, r.time)}.npy", hw.crop(r.lat, r.lon, parts))
            out.append({"storm": r.storm, "time": r.time, **f})
        return out

    def save():
        pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_parquet(OUT)

    with ThreadPoolExecutor(6) as pool:
        for b in range(0, len(times), 60):
            for out in pool.map(one, times[b:b + 60]):
                rows.extend(out)
            save()
            n = min(b + 60, len(times))
            log(f"  Himawari {n}/{len(times)} times, down to {pd.Timestamp(times[n - 1])}, {(time.time() - start) / 60:.0f} min")
            if time.time() - start > HOURS * 3600:
                log(f"Himawari fetch stopped at the {HOURS:g} hour limit")
                break
    log("Himawari fetch finish")


if __name__ == "__main__":
    main()
