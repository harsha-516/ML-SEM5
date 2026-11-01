"""IMERG Late rainfall from NASA, for the flood locations. Late is used throughout because Final arrives months later."""
from __future__ import annotations

import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests
from dotenv import dotenv_values

from .common import DATA, ROOT, log

FIRST, LAST = "2015-04-01", str(pd.Timestamp.now(tz="UTC").date())
IMERG = "https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGDL.07"
LAT0, LAT1, LON0, LON1 = -12, 40, 60, 132  # whole degrees holding all flood locations
session = requests.Session()
session.headers.update({"Authorization": f"Bearer {dotenv_values(ROOT / '.env').get('EARTHDATA_TOKEN', '').strip()}"})


def granules(short: str, first: str = FIRST, last: str = LAST) -> list[dict]:
    out, page = [], 1
    while True:
        r = requests.get("https://cmr.earthdata.nasa.gov/search/granules.json", timeout=60, params={
            "short_name": short, "temporal": f"{first}T00:00:00Z,{last}T23:59:59Z", "page_size": 2000, "page_num": page, "sort_key": "start_date"})
        r.raise_for_status()
        e = r.json()["feed"]["entry"]
        out += e
        if len(e) < 2000:
            return out
        page += 1


def imerg_slab(title: str) -> np.ndarray:
    """Daily rain on the 0.2 degree grid of the whole box, in mm/day, shape lon by lat."""
    y, m = title.split(".")[4][:4], title.split(".")[4][4:6]
    i0, i1, j0, j1 = (LON0 + 180) * 10, (LON1 + 180) * 10 - 1, (LAT0 + 90) * 10, (LAT1 + 90) * 10 - 1
    url = f"{IMERG}/{y}/{m}/{title}.nc4.nc4?precipitation[0][{i0}:2:{i1}][{j0}:2:{j1}]"
    for attempt in range(5):
        try:
            r = session.get(url, timeout=120)
            if r.status_code == 200 and r.content[:4] == b"\x89HDF":
                break
            time.sleep(20 * (attempt + 1))
        except requests.RequestException:
            time.sleep(20 * (attempt + 1))
    else:
        raise RuntimeError(f"no data for {title}")
    with tempfile.NamedTemporaryFile(suffix=".nc4", delete=False) as t:
        t.write(r.content)
    with h5py.File(t.name) as f:
        a = np.squeeze(f["precipitation"][()]).astype(float)  # shape lon, lat
    Path(t.name).unlink()
    a[a < 0] = np.nan
    return a


def imerg_day(title: str, cells: pd.DataFrame) -> np.ndarray:
    """Mean of 5 by 5 points inside each 1 degree cell, in mm/day."""
    a = imerg_slab(title)
    out = []
    for c in cells.itertuples():
        x, y2 = (c.lo - LON0) * 5, (c.la - LAT0) * 5
        out.append(np.nanmean(a[x:x + 5, y2:y2 + 5]) if np.isfinite(a[x:x + 5, y2:y2 + 5]).any() else np.nan)
    return np.array(out)




def title_of(g: dict) -> str:
    return g.get("producer_granule_id", g["title"].split(":")[-1]).removesuffix(".nc4")


def sync(folder: str, cells: pd.DataFrame, first: str) -> pd.DataFrame:
    """Fetch any missing days into a folder and return days by locations. Column order follows cells."""
    out = DATA / folder
    out.mkdir(exist_ok=True)
    jobs = [(t.split(".")[4][:8], t) for t in map(title_of, granules("GPM_3IMERGDL", first))]
    jobs = [j for j in jobs if not (out / f"{j[0]}.npy").exists()]

    def one(job):
        try:
            np.save(out / f"{job[0]}.npy", imerg_day(job[1], cells))
        except Exception as exc:  # the rolling features tolerate a missing day
            log(f"  IMERG {job[0]} failed: {type(exc).__name__}")

    with ThreadPoolExecutor(5) as pool:
        list(pool.map(one, jobs))
    if jobs:
        log(f"IMERG {folder}: {len(jobs)} new days")
    files = sorted(out.glob("*.npy"))
    idx = pd.to_datetime([f.stem for f in files])
    return pd.DataFrame(np.array([np.load(f) for f in files]), index=idx, columns=[(c.la + 0.5, c.lo + 0.5) for c in cells.itertuples()]).sort_index()


def save_grid(day: str, a: np.ndarray) -> None:
    folder = DATA / "imerg_grid"
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder / f"{day}.npz", a=np.nan_to_num(np.round(a * 10), nan=65535).clip(0, 65535).astype(np.uint16))


def load_grid(path: Path) -> np.ndarray:
    a = np.load(path)["a"].astype(np.float32)
    a[a == 65535] = np.nan
    return a / 10


def sync_grid(first: str) -> int:
    """Fetch missing days of the rain grid. Returns how many were added."""
    folder = DATA / "imerg_grid"
    folder.mkdir(exist_ok=True)
    jobs = [(t.split(".")[4][:8], t) for t in map(title_of, granules("GPM_3IMERGDL", first))]
    jobs = [j for j in jobs if not (folder / f"{j[0]}.npz").exists()]
    log(f"IMERG grid: {len(jobs)} days to fetch")

    def one(job):
        try:
            save_grid(job[0], imerg_slab(job[1]))
        except Exception as exc:
            log(f"  IMERG grid {job[0]} failed: {type(exc).__name__}")

    with ThreadPoolExecutor(5) as pool:
        for n, _ in enumerate(pool.map(one, jobs), start=1):
            if n % 200 == 0:
                log(f"  IMERG grid {n}/{len(jobs)} days")
    return len(jobs)
