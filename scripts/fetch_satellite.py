"""Fetch satellite rain and soil moisture for the 60 flood locations, one cached file per day.

IMERG Late is used for all years, never Final, because Final is gauge-adjusted months later and could not run live.
Usage: python scripts/fetch_satellite.py imerg|smap
"""
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests

from hazards import flood
from hazards.common import DATA, log

from hazards.satellite import FIRST, LAST, LAT0, LAT1, LON0, LON1, granules, imerg_day, session


def fetch_imerg() -> None:
    cells = flood.choose_cells(pd.read_parquet(DATA / "groundsource_region.parquet"))
    out = DATA / "imerg"
    out.mkdir(exist_ok=True)
    todo = []
    for g in granules("GPM_3IMERGDL"):
        title = g["producer_granule_id"].removesuffix(".nc4") if "producer_granule_id" in g else g["title"].split(":")[-1].removesuffix(".nc4")
        day = title.split(".")[4][:8]
        if not (out / f"{day}.npy").exists():
            todo.append((day, title))
    log(f"IMERG fetch start: {len(todo)} days to fetch")

    def one(job):
        day, title = job
        try:
            np.save(out / f"{day}.npy", imerg_day(title, cells))
        except Exception as exc:
            log(f"  IMERG {day} failed: {type(exc).__name__}")

    with ThreadPoolExecutor(5) as pool:
        for n, _ in enumerate(pool.map(one, todo), start=1):
            if n % 100 == 0:
                log(f"  IMERG {n}/{len(todo)} days")
    log("IMERG fetch finish")


def ease2(lat: float, lon: float) -> tuple[float, float]:
    """Row and column on the 9 km global EASE-2 grid used by SPL3SMP_E."""
    a, e, size = 6378137.0, 0.081819190843, 9008.055210146
    k0 = np.cos(np.radians(30)) / np.sqrt(1 - (e * np.sin(np.radians(30))) ** 2)
    sp = np.sin(np.radians(lat))
    q = (1 - e**2) * (sp / (1 - (e * sp) ** 2) - np.log((1 - e * sp) / (1 + e * sp)) / (2 * e))
    return (1624 - 1) / 2 - a * q / (2 * k0) / size, (3856 - 1) / 2 + a * k0 * np.radians(lon) / size


def smap_slab(link: str, var: str) -> np.ndarray:
    r0, c0 = ease2(LAT1, LON0)
    r1, c1 = ease2(LAT0, LON1)
    box = f"[{int(r0)}:{int(r1) + 1}][{int(c0)}:{int(c1) + 1}]"
    name = f"/Soil_Moisture_Retrieval_Data_AM/{var}"
    for attempt in range(5):
        try:
            r = session.get(link + ".dap.nc4", params={"dap4.ce": name + box}, timeout=120)
            if r.status_code == 200 and r.content[:4] == b"\x89HDF":
                break
            time.sleep(20 * (attempt + 1))
        except requests.RequestException:
            time.sleep(20 * (attempt + 1))
    else:
        raise RuntimeError("no SMAP data")
    with tempfile.NamedTemporaryFile(suffix=".nc4", delete=False) as t:
        t.write(r.content)
    with h5py.File(t.name) as f:
        found = []
        f.visititems(lambda k, v: found.append(v[()]) if isinstance(v, h5py.Dataset) and k.endswith(var) else None)
    Path(t.name).unlink()
    return found[0].astype(float)


def smap_day(link: str, cells: pd.DataFrame) -> np.ndarray:
    a = smap_slab(link, "soil_moisture")
    a[(a < 0) | (a > 1)] = np.nan
    r0, c0 = ease2(LAT1, LON0)
    out = []
    for c in cells.itertuples():
        ra, cb = ease2(c.la + 1, c.lo)
        rb, cc = ease2(c.la, c.lo + 1)
        v = a[int(round(ra)) - int(r0):int(round(rb)) - int(r0) + 1, int(round(cb)) - int(c0):int(round(cc)) - int(c0) + 1]
        out.append(np.nanmean(v) if np.isfinite(v).any() else np.nan)
    return np.array(out)


def smap_links() -> dict:
    links = {}
    for g in granules("SPL3SMP_E"):
        hrefs = [x["href"] for x in g["links"] if "opendap.earthdata" in x.get("href", "")]
        if hrefs:
            links[g["time_start"][:10].replace("-", "")] = hrefs[0]  # later versions overwrite earlier ones
    return links


def check_smap() -> None:
    """Compare the grid formula with the latitude and longitude stored in a real file."""
    link = next(iter(smap_links().values()))
    la, lo = smap_slab(link, "latitude"), smap_slab(link, "longitude")
    r0, c0 = ease2(LAT1, LON0)
    ok = la > -900
    rows, cols = np.where(ok)
    i = rows.size // 2
    calc = ease2(la[rows[i], cols[i]], lo[rows[i], cols[i]])
    print("stored row col", rows[i] + int(r0), cols[i] + int(c0), "formula", np.round(calc, 2))


def fetch_smap() -> None:
    cells = flood.choose_cells(pd.read_parquet(DATA / "groundsource_region.parquet"))
    out = DATA / "smap"
    out.mkdir(exist_ok=True)
    todo = [(d, u) for d, u in smap_links().items() if not (out / f"{d}.npy").exists()]
    log(f"SMAP fetch start: {len(todo)} days to fetch")

    def one(job):
        day, link = job
        try:
            np.save(out / f"{day}.npy", smap_day(link, cells))
        except Exception as exc:
            log(f"  SMAP {day} failed: {type(exc).__name__}")

    with ThreadPoolExecutor(5) as pool:
        for n, _ in enumerate(pool.map(one, todo), start=1):
            if n % 100 == 0:
                log(f"  SMAP {n}/{len(todo)} days")
    log("SMAP fetch finish")


if __name__ == "__main__":
    {"imerg": fetch_imerg, "smap": fetch_smap, "check": check_smap}[sys.argv[1]]()
