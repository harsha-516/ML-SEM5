"""Upstream river basins for each flood location, from HydroBASINS level 6, and catchment rain from the IMERG grid."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import shapefile
import shapely
from shapely.geometry import box, shape

from .common import DATA, log
from .satellite import LAT0, LON0, load_grid
from .typhoon import _dist_km

FOLDER = DATA / "hydrobasins"
MAX_KM = 1000  # rain further away takes longer to arrive than the features look back
NAMES = ["basin_3", "basin_7", "basin_14", "basin_7_ratio", "basin_14_ratio"]
LAG = 2  # days until a daily IMERG Late file is published


def upstream(start: set[int], next_down: dict[int, int]) -> set[int]:
    """Basins that drain into any start basin, including the start basins."""
    feeders = defaultdict(list)
    for b, d in next_down.items():
        feeders[d].append(b)
    seen, stack = set(start), list(start)
    while stack:
        for b in feeders[stack.pop()]:
            if b not in seen:
                seen.add(b)
                stack.append(b)
    return seen


def grid_points() -> tuple[np.ndarray, np.ndarray]:
    """Centres of the 0.2 degree slab, as lon by lat axes."""
    return LON0 + 0.05 + 0.2 * np.arange(360), LAT0 + 0.05 + 0.2 * np.arange(260)


def load_basins() -> tuple[dict, dict]:
    geom, down = {}, {}
    for path in sorted(FOLDER.glob("*.shp")):
        with shapefile.Reader(str(path)) as r:
            names = [f[0] for f in r.fields[1:]]
            i, j = names.index("HYBAS_ID"), names.index("NEXT_DOWN")
            for sr in r.iterShapeRecords():
                b = int(sr.record[i])
                geom[b], down[b] = shape(sr.shape.__geo_interface__).buffer(0), int(sr.record[j])
    return geom, down


def masks(cells: pd.DataFrame) -> dict:
    """For each location, the grid pixels of its upstream area within MAX_KM."""
    geom, down = load_basins()
    log(f"HydroBASINS: {len(geom)} basins")
    ids = list(geom)
    tree = shapely.STRtree([geom[b] for b in ids])
    xs, ys = grid_points()
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    out = {}
    for c in cells.itertuples():
        here = box(c.lo, c.la, c.lo + 1, c.la + 1)
        hit = {ids[i] for i in tree.query(here) if geom[ids[i]].intersection(here).area > 0.1 * here.area * 0.2}
        lat, lon = c.la + 0.5, c.lo + 0.5
        keep = [b for b in upstream(hit, down) if b in hit or _dist_km(lat, lon, geom[b].centroid.y, geom[b].centroid.x) <= MAX_KM]
        if keep:
            area = shapely.union_all([geom[b] for b in keep])
            m = shapely.contains_xy(area, gx, gy)
        else:
            m = (gx >= c.lo) & (gx < c.lo + 1) & (gy >= c.la) & (gy < c.la + 1)
        m |= (gx >= c.lo) & (gx < c.lo + 1) & (gy >= c.la) & (gy < c.la + 1)
        out[(lat, lon)] = np.flatnonzero(m.ravel())
    return out


def basin_rain(cells: pd.DataFrame, days: list[str] | None = None) -> pd.DataFrame:
    """Mean rain over each upstream area: days by locations."""
    m = masks(cells)
    files = sorted((DATA / "imerg_grid").glob("*.npz"))
    if days is not None:
        files = [f for f in files if f.stem in days]
    rows = []
    for n, f in enumerate(files, start=1):
        a = load_grid(f).ravel()
        rows.append([np.nanmean(a[i]) if np.isfinite(a[i]).any() else np.nan for i in m.values()])
        if n % 500 == 0:
            log(f"  Basin rain {n}/{len(files)} days")
    return pd.DataFrame(rows, index=pd.to_datetime([f.stem for f in files]), columns=list(m)).sort_index()


def features(s: pd.Series, lag: int = LAG) -> pd.DataFrame:
    """Catchment rain, reading only days already published."""
    r = s.asfreq("D").shift(lag)
    f = pd.DataFrame({f"basin_{n}": r.rolling(n, min_periods=max(1, n - 2)).sum() for n in (3, 7, 14)})
    for n in (7, 14):
        f[f"basin_{n}_ratio"] = f[f"basin_{n}"] / f[f"basin_{n}"].rolling(365, min_periods=180).quantile(0.95).shift(1).clip(lower=1)
    return f
