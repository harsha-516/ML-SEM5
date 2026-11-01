"""ERA5 environment around each storm fix. A fix only sees the newest field at or before its time."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from .common import DATA, dist_km, log

FOLDER = DATA / "era5"
NAMES = ["sst_c", "shear_kt", "pi_proxy"]
RADIUS_KM = 500
MS_TO_KT = 1.94384


def field_index(field_times: np.ndarray, fix_times: np.ndarray) -> np.ndarray:
    """Position of the newest field at or before each fix, or -1 when there is none."""
    return np.searchsorted(field_times, fix_times, side="right") - 1


def near_mean(lat2d: np.ndarray, lon2d: np.ndarray, field: np.ndarray, lat: float, lon: float, km: float) -> float:
    """Mean of a field within a radius of the point."""
    d = dist_km(lat, lon, lat2d, lon2d)
    v = field[d <= km]
    return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan


def _open(path) -> xr.Dataset:
    """Open an ERA5 file with standard coordinate names."""
    d = xr.open_dataset(path)
    return d.rename(
        {
            k: v
            for k, v in {"valid_time": "time", "pressure_level": "level", "latitude": "lat", "longitude": "lon"}.items()
            if k in d
        }
    )


def years_ready() -> list[int]:
    """Years with both wind and sea temperature on disk."""
    return sorted(
        y for y in range(1980, 2030) if (FOLDER / f"wind_{y}.nc").exists() and (FOLDER / f"sst_{y}.nc").exists()
    )


def year_features(fixes: pd.DataFrame, year: int) -> pd.DataFrame:
    """ERA5 features for the storm fixes of one year."""
    w, s = _open(FOLDER / f"wind_{year}.nc"), _open(FOLDER / f"sst_{year}.nc")
    t_w, t_s = w.time.to_numpy(), s.time.to_numpy()
    shear = (
        np.hypot(w.u.sel(level=200) - w.u.sel(level=850), w.v.sel(level=200) - w.v.sel(level=850)).to_numpy() * MS_TO_KT
    )
    sst = s.sst.to_numpy()
    if sst.ndim == 4:  # recent ERA5 months can carry an extra experiment dimension
        sst = np.nanmean(sst, axis=1)
    lon2d, lat2d = np.meshgrid(w.lon.to_numpy(), w.lat.to_numpy())
    times = fixes.time.to_numpy()
    iw, js = field_index(t_w, times), field_index(t_s, times)
    rows = []
    for i, f in enumerate(fixes.itertuples()):
        a, b = iw[i], js[i]
        temp = near_mean(lat2d, lon2d, sst[b], f.lat, f.lon, 250) if b >= 0 else np.nan
        cut = near_mean(lat2d, lon2d, shear[a], f.lat, f.lon, RADIUS_KM) if a >= 0 else np.nan
        rows.append((temp - 273.15, cut, temp - 299.15))
    return pd.DataFrame(rows, columns=NAMES, index=fixes.index)


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    """Add ERA5 columns; fixes outside downloaded years stay empty."""
    out = d.copy()
    for c in NAMES:
        out[c] = np.nan
    ready = years_ready()
    for y in ready:
        part = d[d.time.dt.year == y]
        if len(part):
            log(f"  ERA5 features {y}: {len(part)} fixes")
            out.loc[part.index, NAMES] = year_features(part, y).to_numpy()
    return out
