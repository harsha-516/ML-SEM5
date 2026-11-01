"""Himawari band 13 cloud-top features around a storm, read from JMA HSD segments on AWS."""

from __future__ import annotations

import bz2
import struct
import time

import numpy as np
import pandas as pd
import requests

NAMES = ["cold_frac", "tb_mean", "tb_min", "tb_asym"]
COLD_K = 223.15
RADIUS_KM = 200
LINES, SEGMENT = 5500, 550
MAX_DLON = 65
CROP = 64  # pixels a side, about 0.094 degrees each, so 6 degrees across
CROP_KM = 340


def parse_hsd(raw: bytes) -> dict:
    """Header values and the count image of one uncompressed HSD segment."""
    pos, blocks = 0, {}
    total = struct.unpack_from("<H", raw, 3)[0]
    for _ in range(total):
        number, length = struct.unpack_from("<BH", raw, pos)
        blocks[number] = pos
        pos += length
    header = struct.unpack_from("<I", raw, blocks[1] + 70)[0]
    _, _, bits, cols, lines = struct.unpack_from("<BHHHH", raw, blocks[2])
    sub_lon, cfac, lfac, coff, loff = struct.unpack_from("<dIIff", raw, blocks[3] + 3)
    band, wave = struct.unpack_from("<Hd", raw, blocks[5] + 3)
    err, out = struct.unpack_from("<HH", raw, blocks[5] + 15)
    gain, const = struct.unpack_from("<dd", raw, blocks[5] + 19)
    c0, c1, c2, cc0, cc1, cc2, c, h, k = struct.unpack_from("<9d", raw, blocks[5] + 35)
    first = struct.unpack_from("<H", raw, blocks[7] + 5)[0]
    img = np.frombuffer(raw, "<u2", cols * lines, header).reshape(lines, cols)
    return {
        "img": img,
        "cols": cols,
        "lines": lines,
        "first_line": first,
        "sub_lon": sub_lon,
        "cfac": cfac,
        "lfac": lfac,
        "coff": coff,
        "loff": loff,
        "wave": wave,
        "gain": gain,
        "const": const,
        "err": err,
        "out": out,
        "coef": (c0, c1, c2),
        "c": c,
        "h": h,
        "k": k,
    }


def brightness(counts: np.ndarray, s: dict) -> np.ndarray:
    """Counts to brightness temperature in kelvin."""
    bad = (counts == s["err"]) | (counts == s["out"])
    rad = np.maximum(s["gain"] * counts.astype(float) + s["const"], 1e-6)
    lam = s["wave"] * 1e-6
    i = rad * 1e6
    te = (s["h"] * s["c"] / (s["k"] * lam)) / np.log(2 * s["h"] * s["c"] ** 2 / (i * lam**5) + 1)
    tb = s["coef"][0] + s["coef"][1] * te + s["coef"][2] * te**2
    tb[bad] = np.nan
    return tb


def to_pixel(lat: np.ndarray, lon: np.ndarray, s: dict) -> tuple[np.ndarray, np.ndarray]:
    """Latitude and longitude to column and line (1-based), CGMS normalised geostationary projection."""
    phi, dl = np.radians(lat), np.radians(lon - s["sub_lon"])
    cl = np.arctan(0.993305616 * np.tan(phi))
    rl = 6356.7523 / np.sqrt(1 - 0.00669438444 * np.cos(cl) ** 2)
    r1 = 42164 - rl * np.cos(cl) * np.cos(dl)
    r2 = -rl * np.cos(cl) * np.sin(dl)
    r3 = rl * np.sin(cl)
    rn = np.sqrt(r1**2 + r2**2 + r3**2)
    x, y = np.degrees(np.arctan(-r2 / r1)), np.degrees(np.arcsin(-r3 / rn))
    return s["coff"] + x * s["cfac"] / 2**16, s["loff"] + y * s["lfac"] / 2**16


def disc(
    lat: float, lon: float, km: float = RADIUS_KM, step: float = 0.04
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample points within km of the centre, with their distance and bearing."""
    span = km / 111.0
    a = np.arange(-span, span + step, step)
    dy, dx = np.meshgrid(a, a)
    east = dx * np.cos(np.radians(lat))
    r = np.hypot(dy, east) * 111.0
    keep = r <= km
    return lat + dy[keep], lon + dx[keep], r[keep], np.arctan2(east[keep], dy[keep])


def summarise(tb: np.ndarray, bearing: np.ndarray) -> dict:
    """Cold cloud share, mean and minimum, and spread between 8 sectors. A symmetric storm has spread near zero."""
    ok = np.isfinite(tb)
    if ok.mean() < 0.8:
        return dict.fromkeys(NAMES, np.nan)
    tb, bearing = tb[ok], bearing[ok]
    sector = ((bearing + np.pi) / (2 * np.pi) * 8).astype(int) % 8
    means = np.array([tb[sector == k].mean() for k in range(8) if (sector == k).any()])
    return {
        "cold_frac": float((tb < COLD_K).mean()),
        "tb_mean": float(tb.mean()),
        "tb_min": float(tb.min()),
        "tb_asym": float(means.std()),
    }


def segments_for(lat: float, lon: float, sub_lon: float = 140.7) -> list[int]:
    """Segment numbers (1 to 10) holding the disc around a position."""
    s = {"sub_lon": sub_lon, "cfac": 20466275, "lfac": 20466275, "coff": 2750.5, "loff": 2750.5}
    la, lo, _, _ = disc(lat, lon, km=CROP_KM, step=0.2)
    _, line = to_pixel(la, lo, s)
    return sorted({int((v - 1) // SEGMENT) + 1 for v in (line.min(), line.max()) if 0 < v <= LINES})


def url(t, seg: int, of: int = 10) -> str:
    """Address of one Himawari file on NOAA open data."""
    bucket, sat = ("noaa-himawari9", "H09") if t >= pd.Timestamp("2022-12-13 05:00") else ("noaa-himawari8", "H08")
    return (
        f"https://{bucket}.s3.amazonaws.com/AHI-L1b-FLDK/{t:%Y/%m/%d/%H%M}/"
        f"HS_{sat}_{t:%Y%m%d_%H%M}_B13_FLDK_R20_S{seg:02d}{of:02d}.DAT.bz2"
    )


def load_segment(session: requests.Session, t, seg: int, of: int = 10) -> dict | None:
    """Download one image segment, with retries."""
    for attempt in range(4):
        try:
            r = session.get(url(t, seg, of), timeout=180)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return parse_hsd(bz2.decompress(r.content))
        except (requests.RequestException, OSError):
            time.sleep(5 * (attempt + 1))
    return None


def load_image(session: requests.Session, t, segs: list[int]) -> list[dict]:
    """Decoded parts of one image. Older AWS Himawari-8 times hold one whole-disc file instead of 10 segments."""
    parts = [load_segment(session, t, k) for k in segs]
    parts = [x for x in parts if x]
    if parts:
        return parts
    whole = load_segment(session, t, 1, 1)
    return [whole] if whole else []


def crop_name(storm: str, t) -> str:
    """File name of one storm picture."""
    return f"{storm}_{pd.Timestamp(t):%Y%m%d%H}"


def sample(la: np.ndarray, lo: np.ndarray, parts: list[dict]) -> np.ndarray:
    """Brightness temperature at each position, NaN where no segment covers it."""
    col, line = to_pixel(la, lo, parts[0])
    tb = np.full(col.shape, np.nan)
    for s in parts:
        rows = np.round(line).astype(int) - s["first_line"]
        cols = np.round(col).astype(int) - 1
        hit = (rows >= 0) & (rows < s["lines"]) & (cols >= 0) & (cols < s["cols"])
        tb[hit] = brightness(s["img"][rows[hit], cols[hit]], s)
    return tb


def features(lat: float, lon: float, parts: list[dict]) -> dict:
    """Features at a position from the decoded segments of one image."""
    la, lo, _, bearing = disc(lat, lon)
    return summarise(sample(la, lo, parts), bearing)


def crop(lat: float, lon: float, parts: list[dict]) -> np.ndarray:
    """Square picture centred on the storm, north up, 180 to 320 K stored as 0 to 255. Missing pixels are 0."""
    a = (np.arange(CROP) - (CROP - 1) / 2) * 6.0 / CROP
    dy, dx = np.meshgrid(a[::-1], a / max(np.cos(np.radians(lat)), 0.3), indexing="ij")
    tb = sample(lat + dy, lon + dx, parts)
    return np.nan_to_num((tb - 180) / 140 * 255, nan=0).clip(0, 255).astype(np.uint8)
