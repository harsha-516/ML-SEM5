"""Typhoon: will a storm below 64 knots reach 64 knots within 24 hours."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import DATA, compare_models, dist_km, print_summary, write_results

TY_KT = 64
FEATURES = [
    "lat",
    "lon",
    "pressure",
    "wind",
    "month_sin",
    "month_cos",
    "dwind_6",
    "dwind_12",
    "dp_6",
    "dp_12",
    "dp_24",
    "speed_kmh",
    "heading_sin",
    "heading_cos",
    "age_h",
    "grade",
    "max_wind_so_far",
]


def _year(yy: int) -> int:
    """Expand a two-digit best-track year."""
    return 1900 + yy if yy >= 51 else 2000 + yy


def parse_bst(text: str) -> pd.DataFrame:
    """Parse JMA best track fixed-width text into one row per fix."""
    rows, storm, name = [], None, ""
    for line in text.splitlines():
        if line.startswith("66666"):
            storm, name = line[6:10], line[30:50].strip()
            continue
        if storm is None or len(line) < 24:
            continue
        t = pd.Timestamp(year=_year(int(line[:2])), month=int(line[2:4]), day=int(line[4:6]), hour=int(line[6:8]))
        wind, pres = line[33:36].strip(), line[24:28].strip()
        rows.append(
            (
                storm,
                name,
                t,
                int(line[13]),
                int(line[15:18]) / 10,
                int(line[19:23]) / 10,
                float(pres) if pres else np.nan,
                float(wind) if wind and int(wind) > 0 else np.nan,
            )
        )
    return pd.DataFrame(rows, columns=["storm", "name", "time", "grade", "lat", "lon", "pressure", "wind"])


def build_table(tracks: pd.DataFrame, first_year: int = 1977, keep_strong: bool = False) -> pd.DataFrame:
    """6-hourly rows below typhoon force, with the 24-hour label and past-only features."""
    t = tracks.copy()
    t["storm_year"] = t.groupby("storm")["time"].transform("min").dt.year
    t = t[(t.storm_year >= first_year) & (t.time.dt.hour % 6 == 0)]
    t = t.drop_duplicates(["storm", "time"]).sort_values(["storm", "time"]).reset_index(drop=True)
    out = []
    for _, g in t.groupby("storm", sort=False):
        g = g.set_index("time", drop=False)
        idx = pd.date_range(g.index.min(), g.index.max(), freq="6h")
        w, p = g["wind"].reindex(idx), g["pressure"].reindex(idx)
        la, lo = g["lat"].reindex(idx), g["lon"].reindex(idx)
        f = pd.DataFrame(index=idx)
        f["dwind_6"], f["dwind_12"] = w - w.shift(1), w - w.shift(2)
        f["dp_6"], f["dp_12"], f["dp_24"] = p - p.shift(1), p - p.shift(2), p - p.shift(4)
        f["speed_kmh"] = dist_km(la.shift(1), lo.shift(1), la, lo) / 6
        rad = np.arctan2(lo - lo.shift(1), la - la.shift(1))
        f["heading_sin"], f["heading_cos"] = np.sin(rad), np.cos(rad)
        f["age_h"] = (idx - idx[0]).total_seconds() / 3600
        f["max_wind_so_far"] = w.cummax()
        future = pd.concat([w.shift(-i) for i in range(1, 5)], axis=1).max(axis=1)
        f["y"] = (future >= TY_KT).astype(int)
        f["wind_24"] = w.shift(-4)
        out.append(g.join(f))
    d = pd.concat(out, ignore_index=True)
    d = d[d.grade != 6].copy() if keep_strong else d[(d.grade != 6) & ~(d.wind >= TY_KT)].copy()
    m = d.time.dt.month
    d["month_sin"], d["month_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    return d


def load_sets():
    """Load JMA tracks and split them by year."""
    tracks = parse_bst((DATA / "bst_all.txt").read_text())
    df = build_table(tracks)
    return tracks, df[df.storm_year <= 2014], df[df.storm_year.between(2015, 2019)], df[df.storm_year >= 2020]


def run() -> dict:
    """Train, compare and save the typhoon models."""
    tracks, tr, va, te = load_sets()

    def X(d):
        return d[FEATURES].to_numpy(np.float32)

    def base(d):
        return d["wind"].fillna(25).to_numpy(float)

    res = compare_models(
        X(tr), tr.y.to_numpy(), X(va), va.y.to_numpy(), X(te), te.y.to_numpy(), base(va), base(te), FEATURES
    )
    res.update(
        hazard="typhoon",
        unit="6-hourly storm fix below 64 knots",
        label="Reaches 64 knots within 24 hours",
        baseline_name="Current wind speed only",
        storms={
            "train": int(tr.storm.nunique()),
            "validation": int(va.storm.nunique()),
            "test": int(te.storm.nunique()),
        },
        sets={"train": len(tr), "validation": len(va), "test": len(te)},
        periods={"train": "1977 to 2014", "validation": "2015 to 2019", "test": "2020 onward"},
        last_track_time=str(tracks.time.max()),
    )
    write_results("typhoon", res)
    return res


if __name__ == "__main__":
    r = run()
    print(r["storms"])
    print_summary(r)
