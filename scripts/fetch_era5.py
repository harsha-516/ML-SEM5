"""Fetch ERA5 wind and sea temperature by year, newest first.

The CDS rejects extra queued requests per dataset and asks for sequential use, so each dataset
runs one request at a time. Jobs already queued for this account are reused, and files are cached,
so a restart resumes. Usage: python scripts/fetch_era5.py [oldest_year]
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import dotenv_values
from ecmwf.datastores import Client

from hazards.common import log

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "era5"
AREA = [50, 40, -15, 180]
DATASET = {"wind": "reanalysis-era5-pressure-levels", "sst": "reanalysis-era5-single-levels"}


def months_for(year: int) -> list[str]:
    """ERA5 lags by about 6 days, so the current year stops at the last complete month."""
    edge = datetime.now(timezone.utc) - timedelta(days=7)
    last = 12 if year < edge.year else edge.month - 1
    return [f"{m:02d}" for m in range(1, last + 1)]


def request(kind: str, year: int) -> dict:
    base = {"product_type": ["reanalysis"], "year": [str(year)], "month": months_for(year), "day": [f"{d:02d}" for d in range(1, 32)],
            "time": ["00:00", "12:00"], "area": AREA, "grid": [2.0, 2.0], "data_format": "netcdf"}
    if kind == "wind":
        return {**base, "variable": ["u_component_of_wind", "v_component_of_wind"], "pressure_level": ["200", "850"]}
    return {**base, "variable": ["sea_surface_temperature"]}


def open_jobs(client: Client, kind: str) -> dict:
    """Queued or finished jobs of this account for one dataset, by year."""
    out = {}
    for i in client.get_jobs(status=["accepted", "running", "successful"], limit=50).request_ids:
        r = client.get_remote(i)
        q = r.request
        if r.collection_id == DATASET[kind] and len(q.get("year", [])) == 1 and q.get("grid") == [2.0, 2.0] and len(q.get("day", [])) == 31 \
                and q.get("month") == months_for(int(q["year"][0])) and q.get("area") == AREA:
            out.setdefault(int(q["year"][0]), r)
    return out


def fetch(client: Client, kind: str, year: int) -> str:
    path = OUT / f"{kind}_{year}.nc"
    if path.exists():
        return f"{path.name} cached"
    part = path.with_suffix(".part")
    start = time.time()
    for attempt in range(12):
        try:
            job = open_jobs(client, kind).get(year) or client.submit(DATASET[kind], request(kind, year))
            job.download(str(part))
            part.replace(path)
            return f"{path.name} done in {(time.time() - start) / 60:.0f} min, {path.stat().st_size // 1024 // 1024} MB"
        except Exception as exc:  # queue limits and transient failures: wait, then try again
            log(f"ERA5 {path.name} attempt {attempt + 1}: {type(exc).__name__}: {' '.join(str(exc).split())[-200:]}")
            time.sleep(300)
    return f"{path.name} FAILED"


def worker(client: Client, kind: str, years: list[int]) -> None:
    queued = open_jobs(client, kind)  # jobs already at the server come first, or the queue limit blocks every new request
    order = sorted(years, key=lambda y: (y not in queued, -y))
    for n, y in enumerate(order, start=1):
        log(f"ERA5 {kind} {n}/{len(order)}: {fetch(client, kind, y)}")


def main() -> None:
    oldest = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    client = Client(url="https://cds.climate.copernicus.eu/api", key=dotenv_values(ROOT / ".env")["CDSAPI_KEY"].strip())
    OUT.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc).year
    years = list(range(now, oldest - 1, -1))
    log(f"ERA5 fetch start: {len(years)} years for each of 2 datasets, {now} back to {oldest}")
    with ThreadPoolExecutor(2) as pool:
        for f in [pool.submit(worker, client, k, years) for k in DATASET]:
            f.result()
    log("ERA5 fetch finish")


if __name__ == "__main__":
    main()
