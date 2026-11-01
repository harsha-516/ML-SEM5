"""Time one year of ERA5 pressure-level and sea-temperature data at 2 degrees, twice daily."""
import os
import time
from pathlib import Path

import cdsapi
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
os.environ["CDSAPI_URL"] = "https://cds.climate.copernicus.eu/api"
os.environ["CDSAPI_KEY"] = dotenv_values(ROOT / ".env")["CDSAPI_KEY"].strip()
out = ROOT / "data" / "raw" / "era5"
out.mkdir(exist_ok=True)
base = {"product_type": ["reanalysis"], "year": ["2023"], "month": [f"{m:02d}" for m in range(1, 13)], "day": [f"{d:02d}" for d in range(1, 32)],
        "time": ["00:00", "12:00"], "area": [50, 40, -15, 180], "grid": [2.0, 2.0], "data_format": "netcdf"}
c = cdsapi.Client(quiet=True, progress=False)
start = time.time()
c.retrieve("reanalysis-era5-pressure-levels", {**base, "variable": ["u_component_of_wind", "v_component_of_wind"], "pressure_level": ["200", "850"]}, str(out / "wind_2023.nc"))
print("wind", round(time.time() - start), "s", (out / "wind_2023.nc").stat().st_size // 1024, "KB", flush=True)
start = time.time()
c.retrieve("reanalysis-era5-single-levels", {**base, "variable": ["sea_surface_temperature"]}, str(out / "sst_2023.nc"))
print("sst", round(time.time() - start), "s", (out / "sst_2023.nc").stat().st_size // 1024, "KB", flush=True)
