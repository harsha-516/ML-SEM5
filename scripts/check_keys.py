"""Test each key in .env without ever printing a key."""
import os
import re
from pathlib import Path

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
env = {k: (v or "").strip().strip("'\"") for k, v in dotenv_values(ROOT / ".env").items()}
print("keys present:", sorted(k for k, v in env.items() if v))


def clean(text: str) -> str:
    for v in env.values():
        if v:
            text = text.replace(v, "***")
    return text[:220]


if env.get("TELEGRAM_BOT_TOKEN"):
    r = requests.get(f"https://api.telegram.org/bot{env['TELEGRAM_BOT_TOKEN']}/getMe", timeout=20)
    j = r.json()
    print("telegram", r.status_code, j.get("result", {}).get("username") or clean(str(j)))

if env.get("YOUTUBE_API_KEY"):
    r = requests.get("https://www.googleapis.com/youtube/v3/search", timeout=20, params={
        "part": "snippet", "q": "cyclone arnab", "type": "video", "maxResults": 2, "key": env["YOUTUBE_API_KEY"]})
    j = r.json()
    print("youtube", r.status_code, [i["snippet"]["title"][:60] for i in j.get("items", [])] or clean(str(j.get("error", j))))

if env.get("CDSAPI_KEY"):
    os.environ["CDSAPI_URL"] = "https://cds.climate.copernicus.eu/api"
    os.environ["CDSAPI_KEY"] = env["CDSAPI_KEY"]
    import cdsapi
    out = ROOT / "data" / "raw" / "era5_test.nc"
    try:
        cdsapi.Client(quiet=True, progress=False).retrieve("reanalysis-era5-single-levels", {
            "product_type": ["reanalysis"], "variable": ["sea_surface_temperature"], "year": ["2026"], "month": ["10"], "day": ["01"],
            "time": ["00:00"], "area": [20, 80, 10, 95], "data_format": "netcdf"}, str(out))
        print("cds ok", out.stat().st_size, "bytes")
    except Exception as exc:
        print("cds failed:", re.sub(r"\s+", " ", clean(str(exc))))
