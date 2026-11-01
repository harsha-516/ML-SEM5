"""Stitch one fast-loading Earth picture from NASA GIBS tiles, so the globe never shows patchy tiles.

Recent VIIRS days fill each other's orbit gaps, and Blue Marble fills whatever remains.
"""

import io
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import requests
from PIL import Image

Z, TILE = 4, 256
SIZE = TILE * 2**Z
OUT = Path(__file__).resolve().parents[2] / "dashboard" / "public" / "data"
BASE = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best"


def tile(url: str) -> Image.Image | None:
    """Download one map tile, with retries."""
    for _ in range(3):
        try:
            r = requests.get(url, timeout=40)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
                return Image.open(io.BytesIO(r.content)).convert("RGB")
        except (requests.RequestException, OSError):
            pass
    return None


def layer(template: str) -> np.ndarray:
    """Join all tiles of one layer into one picture."""
    canvas = Image.new("RGB", (SIZE, SIZE))
    cells = [(x, y) for y in range(2**Z) for x in range(2**Z)]
    with ThreadPoolExecutor(12) as ex:
        for (x, y), t in zip(cells, ex.map(lambda c: tile(template.format(z=Z, x=c[0], y=c[1])), cells)):
            if t:
                canvas.paste(t, (x * TILE, y * TILE))
    return np.asarray(canvas).copy()


def main() -> None:
    """Build the globe picture."""
    today = datetime.now(UTC).date()
    days = [today - timedelta(days=n) for n in (2, 3)]
    out = layer(
        f"{BASE}/VIIRS_SNPP_CorrectedReflectance_TrueColor/default/{days[0]}/GoogleMapsCompatible_Level9/{{z}}/{{y}}/{{x}}.jpg"
    )
    older = layer(
        f"{BASE}/VIIRS_SNPP_CorrectedReflectance_TrueColor/default/{days[1]}/GoogleMapsCompatible_Level9/{{z}}/{{y}}/{{x}}.jpg"
    )
    marble = layer(
        f"{BASE}/BlueMarble_ShadedRelief_Bathymetry/default/2004-08-01/GoogleMapsCompatible_Level8/{{z}}/{{y}}/{{x}}.jpeg"
    )
    gap = out.sum(axis=2) < 24
    out[gap] = older[gap]
    gap = out.sum(axis=2) < 24
    out[gap] = marble[gap]
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(OUT / "earth.webp", quality=70, method=5)
    (OUT / "earth.json").write_text(json.dumps({"date": str(days[0]), "fill_date": str(days[1])}))
    print((OUT / "earth.webp").stat().st_size // 1024, "KB", days[0], f"gaps left to Blue Marble {gap.mean():.1%}")


if __name__ == "__main__":
    main()
