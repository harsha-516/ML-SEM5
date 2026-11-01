"""Build src/hazards/places.csv from GeoNames (CC BY 4.0): towns over 15,000 people in the bot's Asia box."""
import csv
import io
import zipfile

import requests

BASE = "https://download.geonames.org/export/dump"
OUT = "src/hazards/places.csv"


def get(name: str) -> bytes:
    r = requests.get(f"{BASE}/{name}", timeout=120)
    r.raise_for_status()
    return r.content


def main() -> None:
    admin = dict(line.split("\t")[:2] for line in get("admin1CodesASCII.txt").decode().splitlines())
    countries = {p[0]: p[4] for p in (line.split("\t") for line in get("countryInfo.txt").decode().splitlines() if not line.startswith("#"))}
    rows = []
    with zipfile.ZipFile(io.BytesIO(get("cities15000.zip"))) as z:
        for line in z.read("cities15000.txt").decode().splitlines():
            p = line.split("\t")
            lat, lon = float(p[4]), float(p[5])
            if -16 <= lat <= 61 and 55 <= lon <= 180:
                rows.append((p[2], admin.get(f"{p[8]}.{p[10]}", ""), countries.get(p[8], p[8]), lat, lon))
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "state", "country", "lat", "lon"])
        w.writerows(sorted(rows))
    print(len(rows), "places")


if __name__ == "__main__":
    main()
