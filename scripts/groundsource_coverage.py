import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely

t = pq.read_table("data/raw/groundsource_2026.parquet", columns=["area_km2", "geometry", "start_date", "end_date"]).to_pandas()
g = shapely.from_wkb(t.pop("geometry").to_numpy())
c = shapely.centroid(g)
t["lon"], t["lat"] = shapely.get_x(c), shapely.get_y(c)
t["start"] = pd.to_datetime(t.start_date, errors="coerce")
t["end"] = pd.to_datetime(t.end_date, errors="coerce")
print("all", len(t), "bad dates", int(t.start.isna().sum()), t.start.min(), t.start.max())
r = t[t.lat.between(-12, 50) & t.lon.between(60, 150)].copy()
print("region", len(r), f"{len(r) / len(t):.1%}")
print("area km2 quantiles", r.area_km2.quantile([.1, .5, .9, .99]).round(1).to_dict())
print("duration days median", (r.end - r.start).dt.days.median())
print(r.groupby(r.start.dt.year).size().tail(20).to_string())
r["cell"] = list(zip(np.floor(r.lat).astype(int), np.floor(r.lon).astype(int)))
n = r[r.start.dt.year >= 2015].groupby("cell").size().sort_values(ascending=False)
print("cells with events since 2015", len(n), "top 15", n.head(15).to_dict())
print("cells with 100+ events", int((n >= 100).sum()), "1000+", int((n >= 1000).sum()))
# Sanity: known event, Pakistan floods late Aug 2022, and Assam/Bihar monsoon seasonality
s = r[r.lat.between(24, 30) & r.lon.between(83, 96) & (r.start.dt.year >= 2015)]
print("monthly share, Bihar-Assam box", (s.groupby(s.start.dt.month).size() / len(s)).round(3).to_dict())
p = r[r.lat.between(24, 32) & r.lon.between(66, 72)]
print("Sindh/Balochistan by month 2022", p[p.start.dt.year == 2022].groupby(p.start.dt.month).size().to_dict())
r[["area_km2", "lat", "lon", "start", "end"]].to_parquet("data/raw/groundsource_region.parquet")
