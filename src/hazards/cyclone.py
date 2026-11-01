"""Cyclone: JTWC-scale IBTrACS model across the North Indian and West Pacific basins.

Uses USA_WIND (1-minute winds) so both basins share one scale. All 2022+ storms, including Arnab, stay out of training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

from .common import DATA, Platt, compare_models, conformal_margin, print_summary, tuned, write_results
from .events import cyclone_events
from .live import save_model
from .recal import rolling_margin
from .typhoon import build_table as storm_table

RI_KT = 30
MARGIN_YEARS = 2  # chosen on late validation in scripts/recal_experiments.py

FEATURES = ["lat", "lon", "pressure", "wind", "month_sin", "month_cos", "dwind_6", "dwind_12", "dp_6", "dp_12", "dp_24",
            "speed_kmh", "heading_sin", "heading_cos", "age_h", "grade", "max_wind_so_far", "dist_land", "is_north_indian"]
COLUMNS = ["SID", "SEASON", "BASIN", "NAME", "ISO_TIME", "NATURE", "LAT", "LON", "USA_WIND", "USA_PRES", "DIST2LAND"]


def load_tracks(paths: list) -> pd.DataFrame:
    frames = [pd.read_csv(p, skiprows=[1], usecols=COLUMNS, keep_default_na=False, low_memory=False) for p in paths]
    t = pd.concat(frames, ignore_index=True).drop_duplicates(["SID", "ISO_TIME"])
    for c in ("LAT", "LON", "USA_WIND", "USA_PRES", "DIST2LAND"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    wind = t["USA_WIND"]
    grade = np.select([t["NATURE"] == "ET", wind >= 64, wind >= 48, wind >= 34], [6, 5, 4, 3], default=2)
    return pd.DataFrame({"storm": t["SID"], "name": t["NAME"], "time": pd.to_datetime(t["ISO_TIME"]), "grade": grade, "lat": t["LAT"],
                         "lon": t["LON"], "pressure": t["USA_PRES"], "wind": wind, "dist_land": t["DIST2LAND"],
                         "is_north_indian": (t["BASIN"] == "NI").astype(int)})


def build_table(tracks: pd.DataFrame) -> pd.DataFrame:
    d = storm_table(tracks, first_year=1980)
    return d[d.wind.notna() | d.pressure.notna()]


def load_sets():
    tracks = load_tracks([DATA / "ibtracs_NI.csv", DATA / "ibtracs_WP.csv"])
    df = build_table(tracks)
    return tracks, df, df[df.storm_year <= 2016], df[df.storm_year.between(2017, 2021)], df[df.storm_year >= 2022]


def intensity_models(tracks: pd.DataFrame) -> tuple[dict, dict]:
    """24-hour wind forecast with an 80% range, plus rapid intensification (+30 knots in 24 hours), for every fix."""
    d = storm_table(tracks, first_year=1980, keep_strong=True)
    d = d[d.wind.notna()]
    known = d[d.wind_24.notna()].assign(ri=lambda x: (x.wind_24 - x.wind >= RI_KT).astype(int))
    tr, va, te = known[known.storm_year <= 2016], known[known.storm_year.between(2017, 2021)], known[known.storm_year >= 2022]

    def X(x):
        return x[FEATURES].to_numpy(np.float32)

    common = dict(n_estimators=1500, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=50,
                  tree_method="hist", n_jobs=-1, random_state=42)
    q = {}
    for key, alpha in (("low", 0.1), ("mid", 0.5), ("high", 0.9)):
        q[key] = XGBRegressor(objective="reg:quantileerror", quantile_alpha=alpha, **common).fit(
            X(tr), tr.wind_24, eval_set=[(X(va), va.wind_24)], verbose=False)
    spw = float((tr.ri == 0).sum() / max(tr.ri.sum(), 1))
    ri = XGBClassifier(eval_metric="aucpr", scale_pos_weight=spw, **common).fit(X(tr), tr.ri, eval_set=[(X(va), va.ri)], verbose=False)
    cal = Platt().fit(ri.predict_proba(X(va))[:, 1], va.ri)
    pred = q["mid"].predict(X(te))
    margin = conformal_margin(va.wind_24, q["low"].predict(X(va)), q["high"].predict(X(va)))  # chosen on validation
    raw = (te.wind_24 >= q["low"].predict(X(te))) & (te.wind_24 <= q["high"].predict(X(te)))
    recent = pd.concat([va, te])
    lo_r, hi_r = q["low"].predict(X(recent)), q["high"].predict(X(recent))
    roll = rolling_margin(recent.wind_24.to_numpy(), lo_r, hi_r, recent.storm_year.to_numpy(), MARGIN_YEARS)[len(va):]
    inside = (te.wind_24 >= q["low"].predict(X(te)) - roll) & (te.wind_24 <= q["high"].predict(X(te)) + roll)
    latest = recent.storm_year.to_numpy() > recent.storm_year.max() - MARGIN_YEARS
    live_margin = conformal_margin(recent.wind_24.to_numpy()[latest], lo_r[latest], hi_r[latest])
    ri_te = cal(ri.predict_proba(X(te))[:, 1])
    metrics = {"test_fixes": int(len(te)), "wind_mae_kt": float(np.abs(pred - te.wind_24).mean()),
               "persistence_mae_kt": float(np.abs(te.wind - te.wind_24).mean()), "range_80_coverage": float(inside.mean()), "range_80_uncalibrated": float(raw.mean()), "range_margin_kt": live_margin, "range_validation_margin_kt": margin,
               "range_width_kt": float((q["high"].predict(X(te)) - q["low"].predict(X(te)) + 2 * roll).mean()),
               "ri_base_rate": float(te.ri.mean()), "ri_pr_auc": float(average_precision_score(te.ri, ri_te)),
               "ri_roc_auc": float(roc_auc_score(te.ri, ri_te))}
    return {"table": d, "quantiles": q, "ri": ri, "ri_cal": cal, "margin": margin, "live_margin": live_margin}, metrics


def replay(tracks: pd.DataFrame, m: dict, name: str, year: int) -> dict:
    """Forecast made at each 6-hourly fix of one held-out storm, next to what happened 24 hours later."""
    ids = tracks[(tracks.name == name) & (tracks.time.dt.year == year)].storm.unique()
    rows = m["table"][m["table"].storm.isin(ids)].sort_values("time")
    X = rows[FEATURES].to_numpy(np.float32)
    out = rows.assign(low=m["quantiles"]["low"].predict(X) - m["margin"], mid=m["quantiles"]["mid"].predict(X), high=m["quantiles"]["high"].predict(X) + m["margin"],
                      ri_p=m["ri_cal"](m["ri"].predict_proba(X)[:, 1]))
    num = lambda v: None if pd.isna(v) else round(float(v), 3)  # noqa: E731
    return {"name": name, "year": year, "in_training": False, "track": [
        {"time": str(r.time), "lat": r.lat, "lon": r.lon, "wind_kt": num(r.wind), "pressure": num(r.pressure), "dist_land_km": num(r.dist_land),
         "forecast_wind_24": num(r.mid), "forecast_low": num(r.low), "forecast_high": num(r.high), "actual_wind_24": num(r.wind_24),
         "rapid_intensification_chance": num(r.ri_p)} for r in out.itertuples()]}


def run() -> dict:
    tracks, df, tr, va, te = load_sets()

    def X(d):
        return d[FEATURES].to_numpy(np.float32)

    def base(d):
        return d["wind"].fillna(25).to_numpy(float)

    res = compare_models(X(tr), tr.y.to_numpy(), X(va), va.y.to_numpy(), X(te), te.y.to_numpy(), base(va), base(te), FEATURES, xgb_params=tuned("cyclone"))
    ni = te[te.is_north_indian == 1]
    p_ni = res["_test_proba"][res["winner"]][(te.is_north_indian == 1).to_numpy()]
    m, res["intensity"] = intensity_models(tracks)
    res.update(hazard="cyclone", unit="6-hourly storm fix below 64 knots", label="Reaches 64 knots within 24 hours",
               baseline_name="Current wind speed only",
               storms={"train": int(tr.storm.nunique()), "validation": int(va.storm.nunique()), "test": int(te.storm.nunique())},
               north_indian_test={"storms": int(ni.storm.nunique()), "rows": int(len(ni)), "base_rate": float(ni.y.mean()),
                                  "pr_auc": float(average_precision_score(ni.y, p_ni))},
               sets={"train": int(len(tr)), "validation": int(len(va)), "test": int(len(te))},
               periods={"train": "1980 to 2016", "validation": "2017 to 2021", "test": "2022 onward"},
               note="Recent IBTrACS records are provisional operational estimates, so 2026 storms are rougher than older best tracks.")
    write_results("cyclone", res)
    reach = res["_fitted"]["XGBoost"]
    save_model("cyclone", {"features": FEATURES, "quantiles": m["quantiles"], "margin": m["live_margin"], "ri": m["ri"], "ri_cal": m["ri_cal"], "reach": reach,
                           "reach_cal": Platt().fit(reach.predict_proba(X(va))[:, 1], va.y.to_numpy())})
    cyclone_events(tracks, replay, m)
    return res


if __name__ == "__main__":
    r = run()
    print(r["storms"], r["north_indian_test"])
    print_summary(r)
