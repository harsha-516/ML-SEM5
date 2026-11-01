"""Collect every experiment into results/experiments.json for the Models page. Values come only from results files."""
import json

from hazards.common import RESULTS


def read(name):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def f3(v):
    return "-" if v is None else f"{v:.3f}"


def table(title, columns, rows, note=None):
    return {"title": title, "columns": columns, "rows": rows, "note": note}


def flood():
    out = []
    e, r = read("exp_flood"), read("flood")
    if e:
        out.append(table("Feature sets, XGBoost", ["Features", "Validation", "Test, issued forecasts", "Test, actual rain"],
                         [[x["name"], f3(x["validation"]), f3(x["test_real_forecast"]), f3(x.get("test_actual_rain"))] for x in e["table"]],
                         "Validation uses reanalysis rain for the next 3 days. Actual later rain is not available live."))
        s = e["stage_2024"]
        out.append(table("Final stage on 2024 forecast rows", ["Version", "Sep to Dec 2024", "Test, 2025 onward"],
                         [["Trained on reanalysis", f3(s["check_reanalysis_trained"]), f3(s["test_2025_reanalysis_trained"])],
                          ["Final stage", f3(s["check_stage"]), f3(s["test_2025_stage"])]], "Not kept." if not s["kept"] else "Kept."))
    sat = read("exp_satellite")
    if sat:
        out.append(table("Satellite features", ["Features", "Validation", "Test", "Kept"],
                         [[x["name"], f3(x["validation"]), f3(x["test"]), x["kept"]] for x in sat["table"]], sat["note"]))
    if r and r.get("replays"):
        out.append(table("Event replays", ["Event", "Top chance, 3 days before", "Usual chance", "Warned", "Forecast source"],
                         [[x["id"], f"{x['top_3_days']:.1%}", f"{x['usual']:.1%}", "Yes" if x["warned"] else "No", x["forecast_source"]] for x in r["replays"]],
                         "Warning criterion: at least 1.5 times the usual chance."))
    b = read("exp_basins")
    if b:
        out.append(table("Catchment rain and wider label", ["Version", "Validation, news", "Validation, GDACS", "Test, news", "Test, GDACS", "Kept"],
                         [[x["name"], f3(x["val_news"]), f3(x["val_gdacs"]), f3(x["test_news"]), f3(x["test_gdacs"]), x["kept"]] for x in b["table"]], b["note"]))
    lab = (r or {}).get("label_check") or read("label_check")
    if lab:
        m, c = lab["modis_surges"], lab["modis_controls"]
        out.append(table("Label check", ["Source", "Surges", "Quiet days at same place and month"],
                         [["GDACS flood listed within 500 km", f"{lab['gdacs_agree_surges']:.1%} of {lab['surge_episodes']:,}", f"{lab['gdacs_agree_controls']:.1%}"],
                          ["MODIS flood seen, clear view", f"{m['flood_seen']:.1%} of {m['with_clear_view']}", f"{c['flood_seen']:.1%} of {c['with_clear_view']}"]],
                         f"Labels are news reports. Top 10 of {lab['place_bias']['locations']} locations hold {lab['place_bias']['top_10_share_of_surges']:.0%} of surges. "
                         "Cloud blocks most MODIS views, so that row is weak evidence."))
    return out


def cyclone():
    out = []
    r = read("cyclone")
    for name, title in (("exp_era5", "ERA5 environment features"), ("exp_himawari", "Himawari cloud features")):
        e = read(name)
        if e:
            out.append(table(title, ["Target", "Rows", "Without, validation", "With, validation", "Without, test", "With, test", "Kept"],
                             [[x["target"], f"{x['rows']:,}", x["without_val"], x["with_val"], x["without_test"], x["with_test"], x["kept"]] for x in e["table"]], e["note"]))
    c = read("exp_cnn")
    if c:
        out.append(table("Small CNN on Himawari pictures", ["Target", "Rows", "Plain, validation", "Summaries, validation", "CNN, validation", "Plain, test", "Summaries, test", "CNN, test", "Kept"],
                         [[x["target"], f"{x['rows']:,}", x["plain_val"], x["summary_val"], x["cnn_val"], x["plain_test"], x["summary_test"], x["cnn_test"], x["kept"]] for x in c["table"]], c["note"]))
    if r:
        i = r["intensity"]
        out.append(table("24 h wind range", ["Measure", "Value"],
                         [["Target coverage", "80%"], ["Coverage before calibration", f"{i['range_80_uncalibrated']:.1%}"], ["Coverage after calibration", f"{i['range_80_coverage']:.1%}"],
                          ["Margin from validation", f"{i['range_margin_kt']:.1f} kt"], ["Mean width", f"{i['range_width_kt']:.1f} kt"],
                          ["Wind MAE", f"{i['wind_mae_kt']:.1f} kt"], ["Persistence MAE", f"{i['persistence_mae_kt']:.1f} kt"],
                          ["Rapid intensification PR-AUC", f3(i["ri_pr_auc"])], ["Rapid intensification base rate", f3(i["ri_base_rate"])]]))
    return out


def earthquake():
    r = read("earthquake")
    if not r or "etas" not in r:
        return []
    e = r["etas"]
    w = r["models"][r["winner"]]
    p = e["params"]
    c1, c2 = e["csep_ml"], e["csep_etas"]

    return [table("Benchmark, PR-AUC", ["Model", "Validation", "Test"],
                  [[r["winner"], f3(w["validation"]["pr_auc"]), f3(w["test"]["pr_auc"])], ["ETAS, fitted", f3(e["validation_pr_auc"]), f3(e["test_pr_auc"])],
                   ["Older fixed rule", "-", f3(r.get("rule_test_pr_auc"))]]),
            table("ETAS parameters", ["mu per day", "K", "alpha", "c, days", "p", "D, km"],
                  [[f"{p['mu']:.3f}", f"{p['k']:.4f}", f"{p['alpha']:.2f}", f"{p['c']:.4f}", f"{p['p']:.3f}", f"{p['d']:.1f}"]],
                  f"Maximum likelihood on {e['events']:,} M5+ events, 1980 to 2015. Background share {e['background_share']:.0%}."),
            table("Count of active cell-windows, test", ["Forecast", "Expected", "Observed", "z"],
                  [[n, f"{c['expected_active']:,.0f}", f"{c['observed']:,}", f"{c['z']:.2f}"] for n, c in ((r["winner"], e["count_ml"]), ("ETAS", e["count_etas"]))],
                  "Expected is the sum of chances. Beyond 2 in size, a forecast is too high or too low."),
            table("pyCSEP tests, rates from chances", ["Forecast", "Likelihood quantile", "Spatial quantile"],
                  [[n, f"{c['l_test_quantile']:.3f}", f"{c['s_test_quantile']:.3f}"] for n, c in ((r["winner"], c1), ("ETAS", c2))]),
            *recal_earthquake()]


def recal_earthquake():
    e = read("exp_recal")
    if not e:
        return []
    q = e["earthquake"]
    return [table("Calibration and ETAS blend", ["Version", "Late validation PR-AUC", "Gain z", "Test PR-AUC", "Expected", "Count z"],
                  [[x["name"], f3(x["late_validation_pr_auc"]), f"{x['gain_z']:.2f}", f3(x["test_pr_auc"]), f"{x['expected_active']:,.0f}", f"{x['z']:.2f}"] for x in q["table"]],
                  f"Kept: {q['kept']}. Rule: lower late validation log loss by more than 2 standard errors across windows.")]


def recal_cyclone():
    e = read("exp_recal")
    if not e:
        return []
    c = e["cyclone"]
    return [table("80% range calibration", ["Version", "Late validation coverage", "Test coverage", "Test width, kt"],
                  [[x["name"], f"{x['late_validation']:.1%}", f"{x['test']:.1%}", f"{x['width']:.1f}"] for x in c["table"]], f"Kept: {c['kept']}.")]


if __name__ == "__main__":
    (RESULTS / "experiments.json").write_text(json.dumps({"cyclone": cyclone() + recal_cyclone(), "flood": flood(), "earthquake": earthquake()}, indent=1))
    print("experiments written")
