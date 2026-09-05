"""Diagnostic, no verdict: at LIRF's no-type lane-B rows, does any ranking-visible column move the copy share (y = z_sched) inside the 1 to 4 h bands?"""
import duckdb
import numpy as np
import pandas as pd

pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 400)
con = duckdb.connect()
con.execute("SET TimeZone = 'UTC'")
Q = """
SELECT MVT_ID_mvt AS id, FLIGHT_mvt AS fl, ADES_mvt AS ades, RUNWAY_mvt::VARCHAR AS rwy, STAND_mvt::VARCHAR AS stand, FLIGHT_RULE_mvt AS rule,
  SCHED_TIME_UTC_mvt AS sched, MVT_TIME_UTC_mvt AS toff, TAXITIME_SEC_mvt::DOUBLE AS y, month(MVT_TIME_UTC_mvt) AS month,
  hour(MVT_TIME_UTC_mvt) AS hour, isodow(MVT_TIME_UTC_mvt) AS wday, epoch(MVT_TIME_UTC_mvt - SCHED_TIME_UTC_mvt) AS z_sched
FROM read_parquet('data/training_2025-*.parquet') WHERE PHASE_mvt = 'DEP' AND ADEP_mvt = 'LIRF' AND AOBT_3_flt IS NULL AND AIRCRAFT_TYPE_mvt IS NULL"""
d = con.sql(Q).df()
d["copy"] = (d.y - d.z_sched).abs() <= 600
d["fl_null"] = d.fl.isna()
d["airline"] = d.fl.str[:3]
d["ades_region"] = d.ades.str[:1]
d["stand_hundreds"] = d.stand.str[:1]
d["sched_hour"] = d.sched.dt.hour
d["band"] = pd.cut(d.z_sched, [-1e9, 0, 3600, 7200, 14400, 28800, 43200, 1e9], labels=False)
mid = d[d.band.isin([2, 3])]
print(f"LIRF no-type lane-B rows 2025: {len(d)}, copy share {d['copy'].mean():.3f}; bands 1 to 4 h: {len(mid)} rows, copy share {mid['copy'].mean():.3f}")
print("\ncopy share by month (drift check), all bands / 1 to 4 h bands:")
print(pd.DataFrame({"n": d.groupby("month").size(), "copy": d.groupby("month")["copy"].mean().round(3),
                    "n_mid": mid.groupby("month").size(), "copy_mid": mid.groupby("month")["copy"].mean().round(3)}).to_string())
for col in ("rwy", "rule", "fl_null", "airline", "ades_region", "stand_hundreds", "hour", "sched_hour", "wday"):
    g = mid.groupby(col)["copy"].agg(["size", "mean"]).round(3)
    g = g[g["size"] >= 15].sort_values("mean")
    print(f"\ncopy share in the 1 to 4 h bands by {col} (groups with 15+ rows):")
    print(g.T.to_string())
# The normal rows inside the bands: what does y look like, and is y related to z_sched or the hour?
norm = mid[~mid["copy"]]
print(f"\nnormal rows in the 1 to 4 h bands: n {len(norm)}, y mean {norm.y.mean():.0f}, p50 {norm.y.median():.0f}, p90 {norm.y.quantile(.9):.0f}, "
      f"corr(y, z_sched) {norm.y.corr(norm.z_sched):.3f}, corr(y, hour) {norm.y.corr(norm.hour):.3f}")
print("normal-row y mean by hour of takeoff:")
print(norm.groupby("hour").y.agg(["size", "mean"]).round(0).T.to_string())
