"""H1b: quick one-month structure probe before building the full checks.
Run: PYTHONPATH=. python analysis/hunt/h1b/s01_explore.py
"""
from __future__ import annotations

import pandas as pd

COLS = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "ADEP_mvt", "ADES_mvt",
        "ADEP_flt", "ADES_flt", "ADES_FILED_flt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]

df = pd.read_parquet("data/training_2025-01-01_2025-02-01.parquet", columns=COLS)
print("rows", len(df))
print("PHASE_mvt counts\n", df["PHASE_mvt"].value_counts())
print("ADEP_mvt nunique", df["ADEP_mvt"].nunique(), sorted(df["ADEP_mvt"].dropna().unique().tolist()))
print("ADES_mvt nunique", df["ADES_mvt"].nunique())
print("ADEP_mvt vs ADES_mvt overlap:", set(df["ADEP_mvt"].dropna().unique()) == set(df["ADES_mvt"].dropna().unique()))
dep = df[df["PHASE_mvt"] == "DEP"]
arr = df[df["PHASE_mvt"] == "ARR"]
print("dep rows", len(dep), "arr rows", len(arr))
print("MVT_ID_mvt dtype", df["MVT_ID_mvt"].dtype, "min/max", df["MVT_ID_mvt"].min(), df["MVT_ID_mvt"].max())
print("MVT_ID_mvt nunique vs rows", df["MVT_ID_mvt"].nunique(), len(df))
print("FLIGHT_ID_mvt nunique vs rows", df["FLIGHT_ID_mvt"].nunique(), len(df))
print("MVT_ID_mvt sample sorted head", sorted(df["MVT_ID_mvt"].head(20).tolist()))
print("MVT_ID_mvt first 20 raw order", df["MVT_ID_mvt"].head(20).tolist())
print("FLIGHT_ID_mvt first 20 raw order", df["FLIGHT_ID_mvt"].head(20).tolist())
print("is MVT_ID sorted ascending in file order?", df["MVT_ID_mvt"].is_monotonic_increasing)
print("is FLIGHT_ID sorted ascending in file order?", df["FLIGHT_ID_mvt"].is_monotonic_increasing)
print("is MVT_TIME sorted ascending in file order?", df["MVT_TIME_UTC_mvt"].is_monotonic_increasing)
print("is BLOCK_TIME sorted ascending in file order?", df["BLOCK_TIME_UTC_mvt"].is_monotonic_increasing)
print("FLIGHT_mvt sample", df["FLIGHT_mvt"].head(5).tolist())
print("CALLSIGN_flt sample", df["CALLSIGN_flt"].head(5).tolist())
print("n dup MVT_ID within file", (df["MVT_ID_mvt"].value_counts() > 1).sum())
print("n dup (FLIGHT_ID,PHASE) within file", (df.groupby(["FLIGHT_ID_mvt", "PHASE_mvt"]).size() > 1).sum())
