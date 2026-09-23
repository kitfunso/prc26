"""H1b: confirm the DEP/ARR airport-role hypothesis and eyeball id-vs-time on one month.
Run: PYTHONPATH=. python analysis/hunt/h1b/s02_explore2.py
"""
from __future__ import annotations

import pandas as pd

APTS10 = {"EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"}

COLS = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt",
        "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]
df = pd.read_parquet("data/training_2025-01-01_2025-02-01.parquet", columns=COLS)

dep = df[df["PHASE_mvt"] == "DEP"]
arr = df[df["PHASE_mvt"] == "ARR"]
print("DEP: ADEP in 10-set share", dep["ADEP_mvt"].isin(APTS10).mean(), "ADES in 10-set share", dep["ADES_mvt"].isin(APTS10).mean())
print("ARR: ADES in 10-set share", arr["ADES_mvt"].isin(APTS10).mean(), "ADEP in 10-set share", arr["ADEP_mvt"].isin(APTS10).mean())
print("DEP ADEP_mvt value counts (top 12)\n", dep["ADEP_mvt"].value_counts().head(12))
print("ARR ADES_mvt value counts (top 12)\n", arr["ADES_mvt"].value_counts().head(12))

# id vs time correlation, DEP rows only, in file (row) order
d = dep.reset_index(drop=True).copy()
d["pos"] = range(len(d))
d["mvt_epoch"] = d["MVT_TIME_UTC_mvt"].astype("int64")
print("\nDEP rows: corr(MVT_ID, MVT_TIME) spearman:", d["MVT_ID_mvt"].corr(d["mvt_epoch"], method="spearman"))
print("DEP rows: corr(FLIGHT_ID, MVT_TIME) spearman:", d["FLIGHT_ID_mvt"].corr(d["mvt_epoch"], method="spearman"))
print("DEP rows: corr(row pos, MVT_TIME) spearman:", d["pos"].corr(d["mvt_epoch"], method="spearman"))
blk = d["BLOCK_TIME_UTC_mvt"].astype("int64")
print("DEP rows: corr(MVT_ID, BLOCK_TIME) spearman:", d["MVT_ID_mvt"].corr(blk, method="spearman"))
print("DEP rows: corr(row pos, BLOCK_TIME) spearman:", d["pos"].corr(blk, method="spearman"))
