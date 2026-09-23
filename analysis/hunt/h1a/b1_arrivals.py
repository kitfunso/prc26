"""Session check on bet B1, label-free: do LIRF arrivals with a garbled flight number copy SCHED as BLOCK, 2025 vs 2026?
Run: PYTHONPATH=. python analysis/hunt/h1a/b1_arrivals.py"""
import sys

sys.path.insert(0, "analysis/hunt/h1a")
import numpy as np
import pandas as pd
from common import movements
from lirf_b import fmt

d = movements()
a = d[(d["apt"] == "LIRF") & (d["PHASE_mvt"] == "ARR") & d["BLOCK_TIME_UTC_mvt"].notna()].copy()
a["zs"] = a["MVT_TIME_UTC_mvt"] - a["SCHED_TIME_UTC_mvt"]
a["copy"] = (a["BLOCK_TIME_UTC_mvt"] - a["SCHED_TIME_UTC_mvt"]).abs() <= 10
a["shape"] = np.where(a["FLIGHT_mvt"].map(fmt) == "plain", "plain", "garbled")
a["per"] = np.where(a["split"] == "train", "2025", a["split"].astype(str) + "-m" + a["month"].astype(str).str.zfill(2))
a["band"] = pd.cut(a["zs"] / 3600, [-99, 0.5, 2, 4, 12, 99])
print("LIRF ARR rows with BLOCK, by split:", a["split"].value_counts().to_dict())
print(a.groupby(["band", "shape", "per"], observed=True)["copy"].agg(["mean", "count"]).round(3).to_string())
