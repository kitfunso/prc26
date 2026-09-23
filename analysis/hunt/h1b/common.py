"""Shared helpers for the H1b file-structure hunt. Dev months only (holdout = Jan & Jul 2025)."""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from data import secs

APTS10 = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]

RAW_COLS = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "ADEP_mvt", "ADES_mvt",
            "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
            "AOBT_3_flt", "EOBT_1_flt", "LOBT_flt", "IOBT_flt", "ARVT_1_flt", "ARVT_3_flt",
            "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt", "AIRCRAFT_TYPE_mvt"]

ALL_MONTH_FILES = sorted(glob.glob("data/training_2025-*.parquet"))
HOLDOUT_TOKENS = ("2025-01-01_2025-02-01", "2025-07-01_2025-08-01")
DEV_MONTH_FILES = [p for p in ALL_MONTH_FILES if not any(t in p for t in HOLDOUT_TOKENS)]
HOLDOUT_MONTH_FILES = [p for p in ALL_MONTH_FILES if any(t in p for t in HOLDOUT_TOKENS)]

assert len(DEV_MONTH_FILES) == 10, DEV_MONTH_FILES
assert len(HOLDOUT_MONTH_FILES) == 2, HOLDOUT_MONTH_FILES


def read_month(path: str, cols: list[str] | None = None, dep_only: bool = True) -> pd.DataFrame:
    """Read one raw file in its native physical row order (no sort). Adds `pos` = row index in file."""
    df = pd.read_parquet(path, columns=cols or RAW_COLS)
    df["pos"] = np.arange(len(df))
    if dep_only:
        df = df[df["PHASE_mvt"] == "DEP"].copy()
    df["day"] = df["MVT_TIME_UTC_mvt"].dt.date.astype(str)
    df["apt"] = df["ADEP_mvt"]
    return df


def classify_offblock(df: pd.DataFrame, tol: float = 60.0) -> pd.Series:
    """AOBT / EOBT / SCHED copy (within tol s), DAY_SHIFT (copy of a stamp +-24h*k, k=1,2), else OTHER.
    Only meaningful where BLOCK_TIME_UTC_mvt is known (train rows)."""
    out = pd.Series("OTHER", index=df.index, dtype=object)
    stamps = {"AOBT": "AOBT_3_flt", "EOBT": "EOBT_1_flt", "SCHED": "SCHED_TIME_UTC_mvt"}
    day_hit = pd.Series(False, index=df.index)
    for name, col in stamps.items():
        gap = secs(df["BLOCK_TIME_UTC_mvt"], df[col]).abs()
        direct = gap <= tol
        out[direct & (out == "OTHER")] = name
        for k in (1, 2):
            shifted = (~direct) & ((gap - 86400 * k).abs() <= tol)
            day_hit = day_hit | shifted
    out[(out == "OTHER") & day_hit] = "DAY_SHIFT"
    return out


def spearman(a: pd.Series, b: pd.Series) -> float:
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))
