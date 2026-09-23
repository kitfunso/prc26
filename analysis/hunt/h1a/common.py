"""Shared loader for the H1a hunt: all movements with parsed times, v34 predictions joined, holdout labels masked."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
HOLDOUT = (1, 7)
COLS = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt",
        "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "TAXITIME_SEC_mvt",
        "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt", "ARVT_1_flt", "ARVT_3_flt", "CALLSIGN_flt",
        "AIRCRAFT_OPERATOR_flt", "FLIGHT_RULE_mvt", "split", "has_plan", "month"]
T = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt",
     "AOBT_3_flt", "ARVT_1_flt", "ARVT_3_flt"]


def movements(mask_holdout: bool = True) -> pd.DataFrame:
    """Every movement (both phases, both splits). Times become int64 epoch seconds (NaN when missing).
    With mask_holdout, departures in months 1 and 7 of 2025 lose BLOCK and y, so exploration never sees them."""
    df = pd.read_parquet(ROOT / "data/cache/raw.parquet", columns=COLS)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    for c in T:
        df[c] = (df[c] - epoch).dt.total_seconds()
    df["apt"] = np.where(df["PHASE_mvt"] == "DEP", df["ADEP_mvt"], df["ADES_mvt"])
    df["y"] = df["TAXITIME_SEC_mvt"].astype("float64")
    df["dev"] = (df["split"] == "train") & ~df["month"].isin(HOLDOUT)
    df["ho"] = (df["split"] == "train") & df["month"].isin(HOLDOUT)
    if mask_holdout:
        m = df["ho"] & (df["PHASE_mvt"] == "DEP")
        df.loc[m, ["BLOCK_TIME_UTC_mvt", "y", "TAXITIME_SEC_mvt"]] = np.nan
    return df


def v34_dev() -> pd.DataFrame:
    """v34 out-of-fold predictions on the ten development months, lane A and lane B stacked: id, p."""
    a = pd.read_parquet(ROOT / "data/cache/recal_oof_v34.parquet").rename(columns={"oof": "p"})
    b = pd.read_parquet(ROOT / "data/cache/laneb_oof.parquet").rename(columns={"oof": "p"})
    return pd.concat([a.assign(lane="A"), b.assign(lane="B")], ignore_index=True)


def v34_holdout() -> pd.DataFrame:
    """v34 predictions on the holdout months (no labels in these files): id, p, lane."""
    a = pd.read_parquet(ROOT / "data/cache/recal_holdout_v34.parquet")
    b = pd.read_parquet(ROOT / "data/cache/laneb_holdout.parquet")
    return pd.concat([a.assign(lane="A"), b.assign(lane="B")], ignore_index=True)


def v34_rank() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data/merry-octopus_v34.parquet").rename(
        columns={"MVT_ID_mvt": "id", "TAXITIME_SEC_mvt": "p"})
