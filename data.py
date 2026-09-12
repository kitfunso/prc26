"""Load the challenge parquet files into one frame with parsed times and the lane split."""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
CACHE = DATA / "cache"
TIME_COLS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt",
             "EOBT_1_flt", "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt"]
PLAN_STAMPS = ["AOBT_3_flt", "EOBT_1_flt", "LOBT_flt", "IOBT_flt"]


def _read(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for c in TIME_COLS:
        df[c] = pd.to_datetime(df[c], utc=True)
    return df


def load_raw(refresh: bool = False) -> pd.DataFrame:
    """All 2025 movements plus the 2026 ranking movements, both phases, with a `split` column."""
    CACHE.mkdir(exist_ok=True)
    out = CACHE / "raw.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)
    frames = [_read(p) for p in sorted(glob.glob(str(DATA / "training_*.parquet")))]
    train = pd.concat(frames, ignore_index=True)
    train["split"] = "train"
    rank = _read(str(DATA / "ranking.parquet"))
    rank["split"] = "rank"
    df = pd.concat([train, rank], ignore_index=True)
    df["has_plan"] = df[PLAN_STAMPS].notna().any(axis=1)
    df["month"] = df["MVT_TIME_UTC_mvt"].dt.month
    df["year"] = df["MVT_TIME_UTC_mvt"].dt.year
    df.to_parquet(out)
    return df


def secs(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a - b).dt.total_seconds()


def departures(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["PHASE_mvt"] == "DEP"].copy()


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


if __name__ == "__main__":
    d = load_raw(refresh=True)
    dep = departures(d)
    print(d.shape, d["split"].value_counts().to_dict())
    print("departures:", dep.groupby("split").size().to_dict())
    print("lane B (no plan) departures:", dep[~dep["has_plan"]].groupby("split").size().to_dict())
