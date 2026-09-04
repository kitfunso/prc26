"""Masked view of the movement files and the feature frame every lane shares (protocol section 2)."""
from __future__ import annotations

import glob

import duckdb
import numpy as np
import pandas as pd

DATA = "data"
HOLDOUT_MONTHS = (1, 7)
TRAIN_MONTHS = (2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
CV_FOLDS = ((2, 3), (4, 5), (6, 8), (9, 10), (11, 12))
KEYS = ("id", "apt", "month", "day")
FEATURES = ("hour", "wday", "has_fpl", "z_aobt", "z_eobt", "z_lobt", "z_sched", "z_anchor", "queue")
LABELS = ("y", "c")

# Ranking blanks BLOCK_TIME_UTC_mvt and TAXITIME_SEC_mvt on departures, so the view never selects them.
VIEW = """
SELECT MVT_ID_mvt AS id, ADEP_mvt AS apt, month(MVT_TIME_UTC_mvt) AS month, MVT_TIME_UTC_mvt::DATE AS day,
       hour(MVT_TIME_UTC_mvt) AS hour, isodow(MVT_TIME_UTC_mvt) AS wday, AOBT_3_flt IS NOT NULL AS has_fpl,
       epoch(MVT_TIME_UTC_mvt) AS t_off,
       epoch(MVT_TIME_UTC_mvt) - epoch(AOBT_3_flt) AS z_aobt,
       epoch(MVT_TIME_UTC_mvt) - epoch(EOBT_1_flt) AS z_eobt,
       epoch(MVT_TIME_UTC_mvt) - epoch(LOBT_flt) AS z_lobt,
       epoch(MVT_TIME_UTC_mvt) - epoch(SCHED_TIME_UTC_mvt) AS z_sched
FROM read_parquet({files}) WHERE PHASE_mvt = 'DEP' ORDER BY apt, t_off, id"""

TRUTH = """
SELECT MVT_ID_mvt AS id, TAXITIME_SEC_mvt::DOUBLE AS y
FROM read_parquet({files}) WHERE PHASE_mvt = 'DEP' AND TAXITIME_SEC_mvt IS NOT NULL"""


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    return con


def month_file(month: int) -> str:
    return glob.glob(f"{DATA}/training_2025-{month:02d}-01_*.parquet")[0]


def _files(paths: list[str]) -> str:
    return "[" + ", ".join(f"'{p}'" for p in paths) + "]"


def features(paths: list[str]) -> pd.DataFrame:
    """Feature frame for the departures in the given files; the queue needs every departure of the period."""
    df = connect().sql(VIEW.format(files=_files(paths))).df()
    z_fpl = np.where(df.z_aobt >= 0, df.z_aobt, df.z_eobt)
    df["z_anchor"] = np.where(df.has_fpl, z_fpl, df.z_sched)
    df["queue"] = _queue(df)
    return df[list(KEYS + FEATURES)]


def _queue(df: pd.DataFrame) -> np.ndarray:
    """Other same-airport takeoffs between the anchor off-block time and the own takeoff."""
    t_off = df.t_off.values
    anchor = t_off - df.z_anchor.values
    q = np.zeros(len(df), dtype=np.int32)
    for ix in df.groupby("apt").indices.values():
        s = np.sort(t_off[ix])
        q[ix] = np.searchsorted(s, t_off[ix], "left") - np.searchsorted(s, anchor[ix], "left")
    return np.maximum(q, 0)


def truth(paths: list[str]) -> pd.Series:
    return connect().sql(TRUTH.format(files=_files(paths))).df().set_index("id").y


def load(months: tuple[int, ...]) -> pd.DataFrame:
    """Features joined with truth for the given 2025 months; c is the lane-A label y minus z_aobt."""
    paths = [month_file(m) for m in months]
    df = features(paths).join(truth(paths), on="id", how="inner")
    df["c"] = df.y - df.z_aobt
    return df.reset_index(drop=True)
