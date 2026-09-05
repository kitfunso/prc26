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
CATEGORICAL = ("runway", "stand", "actype", "wake", "segment")
COUNTS = ("queue", "dep_before", "dep_after", "arr_anchor", "arr_off", "push_anchor")
FEATURES = ("hour", "wday", "has_fpl", "z_aobt", "z_eobt", "z_lobt", "z_sched", "z_anchor") + CATEGORICAL + COUNTS
LABELS = ("y", "c")
WINDOW = 900.0

# Ranking blanks BLOCK_TIME_UTC_mvt and TAXITIME_SEC_mvt on departures, so the view never selects them.
VIEW = """
SELECT MVT_ID_mvt AS id, ADEP_mvt AS apt, month(MVT_TIME_UTC_mvt) AS month, MVT_TIME_UTC_mvt::DATE AS day,
       hour(MVT_TIME_UTC_mvt) AS hour, isodow(MVT_TIME_UTC_mvt) AS wday, AOBT_3_flt IS NOT NULL AS has_fpl,
       epoch(MVT_TIME_UTC_mvt) AS t_off,
       epoch(MVT_TIME_UTC_mvt) - epoch(AOBT_3_flt) AS z_aobt,
       epoch(MVT_TIME_UTC_mvt) - epoch(EOBT_1_flt) AS z_eobt,
       epoch(MVT_TIME_UTC_mvt) - epoch(LOBT_flt) AS z_lobt,
       epoch(MVT_TIME_UTC_mvt) - epoch(SCHED_TIME_UTC_mvt) AS z_sched,
       RUNWAY_mvt::VARCHAR AS runway, STAND_mvt::VARCHAR AS stand, AIRCRAFT_TYPE_mvt::VARCHAR AS actype,
       WK_TBL_CAT_flt::VARCHAR AS wake, MARKET_SEGMENT_flt::VARCHAR AS segment
FROM read_parquet({files}) WHERE PHASE_mvt = 'DEP' ORDER BY apt, t_off, id"""

LANDINGS = """
SELECT ADES_mvt AS apt, epoch(MVT_TIME_UTC_mvt) AS t
FROM read_parquet({files}) WHERE PHASE_mvt = 'ARR' ORDER BY apt, t"""

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
    """Feature frame for the departures in the given files; the traffic counts need every movement of the period."""
    con = connect()
    df = con.sql(VIEW.format(files=_files(paths))).df()
    landings = con.sql(LANDINGS.format(files=_files(paths))).df()
    z_fpl = np.where(df.z_aobt >= 0, df.z_aobt, df.z_eobt)
    df["z_anchor"] = np.where(df.has_fpl, z_fpl, df.z_sched)
    for name, col in zip(COUNTS, _traffic(df, landings)):
        df[name] = col
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df[list(KEYS + FEATURES)]


def _between(s: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.searchsorted(s, hi, "right") - np.searchsorted(s, lo, "left")


def _traffic(df: pd.DataFrame, landings: pd.DataFrame) -> list[np.ndarray]:
    """Same-airport counts from sorted event times: takeoffs, landings and anchors around the anchor and the takeoff."""
    t_off = df.t_off.values
    anchor = t_off - df.z_anchor.values
    out = [np.zeros(len(df), dtype=np.int32) for _ in COUNTS]
    land = {apt: g.t.values for apt, g in landings.groupby("apt")}
    for apt, ix in df.groupby("apt").indices.items():
        s_dep, s_anc, s_arr = np.sort(t_off[ix]), np.sort(anchor[ix]), land.get(apt, np.empty(0))
        a, t = anchor[ix], t_off[ix]
        out[0][ix] = np.maximum(np.searchsorted(s_dep, t, "left") - np.searchsorted(s_dep, a, "left"), 0)
        out[1][ix] = np.searchsorted(s_dep, a, "left") - np.searchsorted(s_dep, a - WINDOW, "left")
        out[2][ix] = np.searchsorted(s_dep, t + WINDOW, "right") - np.searchsorted(s_dep, t, "right")
        out[3][ix] = _between(s_arr, a - WINDOW, a + WINDOW)
        out[4][ix] = _between(s_arr, t - WINDOW, t + WINDOW)
        out[5][ix] = _between(s_anc, a - WINDOW, a + WINDOW) - 1
    return out


def truth(paths: list[str]) -> pd.Series:
    return connect().sql(TRUTH.format(files=_files(paths))).df().set_index("id").y


def load(months: tuple[int, ...]) -> pd.DataFrame:
    """Features joined with truth for the given 2025 months; c is the lane-A label y minus z_aobt."""
    paths = [month_file(m) for m in months]
    df = features(paths).join(truth(paths), on="id", how="inner")
    df["c"] = df.y - df.z_aobt
    return df.reset_index(drop=True)
