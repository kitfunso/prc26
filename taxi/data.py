"""Masked view of the movement files and the feature frame every lane shares (protocol section 2)."""
from __future__ import annotations

import glob

import duckdb
import holidays
import numpy as np
import pandas as pd

DATA = "data"
HOLDOUT_MONTHS = (1, 7)
TRAIN_MONTHS = (2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
CV_FOLDS = ((2, 3), (4, 5), (6, 8), (9, 10), (11, 12))
KEYS = ("id", "apt", "month", "day", "t_off")
CATEGORICAL = ("runway", "stand", "actype", "wake", "segment")
COUNTS = ("queue", "dep_before", "dep_after", "arr_anchor", "arr_off", "push_anchor")
FEATURES = ("hour", "wday", "has_fpl", "z_aobt", "z_eobt", "z_lobt", "z_sched", "z_anchor") + CATEGORICAL + COUNTS
# Plan-provenance columns (D1): outside FEATURES so the champions' column lists stay as declared.
PLAN_CAT = ("operator", "ftype", "frule", "ades")
PLAN = PLAN_CAT + ("dur", "slip", "d_ae", "d_al", "d_es", "aobt_sec", "aobt_m5", "eobt_m5", "lobt_sec")
# Stand, type and operator columns (L3, D2, L5): built from the same file's rows, so the ranking file carries them too.
STAND = ("notyp", "d_as", "occ", "t_occ", "airline")
# R1 runway share and H1 holiday flags: outside FEATURES for the same reason as PLAN.
EXTRA = ("rwy_share", "hol", "hol_prev", "hol_next", "rq", "rec_taxi", "d_rec")
COUNTRY = dict(ED="DE", EG="GB", EH="NL", LE="ES", LF="FR", LI="IT", LS="CH", LT="TR")
LABELS = ("y", "c")
WINDOW = 900.0
HOUR = 3600.0
RECENT = 10  # Q1: takeoffs averaged for the recent taxi level, not swept

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
       WK_TBL_CAT_flt::VARCHAR AS wake, MARKET_SEGMENT_flt::VARCHAR AS segment,
       AIRCRAFT_OPERATOR_flt::VARCHAR AS operator, FLIGHT_TYPE_flt::VARCHAR AS ftype,
       FLIGHT_RULE_mvt::VARCHAR AS frule, ADES_flt::VARCHAR AS ades,
       epoch(ARVT_1_flt) - epoch(EOBT_1_flt) AS dur, epoch(EOBT_1_flt) - epoch(IOBT_flt) AS slip,
       epoch(AOBT_3_flt) - epoch(EOBT_1_flt) AS d_ae, epoch(AOBT_3_flt) - epoch(LOBT_flt) AS d_al,
       epoch(EOBT_1_flt) - epoch(SCHED_TIME_UTC_mvt) AS d_es,
       epoch(AOBT_3_flt) % 60 AS aobt_sec, (epoch(AOBT_3_flt) % 300 = 0)::INT AS aobt_m5,
       (epoch(EOBT_1_flt) % 300 = 0)::INT AS eobt_m5, epoch(LOBT_flt) % 60 AS lobt_sec,
       AIRCRAFT_TYPE_mvt IS NULL AS notyp, epoch(AOBT_3_flt) - epoch(SCHED_TIME_UTC_mvt) AS d_as,
       coalesce(substr(FLIGHT_mvt, 1, 3), 'NA') AS airline
FROM read_parquet({files}) WHERE PHASE_mvt = 'DEP' ORDER BY apt, t_off, id"""

LANDINGS = """
SELECT ADES_mvt AS apt, epoch(MVT_TIME_UTC_mvt) AS t
FROM read_parquet({files}) WHERE PHASE_mvt = 'ARR' ORDER BY apt, t"""

ONBLOCK = """
SELECT ADES_mvt AS apt, STAND_mvt::VARCHAR AS stand, epoch(BLOCK_TIME_UTC_mvt) AS t
FROM read_parquet({files}) WHERE PHASE_mvt = 'ARR' AND STAND_mvt IS NOT NULL AND BLOCK_TIME_UTC_mvt IS NOT NULL ORDER BY apt, stand, t"""

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


def anchor(df: pd.DataFrame) -> np.ndarray:
    """A1 (R6 amended): the first non-negative of the AOBT, EOBT and schedule gaps; a departure with none anchors at its own takeoff (F11)."""
    has, z = df.has_fpl.values, np.zeros(len(df))
    for v, ok in ((df.z_sched.values, np.ones(len(df), bool)), (df.z_eobt.values, has), (df.z_aobt.values, has)):
        m = ok & (v >= 0)  # a missing gap is NaN and never passes; the most preferred gap writes last
        z[m] = v[m]
    return z


def features(paths: list[str], window: float = WINDOW) -> pd.DataFrame:
    """Feature frame for the departures in the given files; the traffic counts need every movement of the period."""
    con = connect()
    df = con.sql(VIEW.format(files=_files(paths))).df()
    landings = con.sql(LANDINGS.format(files=_files(paths))).df()
    df["z_anchor"] = anchor(df)
    for name, col in zip(COUNTS, _traffic(df, landings, window)):
        df[name] = col
    df["occ"], df["t_occ"] = _occupancy(df, con.sql(ONBLOCK.format(files=_files(paths))).df())
    df["rwy_share"] = _rwy_share(df)
    df["rq"], df["rec_taxi"] = _local(df)
    df["d_rec"] = df.z_aobt.values - df.rec_taxi.values
    df["hol"], df["hol_prev"], df["hol_next"] = _holidays(df)
    df["notyp"] = df.notyp.astype(np.int8)
    for c in CATEGORICAL + PLAN_CAT:
        df[c] = df[c].astype("category")
    return df[list(KEYS + FEATURES + PLAN + STAND + EXTRA)]


def _rwy_share(df: pd.DataFrame) -> np.ndarray:
    """R1: share of the airport's takeoffs on this runway in the hour before the anchor; NaN when that hour is empty or the runway is missing."""
    t_off, anc, out = df.t_off.values, df.t_off.values - df.z_anchor.values, np.full(len(df), np.nan)
    for apt, ix in df.groupby("apt").indices.items():
        s_all, a = np.sort(t_off[ix]), anc[ix]
        den = np.searchsorted(s_all, a, "left") - np.searchsorted(s_all, a - HOUR, "left")
        for jx in df.iloc[ix].groupby("runway").indices.values():
            s = np.sort(t_off[ix[jx]])
            num = np.searchsorted(s, a[jx], "left") - np.searchsorted(s, a[jx] - HOUR, "left")
            out[ix[jx]] = np.where(den[jx] > 0, num / np.maximum(den[jx], 1), np.nan)
    return out


def _local(df: pd.DataFrame, k: int = RECENT) -> tuple[np.ndarray, np.ndarray]:
    """Q1: rq = same-runway takeoffs in the window before the anchor; rec_taxi = mean plan taxi (z_aobt) of the last k airport takeoffs that left
    the ground before the anchor. Both are rates read at the anchor, so neither re-encodes the anchor-to-takeoff window (LK1)."""
    t_off, anc = df.t_off.values, df.t_off.values - df.z_anchor.values
    rq, rec = np.full(len(df), np.nan), np.full(len(df), np.nan)
    for apt, ix in df.groupby("apt").indices.items():
        for jx in df.iloc[ix].groupby("runway").indices.values():
            s = np.sort(t_off[ix[jx]])
            a = anc[ix[jx]]
            rq[ix[jx]] = np.searchsorted(s, a, "left") - np.searchsorted(s, a - WINDOW, "left")
        f = ix[df.has_fpl.values[ix]]
        o = np.argsort(t_off[f])
        s, cs = t_off[f][o], np.concatenate([[0.0], np.cumsum(df.z_aobt.values[f][o])])
        pos = np.searchsorted(s, anc[ix], "left")
        lo = np.maximum(pos - k, 0)
        rec[ix] = np.where(pos - lo >= 3, (cs[pos] - cs[lo]) / np.maximum(pos - lo, 1), np.nan)
    return rq, rec


def _holidays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """H1: public holiday in the airport's country on the takeoff day; hol_prev = yesterday was one, hol_next = tomorrow is one."""
    day = pd.to_datetime(df.day).dt.normalize()
    years, out = sorted(day.dt.year.unique()), np.zeros((3, len(df)), dtype=np.int8)
    for apt, ix in df.groupby("apt").indices.items():
        h = pd.to_datetime(sorted(holidays.country_holidays(COUNTRY[str(apt)[:2]], years=years)))
        for k, off in enumerate((0, 1, -1)):
            out[k, ix] = day.iloc[ix].isin(h + pd.Timedelta(days=off)).values
    return out[0], out[1], out[2]


def _occupancy(df: pd.DataFrame, onblock: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """First arrival parked on the departure's stand after its anchor: occ = it parked before takeoff, t_occ = takeoff minus that on-block."""
    t_off, anc = df.t_off.values, df.t_off.values - df.z_anchor.values
    nxt = np.full(len(df), np.inf)
    parked = {k: g.t.values for k, g in onblock.groupby(["apt", "stand"])}
    for key, ix in df.groupby(["apt", "stand"]).indices.items():
        s = parked.get(key)
        if s is None:
            continue
        pos = np.searchsorted(s, anc[ix], "right")
        nxt[ix] = np.where(pos < len(s), s[np.minimum(pos, len(s) - 1)], np.inf)
    occ = nxt < t_off
    return occ.astype(np.int8), np.where(occ, t_off - nxt, np.nan)


def _between(s: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.searchsorted(s, hi, "right") - np.searchsorted(s, lo, "left")


def _traffic(df: pd.DataFrame, landings: pd.DataFrame, window: float = WINDOW) -> list[np.ndarray]:
    """Same-airport counts from sorted event times: takeoffs, landings and anchors within window s of the anchor and the takeoff."""
    t_off = df.t_off.values
    anchor = t_off - df.z_anchor.values
    out = [np.zeros(len(df), dtype=np.int32) for _ in COUNTS]
    land = {apt: g.t.values for apt, g in landings.groupby("apt")}
    for apt, ix in df.groupby("apt").indices.items():
        s_dep, s_anc, s_arr = np.sort(t_off[ix]), np.sort(anchor[ix]), land.get(apt, np.empty(0))
        a, t = anchor[ix], t_off[ix]
        out[0][ix] = np.maximum(np.searchsorted(s_dep, t, "left") - np.searchsorted(s_dep, a, "left"), 0)
        out[1][ix] = np.searchsorted(s_dep, a, "left") - np.searchsorted(s_dep, a - window, "left")
        out[2][ix] = np.searchsorted(s_dep, t + window, "right") - np.searchsorted(s_dep, t, "right")
        out[3][ix] = _between(s_arr, a - window, a + window)
        out[4][ix] = _between(s_arr, t - window, t + window)
        out[5][ix] = _between(s_anc, a - window, a + window) - 1
    return out


def truth(paths: list[str]) -> pd.Series:
    return connect().sql(TRUTH.format(files=_files(paths))).df().set_index("id").y


def load(months: tuple[int, ...], window: float = WINDOW) -> pd.DataFrame:
    """Features joined with truth for the given 2025 months; c is the lane-A label y minus z_aobt."""
    paths = [month_file(m) for m in months]
    df = features(paths, window).join(truth(paths), on="id", how="inner")
    df["c"] = df.y - df.z_aobt
    return df.reset_index(drop=True)
