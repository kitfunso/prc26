"""Feature frame for every departure (train and rank), cached. Month is never a feature."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import CACHE, departures, load_raw, secs
from weather import WX, join

W = 900.0
CAT = ["apt", "runway", "stand", "actype", "wake", "segment", "operator", "ftype", "frule", "ades", "airline"]
NUM = ["hour", "wday", "mod", "has_plan", "notyp", "z_aobt", "z_eobt", "z_lobt", "z_iobt", "z_sched", "z_anchor",
       "d_ae", "d_al", "d_es", "d_as", "dur", "slip", "aobt_sec", "aobt_m5", "eobt_m5", "lobt_sec",
       "queue", "dep_before", "dep_after", "arr_anchor", "arr_off", "rq", "rwy_share", "push_anchor"]
DAY = ["n_day", "nB_day", "b_share", "day_dae_med", "day_zs_med", "day_za_med", "h_nB"]
OCC = ["occ", "t_occ"]


def anchor(df: pd.DataFrame) -> np.ndarray:
    """Best guess of the off-block gap: first non-negative of AOBT, EOBT, schedule gap, else 0 (takeoff itself)."""
    z = np.zeros(len(df))
    for c in ("z_sched", "z_eobt", "z_aobt"):
        v = df[c].values
        m = np.isfinite(v) & (v >= 0)
        z[m] = v[m]
    return z


def _between(s: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.searchsorted(s, hi, "right") - np.searchsorted(s, lo, "left")


def traffic(dep: pd.DataFrame, arr: pd.DataFrame) -> pd.DataFrame:
    """Counts around the anchor (estimated off-block) and the takeoff, from movements visible in the same file."""
    t, a = dep["t_off"].values, dep["t_off"].values - dep["z_anchor"].values
    out = {k: np.zeros(len(dep)) for k in ("queue", "dep_before", "dep_after", "arr_anchor", "arr_off", "rq", "rwy_share", "push_anchor")}
    land = {k: np.sort(g["t"].values) for k, g in arr.groupby("apt")}
    for apt, ix in dep.groupby("apt").indices.items():
        s = np.sort(t[ix]); sa = np.sort(a[ix]); l = land.get(apt, np.array([]))
        out["queue"][ix] = np.maximum(np.searchsorted(s, t[ix], "left") - np.searchsorted(s, a[ix], "left"), 0)
        out["dep_before"][ix] = _between(s, a[ix] - W, a[ix])
        out["dep_after"][ix] = _between(s, t[ix], t[ix] + W)
        out["arr_anchor"][ix] = _between(l, a[ix] - W, a[ix])
        out["arr_off"][ix] = _between(l, t[ix] - W, t[ix])
        out["push_anchor"][ix] = _between(sa, a[ix] - W, a[ix])
        den = _between(s, t[ix] - 3600, t[ix])
        for jx in dep.iloc[ix].groupby("runway", observed=True).indices.values():
            sr = np.sort(t[ix][jx])
            out["rq"][ix[jx]] = _between(sr, t[ix][jx] - W, t[ix][jx])
            out["rwy_share"][ix[jx]] = _between(sr, t[ix][jx] - 3600, t[ix][jx]) / np.maximum(den[jx], 1)
    return pd.DataFrame(out, index=dep.index)


def occupancy(dep: pd.DataFrame, blk: pd.DataFrame) -> pd.DataFrame:
    """First arrival that parks on the departure's stand after its anchor: did it park before takeoff, and how long before."""
    t, a = dep["t_off"].values, dep["t_off"].values - dep["z_anchor"].values
    nxt = np.full(len(dep), np.inf)
    parked = {k: np.sort(g["t"].values) for k, g in blk.groupby(["apt", "stand"])}
    for key, ix in dep.groupby(["apt", "stand"], observed=True).indices.items():
        s = parked.get(key)
        if s is not None and len(s):
            pos = np.minimum(np.searchsorted(s, a[ix], "right"), len(s) - 1)
            nxt[ix] = np.where(pos < len(s), s[pos], np.inf)
    occ = nxt < t
    return pd.DataFrame({"occ": occ.astype(int), "t_occ": np.where(occ, t - nxt, np.nan)}, index=dep.index)


def daylevel(f: pd.DataFrame) -> pd.DataFrame:
    """Airport-day aggregates from the same file: outage days (many rows without a plan) change the off-block mechanism."""
    g = f.groupby(["split", "apt", "day"])
    out = pd.DataFrame(index=f.index)
    out["n_day"] = g["id"].transform("size")
    out["nB_day"] = g["has_plan"].transform(lambda s: (s == 0).sum())
    out["b_share"] = out["nB_day"] / out["n_day"]
    out["day_dae_med"] = g["d_ae"].transform("median")
    out["day_zs_med"] = g["z_sched"].transform("median")
    out["day_za_med"] = g["z_aobt"].transform("median")
    out["h_nB"] = f.groupby(["split", "apt", "day", "hour"])["has_plan"].transform(lambda s: (s == 0).sum())
    return out


def build(refresh: bool = False) -> pd.DataFrame:
    out = CACHE / "feat.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)
    raw = load_raw()
    dep = departures(raw)
    ep = lambda c: secs(dep[c], pd.Timestamp(0, tz="UTC"))  # noqa: E731  (parquet stamps are ms, so never astype int64)
    f = pd.DataFrame(index=dep.index)
    f["id"] = dep["MVT_ID_mvt"]; f["split"] = dep["split"]; f["month"] = dep["month"]; f["day"] = dep["MVT_TIME_UTC_mvt"].dt.date.astype(str)
    f["y"] = dep["TAXITIME_SEC_mvt"].astype(float)
    f["t_off"] = ep("MVT_TIME_UTC_mvt")
    f["hour"] = dep["MVT_TIME_UTC_mvt"].dt.hour; f["wday"] = dep["MVT_TIME_UTC_mvt"].dt.dayofweek
    f["mod"] = dep["MVT_TIME_UTC_mvt"].dt.hour * 60 + dep["MVT_TIME_UTC_mvt"].dt.minute
    f["has_plan"] = dep["has_plan"].astype(int); f["notyp"] = dep["AIRCRAFT_TYPE_mvt"].isna().astype(int)
    for k, c in (("aobt", "AOBT_3_flt"), ("eobt", "EOBT_1_flt"), ("lobt", "LOBT_flt"), ("iobt", "IOBT_flt"), ("sched", "SCHED_TIME_UTC_mvt")):
        f[f"z_{k}"] = f["t_off"] - ep(c)
    f["z_anchor"] = anchor(f)
    f["d_ae"] = ep("AOBT_3_flt") - ep("EOBT_1_flt"); f["d_al"] = ep("AOBT_3_flt") - ep("LOBT_flt")
    f["d_es"] = ep("EOBT_1_flt") - ep("SCHED_TIME_UTC_mvt"); f["d_as"] = ep("AOBT_3_flt") - ep("SCHED_TIME_UTC_mvt")
    f["dur"] = ep("ARVT_1_flt") - ep("EOBT_1_flt"); f["slip"] = ep("EOBT_1_flt") - ep("IOBT_flt")
    f["aobt_sec"] = ep("AOBT_3_flt") % 60; f["aobt_m5"] = (ep("AOBT_3_flt") % 300 == 0).astype(int)
    f["eobt_m5"] = (ep("EOBT_1_flt") % 300 == 0).astype(int); f["lobt_sec"] = ep("LOBT_flt") % 60
    f["apt"] = dep["ADEP_mvt"]; f["runway"] = dep["RUNWAY_mvt"]; f["stand"] = dep["STAND_mvt"]; f["actype"] = dep["AIRCRAFT_TYPE_mvt"]
    f["wake"] = dep["WK_TBL_CAT_flt"]; f["segment"] = dep["MARKET_SEGMENT_flt"]; f["operator"] = dep["AIRCRAFT_OPERATOR_flt"]
    f["ftype"] = dep["FLIGHT_TYPE_flt"]; f["frule"] = dep["FLIGHT_RULE_mvt"]; f["ades"] = dep["ADES_flt"]
    f["airline"] = dep["FLIGHT_mvt"].astype(str).str[:3].where(dep["FLIGHT_mvt"].notna(), "NA")
    arr = raw[raw["PHASE_mvt"] == "ARR"][["ADES_mvt", "MVT_TIME_UTC_mvt", "STAND_mvt", "BLOCK_TIME_UTC_mvt"]].rename(columns={"ADES_mvt": "apt"})
    arr["t"] = secs(arr["MVT_TIME_UTC_mvt"], pd.Timestamp(0, tz="UTC"))
    blk = arr.dropna(subset=["STAND_mvt", "BLOCK_TIME_UTC_mvt"]).assign(stand=lambda d: d["STAND_mvt"].astype(str))
    blk["t"] = secs(blk["BLOCK_TIME_UTC_mvt"], pd.Timestamp(0, tz="UTC"))
    f["stand"] = f["stand"].astype(str)
    f = pd.concat([f, traffic(f, arr), daylevel(f), occupancy(f, blk), join(f)], axis=1)
    for c in CAT:
        f[c] = f[c].astype(str).fillna("NA").astype("category")
    f = f.reset_index(drop=True)
    f.to_parquet(out)
    return f


if __name__ == "__main__":
    f = build(refresh=True)
    print(f.shape); print(f.describe().T.round(1).to_string())
