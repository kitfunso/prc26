"""Feature frame for every departure (train and rank), cached. Month is never a feature."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import CACHE, departures, load_raw, secs
from weather import WX, join

W = 900.0
TOL = 60.0                # arrival copy tolerance, same as mix.label
TAIL = 1800.0             # arrival taxi-in above this = tail-of-queue arrival
WIN = 7200.0              # +-2h window around the departure's takeoff, for arrival shares
STAND_LOOKBACK = 86400.0  # previous-arrival lookback cap on the same stand
CAT = ["apt", "runway", "stand", "actype", "wake", "segment", "operator", "ftype", "frule", "ades", "airline"]
NUM = ["hour", "wday", "mod", "has_plan", "notyp", "z_aobt", "z_eobt", "z_lobt", "z_iobt", "z_sched", "z_anchor",
       "d_ae", "d_al", "d_es", "d_as", "dur", "slip", "aobt_sec", "aobt_m5", "eobt_m5", "lobt_sec",
       "queue", "dep_before", "dep_after", "arr_anchor", "arr_off", "rq", "rwy_share", "push_anchor"]
DAY = ["n_day", "nB_day", "b_share", "day_dae_med", "day_zs_med", "day_za_med", "h_nB"]
OCC = ["occ", "t_occ"]
ARR = ["ad_arr_a1", "ad_arr_sc", "ad_arr_tail", "ad_n_arr", "w_arr_a1", "w_arr_sc", "w_arr_tail", "w_n_arr",
       "pa_a1", "pa_sc", "pa_taxi_in", "pa_gap", "pa_missing", "al_arr_a1", "al_arr_sc", "al_n_arr"]
WS_CLIP, WS_TAIL = 1800.0, 600.0  # gap clip; lanea's |c| tail threshold, reused for the gap shares
WS_SCALES = {"wz": (("split", "apt", "day"), 3600.0), "wa": (("split", "apt", "day", "airline"), 3600.0),
             "wz30": (("split", "apt", "day"), 1800.0), "wz3h": (("split", "apt", "day"), 10800.0)}
WZ = ["wz_n", "wz_dz", "wz_dz_sd", "wz_hi", "wz_lo", "wz_dae", "wz_dae_sd", "wz_dae0",
      "wa_n", "wa_dz", "wa_dae", "wa_dae_sd", "wz30_dz", "wz30_dae", "wz30_dae_sd",
      "wz3h_dz", "wz3h_dae", "wz3h_dae_sd", "wz_dal", "wz_dal_sd"]
PW = ["pw_n", "pw_m", "pw_hi", "pr_m", "pw30_m"]
# duplicated from recal.W: importing recal here would cycle (recal -> train -> features)
PW_BLEND = {"ra_cb_wz": 0.5223, "ra_lgb_wz": 0.1564, "nn3_wz": 0.3213}


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
            pos = np.searchsorted(s, a[ix], "right")
            nxt[ix] = np.where(pos < len(s), s[np.minimum(pos, len(s) - 1)], np.inf)
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


def arr_recs(raw: pd.DataFrame) -> pd.DataFrame:
    """Arrival movements with block time present, plus copy/tail flags (arr_a3 is diagnostic-only, kept for analysis/arrmode.py)."""
    a = raw[raw["PHASE_mvt"] == "ARR"].dropna(subset=["BLOCK_TIME_UTC_mvt"]).rename(columns={"ADES_mvt": "apt"}).copy()
    a["day"] = a["MVT_TIME_UTC_mvt"].dt.date.astype(str)
    a["t_block"] = secs(a["BLOCK_TIME_UTC_mvt"], pd.Timestamp(0, tz="UTC"))
    a["airline"] = a["FLIGHT_mvt"].astype(str).str[:3].where(a["FLIGHT_mvt"].notna(), "NA")
    a["a1"] = (secs(a["BLOCK_TIME_UTC_mvt"], a["ARVT_1_flt"]).abs() < TOL).astype(int)
    a["sc"] = (secs(a["BLOCK_TIME_UTC_mvt"], a["SCHED_TIME_UTC_mvt"]).abs() < TOL).astype(int)
    a["a3"] = (secs(a["BLOCK_TIME_UTC_mvt"], a["ARVT_3_flt"]).abs() < TOL).astype(int)
    a["tail"] = (a["TAXITIME_SEC_mvt"].astype(float) > TAIL).astype(int)
    return a


def _arr_day(a: pd.DataFrame) -> pd.DataFrame:
    """Airport-day arrival shares, one row per (split, apt, day)."""
    g = a.groupby(["split", "apt", "day"], observed=True)
    return g.agg(ad_arr_a1=("a1", "mean"), ad_arr_sc=("sc", "mean"), ad_arr_tail=("tail", "mean"),
                 ad_n_arr=("a1", "size")).reset_index()


def _arr_airline_day(a: pd.DataFrame) -> pd.DataFrame:
    """Airport-day-airline arrival shares, one row per (split, apt, day, airline)."""
    g = a.groupby(["split", "apt", "day", "airline"], observed=True)
    return g.agg(al_arr_a1=("a1", "mean"), al_arr_sc=("sc", "mean"), al_n_arr=("a1", "size")).reset_index()


def _arr_window(f: pd.DataFrame, a: pd.DataFrame) -> pd.DataFrame:
    """+-2h arrival shares around each departure's takeoff, same-apt arrivals only (traffic()'s searchsorted pattern)."""
    n = len(f)
    out = {k: np.full(n, np.nan) for k in ("w_arr_a1", "w_arr_sc", "w_arr_tail")}
    cnt = np.zeros(n)
    t = f["t_off"].values
    for apt, ix in f.groupby("apt", observed=True).indices.items():
        m = a["apt"].values == apt
        if not m.any():
            continue
        order = np.argsort(a["t_block"].values[m])
        tb = a["t_block"].values[m][order]
        ca1 = np.concatenate([[0], np.cumsum(a["a1"].values[m][order])])
        csc = np.concatenate([[0], np.cumsum(a["sc"].values[m][order])])
        ctail = np.concatenate([[0], np.cumsum(a["tail"].values[m][order])])
        lo = np.searchsorted(tb, t[ix] - WIN, "left")
        hi = np.searchsorted(tb, t[ix] + WIN, "right")
        c = hi - lo
        cnt[ix] = c
        with np.errstate(invalid="ignore", divide="ignore"):
            out["w_arr_a1"][ix] = (ca1[hi] - ca1[lo]) / c
            out["w_arr_sc"][ix] = (csc[hi] - csc[lo]) / c
            out["w_arr_tail"][ix] = (ctail[hi] - ctail[lo]) / c
    res = pd.DataFrame(out, index=f.index)
    res["w_n_arr"] = cnt
    return res


def _arr_prev_stand(f: pd.DataFrame, a: pd.DataFrame) -> pd.DataFrame:
    """Last arrival that blocked on the same stand before the departure's anchor (occupancy()'s dict-lookup pattern)."""
    st = a.dropna(subset=["STAND_mvt"]).assign(stand=lambda d: d["STAND_mvt"].astype(str))
    groups = {}
    for key, g in st.groupby(["split", "apt", "stand"], observed=True):
        order = np.argsort(g["t_block"].values)
        groups[key] = (g["t_block"].values[order], g["a1"].values[order], g["sc"].values[order],
                        g["TAXITIME_SEC_mvt"].values[order].astype(float))
    n = len(f)
    pa_a1, pa_sc = np.full(n, np.nan), np.full(n, np.nan)
    pa_taxi, pa_gap, missing = np.full(n, np.nan), np.full(n, np.nan), np.ones(n)
    anchor_t = f["t_off"].values - f["z_anchor"].values
    for key, ix in f.groupby(["split", "apt", "stand"], observed=True).indices.items():
        found = groups.get(key)
        if found is None:
            continue
        tb, a1v, scv, taxiv = found
        ix = np.asarray(ix)
        pos = np.searchsorted(tb, anchor_t[ix], "right") - 1
        ok = pos >= 0
        p = np.clip(pos, 0, len(tb) - 1)
        gap = np.where(ok, anchor_t[ix] - tb[p], np.nan)
        within = ok & (gap <= STAND_LOOKBACK)
        pa_a1[ix[within]] = a1v[p[within]]; pa_sc[ix[within]] = scv[p[within]]
        pa_taxi[ix[within]] = taxiv[p[within]]; pa_gap[ix[within]] = gap[within]
        missing[ix[within]] = 0
    return pd.DataFrame({"pa_a1": pa_a1, "pa_sc": pa_sc, "pa_taxi_in": pa_taxi, "pa_gap": pa_gap,
                          "pa_missing": missing}, index=f.index)


def arrmode(f: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """16 arrival recorder-mode features (docs/ARRMODE.md): arrivals share the departure's airport-day recorder,
    and their block times are never hidden, so their copy behaviour is a visible proxy for the departure stamp."""
    a = arr_recs(raw)
    day, al = _arr_day(a), _arr_airline_day(a)
    keys = pd.DataFrame({"split": f["split"].values, "apt": f["apt"].astype(str).values, "day": f["day"].values})
    dj = keys.merge(day, on=["split", "apt", "day"], how="left")
    aj = keys.assign(airline=f["airline"].astype(str).values).merge(al, on=["split", "apt", "day", "airline"], how="left")
    win, pa = _arr_window(f, a), _arr_prev_stand(f, a)
    out = pd.concat([dj[["ad_arr_a1", "ad_arr_sc", "ad_arr_tail", "ad_n_arr"]].reset_index(drop=True),
                      win.reset_index(drop=True), pa.reset_index(drop=True),
                      aj[["al_arr_a1", "al_arr_sc", "al_n_arr"]].reset_index(drop=True)], axis=1)
    out.index = f.index
    return out


def window_state(f: pd.DataFrame, keys: tuple[str, ...], win: float, pre: str) -> pd.DataFrame:
    """Leave-one-out +-win stats within `keys` over neighbours with the stamps; z_aobt, d_ae, d_al only, never y."""
    za, dae, dal = f["z_aobt"].values, f["d_ae"].values, f["d_al"].values
    ok = np.isfinite(za) & np.isfinite(dae)
    okl = ok & np.isfinite(dal)
    norm = f["z_aobt"].groupby([f["apt"], f["runway"], f["hour"]], observed=True).transform("median").values
    dz = np.where(ok, np.clip(za - norm, -WS_CLIP, WS_CLIP), 0.0)
    de = np.where(ok, np.clip(dae, -WS_CLIP, WS_CLIP), 0.0)
    dl = np.where(okl, np.clip(dal, -WS_CLIP, WS_CLIP), 0.0)
    q = {"n": ok, "dz": dz, "dz2": dz ** 2, "hi": ok & (dz > WS_TAIL), "lo": ok & (dz < -WS_TAIL),
         "de": de, "de2": de ** 2, "de0": ok & (np.abs(de) < 60), "nl": okl, "dl": dl, "dl2": dl ** 2}
    q = {k: np.asarray(v, dtype=float) for k, v in q.items()}
    s = {k: np.zeros(len(f)) for k in q}
    t = f["t_off"].values
    for ix in f.groupby(list(keys), observed=True).indices.values():
        o = ix[np.argsort(t[ix], kind="stable")]
        ts = t[o]
        lo, hi = np.searchsorted(ts, ts - win, "left"), np.searchsorted(ts, ts + win, "right")
        for k, v in q.items():
            cum = np.concatenate([[0.0], np.cumsum(v[o])])
            s[k][o] = cum[hi] - cum[lo] - v[o]
    n, nl = s["n"], s["nl"]
    with np.errstate(invalid="ignore", divide="ignore"):
        m, e, l = s["dz"] / n, s["de"] / n, s["dl"] / nl
        d = {"n": n, "dz": m, "dz_sd": np.sqrt(np.maximum(s["dz2"] / n - m ** 2, 0)), "hi": s["hi"] / n,
             "lo": s["lo"] / n, "dae": e, "dae_sd": np.sqrt(np.maximum(s["de2"] / n - e ** 2, 0)),
             "dae0": s["de0"] / n, "dal": l, "dal_sd": np.sqrt(np.maximum(s["dl2"] / nl - l ** 2, 0))}
    return pd.DataFrame({f"{pre}_{k}": v for k, v in d.items()}, index=f.index)


def winstate(f: pd.DataFrame) -> pd.DataFrame:
    """Recorder state around each departure (docs/WINDOW_STATE.md): neighbours' AOBT gaps stand in for their unseen c."""
    return pd.concat([window_state(f, k, w, p) for p, (k, w) in WS_SCALES.items()], axis=1)[WZ]


def loo_mean(f: pd.DataFrame, keys: tuple[str, ...], win: float, v: np.ndarray, ok: np.ndarray, pre: str) -> pd.DataFrame:
    """Leave-one-out count, mean and share above 600 of v over neighbours with ok in the same keys within +-win."""
    t = f["t_off"].values
    q = {"n": ok.astype(float), "s": np.where(ok, v, 0.0), "hi": (ok & (v > 600)).astype(float)}
    s = {k: np.zeros(len(f)) for k in q}
    for ix in f.groupby(list(keys), observed=True).indices.values():
        o = ix[np.argsort(t[ix], kind="stable")]; ts = t[o]
        lo, hi = np.searchsorted(ts, ts - win, "left"), np.searchsorted(ts, ts + win, "right")
        for k, x in q.items():
            cum = np.concatenate([[0.0], np.cumsum(x[o])]); s[k][o] = cum[hi] - cum[lo] - x[o]
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.DataFrame({f"{pre}_n": s["n"], f"{pre}_m": s["s"] / s["n"], f"{pre}_hi": s["hi"] / s["n"]}, index=f.index)


def pw_cols() -> pd.DataFrame:
    """Windows of the neighbours' predicted residual (v28 blend minus z_aobt), for every train and rank row
    (docs/WINDOW_STATE.md): id plus the 5 PW columns, cached since the blend inputs never change underneath it."""
    out = CACHE / "pw.parquet"
    if out.exists():
        return pd.read_parquet(out)
    f = build()
    rk = (f["split"] == "rank").values
    p = np.zeros(len(f))
    for n, w in PW_BLEND.items():
        tr = pd.concat([pd.read_parquet(CACHE / f"{n}_oof.parquet").rename(columns={"oof": "p"}),
                        pd.read_parquet(CACHE / f"{n}_holdout.parquet")]).set_index("id")["p"]
        rr = pd.read_parquet(f"data/{n}_rank.parquet").set_index("MVT_ID_mvt")["TAXITIME_SEC_mvt"]
        p[~rk] += w * f.loc[~rk, "id"].map(tr).values
        p[rk] += w * f.loc[rk, "id"].map(rr).values
    p = np.maximum(p, 0)
    c = p - f["z_aobt"].values
    ok = (f["has_plan"].values == 1) & np.isfinite(c)
    c = np.where(ok, np.clip(c, -1800, 1800), 0.0)
    res = pd.concat([loo_mean(f, ("split", "apt", "day"), 3600.0, c, ok, "pw"),
                      loo_mean(f, ("split", "apt", "day", "runway"), 3600.0, c, ok, "pr")[["pr_m"]],
                      loo_mean(f, ("split", "apt", "day"), 1800.0, c, ok, "pw30")[["pw30_m"]]], axis=1)
    out_df = pd.concat([f[["id"]], res], axis=1)
    tmp = out.with_suffix(".tmp.parquet")
    out_df.to_parquet(tmp); tmp.replace(out)
    return out_df


def with_pw(f: pd.DataFrame) -> pd.DataFrame:
    """Join the 5 PW columns onto f by id (docs/WINDOW_STATE.md, the neighbours' predicted residual)."""
    return f.merge(pw_cols(), on="id", how="left")


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
    f = pd.concat([f, traffic(f, arr), daylevel(f), occupancy(f, blk), join(f), arrmode(f, raw)], axis=1)
    for c in CAT:
        f[c] = f[c].astype(str).fillna("NA").astype("category")
    f = f.reset_index(drop=True)
    f = pd.concat([f, winstate(f)], axis=1)  # after the CAT cast: a missing runway is then its own level, not a dropped key
    tmp = out.with_suffix(".tmp.parquet")  # another process may be reading feat.parquet: write, then atomic rename
    f.to_parquet(tmp)
    tmp.replace(out)
    return f


if __name__ == "__main__":
    f = build(refresh=True)
    print(f.shape); print(f.describe().T.round(1).to_string())
