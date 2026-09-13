"""Feature groups for the lane-A stamp classifier, on top of CAT+NUM+DAY+ARR (docs/FE.md):
group 1 = exact-stamp fingerprints (row-level, no label, cached at data/cache/fe_fp.parquet);
groups 2+3 = norm-relative gaps and shrunk mechanism target encodings, both fit-row dependent
(fit_apply) so the test harness can apply them out-of-fold (oof_apply, day-blocked)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import CACHE, departures, load_raw, secs
from mix import label

FP = ["lobt_eq_eobt", "iobt_eq_eobt", "aobt_eq_eobt", "d_ae_m5", "d_ae_m1", "eobt_sec", "sched_eq_eobt", "lobt_min5"]
G2 = ["za_rel", "ze_rel", "zs_rel", "ze_over_za"]
G3 = ["fl_aobt_rate", "fl_eobt_rate", "fl_sched_rate", "fl_n",
      "st_aobt_rate", "st_eobt_rate", "st_sched_rate", "st_n",
      "al_aobt_rate", "al_eobt_rate", "al_sched_rate", "al_n"]
GROUPS = {"fl": ["FLIGHT_mvt"], "st": ["apt", "stand"], "al": ["apt", "airline"]}
PRIOR_N = 20.0  # smoothing weight toward the airport rate for sparse flight/stand/airline groups


def build_fp(refresh: bool = False) -> pd.DataFrame:
    """Group 1: exact-equality fingerprints from each row's own raw stamps (no label, no OOF needed)."""
    out = CACHE / "fe_fp.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)
    dep = departures(load_raw())
    ep = lambda c: secs(dep[c], pd.Timestamp(0, tz="UTC"))  # noqa: E731
    lobt, eobt, iobt, aobt, sched = ep("LOBT_flt"), ep("EOBT_1_flt"), ep("IOBT_flt"), ep("AOBT_3_flt"), ep("SCHED_TIME_UTC_mvt")
    d_ae = aobt - eobt
    f = pd.DataFrame({"id": dep["MVT_ID_mvt"].values})
    f["lobt_eq_eobt"] = (lobt == eobt).astype(int).values
    f["iobt_eq_eobt"] = (iobt == eobt).astype(int).values
    f["aobt_eq_eobt"] = (aobt == eobt).astype(int).values
    f["d_ae_m5"] = (d_ae % 300 == 0).astype(int).values
    f["d_ae_m1"] = (d_ae % 60 == 0).astype(int).values
    f["eobt_sec"] = (eobt % 60 == 0).astype(int).values
    f["sched_eq_eobt"] = (sched == eobt).astype(int).values
    f["lobt_min5"] = (lobt % 300 == 0).astype(int).values
    f.to_parquet(out)
    return f


def flight_ids() -> pd.DataFrame:
    """Full FLIGHT_mvt callsign per departure id (finer-grained than the truncated `airline` CAT column)."""
    dep = departures(load_raw())
    fl = dep["FLIGHT_mvt"].astype(str).where(dep["FLIGHT_mvt"].notna(), "NA")
    return pd.DataFrame({"id": dep["MVT_ID_mvt"].values, "FLIGHT_mvt": fl.values})


def _rates(fit_a: pd.DataFrame, lab: np.ndarray, keys: list[str]) -> pd.DataFrame:
    """Per-group class-0/1/2 rate and row count among lane-A fit rows."""
    d = fit_a[keys].reset_index(drop=True).copy()
    for k in range(3):
        d[f"_r{k}"] = (lab == k).astype(int)
    g = d.groupby(keys, observed=True).agg(_n=(keys[0], "size"), _r0=("_r0", "mean"), _r1=("_r1", "mean"), _r2=("_r2", "mean"))
    return g.reset_index()


def fit_apply(fit_df: pd.DataFrame, apply_df: pd.DataFrame) -> pd.DataFrame:
    """Groups 2+3: fit norm-taxi and stamp-rate stats on fit_df's lane-A rows, apply (out-of-sample) to apply_df."""
    fit_a = fit_df[fit_df["has_plan"] == 1]
    lab = label(fit_a)
    out = pd.DataFrame(index=apply_df.index)

    norm = fit_a[lab == 3].groupby(["apt", "runway", "hour"], observed=True)["y"].median().rename("_norm").reset_index()
    nt = apply_df[["apt", "runway", "hour"]].reset_index(drop=True).merge(norm, on=["apt", "runway", "hour"], how="left")["_norm"].values
    out["za_rel"] = apply_df["z_aobt"].values - nt
    out["ze_rel"] = apply_df["z_eobt"].values - nt
    out["zs_rel"] = apply_df["z_sched"].values - nt
    ratio = (apply_df["z_eobt"] / apply_df["z_aobt"]).replace([np.inf, -np.inf], np.nan).clip(0, 10)
    out["ze_over_za"] = ratio.values

    apt_rate = _rates(fit_a, lab, ["apt"])
    rate_cols = ["_n", "_r0", "_r1", "_r2"]
    base = apply_df[["apt"]].reset_index(drop=True).merge(apt_rate, on="apt", how="left")
    base[rate_cols] = base[rate_cols].fillna(0.0)  # unseen apt (should not happen): 0 prior, not a categorical column
    for prefix, keys in GROUPS.items():
        g = _rates(fit_a, lab, keys)
        m = apply_df[keys].reset_index(drop=True).merge(g, on=keys, how="left")
        n = m["_n"].fillna(0.0).values
        for k, name in enumerate(("aobt", "eobt", "sched")):
            raw = m[f"_r{k}"].fillna(0.0).values
            prior = base[f"_r{k}"].values  # airport rate: the smoothing prior, and the unseen-key fallback (n=0)
            out[f"{prefix}_{name}_rate"] = (n * raw + PRIOR_N * prior) / (n + PRIOR_N)
        out[f"{prefix}_n"] = n
    return out


def oof_apply(df: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> pd.DataFrame:
    """5-fold out-of-fold fit_apply, blocked by day, so a training row's own label never reaches its own encoding."""
    days = np.sort(df["day"].unique())
    block = pd.Series(np.random.default_rng(seed).permutation(len(days)) % n_splits, index=days)
    day_block = df["day"].map(block).values
    parts = [fit_apply(df[day_block != b], df[day_block == b]) for b in range(n_splits)]
    return pd.concat(parts).reindex(df.index)
