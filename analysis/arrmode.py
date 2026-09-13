"""Arrival recorder-mode features: thin caller over features.py's production build (docs/ARRMODE.md).
Kept for the diagnostic sanity() report and the standalone cache `data/cache/arrmode.parquet`.
`PYTHONPATH=/c/Users/skf_s/prc26 python analysis/arrmode.py`"""
from __future__ import annotations

import pandas as pd

from data import CACHE, load_raw
from features import ARR, TOL, arr_recs
from features import build as build_dep


def _demean(df: pd.DataFrame, apt: pd.Series, cols: list[str]) -> pd.DataFrame:
    return df[cols].sub(df.groupby(apt)[cols].transform("mean"))


def sanity(dep: pd.DataFrame, a: pd.DataFrame, f: pd.DataFrame) -> None:
    """Non-null shares per column, and the within-airport arrival/dep-copy-share correlation table."""
    print("non-null share by split:")
    for c in ARR:
        print(f"  {c}: " + ", ".join(f"{s}={f.loc[dep['split'].values == s, c].notna().mean():.3f}"
                                      for s in ("train", "rank")))

    # independent (non-mutually-exclusive) per-stamp match, not mix.label's priority class: this is what
    # reproduces the reference correlation table to 2dp, unlike the mixture-model class share.
    lane_a = dep[dep["has_plan"] == 1]
    aobt = (lane_a["y"] - lane_a["z_aobt"]).abs() < TOL
    eobt = (lane_a["y"] - lane_a["z_eobt"]).abs() < TOL
    sched = (lane_a["y"] - lane_a["z_sched"]).abs() < TOL
    dep_day = pd.DataFrame({"apt": lane_a["apt"].astype(str).values, "day": lane_a["day"].values,
                             "aobt": aobt.values, "eobt": eobt.values, "sched": sched.values}).groupby(["apt", "day"]).agg(
        dep_aobt=("aobt", "mean"), dep_eobt=("eobt", "mean"), dep_sched=("sched", "mean"), n_dep=("aobt", "size"))
    arr_tr = a[a["split"] == "train"]
    arr_day = arr_tr.groupby(["apt", "day"]).agg(arr_a1=("a1", "mean"), arr_sched=("sc", "mean"),
                                                  arr_a3=("a3", "mean"), n_arr=("a1", "size"))
    j = dep_day.join(arr_day, how="inner")
    j = j[(j["n_dep"] >= 20) & (j["n_arr"] >= 20)]
    apt = j.index.get_level_values("apt")
    dm = _demean(j.reset_index(drop=True), pd.Series(apt), ["dep_aobt", "dep_eobt", "dep_sched", "arr_a1", "arr_sched", "arr_a3"])
    print(f"\nwithin-airport correlation, {len(j)} airport-days (train, n_arr>=20, n_dep>=20):")
    tbl = pd.DataFrame({dc: [dm[ac].corr(dm[dc]) for ac in ("arr_a1", "arr_sched", "arr_a3")]
                        for dc in ("dep_aobt", "dep_eobt", "dep_sched")}, index=["arr_a1", "arr_sched", "arr_a3"])
    print(tbl.round(2).to_string())


def build(refresh: bool = False) -> pd.DataFrame:
    out = CACHE / "arrmode.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)
    dep = build_dep()
    f = pd.concat([dep[["id"]], dep[ARR]], axis=1)
    sanity(dep, arr_recs(load_raw()), f)
    f.to_parquet(out)
    return f


if __name__ == "__main__":
    f = build(refresh=True)
    print(f.shape)
