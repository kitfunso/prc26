"""Are LIRF schedule copies clustered in time (a recording outage)? Label-free proxy: the share of LIRF ARRIVALS near
the departure whose in-block equals their schedule. Arrival in-blocks are visible on rank rows.
Run: PYTHONPATH=. python analysis/hunt/h1a/burst.py"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import OUT, movements

WIN = (1800, 3600, 7200)


def arrival_copy_share(dep_t: np.ndarray, arr_t: np.ndarray, arr_copy: np.ndarray, win: float) -> tuple[np.ndarray, np.ndarray]:
    """Share and count of arrivals (by in-block time) within +-win seconds of each departure time."""
    o = np.argsort(arr_t)
    t, c = arr_t[o], np.r_[0, np.cumsum(arr_copy[o])]
    lo, hi = np.searchsorted(t, dep_t - win), np.searchsorted(t, dep_t + win)
    n = hi - lo
    return np.where(n > 0, (c[hi] - c[lo]) / np.maximum(n, 1), np.nan), n


def lirf_frame(apt: str = "LIRF") -> pd.DataFrame:
    df = movements()
    a = df[(df["apt"] == apt) & (df["PHASE_mvt"] == "ARR") & df["BLOCK_TIME_UTC_mvt"].notna()]
    acopy = ((a["BLOCK_TIME_UTC_mvt"] - a["SCHED_TIME_UTC_mvt"]).abs() <= 10).values.astype(float)
    d = df[(df["apt"] == apt) & (df["PHASE_mvt"] == "DEP")].copy()
    for w in WIN:
        d[f"acs{w // 60}"], d[f"acn{w // 60}"] = arrival_copy_share(
            d["MVT_TIME_UTC_mvt"].values, a["BLOCK_TIME_UTC_mvt"].values, acopy, w)
    d["zs"] = d["MVT_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]
    d["copy"] = ((d["y"] - d["zs"]).abs() <= 10).where(d["y"].notna())
    return d


def main() -> None:
    d = lirf_frame()
    d.to_parquet(OUT / "lirf_burst.parquet")
    dev = d[d["dev"]].copy()
    dev["lane"] = np.where(dev["has_plan"], "A", "B")
    pd.set_option("display.width", 250)
    for w in WIN:
        c = f"acs{w // 60}"
        dev["bin"] = pd.cut(dev[c], [-0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0])
        for zmin in (-1e9, 3600):
            s = dev[dev["zs"] > zmin]
            t = s.groupby(["lane", "bin"], observed=True)["copy"].agg(["size", "mean"]).unstack("lane")
            print(f"window +-{w // 60} min, zs > {zmin:.0f}: departure copy rate by arrival-copy share\n", t.round(3).to_string())


if __name__ == "__main__":
    main()
