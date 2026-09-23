"""H1b item 2: are MVT_ID_mvt / FLIGHT_ID_mvt sequential, hashed or time-encoded?
Global (month-level) correlation conflates calendar days (which MVT_TIME already gives on rank rows).
The real test is the WITHIN-(apt,day) pooled rank correlation, same method as item 1's row order check.
Run: PYTHONPATH=. python analysis/hunt/h1b/s2_ids.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import DEV_MONTH_FILES, read_month
from data import secs


def add_ranks(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = df.groupby(["apt", "day"])
    for c in cols:
        df[f"r_{c}"] = g[c].rank()
    return df


def pooled_corr(df: pd.DataFrame, a: str, b: str) -> float:
    return float(df[a].corr(df[b]))


def month_report(df: pd.DataFrame, label: str, has_block: bool) -> pd.DataFrame:
    cols = ["mvt_i", "sch_i"] + (["blk_i"] if has_block else [])
    add_ranks(df, ["fid", "mid"] + cols)
    print(f"\n--- {label}: n={len(df)} ---")
    print("global spearman FLIGHT_ID vs MVT_TIME  :", float(df["FLIGHT_ID_mvt"].corr(df["mvt_i"], method="spearman")))
    print("within(apt,day) rank-corr FLIGHT_ID vs MVT_TIME  :", pooled_corr(df, "r_fid", "r_mvt_i"))
    print("within(apt,day) rank-corr FLIGHT_ID vs SCHED_TIME:", pooled_corr(df, "r_fid", "r_sch_i"))
    print("within(apt,day) rank-corr MVT_ID    vs MVT_TIME  :", pooled_corr(df, "r_mid", "r_mvt_i"))
    print("within(apt,day) rank-corr MVT_ID    vs SCHED_TIME:", pooled_corr(df, "r_mid", "r_sch_i"))
    if has_block:
        print("within(apt,day) rank-corr FLIGHT_ID vs BLOCK_TIME:", pooled_corr(df, "r_fid", "r_blk_i"))
        print("within(apt,day) rank-corr MVT_ID    vs BLOCK_TIME:", pooled_corr(df, "r_mid", "r_blk_i"))
        print("  ...vs SCHED->BLOCK baseline (how well SCHED alone ranks BLOCK):",
              pooled_corr(df, "r_sch_i", "r_blk_i"))
    return df


def main() -> None:
    parts = []
    for path in DEV_MONTH_FILES:
        df = read_month(path)
        df["fid"] = df["FLIGHT_ID_mvt"]
        df["mid"] = df["MVT_ID_mvt"]
        df["mvt_i"] = df["MVT_TIME_UTC_mvt"]
        df["blk_i"] = df["BLOCK_TIME_UTC_mvt"]
        df["sch_i"] = df["SCHED_TIME_UTC_mvt"]
        parts.append(df)
    d = pd.concat(parts, ignore_index=True)
    month_report(d, "dev months pooled", has_block=True)

    # FLIGHT_ID gap structure (dedup: FLIGHT_ID repeats across DEP/ARR of the same flight)
    fid_sorted = np.sort(d["FLIGHT_ID_mvt"].dropna().unique())
    gaps = np.diff(fid_sorted)
    print("\nFLIGHT_ID unique-value gap stats (dev, sorted):",
          "median", np.median(gaps), "mean", gaps.mean(), "p90", np.percentile(gaps, 90), "max", gaps.max())
    mid_sorted = np.sort(d["MVT_ID_mvt"].dropna().unique())
    mgaps = np.diff(mid_sorted)
    print("MVT_ID unique-value gap stats (dev, sorted):",
          "median", np.median(mgaps), "mean", mgaps.mean(), "p90", np.percentile(mgaps, 90), "max", mgaps.max())

    # label-free: ranking.parquet (2026)
    rk = read_month("data/ranking.parquet", dep_only=True)
    rk["fid"] = rk["FLIGHT_ID_mvt"]
    rk["mid"] = rk["MVT_ID_mvt"]
    rk["mvt_i"] = rk["MVT_TIME_UTC_mvt"]
    rk["sch_i"] = rk["SCHED_TIME_UTC_mvt"]
    month_report(rk, "ranking.parquet (2026, label-free)", has_block=False)


if __name__ == "__main__":
    main()
