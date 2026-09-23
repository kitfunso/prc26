"""H1b item 1: does physical row order track MVT_TIME, BLOCK_TIME or SCHED_TIME within airport-day?
Pooled within-(apt,day) rank correlation: rank each of pos/MVT_TIME/BLOCK_TIME/SCHED_TIME inside each
(apt,day) group, then correlate the rank columns across all rows (equivalent to a group-stratified
Spearman, weighted by group size). Dev months (train) + label-free check on ranking/submitting.
Run: PYTHONPATH=. python analysis/hunt/h1b/s1_row_order.py
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


def main() -> None:
    rows = []
    parts = []
    for path in DEV_MONTH_FILES:
        df = read_month(path)
        df["mvt_i"] = df["MVT_TIME_UTC_mvt"]
        df["blk_i"] = df["BLOCK_TIME_UTC_mvt"]
        df["sch_i"] = df["SCHED_TIME_UTC_mvt"]
        df["y"] = secs(df["MVT_TIME_UTC_mvt"], df["BLOCK_TIME_UTC_mvt"])
        add_ranks(df, ["pos", "mvt_i", "blk_i", "sch_i"])
        keep = df[["apt", "day", "r_pos", "r_mvt_i", "r_blk_i", "r_sch_i", "y"]].copy()
        keep["month"] = path
        parts.append(keep)
    d = pd.concat(parts, ignore_index=True)
    print(f"dev DEP rows: {len(d)}")
    print("pooled within(apt,day) rank-corr(pos, MVT_TIME)   =", pooled_corr(d, "r_pos", "r_mvt_i"))
    print("pooled within(apt,day) rank-corr(pos, BLOCK_TIME) =", pooled_corr(d, "r_pos", "r_blk_i"))
    print("pooled within(apt,day) rank-corr(pos, SCHED_TIME) =", pooled_corr(d, "r_pos", "r_sch_i"))

    for lo, label in [(3600, "|y| > 1h"), (43200, "|y| > 12h")]:
        sub = d[d["y"].abs() > lo]
        n = len(sub)
        if n < 20:
            print(f"[{label}] n={n} too few for a stable correlation")
            continue
        print(f"[{label}] n={n}: rank-corr(pos,MVT)={pooled_corr(sub,'r_pos','r_mvt_i'):.4f} "
              f"rank-corr(pos,BLOCK)={pooled_corr(sub,'r_pos','r_blk_i'):.4f} "
              f"rank-corr(pos,SCHED)={pooled_corr(sub,'r_pos','r_sch_i'):.4f}")

    # label-free check: ranking.parquet order vs MVT_TIME / SCHED_TIME
    rk = read_month("data/ranking.parquet", dep_only=True)
    rk["mvt_i"] = rk["MVT_TIME_UTC_mvt"]
    rk["sch_i"] = rk["SCHED_TIME_UTC_mvt"]
    add_ranks(rk, ["pos", "mvt_i", "sch_i"])
    print(f"\nranking.parquet DEP rows: {len(rk)}")
    print("pooled within(apt,day) rank-corr(pos, MVT_TIME)   =", pooled_corr(rk, "r_pos", "r_mvt_i"))
    print("pooled within(apt,day) rank-corr(pos, SCHED_TIME) =", pooled_corr(rk, "r_pos", "r_sch_i"))

    # submitting.parquet: only MVT_ID_mvt + TAXITIME_SEC_mvt; join to ranking.parquet for timestamps
    sub = pd.read_parquet("data/submitting.parquet")
    sub["pos"] = np.arange(len(sub))
    print(f"\nsubmitting.parquet rows: {len(sub)}  TAXITIME_SEC_mvt non-null: {sub['TAXITIME_SEC_mvt'].notna().sum()}")
    j = sub.merge(rk[["MVT_ID_mvt", "apt", "day", "mvt_i", "sch_i"]], on="MVT_ID_mvt", how="left")
    print("submitting rows matched to a ranking DEP row:", j["apt"].notna().sum(), "of", len(j))
    j = j.dropna(subset=["apt"])
    add_ranks(j, ["pos", "mvt_i", "sch_i"])
    print("pooled within(apt,day) rank-corr(submitting pos, MVT_TIME)   =", pooled_corr(j, "r_pos", "r_mvt_i"))
    print("pooled within(apt,day) rank-corr(submitting pos, SCHED_TIME) =", pooled_corr(j, "r_pos", "r_sch_i"))
    common_ids = set(sub["MVT_ID_mvt"]) & set(rk["MVT_ID_mvt"])
    rk_order = rk[rk["MVT_ID_mvt"].isin(common_ids)].sort_values("pos")["MVT_ID_mvt"].tolist()
    sub_order = sub[sub["MVT_ID_mvt"].isin(common_ids)]["MVT_ID_mvt"].tolist()
    print("submitting order identical to ranking-file DEP order restricted to shared ids:", rk_order == sub_order)


if __name__ == "__main__":
    main()
