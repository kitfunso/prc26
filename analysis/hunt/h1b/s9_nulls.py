"""H1b item 6: null patterns. (1) for every raw column, train-dev DEP non-null rate vs rank DEP
non-null rate -- flags anything blanked on rank beyond BLOCK_TIME_UTC_mvt/TAXITIME_SEC_mvt/label.
(2) in training, does null rate on AOBT/EOBT/LOBT/IOBT differ by off-block class (a null stamp
can't have been copied from, so class should skew toward OTHER when the stamp is missing).
Run: PYTHONPATH=. python analysis/hunt/h1b/s9_nulls.py
"""
from __future__ import annotations

import pandas as pd

from common import DEV_MONTH_FILES, RAW_COLS, classify_offblock, read_month


def main() -> None:
    parts = [read_month(p) for p in DEV_MONTH_FILES]
    d = pd.concat(parts, ignore_index=True)
    rk = read_month("data/ranking.parquet", dep_only=True)

    print(f"dev DEP rows: {len(d)}  rank DEP rows: {len(rk)}")
    print("\n--- non-null rate by column, train-dev DEP vs rank DEP ---")
    cols = [c for c in RAW_COLS if c in d.columns and c in rk.columns]
    for c in cols:
        tr = d[c].notna().mean()
        rn = rk[c].notna().mean()
        flag = "  <-- DIFFERS" if abs(tr - rn) > 0.001 else ""
        print(f"{c:20s} train={tr:.4%}  rank={rn:.4%}{flag}")

    print("\n--- (train) null rate on AOBT/EOBT/LOBT/IOBT by off-block class ---")
    d["cls"] = classify_offblock(d)
    for stamp_col in ("AOBT_3_flt", "EOBT_1_flt", "LOBT_flt", "IOBT_flt"):
        print(f"\n{stamp_col}:")
        for cls, sub in d.groupby("cls"):
            if len(sub) < 200:
                continue
            print(f"  {cls:10s} n={len(sub):8d}  null rate={sub[stamp_col].isna().mean():.4%}")

    print("\n--- overall null-row profile: rows missing AOBT and/or EOBT, by class (train) ---")
    miss_aobt = d["AOBT_3_flt"].isna()
    miss_eobt = d["EOBT_1_flt"].isna()
    print(f"missing AOBT: {miss_aobt.sum()} ({miss_aobt.mean():.4%})  missing EOBT: {miss_eobt.sum()} ({miss_eobt.mean():.4%})  "
          f"missing both: {(miss_aobt & miss_eobt).sum()}")
    print(d.loc[miss_aobt, "cls"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
