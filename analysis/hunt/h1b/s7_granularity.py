"""H1b item 4: stamp granularity/rounding. Two questions --
(a) sanity: does BLOCK_TIME's own seconds-of-minute go to :00 far more often on copy-class rows
    than OTHER (physical) rows -- validates the classify_offblock premise with a second, independent
    signal (train only, BLOCK_TIME known).
(b) the real test: MVT_TIME_UTC_mvt (takeoff) and SCHED_TIME_UTC_mvt are both available on RANK rows.
    Does their raw seconds-of-minute value carry residual signal on top of v34's OOF (i.e. does the
    model currently under-use rounding it doesn't see as a feature)? Fit a group-mean-by-second
    correction on 8 dev months, evaluate held out on 2 -- identical discipline to s5.
Run: PYTHONPATH=. python analysis/hunt/h1b/s7_granularity.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import DEV_MONTH_FILES, classify_offblock, read_month

FIT_MONTHS = [p for p in DEV_MONTH_FILES if any(t in p for t in
              ("2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-08", "2025-09", "2025-10"))]
TEST_MONTHS = [p for p in DEV_MONTH_FILES if p not in FIT_MONTHS]


def sec_of_min(col: pd.Series) -> pd.Series:
    return col.dt.second + col.dt.microsecond / 1e6


def load_raw() -> pd.DataFrame:
    parts = []
    for path in DEV_MONTH_FILES:
        df = read_month(path)
        df["cls"] = classify_offblock(df)
        df["mvt_sec"] = sec_of_min(df["MVT_TIME_UTC_mvt"])
        df["sched_sec"] = sec_of_min(df["SCHED_TIME_UTC_mvt"])
        df["block_sec"] = sec_of_min(df["BLOCK_TIME_UTC_mvt"])
        df = df.rename(columns={"MVT_ID_mvt": "id"})
        parts.append(df[["id", "apt", "day", "cls", "mvt_sec", "sched_sec", "block_sec"]])
    return pd.concat(parts, ignore_index=True)


def fit_lut(fit_df: pd.DataFrame, key_col: str) -> pd.Series:
    key = fit_df[key_col].round().clip(0, 59).astype(int)
    return fit_df["resid"].groupby(key).mean()


def apply_lut(test_df: pd.DataFrame, key_col: str, lut: pd.Series) -> np.ndarray:
    key = test_df[key_col].round().clip(0, 59).astype(int)
    return key.map(lut).fillna(0.0).to_numpy()


def main() -> None:
    d = load_raw()
    print(f"dev rows: {len(d)}")
    print(f"MVT_TIME microsecond always 0 (whole-second precision only): {(d['mvt_sec'] == d['mvt_sec'].round()).mean():.4%}")

    print("\n--- (a) BLOCK_TIME seconds-of-minute == 0 (:00), by off-block class ---")
    for cls, sub in d.groupby("cls"):
        print(f"{cls:10s} n={len(sub):8d}  block_sec==0: {(sub['block_sec'] == 0).mean():.2%}  "
              f"mvt_sec==0: {(sub['mvt_sec'] == 0).mean():.2%}  sched_sec==0: {(sub['sched_sec'] == 0).mean():.2%}")

    # rank-side availability check: are AOBT/EOBT/SCHED/MVT non-null on rank rows? (z_aobt etc already
    # used as v34 features, so this should hold -- verifying rather than assuming)
    rk = pd.read_parquet("data/ranking.parquet",
                          columns=["PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt"])
    rk = rk[rk["PHASE_mvt"] == "DEP"]
    print(f"\nrank DEP rows non-null: MVT_TIME={rk['MVT_TIME_UTC_mvt'].notna().mean():.2%} "
          f"SCHED_TIME={rk['SCHED_TIME_UTC_mvt'].notna().mean():.2%} "
          f"AOBT={rk['AOBT_3_flt'].notna().mean():.2%} EOBT={rk['EOBT_1_flt'].notna().mean():.2%}")

    oof = pd.concat([
        pd.read_parquet("data/cache/recal_oof_v34.parquet", columns=["id", "oof"]),
        pd.read_parquet("data/cache/laneb_oof.parquet", columns=["id", "oof"]),
    ], ignore_index=True)
    y = pd.read_parquet("data/cache/feat.parquet", columns=["id", "y"])
    d = d.merge(oof, on="id").merge(y, on="id")
    d["resid"] = d["y"] - d["oof"]

    fit_ids, test_ids = set(), set()
    for p in FIT_MONTHS:
        fit_ids |= set(pd.read_parquet(p, columns=["MVT_ID_mvt"])["MVT_ID_mvt"])
    for p in TEST_MONTHS:
        test_ids |= set(pd.read_parquet(p, columns=["MVT_ID_mvt"])["MVT_ID_mvt"])
    fit = d[d["id"].isin(fit_ids)]
    test = d[d["id"].isin(test_ids)]

    base_sse = float((test["resid"] ** 2).sum())
    print(f"\n--- (b) seconds-of-minute bias correction, fit={len(fit)} test={len(test)} ---")
    print(f"test baseline (v34 OOF) SSE={base_sse:.6g}")
    for key_col in ("mvt_sec", "sched_sec"):
        lut = fit_lut(fit, key_col)
        corr = apply_lut(test, key_col, lut)
        sse = float(((test["resid"].to_numpy() - corr) ** 2).sum())
        print(f"corrector={key_col:10s} new SSE={sse:.6g} delta={base_sse - sse:+.6g} ({(base_sse-sse)/1e6:+.2f}M)")


if __name__ == "__main__":
    main()
