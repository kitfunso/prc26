"""H1b, tying items 1-7 together: the task's own framing says competitors' 8.7-21.5s one-step jumps
most plausibly come from fixing a handful of catastrophic rows (~7.5B SSE / ~37s board each for a
one-day miss). Does a STRUCTURAL cue (id/cross-link/null pattern -- not a feature-engineering gap
like dur/slip which v34 already has) mark these rows, visible on rank data? Dev months only.
Run: PYTHONPATH=. python analysis/hunt/h1b/s10_extreme_rows.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import DEV_MONTH_FILES, classify_offblock, read_month
from data import secs


def main() -> None:
    parts = [read_month(p) for p in DEV_MONTH_FILES]
    d = pd.concat(parts, ignore_index=True)
    d["cls"] = classify_offblock(d)
    d["y"] = secs(d["MVT_TIME_UTC_mvt"], d["BLOCK_TIME_UTC_mvt"])
    d = d.rename(columns={"MVT_ID_mvt": "id", "FLIGHT_ID_mvt": "fid"})

    oof = pd.concat([
        pd.read_parquet("data/cache/recal_oof_v34.parquet", columns=["id", "oof"]),
        pd.read_parquet("data/cache/laneb_oof.parquet", columns=["id", "oof"]),
    ], ignore_index=True)
    d = d.merge(oof, on="id", how="inner")
    d["resid"] = d["y"] - d["oof"]

    print(f"dev DEP rows matched to OOF: {len(d)}")
    extreme = d[d["y"].abs() > 43200].copy()
    print(f"\n|y| > 12h (43200s): n={len(extreme)} ({len(extreme)/len(d):.4%})")
    print(f"SSE share of these rows (true y vs v34 oof): "
          f"{float((extreme['resid']**2).sum() / (d['resid']**2).sum()):.4%}")
    print(f"of which within 900s of an exact multiple of 86400s (clean day-shift): "
          f"{((extreme['y'].abs() % 86400).clip(upper=86400 - (extreme['y'].abs() % 86400)) < 900).sum()}")

    print("\noff-block class of extreme rows:")
    print(extreme["cls"].value_counts())
    print("\noff-block class, ALL dev rows (baseline for comparison):")
    print(d["cls"].value_counts(normalize=True))

    print("\nis v34's OOF already large on these rows (does it partially see the problem)?")
    print(f"  median |oof| on extreme rows: {extreme['oof'].abs().median():.0f}s "
          f"vs median |y|: {extreme['y'].abs().median():.0f}s")
    print(f"  median |resid| on extreme rows: {extreme['resid'].abs().median():.0f}s")

    # structural cues available on RANK rows: null AOBT/EOBT, MVT hour-of-day, gap to SCHED, has FLIGHT_ID
    extreme["null_plan"] = extreme["AOBT_3_flt"].isna()
    d["null_plan"] = d["AOBT_3_flt"].isna()
    print(f"\nnull AOBT (no-plan) rate: extreme={extreme['null_plan'].mean():.4%} "
          f"all-dev={d['null_plan'].mean():.4%}")

    gap_sched = secs(d["MVT_TIME_UTC_mvt"], d["SCHED_TIME_UTC_mvt"])
    d["gap_sched_abs"] = gap_sched.abs()
    extreme_gap = d.loc[extreme.index, "gap_sched_abs"]
    print(f"\n|MVT_TIME - SCHED_TIME| (already-modeled slip/dur signal): "
          f"extreme median={extreme_gap.median():.0f}s  all-dev median={d['gap_sched_abs'].median():.0f}s")
    # does a large sched gap alone identify these rows well enough to flag them on rank data?
    thresh = d["gap_sched_abs"].quantile(0.99)
    catch_rate = d.loc[extreme.index, "gap_sched_abs"].gt(thresh).mean()
    print(f"  top-1% sched-gap flag (threshold={thresh:.0f}s) catches {catch_rate:.2%} of the "
          f"extreme-y rows (vs 1.00% of all rows flagged by construction)")

    print("\napt distribution of extreme rows vs all-dev:")
    print(pd.concat([extreme["apt"].value_counts(normalize=True).rename("extreme"),
                      d["apt"].value_counts(normalize=True).rename("all_dev")], axis=1))


if __name__ == "__main__":
    main()
