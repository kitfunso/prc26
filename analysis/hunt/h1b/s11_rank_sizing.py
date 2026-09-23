"""H1b: label-free sizing of the LIRF/no-plan/huge-slip regime on the RANK set (context only --
no y used, no fix attempted; this regime is already owned by the existing lane-B mechanism model,
which is off-limits to me per the fresh-eyes rule). Just counts, for the report's SSE-at-stake table.
Run: PYTHONPATH=. python analysis/hunt/h1b/s11_rank_sizing.py
"""
from __future__ import annotations

import pandas as pd

from common import read_month
from data import secs


def main() -> None:
    rk = read_month("data/ranking.parquet", dep_only=True)
    null_aobt = rk["AOBT_3_flt"].isna()
    is_lirf = rk["apt"] == "LIRF"
    gap_sched = secs(rk["MVT_TIME_UTC_mvt"], rk["SCHED_TIME_UTC_mvt"]).abs()

    print(f"rank DEP rows: {len(rk)}")
    print(f"null AOBT (no-plan): {null_aobt.sum()} ({null_aobt.mean():.4%})")
    print(f"LIRF: {is_lirf.sum()} ({is_lirf.mean():.4%})")
    print(f"null AOBT & LIRF: {(null_aobt & is_lirf).sum()}")
    print(f"null AOBT & LIRF & |MVT-SCHED| > 9083s (dev's top-1% threshold): "
          f"{(null_aobt & is_lirf & (gap_sched > 9083)).sum()}")


if __name__ == "__main__":
    main()
