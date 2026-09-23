"""H1b item 3 follow-up: what IS the DEP+ARR same-FLIGHT_ID pairing found in s6 (50922 cases)?
Verify causal ordering (does ARR event follow DEP event, i.e. same movement observed leaving
origin then arriving at destination) and inspect the ADEP/ADES semantics + the 4 train/rank
FLIGHT_ID "overlaps" (near-certainly coincidence in a large id space, confirm not exploitable).
Label-free: uses ranking.parquet only for the pairing check (no y anywhere here).
Run: PYTHONPATH=. python analysis/hunt/h1b/s6b_croslink_followup.py
"""
from __future__ import annotations

import pandas as pd

from common import DEV_MONTH_FILES
from data import secs

COLS = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt",
        "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "SCHED_TIME_UTC_mvt"]


def main() -> None:
    rk = pd.read_parquet("data/ranking.parquet", columns=COLS)

    dep = rk[rk["PHASE_mvt"] == "DEP"]
    arr = rk[rk["PHASE_mvt"] == "ARR"]
    dep_dup = dep["FLIGHT_ID_mvt"].value_counts()
    arr_dup = arr["FLIGHT_ID_mvt"].value_counts()
    print(f"FLIGHT_ID_mvt with >1 DEP row: {(dep_dup > 1).sum()}  with >1 ARR row: {(arr_dup > 1).sum()}")

    # keep exactly one DEP + one ARR row per FLIGHT_ID (first by MVT_TIME) for a clean 1:1 pairing
    dep = dep.sort_values("MVT_TIME_UTC_mvt").drop_duplicates("FLIGHT_ID_mvt").set_index("FLIGHT_ID_mvt")
    arr = arr.sort_values("MVT_TIME_UTC_mvt").drop_duplicates("FLIGHT_ID_mvt").set_index("FLIGHT_ID_mvt")
    both_idx = dep.index.intersection(arr.index)
    print(f"FLIGHT_ID_mvt with (>=1 DEP) and (>=1 ARR) row in ranking.parquet: {len(both_idx)}")

    d = dep.loc[both_idx]
    a = arr.loc[both_idx]

    same_route = (d["ADEP_mvt"].to_numpy() == a["ADEP_mvt"].to_numpy())
    print(f"ARR.ADEP == DEP.ADEP (same origin tag on both rows): {same_route.sum()} of {len(d)}")
    dest_is_dep_ades = (a["ADES_mvt"].to_numpy() == d["ADES_mvt"].to_numpy())
    print(f"ARR.ADES == DEP.ADES (same destination tag on both rows): {dest_is_dep_ades.sum()} of {len(d)}")

    gap = secs(a["MVT_TIME_UTC_mvt"].reset_index(drop=True), d["MVT_TIME_UTC_mvt"].reset_index(drop=True)).to_numpy()
    print(f"\ngap = ARR.MVT_TIME - DEP.MVT_TIME (seconds): "
          f"min={gap.min():.0f} p1={pd.Series(gap).quantile(.01):.0f} median={pd.Series(gap).median():.0f} "
          f"p99={pd.Series(gap).quantile(.99):.0f} max={gap.max():.0f}")
    print(f"gap <= 0 (arrival recorded at/before the linked departure event): {(gap <= 0).sum()} of {len(gap)}")
    print(f"gap in a plausible single-flight-duration window (0, 6h]: {((gap > 0) & (gap <= 21600)).sum()} of {len(gap)}")

    # the 4 train/rank FLIGHT_ID "overlaps" -- inspect directly, are these real or coincidence
    train_ids = set()
    for p in DEV_MONTH_FILES:
        train_ids |= set(pd.read_parquet(p, columns=["FLIGHT_ID_mvt"])["FLIGHT_ID_mvt"].dropna())
    rank_ids = set(rk["FLIGHT_ID_mvt"].dropna())
    overlap = train_ids & rank_ids
    print(f"\ntrain(dev)/rank FLIGHT_ID_mvt overlap: {sorted(overlap)}")
    print(f"train FLIGHT_ID_mvt range: [{min(train_ids):.0f}, {max(train_ids):.0f}]  n_unique={len(train_ids)}")
    print(f"rank  FLIGHT_ID_mvt range: [{min(rank_ids):.0f}, {max(rank_ids):.0f}]  n_unique={len(rank_ids)}")


if __name__ == "__main__":
    main()
