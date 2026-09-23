"""H1b item 3, final check: are the 4 train(dev)/rank FLIGHT_ID_mvt "overlaps" found in s6b a
genuine identity match (same flight instance, so the dev-2025 side's real label would leak the
rank-2026 row) or numeric coincidence in a shared id range? Cheap, high-value if real.
Run: PYTHONPATH=. python analysis/hunt/h1b/s6c_id_overlap_check.py
"""
from __future__ import annotations

import pandas as pd

from common import DEV_MONTH_FILES
from data import secs

OVERLAP_IDS = [294566689.0, 294566709.0, 294566991.0, 294567225.0]
COLS = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt",
        "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]


def main() -> None:
    train_hits = []
    for p in DEV_MONTH_FILES:
        df = pd.read_parquet(p, columns=COLS)
        hit = df[df["FLIGHT_ID_mvt"].isin(OVERLAP_IDS)]
        if len(hit):
            hit = hit.copy()
            hit["src"] = p
            train_hits.append(hit)
    train_hits = pd.concat(train_hits, ignore_index=True) if train_hits else pd.DataFrame()
    print("train(dev) rows for the 4 overlap ids:")
    print(train_hits[["FLIGHT_ID_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt",
                       "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "src"]].to_string())

    rk = pd.read_parquet("data/ranking.parquet", columns=COLS)
    rank_hits = rk[rk["FLIGHT_ID_mvt"].isin(OVERLAP_IDS)]
    print("\nrank(2026) rows for the 4 overlap ids:")
    print(rank_hits[["FLIGHT_ID_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt",
                      "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]].to_string())

    # identity test: same FLIGHT_mvt/CALLSIGN and a plausible (small) time gap => real match, not coincidence
    if len(train_hits) and len(rank_hits):
        m = train_hits.merge(rank_hits, on="FLIGHT_ID_mvt", suffixes=("_train", "_rank"))
        if len(m):
            m["same_flightnum"] = m["FLIGHT_mvt_train"] == m["FLIGHT_mvt_rank"]
            m["gap_days"] = secs(m["MVT_TIME_UTC_mvt_rank"], m["MVT_TIME_UTC_mvt_train"]) / 86400
            print("\nidentity check (train row vs rank row, same FLIGHT_ID):")
            print(m[["FLIGHT_ID_mvt", "FLIGHT_mvt_train", "FLIGHT_mvt_rank", "same_flightnum",
                     "PHASE_mvt_train", "PHASE_mvt_rank", "gap_days"]].to_string())


if __name__ == "__main__":
    main()
