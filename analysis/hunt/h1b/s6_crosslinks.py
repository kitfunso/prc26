"""H1b item 3 (cross-links) + parts of item 6 (nulls): the cheap, high-value checks first --
does BLOCK_TIME/TAXITIME ever leak non-null on a rank row (any phase), do ids repeat across files,
do exact-duplicate rows exist, and does a same-flight ARR row in the SAME file carry a real block/in
time that a DEP row's off-block could be read off from. Uses ALL 12 raw training files + ranking.parquet
(id/structure only, no label use on rank -- this is the leak-integrity check itself).
Run: PYTHONPATH=. python analysis/hunt/h1b/s6_crosslinks.py
"""
from __future__ import annotations

import pandas as pd

from common import DEV_MONTH_FILES

# Dev months only + the 2026 rank file. Jan/Jul 2025 (holdout) are deliberately excluded from this
# discovery pass -- see REPORT.md methodology note on the holdout-read cap.
ALL_FILES = DEV_MONTH_FILES + ["data/ranking.parquet"]
COLS = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt",
        "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]


def main() -> None:
    frames = []
    for p in ALL_FILES:
        df = pd.read_parquet(p, columns=COLS)
        df["src"] = p
        frames.append(df)
        tag = "RANK" if "ranking" in p else "train"
        print(f"{p}: n={len(df)}  BLOCK non-null: DEP={df.loc[df.PHASE_mvt=='DEP','BLOCK_TIME_UTC_mvt'].notna().sum()} "
              f"ARR={df.loc[df.PHASE_mvt=='ARR','BLOCK_TIME_UTC_mvt'].notna().sum()}  "
              f"TAXITIME non-null: DEP={df.loc[df.PHASE_mvt=='DEP','TAXITIME_SEC_mvt'].notna().sum()} "
              f"ARR={df.loc[df.PHASE_mvt=='ARR','TAXITIME_SEC_mvt'].notna().sum()}  [{tag}]")

    d = pd.concat(frames, ignore_index=True)
    print(f"\ntotal rows all files: {len(d)}")

    # A) leak check on the rank file specifically, both phases
    rk = d[d["src"].str.contains("ranking")]
    leak = rk[rk["BLOCK_TIME_UTC_mvt"].notna() | rk["TAXITIME_SEC_mvt"].notna()]
    print(f"\nranking.parquet rows with a non-null BLOCK_TIME or TAXITIME (either phase): {len(leak)} of {len(rk)}")
    if len(leak):
        print(leak[["MVT_ID_mvt", "PHASE_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]].head(20))

    # B) MVT_ID_mvt duplicated across files (or within one)
    dup_id = d["MVT_ID_mvt"].value_counts()
    dup_id = dup_id[dup_id > 1]
    print(f"\nMVT_ID_mvt values appearing more than once across all 13 files: {len(dup_id)}")
    if len(dup_id):
        ex = d[d["MVT_ID_mvt"].isin(dup_id.index[:5])].sort_values("MVT_ID_mvt")
        print(ex[["MVT_ID_mvt", "PHASE_mvt", "src", "MVT_TIME_UTC_mvt"]])

    # C) exact duplicate (FLIGHT_ID, PHASE, MVT_TIME) rows, within-file and cross-file
    key = d.groupby(["FLIGHT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"]).size()
    key = key[key > 1]
    print(f"\n(FLIGHT_ID, PHASE, MVT_TIME) combos appearing more than once anywhere: {len(key)} "
          f"(rows involved: {int(key.sum())})")

    # D) train/rank FLIGHT_ID_mvt overlap (2025 vs 2026 -- should be none)
    train_fid = set(d.loc[~d["src"].str.contains("ranking"), "FLIGHT_ID_mvt"].dropna())
    rank_fid = set(rk["FLIGHT_ID_mvt"].dropna())
    print(f"\nFLIGHT_ID_mvt overlap train(2025) vs rank(2026): {len(train_fid & rank_fid)}")

    # E) within ranking.parquet: flights with both a DEP and an ARR row -- does the ARR side ever
    # carry real timing that the DEP side lacks?
    both = rk.groupby("FLIGHT_ID_mvt")["PHASE_mvt"].apply(lambda s: set(s))
    both = both[both.apply(lambda s: {"DEP", "ARR"} <= s)]
    print(f"\nrank FLIGHT_ID_mvt with both a DEP and an ARR row in ranking.parquet: {len(both)}")
    if len(both):
        sample = rk[rk["FLIGHT_ID_mvt"].isin(both.index[:5])].sort_values(["FLIGHT_ID_mvt", "PHASE_mvt"])
        print(sample[["FLIGHT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt",
                       "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]])


if __name__ == "__main__":
    main()
