"""Training-row features on the final build must equal the v53 build's; rank rows may differ only by the re-jittered stamps."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from features import ARR, CAT, DAY, NUM, OCC, WX, WZ

COLS = list(dict.fromkeys(NUM + DAY + OCC + WX + ARR + WZ))


def compare(a: pd.DataFrame, b: pd.DataFrame, label: str) -> None:
    m = a.merge(b, on="id", suffixes=("_a", "_b"))
    print(f"{label}: {len(a)} vs {len(b)} rows, {len(m)} matched")
    bad = {}
    for c in COLS:
        x, y = m[f"{c}_a"].astype(float).values, m[f"{c}_b"].astype(float).values
        diff = ~((x == y) | (np.isnan(x) & np.isnan(y)) | np.isclose(x, y, rtol=1e-9, atol=1e-9))
        if diff.any():
            bad[c] = int(diff.sum())
    for c in CAT:
        diff = m[f"{c}_a"].astype(str).values != m[f"{c}_b"].astype(str).values
        if diff.any():
            bad[c] = int(diff.sum())
    print(f"  columns differing: {bad if bad else 'none'}")


if __name__ == "__main__":
    old = pd.read_parquet(sys.argv[1])
    new = pd.read_parquet("data/cache/feat.parquet")
    compare(old[old["split"] == "train"], new[new["split"] == "train"], "train")
    ro, rn = old[old["split"] == "rank"], new[new["split"] == "rank"]
    compare(ro, rn[rn["id"].isin(ro["id"])], "rank Jan+Jul")
    print("rank months:", rn["month"].value_counts().sort_index().to_dict(), "lane B:", int((rn["has_plan"] == 0).sum()))
