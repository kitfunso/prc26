"""S1: the ranking file from all 12 months of 2025 with the frozen lane champions (protocol section 4)."""
from __future__ import annotations

import hashlib
import sys

import numpy as np
import pandas as pd

from taxi.baselines import b4
from taxi.data import HOLDOUT_MONTHS, TRAIN_MONTHS, features, load
from taxi.m1 import chosen_cells, predict, train
from taxi.submit import build_submission

RANKING = "data/ranking.parquet"


def main(out: str) -> None:
    tr, rk = load(TRAIN_MONTHS + HOLDOUT_MONTHS), features([RANKING])
    cell, rounds = chosen_cells()["A"]
    pred = pd.Series(np.nan, index=rk.id.values)
    for lane in ("A", "B"):
        m = rk.has_fpl.values == (lane == "A")
        trl, rkl = tr[tr.has_fpl == (lane == "A")].reset_index(drop=True), rk[m].reset_index(drop=True)
        p = predict(train(trl, lane, cell, rounds=rounds), rkl, lane) if lane == "A" else b4(trl, rkl, lane)
        pred[rk.id.values[m]] = p
        print(f"lane {lane}: train rows {len(trl)}, ranking rows {len(rkl)}, train y mean {trl.y.mean():.1f}, "
              f"pred mean {p.mean():.1f}, pred p50 {np.median(p):.1f}, pred p99 {np.quantile(p, .99):.1f}, "
              f"z_anchor mean train {trl.z_anchor.mean():.1f} ranking {rkl.z_anchor.mean():.1f}", flush=True)
    sub = build_submission(pred, out)
    digest = hashlib.sha256(open(out, "rb").read()).hexdigest()
    print(f"{out}: {len(sub)} rows, sha256 {digest}, spec lane A M1 {cell} {rounds} rounds, lane B B4")


if __name__ == "__main__":
    main(sys.argv[1] if sys.argv[1:] else "data/merry-octopus_v1.parquet")
