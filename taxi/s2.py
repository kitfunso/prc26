"""S2/S3: the ranking file from all 12 months of 2025 with the lane champions (lane A by name, lane B B4). Build only; upload is a Keith gate."""
from __future__ import annotations

import hashlib
import sys

import numpy as np
import pandas as pd

from taxi.baselines import b4
from taxi.champ import CHAMPIONS_A
from taxi.data import HOLDOUT_MONTHS, TRAIN_MONTHS, features, load
from taxi.m1 import chosen_cells, final_rounds
from taxi.m2 import predict_m2, stages
from taxi.submit import build_submission
from taxi.w1 import coverage
from taxi.weather import join, load_metar

RANKING = "data/ranking.parquet"


def main(out: str, champ: str) -> None:
    cols, tag, doc = CHAMPIONS_A[champ]
    rounds = final_rounds(tag, doc)
    metar = load_metar()
    tr, rk = (join(d, metar) for d in (load(TRAIN_MONTHS + HOLDOUT_MONTHS), features([RANKING])))
    print(coverage(rk, "ranking"), flush=True)
    cell, _ = chosen_cells()["A"]
    pred = pd.Series(np.nan, index=rk.id.values)
    for lane in ("A", "B"):
        m = rk.has_fpl.values == (lane == "A")
        trl, rkl = tr[tr.has_fpl == (lane == "A")].reset_index(drop=True), rk[m].reset_index(drop=True)
        p = predict_m2(stages(trl, cell, rounds=rounds, cols=cols), rkl)[0] if lane == "A" else b4(trl, rkl, lane)
        pred[rk.id.values[m]] = p
        print(f"lane {lane}: train rows {len(trl)}, ranking rows {len(rkl)}, train y mean {trl.y.mean():.1f}, "
              f"pred mean {p.mean():.1f}, pred p50 {np.median(p):.1f}, pred p99 {np.quantile(p, .99):.1f}, "
              f"negative preds {int((p < 0).sum())}", flush=True)
    sub = build_submission(pred, out)
    digest = hashlib.sha256(open(out, "rb").read()).hexdigest()
    print(f"{out}: {len(sub)} rows, sha256 {digest}, spec lane A {champ} {cell} rounds {rounds} on {len(cols)} columns, lane B B4")


if __name__ == "__main__":
    main(sys.argv[1] if sys.argv[1:] else "data/merry-octopus_v11.parquet", sys.argv[2] if sys.argv[2:] else "D1")
