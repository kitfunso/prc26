"""S2/S3: the ranking file from all 12 months of 2025 with the lane champions (lane A by name or from a parent file, lane B the registry's current).
Build only; upload is a Keith gate."""
from __future__ import annotations

import hashlib
import sys

import numpy as np
import pandas as pd

from taxi import l6, l8
from taxi.champ import CHAMPIONS_A, COMPOSED_A, CURRENT, predict_b, refit_predict
from taxi.data import HOLDOUT_MONTHS, TRAIN_MONTHS, features, load
from taxi.m1 import final_rounds
from taxi.submit import build_submission
from taxi.w1 import coverage
from taxi.weather import join, load_metar

RANKING = "data/ranking.parquet"


def main(out: str, champ: str, floor: float | None = None, parent: str | None = None) -> None:
    """parent: a built file whose lane-A values are reused by id (GPU refits do not reproduce), so only lane B is fit."""
    # A composed champion (SB3) has no single cell or round count: its parts each carry their own, so only its module is quoted.
    cols, tag, doc, kind = ((), champ, "", "composed") if champ in COMPOSED_A else CHAMPIONS_A[champ]
    rounds = final_rounds(tag, doc) if parent is None and doc else None
    if parent is not None:
        pp = pd.read_parquet(parent)
        lane_a = pd.Series(pp.TAXITIME_SEC_mvt.values.astype("float64"), index=pp.MVT_ID_mvt.values)
    metar = load_metar()
    tr, rk = (join(d, metar) for d in (load(TRAIN_MONTHS + HOLDOUT_MONTHS), features([RANKING])))
    print(coverage(rk, "ranking"), flush=True)
    pred = pd.Series(np.nan, index=rk.id.values)
    for lane in ("A", "B"):
        m = rk.has_fpl.values == (lane == "A")
        trl, rkl = tr[tr.has_fpl == (lane == "A")].reset_index(drop=True), rk[m].reset_index(drop=True)
        if lane == "B":
            p = predict_b(CURRENT["B"], trl, rkl)
            if CURRENT["B"] in l6.CELLS or CURRENT["B"] in l8.CELLS:
                print(f"lane B gate: non-copy training rows per band {l6.gate_counts(trl, rkl)}, mixture where >= {l6.MIN_NONCOPY}", flush=True)
        else:
            p = refit_predict(champ, trl, rkl) if parent is None else lane_a.reindex(rkl.id.values).values
            if parent is not None and np.isnan(p).any():
                raise ValueError(f"{int(np.isnan(p).sum())} lane-A ids missing from {parent}")
        pred[rk.id.values[m]] = p
        print(f"lane {lane}: train rows {len(trl)}, ranking rows {len(rkl)}, train y mean {trl.y.mean():.1f}, "
              f"pred mean {p.mean():.1f}, pred p50 {np.median(p):.1f}, pred p99 {np.quantile(p, .99):.1f}, "
              f"negative preds {int((p < 0).sum())}", flush=True)
    sub = build_submission(pred, out, floor=floor)
    digest = hashlib.sha256(open(out, "rb").read()).hexdigest()
    src = f"from {parent}" if parent else (COMPOSED_A[champ] if champ in COMPOSED_A else f"rounds {rounds} on {len(cols)} columns")
    print(f"{out}: {len(sub)} rows, sha256 {digest}, spec lane A {champ} ({kind}) {src}, lane B {CURRENT['B']}, floor {floor}")


if __name__ == "__main__":
    main(sys.argv[1] if sys.argv[1:] else "data/merry-octopus_v11.parquet", sys.argv[2] if sys.argv[2:] else "D1",
         float(sys.argv[3]) if sys.argv[3:] else None, sys.argv[4] if sys.argv[4:] else None)
