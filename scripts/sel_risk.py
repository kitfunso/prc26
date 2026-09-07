"""Diagnostic, no verdict: can a label-free learned risk model beat raw prediction at picking lane B's biggest errors?

Writes data/sel_risk.parquet so taxi.pb can report the row without refitting; the answer is no, and the row records that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from taxi.cb1 import lane_b
from taxi.champ import CURRENT, predict_b
from taxi.l2 import floored
from taxi.pb import N_RANK, V15_BOARD
from taxi.probes import groups

OUT = "data/sel_risk.parquet"
CAT = ["apt", "runway", "stand", "actype", "wake", "segment", "operator", "ftype", "frule", "ades"]
KS = (5, 10, 20, 40, 75)


def frame(d: pd.DataFrame, pred: np.ndarray, cols: list[str]) -> pd.DataFrame:
    x = d[cols].copy()
    for c in CAT:
        x[c] = x[c].astype("category")
    x["pred"] = pred
    return x


def main() -> None:
    trb, teb = lane_b()
    p_te = floored(predict_b(CURRENT["B"], trb, teb))
    e_te = p_te - teb.y.values
    sse = float((e_te ** 2).sum())

    oof = np.full(len(trb), np.nan)
    for m in sorted(trb.month.unique()):
        hold = trb.month.values == m
        oof[hold] = floored(predict_b(CURRENT["B"], trb[~hold], trb[hold]))
    print(f"OOF built on {len(trb):,} lane-B training rows, nan {int(np.isnan(oof).sum())}")

    cols = [c for c in trb.columns if c not in {"id", "y"} and trb[c].dtype.kind in "ifb"] + CAT
    g = LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=0)
    g.fit(frame(trb, oof, cols), np.log1p(np.abs(oof - trb.y.values)), categorical_feature=CAT)
    risk = g.predict(frame(teb, p_te, cols))
    pd.DataFrame({"id": teb.id.values, "risk": risk}).to_parquet(OUT, index=False)

    n_board_b = len(groups()["laneB"])

    def board(rem: float) -> float:
        return V15_BOARD - float(np.sqrt((N_RANK * V15_BOARD ** 2 - rem * n_board_b / len(e_te)) / N_RANK))

    print(f"\n{'selector':30s} " + " ".join(f"{'k=' + str(k):>13s}" for k in KS))
    for name, sc in (("prediction", p_te), ("predicted |error|", risk), ("oracle |error|", np.abs(e_te))):
        o = np.argsort(-np.asarray(sc, dtype="float64"))
        cells = [f"{float((e_te[o[:k]] ** 2).sum()) / sse:5.1%}/{board(float((e_te[o[:k]] ** 2).sum())):5.2f}" for k in KS]
        print(f"{name:30s} " + " ".join(f"{c:>13s}" for c in cells))
    sp = pd.Series(risk).corr(pd.Series(np.abs(e_te)), method="spearman")
    print(f"\nSpearman(risk, |e|) = {sp:.4f}; wrote {OUT}")


if __name__ == "__main__":
    main()
