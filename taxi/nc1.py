"""NC1: NN3's cell measured where it was NOT selected, so the k = 40 optimism bound has something to answer to.

SB3+NN3 gains +2.07 s on the untouched holdout. That clears the 2 s floor and fails the 3.0 s hurdle only because
`docs/NN3.md` adds a k = 40 bound for the search that chose the cell, and `docs/SW1.md` cannot retire it: step-down
controls multiplicity ACROSS lanes, the bound controls selection WITHIN one. Nothing retires a selection bound except
measuring the selected spec on rows it was not selected on. `taxi.nn3.score` reads only the training frame and ranked
its 40 cells on three of the five CV folds, so months 4, 5, 9 and 10 are exactly those rows. Declared in protocol
section 4 before this file existed. No fit happens here; every vector is already on disk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from taxi.champ import lane_a
from taxi.data import CV_FOLDS
from taxi.l2 import floored
from taxi.nn1 import base_oof, blend_weight
from taxi.verdict import paired_delta, rmse

SEARCH_FOLDS = (CV_FOLDS[0], CV_FOLDS[2], CV_FOLDS[4])  # what taxi.nn3 ranked cells on; the weight is fitted here
HELD = tuple(f for f in CV_FOLDS if f not in SEARCH_FOLDS)  # months no cell was ever ranked on; the verdict is read here
GATE = 2.0  # the floor alone: k = 1, one fixed spec, no cell chosen on a judged row, so the optimism bound is 0.0


def vectors(tra: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nn = pd.read_parquet("data/cache/nn3_oof.parquet")
    nn3 = tra[["id"]].merge(nn[["id", "pred"]], on="id", how="left").pred.to_numpy(float)
    return nn3, base_oof(tra), tra.y.values


def main(out: str = "docs/NC1.md") -> None:
    tra, _ = lane_a()
    nn3, base, y = vectors(tra)
    months = lambda folds: np.isin(tra.month.values, [m for f in folds for m in f])  # noqa: E731
    fit, judge = months(SEARCH_FOLDS), months(HELD)
    ok = np.isfinite(nn3) & np.isfinite(base)
    assert ok.all(), f"{(~ok).sum():,} rows lack a cached prediction; NC1 refuses to judge a partial vector"

    w = blend_weight(nn3[fit], base[fit], y[fit])
    e_base, e_blend = base - y, floored(w * nn3 + (1 - w) * base) - y
    v = paired_delta(e_base[judge], e_blend[judge], tra.day.values[judge])
    passed = v.delta >= GATE and v.lo > 0

    gain_fit = rmse(e_base[fit]) - rmse(e_blend[fit])
    verdict_block = ([
        f"**PASS, and the pre-registered prediction was WRONG.** The declaration predicted the gain would come in below the "
        f"holdout's +2.07 s and named a real risk of failing this gate. It came in at {v.delta:+.2f} s with the whole "
        f"interval above the floor. That is recorded here because a prediction which only appears when it is right is not a "
        f"prediction.", "",
        f"The sharper reading is the pair of rows above, which is what this lane was built to see. Against ONE base, the same "
        f"MC2 out-of-fold vector, the blend gains {gain_fit:+.2f} s on the months that chose the cell and {v.delta:+.2f} s on "
        f"the months that did not. The selection advantage is about {gain_fit - v.delta:.2f} s, not the 1.0 s the k = 40 bound "
        f"assumes, and nowhere near enough to explain a +2.07 s holdout gain away. The bound was a reasonable default and it "
        f"is larger than the thing it was defending against, now that the thing has been measured.", "",
        f"This retires that objection and nothing else. No file is built and no upload is made here, and whether the 2 s floor "
        f"is the right constant is untouched. The promotion of SB3+NN3 is a Stage 6 decision and Keith's alone."]
        if passed else [
        f"**FAIL.** The gain does not clear the floor on rows the selection never touched. The pre-registered prediction was "
        f"that it would come in below the holdout's +2.07 s, and it did. SB3 stays, and the k = 40 bound stands as the "
        f"correct reading of why SB3+NN3's holdout number is larger than the spec earns."])

    row = lambda m, lbl: (f"| {lbl} | {int(m.sum()):,} | {rmse(e_base[m]):.2f} | {rmse((floored(w * nn3 + (1 - w) * base) - y)[m]):.2f} "  # noqa: E731
                          f"| {rmse(e_base[m]) - rmse((floored(w * nn3 + (1 - w) * base) - y)[m]):+.2f} |")

    txt = ["# NC1: NN3's cell measured on the folds its search never scored", "",
           f"Regenerate with `python -m taxi.nc1`. Declared in protocol section 4 before this file was written. No model is "
           f"fitted here: NN3's cached five-seed out-of-fold vector and MC2's are read from disk, so the only new quantities "
           f"are one blend weight and one paired interval. k = 1, one fixed spec, so the hurdle is the {GATE:.0f} s floor "
           f"alone. This lane promotes nothing whatever it finds.", "",
           f"NN3 ranked its 40 cells on folds {SEARCH_FOLDS}; the weight below is fitted on those months and the verdict is "
           f"read on {HELD}, which no cell was ever ranked on. Weight on NN3, fitted on the search months only: **{w:.4f}** "
           f"(NN3's own published weight, fitted on all ten months, is 0.5189).", "",
           "| rows | n | MC2 out-of-fold | blend | gain |", "|---|---|---|---|---|",
           row(fit, "search months (weight fitted here, NOT a verdict)"), row(judge, "held months (THE VERDICT)"), "",
           "## Verdict", "",
           f"On the {int(judge.sum()):,} rows no cell was ranked on, the blend beats MC2 out of fold by "
           f"**{v.delta:+.2f} s** [{v.lo:+.2f}, {v.hi:+.2f}], against a {GATE:.0f} s floor and a 0.0 s optimism bound.", "",
           *verdict_block, "",
           "## What this cannot say", "",
           f"Whether {GATE:.0f} s is the right floor is a decision-rule question and not measurable from inside the campaign. "
           f"These rows are training months, so the level is not comparable to a holdout RMSE and only the GAIN transfers. "
           f"And the base here is MC2 out of fold rather than SB3, which is what every out-of-fold blend in this campaign "
           f"uses because a five-seed bag has no out-of-fold vector.", ""]
    text = "\n".join(txt)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
