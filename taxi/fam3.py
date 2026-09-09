"""FAM3: the last EMPTY row in the model-class register, lane A.

`taxi/register.py` marked explainable boosting (GA2M) empty for one reason only, that `interpret` was not installed here. That is a
closure by argument, which is the shape Stage 2 says reads as "passed". One arm, one fixed cell, FAM2's harness unchanged.
"""
from __future__ import annotations

import gc
import time

import numpy as np
import pandas as pd

from taxi.champ import lane_a
from taxi.fam1 import BUDGET_S, CATCOLS, FOLDS, GATE, NUMS, _tree_block, oof, score_arm
from taxi.l2 import floored
from taxi.nn1 import base_oof
from taxi.sb3 import predict as sb3_predict
from taxi.verdict import paired_delta, rmse
from taxi.w1 import verdict_line

K = 1
CELL = dict(interactions=32, outer_bags=1, max_bins=256, random_state=0)
OOF = "data/cache/fam3_ebm_oof.parquet"


def arm_ebm(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    """GA2M: additive in every single feature, plus a fixed budget of pairwise terms and nothing deeper."""
    from interpret.glassbox import ExplainableBoostingRegressor

    a, b = _tree_block(tr, te)
    types = ["continuous"] * len(NUMS) + ["nominal"] * len(CATCOLS)
    m = ExplainableBoostingRegressor(feature_types=types, n_jobs=-1, **CELL).fit(a, tr.c.values)
    del a
    gc.collect()
    return np.asarray(m.predict(b)).ravel()


def main(out: str = "docs/FAM3.md") -> None:
    tra, tea = lane_a()
    t0 = time.time()
    print("| arm | own RMSE | blend | MC2 base | agreement | min |\n|---|---|---|---|---|---|", flush=True)
    p, spent, note = oof(tra, arm_ebm, BUDGET_S)
    pd.DataFrame({"id": tra.id.values, "pred": p}).to_parquet(OOF, index=False)
    r = score_arm(tra, base_oof(tra), "ebm", "explainable boosting (GA2M)", p, spent, note)
    gain = r["base"] - r["blend"]

    verd = []
    if gain >= GATE:
        y, sb3 = tea.y.values, sb3_predict(tra, tea)
        lvl = floored(arm_ebm(tra, tea) + tea.z_aobt.values)
        v = paired_delta(sb3 - y, floored(r["w"] * lvl + (1 - r["w"]) * sb3) - y, tea.day.values)
        verd = [f"- **FAM3-ebm + SB3** at w={r['w']:.4f}, holdout {rmse(lvl - y):.2f} s alone: "
                + verdict_line("FAM3-ebm", "SB3", v, k=K)]

    text = "\n".join([
        "# FAM3: explainable boosting, the register's last EMPTY row, lane A screen", "",
        f"Regenerate with `python -m taxi.fam3`. Declared in protocol section 4 before the code was written. One arm, one FIXED "
        f"cell {CELL}, no search, k = {K}. Folds are the three declared month pairs {FOLDS}; the comparator is MC2's cached "
        f"out-of-fold vector on those rows, the same one FAM1 and FAM2 used. Total {(time.time() - t0) / 60:.0f} min.", "",
        "`interpret` 0.7.8 is now installed, which is the only thing that ever kept this row empty. A GA2M is additive in every "
        "single feature plus a fixed budget of pairwise terms, so it is the one named family whose function class the champion's "
        "depth-8 boosting cannot contain.", "",
        "## The screen", "",
        f"Gate declared in advance and identical to FAM1 and FAM2: the arm meets the untouched holdout only if its blend with MC2 "
        f"beats MC2 alone by {GATE:.1f} s on these rows. A failure is a closed register row and costs no holdout fit.", "",
        "| arm | family | own RMSE | blend with MC2 | gain over MC2 | unclipped weight | agreement | outcome |",
        "|---|---|---|---|---|---|---|---|",
        f"| FAM3-ebm | {r['label']} | {r['level']:.2f} | {r['blend']:.2f} | {gain:+.2f} | {r['raw']:+.4f} | {r['agree']:.4f} | "
        f"{'PASS' if gain >= GATE else 'closed at this cell'}{'; ' + r['note'] if r['note'] else ''} |", "",
        ("## Verdict against SB3" if verd else "## The arm failed the declared gate"), "",
        *(verd or [f"The blend with MC2 buys {gain:+.2f} s against a {GATE:.1f} s gate, so the arm never met the holdout and the "
                   f"register row closes by measurement rather than by a missing library."]), ""])
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
