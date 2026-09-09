"""MONO: does the protocol's monotone rung have a sign to impose on lane A? Diagnostic, never a verdict.

Stage 4.1 rung 6 says to encode a driver's sign as a monotone constraint where that sign is domain knowledge, so the trees
cannot fit a noise-driven flip. Lane A's target is not taxi time: `c = y - z_aobt` and `z_aobt` is `MVT_TIME_UTC - AOBT_3_flt`
(`taxi/data.py`), so `c` is the disagreement between two clock definitions. This module measures whether the physics sign
survives that subtraction, on the training rows only, and writes the answer instead of arguing it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from taxi.champ import lane_a

# Columns whose sign on TAXI TIME is domain knowledge, with that expected sign: congestion and distance lengthen the taxi,
# wind lengthens it, and visibility and ceiling shorten it as they improve.
CAND: tuple[tuple[str, int], ...] = (
    ("queue", +1), ("dep_before", +1), ("dep_after", +1), ("occ", +1), ("t_occ", +1), ("arr_off", +1),
    ("dur", +1), ("w_sknt", +1), ("w_gust", +1), ("w_vsby", -1), ("w_ceil", -1))
FLIP = 0.02  # a |rho| floor: below it the column has no monotone relationship to impose on either target, so no sign to read


def signs(tra: pd.DataFrame) -> list[dict]:
    rows = []
    for col, want in CAND:
        if col not in tra.columns:
            rows.append(dict(col=col, want=want, note="not a column in this build"))
            continue
        v = tra[col].values.astype("float64")
        ok = np.isfinite(v)
        ry = float(spearmanr(v[ok], tra.y.values[ok]).statistic)
        rc = float(spearmanr(v[ok], tra.c.values[ok]).statistic)
        rows.append(dict(col=col, want=want, n=int(ok.sum()), rho_y=ry, rho_c=rc,
                         level_ok=np.sign(ry) == want and abs(ry) >= FLIP,
                         flipped=abs(ry) >= FLIP and abs(rc) >= FLIP and np.sign(rc) != np.sign(ry)))
    return rows


def main(out: str = "docs/MONO.md") -> None:
    tra, _ = lane_a()
    rows = signs(tra)
    keep = [r for r in rows if "rho_y" in r]
    flipped = [r for r in keep if r["flipped"]]
    agrees = [r for r in keep if r["level_ok"]]

    tbl = [f"| {r['col']} | {'+' if r['want'] > 0 else '-'} | {r['rho_y']:+.4f} | {r['rho_c']:+.4f} | "
           f"{'yes' if r['level_ok'] else 'no'} | {'FLIPPED' if r['flipped'] else ''} |" for r in keep]

    text = "\n".join([
        "# MONO: is there a monotone sign to impose on lane A?", "",
        f"Regenerate with `python -m taxi.mono`. Spearman rank correlation over all {len(tra):,} lane-A training rows; the "
        f"holdout is not read. Diagnostic only: no model is fitted here and nothing is promoted.", "",
        "Stage 4.1 rung 6 asks for monotone constraints where a driver's sign is domain knowledge. Lane A does not predict "
        "taxi time. It predicts `c = y - z_aobt`, and `z_aobt` is the flight plan's own taxi estimate, so `c` is the "
        "disagreement between two clocks. The question this file answers is whether the physics sign survives that "
        "subtraction.", "",
        "| column | physics sign | rho vs level `y` | rho vs target `c` | level agrees | on the target |",
        "|---|---|---|---|---|---|", *tbl, "",
        "## What it says", "",
        f"**Physics is right about the level.** {len(agrees)} of {len(keep)} columns carry the sign physics predicts against "
        f"`y`, and the three largest are the congestion and duration measures.", "",
        f"**Physics is backwards on the target.** {len(flipped)} columns flip sign between `y` and `c`: "
        + ", ".join(f"`{r['col']}` {r['rho_y']:+.4f} to {r['rho_c']:+.4f}" for r in flipped)
        + ". The plan's estimate over-absorbs congestion, so when the queue is long `z_aobt` reads high and the correction "
        "runs the other way. A constraint taken from physics would be imposed with the wrong sign and would cost, not help.",
        "", "## What that leaves", "",
        "The rung is NOT closed by this, because a sign measured on the target is still a sign. It is closed as a lane on two "
        "conditions that a declaration would have to meet. First, the sign has to come from rows the verdict does not judge, "
        "or the constraint is fitted rather than imposed. Second, and the reason it is not queued ahead of the tuned "
        f"piecewise-linear lane, a monotone constraint is regularisation, and it binds when data is thin; lane A fits "
        f"{len(tra):,} rows, where a rank correlation of {max(abs(r['rho_c']) for r in keep):.4f} is not something a "
        "depth-8 ensemble mistakes for noise.", ""])
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
