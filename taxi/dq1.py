"""DQ1 (protocol section 4): lane A with the hand-set 600 s split removed, three losses, k = 3.

`taxi.m2.TAIL` is a constant a human chose and every lane-A champion since M2 has inherited it. Under squared error the
best prediction is the conditional MEAN, so a model that estimates the conditional distribution and takes its mean needs
no split. The three arms share columns, cell and stopping, so they differ in the loss alone.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from taxi.champ import COLS_D2, lane_a
from taxi.l2 import floored
from taxi.m3 import cached
from taxi.mc1 import CELL, MAX_ITER, fit, pool
from taxi.sb3 import predict as sb3_predict
from taxi.verdict import paired_delta, rmse
from taxi.w1 import verdict_line

CACHE, VAL_FRAC, BAG_CREDIT, BUDGET_S = "data/cache/dq1", 0.10, 1.07, 6 * 3600
GRID = {20: tuple((i - 0.5) / 20 for i in range(1, 21)), 10: tuple((i - 0.5) / 10 for i in range(1, 11))}


def split(tra: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forward-in-time stopping split: the last 10% of training DAYS, never a random slice and never the holdout."""
    d = np.sort(pd.unique(tra.day))
    m = tra.day.values >= d[int(len(d) * (1 - VAL_FRAC))]
    return tra[~m], tra[m]


def one(tag: str, tra: pd.DataFrame, tea: pd.DataFrame, loss: str) -> pd.DataFrame:
    """One single-stage fit of `c` at CELL on D2's columns; returns the cached holdout prediction of `c`."""
    def make() -> tuple[np.ndarray, float, float]:
        f, va = split(tra)
        t0 = time.time()
        m = fit("reg", f, f.c.values, (va, va.c.values), MAX_ITER, COLS_D2, loss=loss)
        p = np.asarray(m.predict(pool(tea, COLS_D2)))
        # m3.cached names its third slot vram; here it carries the tree count, which is what this lane needs recorded.
        return (p[:, 0] if p.ndim == 2 else p), time.time() - t0, float(m.tree_count_)
    c = cached(f"{CACHE}/{tag}.parquet", tea.id.values, make)
    print(f"| {tag} | {loss} | {c.vram.iloc[0]:.0f} | {c.dt.iloc[0] / 60:.1f} |", flush=True)
    return c


def main(out: str = "docs/DQ1.md") -> None:
    tra, tea = lane_a()
    y, z = tea.y.values, tea.z_aobt.values
    e0 = sb3_predict(tra, tea) - y

    print("| arm | loss | trees | min |\n|---|---|---|---|", flush=True)
    ctl = one("c", tra, tea, "RMSE")
    gau = one("g", tra, tea, "RMSEWithUncertainty")

    per = max(ctl.dt.iloc[0], gau.dt.iloc[0])  # the slower of the two arms already measured, never an assumed rate
    left = BUDGET_S - (ctl.dt.iloc[0] + gau.dt.iloc[0])
    k = next((n for n in sorted(GRID, reverse=True) if n * per <= left), 0)  # declared TIME trigger, never a result trigger
    knots = [one(f"q{a:.4f}", tra, tea, f"Quantile:alpha={a:.4f}") for a in GRID.get(k, ())]
    qc = np.mean([q.pred.values for q in knots], axis=0) if knots else None

    arms = {"DQ1c (RMSE, no split)": ctl.pred.values, "DQ1g (Gaussian head)": gau.pred.values}
    if qc is not None:
        arms[f"DQ1q ({k} quantiles)"] = qc
    skipped = ("" if qc is not None else
               f"**DQ1q NOT RUN.** The two arms above cost {(ctl.dt.iloc[0] + gau.dt.iloc[0]) / 3600:.1f} h of the declared "
               f"{BUDGET_S / 3600:.0f} h, and the slowest of them prices even the {min(GRID)}-knot rung at "
               f"{min(GRID) * per / 3600:.1f} h against {max(left, 0) / 3600:.1f} h left. The TIME trigger fired all the way "
               f"down the ladder, so the quantile arm is a NOT-DONE row with a measured reason, not a silent omission.")
    rows, verd = [], []
    for name, c in arms.items():
        e = floored(c + z) - y
        v = paired_delta(e0, e, tea.day.values)
        rows.append(f"| {name} | {rmse(e):.2f} | {c.mean():+.1f} | {tea.c.values.mean():+.1f} | {c.mean() - tea.c.values.mean():+.1f} |")
        verd.append(f"- **{name}**: " + verdict_line(name.split()[0], "SB3", v, k=3))
    pair = [f"| {n} minus DQ1c | {rmse(floored(arms[n] + z) - y) - rmse(floored(ctl.pred.values + z) - y):+.2f} |"
            for n in list(arms)[1:]]

    text = "\n".join([
        "# DQ1: the conditional distribution instead of the hand-set 600 s split, lane A, Jan + Jul 2025 holdout", "",
        f"Regenerate with `python -m taxi.dq1`. Three arms, k = 3, protocol section 4. All three fit `c` in ONE stage on D2's "
        f"{len(COLS_D2)} columns at `taxi.mc1.CELL` unchanged, up to {MAX_ITER} iterations, early-stopped on the last "
        f"{VAL_FRAC:.0%} of training DAYS, with the plan gap added back and the C1 floor applied. They differ in the LOSS alone. "
        f"Comparator SB3, {rmse(e0):.4f} s. The arms are single-seed against a five-seed champion bag, which `docs/SEEDS.md` "
        f"prices at {BAG_CREDIT} s; any arm inside that of SB3 is re-run bagged rather than called a tie.", "",
        "## Arms", "", "| arm | holdout RMSE | mean predicted c | mean realised c | midpoint-rule bias |",
        "|---|---|---|---|---|", *rows, "",
        "## The loss, isolated", "",
        "DQ1c carries both the removed split and the changed stopping, so it is not a clean reading of either. The arm-to-arm "
        "contrasts are, because the three share everything but the loss:", "",
        "| contrast | RMSE change (negative is better) |", "|---|---|", *pair, "",
        "## Verdicts against SB3", "", *verd, "", skipped])
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
