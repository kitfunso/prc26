"""SB2: do three parked positive ties add?

SB1 (five-seed bag, +1.0 s), ST1a (tail-gated weight toward D1, +0.6 s) and C1's floor (+0.5 s) each clear zero across their whole interval and each
sit under the 2 s promotion floor alone. Every constant here was fixed by one of those runs, so this composes them and takes one verdict at k = 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from taxi.champ import champion_holdout, lane_a
from taxi.l2 import floored
from taxi.m2 import is_tail
from taxi.verdict import paired_delta, rmse
from taxi.w1 import verdict_line

SEEDS = ("data/cache/mc2_holdout.parquet", *(f"data/cache/sb1_seed{i}_holdout.parquet" for i in range(1, 5)))
W_TAIL, W_BODY = 1.0, 0.7


def bag(tea: pd.DataFrame) -> np.ndarray:
    """The mean of MC2's five seed predictions, each checked against the holdout row order before it is used."""
    out = []
    for path in SEEDS:
        c = pd.read_parquet(path)
        assert np.array_equal(c.id.values, tea.id.values), path
        out.append(c.pred.values)
    return np.mean(out, axis=0)


def main() -> None:
    raise RuntimeError("SB2 is withdrawn (H53): its ST1a gate reads is_tail(tea), which is c = y - z_aobt, the predicted row's own label. "
                       "docs/SB2.md keeps the contaminated numbers as history.")
    tra, tea = lane_a()
    y, hd = tea.y.values, champion_holdout("D1", tra, tea)
    base = floored(champion_holdout("MC2", tra, tea))
    b = bag(tea)
    w = np.where(is_tail(tea), W_TAIL, W_BODY)
    steps = {"MC2 floored (champion)": base, "seed bag": floored(b), "seed bag + ST1a gate": floored(w * b + (1 - w) * hd)}
    sb2 = steps["seed bag + ST1a gate"]
    v = paired_delta(base - y, sb2 - y, tea.day.values)
    vt, jan, jul = is_tail(tea), tea.month.values == 1, tea.month.values == 7
    row = lambda n, p: f"| {n} | {rmse(p - y):.2f} | {rmse((p - y)[jan]):.1f} | {rmse((p - y)[jul]):.1f} | {rmse((p - y)[vt]):.1f} | {rmse((p - y)[~vt]):.1f} |"  # noqa: E731

    out = ["# SB2: three parked ties composed, Jan + Jul 2025 development holdout", "",
           "Regenerate with `python -m taxi.sb2`. Every constant was fixed by an earlier declared run (SB1's five seeds, ST1a's out-of-fold weights "
           "1.0 on tail rows and 0.7 on body rows, C1's floor at 0 s), so nothing is chosen here and k = 1. All predictions are cached; no GPU.", "",
           f"## Each step added, on the same {len(tea):,} rows", "",
           "| model | RMSE | Jan | Jul | tail rows | body rows |", "|---|---|---|---|---|---|"]
    out += [row(n, p) for n, p in steps.items()]
    out += ["", verdict_line("SB2", "MC2", v, k=1)]
    text = "\n".join(out) + "\n"
    with open("docs/SB2.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
