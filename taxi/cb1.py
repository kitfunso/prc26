"""CB1: the lane-B promotion chain judged end to end, and the L8c step's robustness to the extreme rows.

Answers three gaps the Stage 5.7 critique found in the S4 gate: no verdict of the current champion against the baseline the chain
started from (each link priced its own k, the chain never did), no drop-k table for the L8c step (only for L6a), and no forward-fold
row for L8c (`taxi.fwd` writes the champion current when it last ran).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from taxi.baselines import b4
from taxi.champ import predict_b
from taxi.data import HOLDOUT_MONTHS, TRAIN_MONTHS, load
from taxi.l2 import floored
from taxi.m1 import expected_max
from taxi.verdict import paired_delta, rmse

CHAIN = ("B4", "L4b", "L6a", "L8c")
DROPS = (0, 1, 2, 5, 20)
CHAIN_CELLS = 25  # every lane-B cell declared since B4 (protocol section 4 lanes L1-L9), priced once for the whole chain


def lane_b() -> tuple[pd.DataFrame, pd.DataFrame]:
    return tuple(d[~d.has_fpl].reset_index(drop=True) for d in (load(TRAIN_MONTHS), load(HOLDOUT_MONTHS)))


def errors(trb: pd.DataFrame, teb: pd.DataFrame) -> dict[str, np.ndarray]:
    y = teb.y.values
    return {n: floored(b4(trb, teb, "B") if n == "B4" else predict_b(n, trb, teb)) - y for n in CHAIN}


def drop_table(teb: pd.DataFrame, e0: np.ndarray, e1: np.ndarray, a: str, b: str) -> list[str]:
    """RMSE of both models after removing the k rows the CHAMPION misses worst, the ordering LB-CONC's control rows reproduce."""
    order = np.argsort(np.abs(e0))[::-1]
    out = [f"| rows dropped | n | {a} | {b} | improvement [95%] |", "|---|---|---|---|---|"]
    for k in DROPS:
        keep = np.ones(len(teb), bool)
        keep[order[:k]] = False
        v = paired_delta(e0[keep], e1[keep], teb.day.values[keep])
        out.append(f"| {k} | {int(keep.sum())} | {v.champion:.1f} | {v.challenger:.1f} | {v.delta:+.1f} [{v.lo:+.1f}, {v.hi:+.1f}] |")
    return out


def main() -> None:
    trb, teb = lane_b()
    e = errors(trb, teb)
    days = teb.day.values
    chain = paired_delta(e["B4"], e["L8c"], days)
    bound = expected_max(CHAIN_CELLS) * (chain.hi - chain.lo) / 3.92
    steps = [(CHAIN[i], CHAIN[i + 1]) for i in range(len(CHAIN) - 1)]
    lines = ["# CB1: the lane-B chain judged end to end (holdout Jan+Jul 2025, floored)", "",
             f"Regenerate with `python -m taxi.cb1`. Train months {TRAIN_MONTHS}, {len(teb)} lane-B holdout rows, every prediction floored at 0 s "
             "(H22). Paired day-block bootstrap, 2,000 reps, seed 0.", "",
             "## 1. The whole chain in one verdict (H45)", "",
             "Each promotion priced its own cell count; nothing priced the chain. This is the contrast the chain claims, judged once.", "",
             "| champion | challenger | RMSE champion | RMSE challenger | improvement [95%] |", "|---|---|---|---|---|",
             f"| B4 floored | L8c floored | {chain.champion:.1f} | {chain.challenger:.1f} | {chain.delta:+.1f} [{chain.lo:+.1f}, {chain.hi:+.1f}] |", "",
             f"Priced once for the whole lane at k = {CHAIN_CELLS} declared cells: expected max {expected_max(CHAIN_CELLS):.2f} x SE gives a bound of "
             f"{bound:.1f} s against an observed {chain.delta:+.1f} s. Verdict: "
             f"{'clears the chain bound' if chain.delta >= bound + 2.0 and chain.lo > 0 else 'does NOT clear the chain bound'}.", "",
             "## 2. Each step, for the record", "", "| step | RMSE champion | RMSE challenger | improvement [95%] |", "|---|---|---|---|"]
    for a, b in steps:
        v = paired_delta(e[a], e[b], days)
        lines.append(f"| {a} to {b} | {v.champion:.1f} | {v.challenger:.1f} | {v.delta:+.1f} [{v.lo:+.1f}, {v.hi:+.1f}] |")
    lines += ["", "## 3. Does the L8c step survive the extreme rows?", "",
              "Rows removed in descending order of the champion's absolute error, the construction the control table below reproduces from `docs/LB-CONC.md`. If the lead collapses as the biggest "
              "truths leave, the step is a level artifact of a handful of rows.", ""]
    lines += drop_table(teb, e["L6a"], e["L8c"], "L6a", "L8c")
    lines += ["", "The same table for the earlier step, as a control (LB-CONC reports the first two rows):", ""]
    lines += drop_table(teb, e["B4"], e["L6a"], "B4", "L6a")
    text = "\n".join(lines) + "\n"
    with open("docs/CB1.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
