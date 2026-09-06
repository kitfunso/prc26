"""CA1: the lane-A promotion chain judged end to end, and the MC2 step's robustness to the extreme rows.

The lane-A twin of CB1. Section 6 of the S4 gate names both as missing: every lane-A link priced its own cell count and nothing
priced the chain, and lane A's top 1% of rows carry 40.8% of the lane's squared error with no test that the lead survives them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from taxi.baselines import b4
from taxi.champ import champion_holdout, lane_a
from taxi.data import COUNTS
from taxi.l2 import floored
from taxi.m1 import expected_max
from taxi.verdict import paired_delta, rmse

DROPS = (0, 1, 2, 5, 20, 100, 500, 3390)
CHAIN_CELLS = 39  # every lane-A cell declared since B4 and run (protocol section 4 CA1 row lists the count per lane)
SNAP = ("z_anchor", *COUNTS, "occ", "t_occ")
PRE = "data/cache/anchor.pre-a1.parquet"


def on_old_anchors(tra: pd.DataFrame, tea: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """The same two frames with every anchor-dependent column restored from the snapshot `taxi.a1 before` wrote."""
    pre = pd.read_parquet(PRE).set_index("id")
    out = []
    for d in (tra, tea):
        o = d.copy()
        for c in SNAP:
            o[c] = pre[c].reindex(d.id.values).values
        out.append(o)
    changed = {c: int((pre[c].reindex(tea.id.values).fillna(-1).values != tea[c].fillna(-1).values).sum()) for c in SNAP}
    return out[0], out[1], changed


def drop_table(tea: pd.DataFrame, e0: np.ndarray, e1: np.ndarray, a: str, b: str) -> list[str]:
    """RMSE of both models after removing the k rows the CHAMPION misses worst, the same construction CB1 uses in lane B."""
    order = np.argsort(np.abs(e0))[::-1]
    out = [f"| rows dropped | n | {a} | {b} | improvement [95%] |", "|---|---|---|---|---|"]
    for k in DROPS:
        keep = np.ones(len(tea), bool)
        keep[order[:k]] = False
        v = paired_delta(e0[keep], e1[keep], tea.day.values[keep])
        out.append(f"| {k} | {int(keep.sum())} | {v.champion:.1f} | {v.challenger:.1f} | {v.delta:+.1f} [{v.lo:+.1f}, {v.hi:+.1f}] |")
    return out


def main() -> None:
    tra, tea = lane_a()
    y = tea.y.values
    e = {"B4": floored(b4(tra, tea, "A")) - y,
         "D1": floored(champion_holdout("D1", tra, tea)) - y,
         "MC2": floored(champion_holdout("MC2", tra, tea)) - y}
    days = tea.day.values
    chain = paired_delta(e["B4"], e["MC2"], days)
    bound = expected_max(CHAIN_CELLS) * (chain.hi - chain.lo) / 3.92
    lines = ["# CA1: the lane-A chain judged end to end (holdout Jan+Jul 2025, floored)", "",
             f"Regenerate with `python -m taxi.ca1`. {len(tea)} lane-A holdout rows, every prediction floored at 0 s (the file's C1 option; the floor "
             "was a tie in this lane, H21). Paired day-block bootstrap, 2,000 reps, seed 0.", "",
             "## 1. The whole chain in one verdict", "",
             "B4 is where lane A started. Each promotion since priced its own cell count; nothing priced the chain the champion actually rode.", "",
             "| champion | challenger | RMSE champion | RMSE challenger | improvement [95%] |", "|---|---|---|---|---|",
             f"| B4 floored | MC2 floored | {chain.champion:.1f} | {chain.challenger:.1f} | {chain.delta:+.1f} [{chain.lo:+.1f}, {chain.hi:+.1f}] |", "",
             f"Priced once for the whole lane at k = {CHAIN_CELLS} declared and run cells: expected max {expected_max(CHAIN_CELLS):.2f} x SE gives a bound of "
             f"{bound:.1f} s against an observed {chain.delta:+.1f} s. Verdict: "
             f"{'clears the chain bound' if chain.delta >= bound + 2.0 and chain.lo > 0 else 'does NOT clear the chain bound'}.", "",
             "## 2. Does the MC2 step survive the extreme rows?", "",
             "Rows removed in descending order of the champion's absolute error. The last row drops the top 1% of the lane, which `docs/M1.md` measures "
             "at 40.8% of the lane's squared error. If the lead collapses as those rows leave, the step is a level artifact of the tail.", ""]
    lines += drop_table(tea, e["D1"], e["MC2"], "D1", "MC2")
    lines += ["", "The same table for the whole chain:", ""]
    lines += drop_table(tea, e["B4"], e["MC2"], "B4", "MC2")
    tra_o, tea_o, changed = on_old_anchors(tra, tea)
    old = floored(b4(tra_o, tea_o, "A")) - y
    a1 = paired_delta(old, e["B4"], days)
    lines += ["", "## 3. The baseline moved under A1", "",
              "B4 was measured on 2026-09-04, before the A1 anchor guard. A1 was re-verdicted against the champions (H29 to H32) and never against B4, "
              "so the chain's starting point above is not the number the ledger records. Both fits below use the anchor-dependent columns "
              f"`{', '.join(SNAP)}` restored from the snapshot `python -m taxi.a1 before` wrote.", "",
              "| B4 floored on | RMSE | paired, new minus old [95%] |", "|---|---|---|",
              f"| the pre-A1 anchors | {rmse(old):.1f} | |",
              f"| today's anchors | {rmse(e['B4']):.1f} | {a1.delta:+.1f} [{a1.lo:+.1f}, {a1.hi:+.1f}] |", "",
              f"A1 changed z_anchor on {changed['z_anchor']} of {len(tea):,} lane-A holdout rows and the queue count on {changed['queue']}, and it costs the "
              f"linear baseline {-a1.delta:.1f} s. Rows changed per column: {changed}. The guard is a data-honesty fix (F11: an anchor after takeoff cannot "
              "be known at prediction time), so it stays; the consequence for reporting is that every improvement quoted against B4 on today's data is "
              f"about {-a1.delta:.0f} s larger than the same improvement on the anchors B4 was measured on. Section 1's chain reads "
              f"{chain.delta:+.1f} s today and {rmse(old) - chain.challenger:+.1f} s against the old baseline; both clear the {bound:.1f} s bound."]
    text = "\n".join(lines) + "\n"
    with open("docs/CA1.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
