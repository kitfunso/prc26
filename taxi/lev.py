"""LEV (diagnostic, never a verdict): what a lane second buys in composite seconds, and what the parked ties are worth once converted.

Every verdict in this campaign is measured in LANE RMSE and judged against a 2 s floor expressed in the same unit, but the board scores one
composite over both lanes. The two units are not interchangeable, and the ties parked under the floor have never been priced in the unit that
decides the ranking. This does that conversion and prices them. No cell is declared.
"""
from __future__ import annotations

import numpy as np

from taxi.cb1 import lane_b
from taxi.champ import CURRENT, champion_holdout, lane_a, predict_b
from taxi.l2 import floored

# Ledger id, lane, effect and interval as recorded in docs/TRIAL-LEDGER.md; SB2 already contains SB1 and ST1a, so only it is counted.
PARKED = [
    ("H34", "A", "E2, the fixed-weight MC2/D1 blend", 0.2, (0.1, 0.3), True),
    ("H35", "A", "SB1, the five-seed bag", 1.0, (0.7, 1.3), False),
    ("H47", "A", "ST1a, the row-dependent MC2/D1 gate", 0.6, (0.5, 0.8), False),
    ("H48", "A", "SB2, SB1 then ST1a then the C1 floor", 1.3, (1.0, 1.7), True),
    ("H38", "A", "G1, target encoding of airport/stand/runway", 0.2, (-0.2, 0.6), True),
    ("H7", "A", "B3a", 19.5, (-17.9, 73.7), True),
    ("H10", "A", "B3d", 33.0, (-2.5, 86.3), True),
    ("H15", "B", "W1-B, the five weather columns", 9.9, (-2.7, 22.6), True),
    ("H17", "B", "L1a", 2.1, (-0.2, 5.3), True),
]


def main() -> None:
    tra, tea = lane_a()
    trb, teb = lane_b()
    ea = floored(champion_holdout(CURRENT["A"], tra, tea)) - tea.y.values
    eb = floored(predict_b(CURRENT["B"], trb, teb)) - teb.y.values
    na, nb = len(ea), len(eb)
    n = na + nb
    sa, sb = float((ea ** 2).sum()), float((eb ** 2).sum())
    ra, rb = float(np.sqrt(sa / na)), float(np.sqrt(sb / nb))
    comp = float(np.sqrt((sa + sb) / n))
    ka, kb = ra * na / (comp * n), rb * nb / (comp * n)

    def priced(lane: str, g: float) -> float:
        """Composite seconds freed by g seconds of that lane's RMSE, exactly rather than through the local derivative."""
        r = (ra - g, rb) if lane == "A" else (ra, rb - g)
        return comp - float(np.sqrt((r[0] ** 2 * na + r[1] ** 2 * nb) / n))

    lines = ["# LEV: a lane second is not a composite second (diagnostic, never a verdict)", "",
             f"Regenerate with `python -m taxi.lev`. Champions {CURRENT['A']} and {CURRENT['B']}, floored, holdout Jan + Jul 2025. No cell is "
             "declared. The point is that the floor and every verdict interval are quoted in lane RMSE while the board is scored in composite "
             "RMSE, and the two differ by a factor of eight between the lanes.", "",
             "## 1. The conversion", "",
             "| lane | rows | RMSE | share of composite squared error | one second of it, in composite seconds |", "|---|---|---|---|---|",
             f"| A | {na:,} | {ra:,.2f} s | {sa / (sa + sb) * 100:.1f}% | {ka:.3f} |",
             f"| B | {nb:,} | {rb:,.2f} s | {sb / (sa + sb) * 100:.1f}% | {kb:.3f} |",
             f"| both | {n:,} | {comp:,.2f} s | 100% | 1.000 |", "",
             f"One second of lane-A RMSE is worth {ka / kb:.1f} seconds of lane-B RMSE, because lane A carries {na / nb:.0f} times the rows. The "
             "shares of squared error are near-equal, so a given PERCENTAGE gain pays about the same either side: "
             f"1% of lane A is {priced('A', ra * 0.01):.2f} s of composite, 1% of lane B is {priced('B', rb * 0.01):.2f} s. Lane B looks like half "
             "the problem and is, but its seconds are cheap and its percentages are not reachable, for the reason section 3 gives.", "",
             "## 2. Every parked tie, priced in the unit the board scores", "",
             "Effects and intervals are quoted from `docs/TRIAL-LEDGER.md`; only the conversion is computed here.", "",
             "| ledger | lane | cell | lane effect | interval | composite | counted |", "|---|---|---|---|---|---|---|"]
    total = 0.0
    for lid, lane, cell, g, (lo, hi), _ in PARKED:
        c = priced(lane, g)
        keep = lid == "H48"
        if keep:
            total += c
        note = "yes" if keep else ("inside H48" if lid in ("H35", "H47") else "interval crosses zero" if lo <= 0 else "overlaps ST1a's blend")
        lines.append(f"| {lid} | {lane} | {cell} | {g:+.1f} s | [{lo:+.1f}, {hi:+.1f}] | {c:+.2f} s | {note} |")
    lines += ["", f"**The whole parked inventory is worth {total:+.2f} s of composite.** SB1 and ST1a are not separate gains; SB2 IS their "
              "composition and H48 measures the stack. Of the rest, five intervals cross zero and are noise, and E2's fixed blend reaches for the "
              "same D1 information ST1a already uses inside SB2.", "",
              "## 3. Why lane B's half of the error is not half of the opportunity", ""]
    o = np.argsort(-(eb ** 2))
    for k in (1, 2, 5):
        keep = np.ones(nb, bool)
        keep[o[:k]] = False
        c2 = float(np.sqrt((sa + float((eb[keep] ** 2).sum())) / (na + int(keep.sum()))))
        lines.append(f"- Deleting lane B's worst {k} row{'s' if k > 1 else ''} takes the composite to {c2:,.2f} s, a gain of {comp - c2:,.2f} s.")
    lines += ["", "Those rows are the day-shift artifact of `docs/DE.md` and flag F21 prices every route to them at zero. So lane B's 49% share of "
              "squared error is mostly unreachable, and the lane-B percentage that section 1 costs at par cannot actually be bought. What is left "
              "in lane B after the artifacts is a few seconds of composite at best.", "",
              "## 4. What this changes", "",
              "- **The 2 s floor is not one floor.** In composite seconds it is "
              f"{priced('A', 2.0):.2f} s on lane A and {priced('B', 2.0):.3f} s on lane B, an {priced('A', 2.0) / priced('B', 2.0):.0f}-fold "
              "difference in what the same rule demands. Flag F18 already records that the floor is about 10 paired SE in lane A and about 0.12 in "
              "lane B; this is the same fact in the board's own unit. Changing it is a decision-rule change and stays with Keith.", "",
              f"- **The parked inventory is small.** {total:.2f} s of composite against a projected residual gap to the leader of 5.4 to 10.1 s "
              "(protocol section 4, the two untested S4 anchors at 282.7 and 287.4 against the leader's 277.29). Releasing every parked tie closes "
              f"under a fifth of the smaller anchor's gap and none of the larger one's.", "",
              "- **Lane A is where the reachable seconds are**, not because it holds more error, but because its error is spread over "
              f"{na:,} rows that a model can move, while half of lane B's sits on two rows that no ranking-visible column separates (LK1, F17, F21)."]
    text = "\n".join(lines) + "\n"
    with open("docs/LEV.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
