"""BL: what the one board number can and cannot say about the lane split.

Diagnostic, never a verdict. Three results. The lane SSE split is not measurable from the leaderboard at any number of
submissions (proof in the report), so the board fixes only a one-parameter family of (lane A, lane B) pairs. v15's
per-lane board error sums ARE exact, because S1's were measured and both files sit on disk. And v15's projected board
score is almost flat along that family, which prices the S4 anchors far better than the two scalar guesses on the gate.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

RANKING = "data/ranking.parquet"
# docs/TRIAL-LEDGER.md section A (S1) and section C (P1, P3s, P5s).
S1_BOARD, LEADER = 325.3982, 277.29
S1_HOLD = {"A": 259.5, "B": 2074.1}
V15_HOLD = {"A": 232.1502, "B": 1808.1729}
S1_SUM = {"B": -1_041_701.0, "A": +893_396.0}


def lane_ids() -> dict[str, pd.Index]:
    q = "SELECT MVT_ID_mvt FROM read_parquet('%s') WHERE PHASE_mvt = 'DEP' AND AOBT_3_flt IS %s NULL"
    return {k: pd.Index(duckdb.sql(q % (RANKING, s)).df().MVT_ID_mvt) for k, s in (("A", "NOT"), ("B", ""))}


def sse(rmse: float, n: int) -> float:
    return rmse * rmse * n


def main() -> None:
    ids = lane_ids()
    n = {k: len(v) for k, v in ids.items()}
    nt = n["A"] + n["B"]
    comp = lambda a, b: float(np.sqrt((sse(a, n["A"]) + sse(b, n["B"])) / nt))  # noqa: E731
    p = {v: pd.read_parquet(f"data/merry-octopus_v{v}.parquet").set_index("MVT_ID_mvt").TAXITIME_SEC_mvt for v in (1, 15)}
    d = p[15] - p[1]
    v15_sum = {k: S1_SUM[k] + float(d.loc[ids[k]].sum()) for k in ("A", "B")}

    board = sse(S1_BOARD, nt)

    def lane_b(a: float) -> float:
        """The board lane B that S1's board score forces, once board lane A is named."""
        return float(np.sqrt((board - sse(a, n["A"])) / n["B"]))

    rows, proj = [], []
    for a in range(190, 311, 15):
        b = lane_b(float(a))
        mul = comp(V15_HOLD["A"] * a / S1_HOLD["A"], V15_HOLD["B"] * b / S1_HOLD["B"])
        add = comp(V15_HOLD["A"] + a - S1_HOLD["A"], max(V15_HOLD["B"] + b - S1_HOLD["B"], 0.0))
        proj += [mul, add]
        rows.append(f"| {a} | {b:,.0f} | {a / S1_HOLD['A']:.3f} | {b / S1_HOLD['B']:.3f} | {mul:.2f} | {add:.2f} |")

    lrows = [f"| {a} | {float(np.sqrt((sse(LEADER, nt) - sse(float(a), n['A'])) / n['B'])):,.0f} |"
             for a in range(150, 281, 20) if sse(LEADER, nt) > sse(float(a), n["A"])]

    eq = float(np.sqrt(board / (sse(S1_HOLD["A"], n["A"]) + sse(S1_HOLD["B"], n["B"]))))
    va, vb = V15_HOLD["A"] * eq, V15_HOLD["B"] * eq
    mid = comp(va, vb)
    # The leader's lane held equal to ours in turn: the other lane is then forced.
    ld = sse(LEADER, nt)
    lb_at_va = float(np.sqrt((ld - sse(va, n["A"])) / n["B"]))
    la_at_vb = float(np.sqrt((ld - sse(vb, n["B"])) / n["A"]))
    gain = mid - float(np.sqrt((sse(mid, nt) - n["B"] * (v15_sum["B"] / n["B"]) ** 2) / nt))

    out = [
        "# BL: the lane split behind one board number", "",
        "Regenerate with `python -m taxi.bl`. Arithmetic only, no fit. Diagnostic: it declares no cell and decides no verdict.", "",
        f"Ranking DEP rows {nt:,}: lane A {n['A']:,}, lane B {n['B']:,}. S1 scored {S1_BOARD} "
        f"(total SSE {board:,.0f} s^2); the leader is at {LEADER}.", "",
        "## 1. The split is not measurable, at any number of submissions", "",
        "A submission changes our predictions from `p` to `p'` and returns one number. The difference of the two total SSEs is",
        "`sum(p'^2 - p^2) - 2 sum((p' - p) y)`, and we know `p` and `p'` exactly, so every probe returns one **linear** functional",
        "of the truth vector and nothing else. A lane's SSE is quadratic in the truths. Writing the four unknowns (each lane's SSE",
        "and each lane's sum of squared truths) against the four available files (ours, all zeros, zeros in A only, zeros in B only)",
        "gives a rank-three system, so the split is unidentifiable however many files are spent on it. The board fixes exactly one",
        "equation, `n_A x rmse_A^2 + n_B x rmse_B^2 = SSE`. Every probe design that asks which lane is costing us is closed by this.", "",
        "## 2. What is exact: v15's per-lane board error sums", "",
        "S1's per-lane sums were measured by P1 and P3s/P5s, and v15 sits on disk beside S1, so the difference of the two",
        "prediction vectors carries the measurement across with no upload. These are 2026 ranking-set numbers for a file that has",
        "never been scored.", "",
        "| file | lane A sum (s) | lane A mean (s) | lane B sum (s) | lane B mean (s) |", "|---|---|---|---|---|",
        f"| S1 (board {S1_BOARD}) | {S1_SUM['A']:+,.0f} | {S1_SUM['A'] / n['A']:+.2f} | {S1_SUM['B']:+,.0f} | {S1_SUM['B'] / n['B']:+.1f} |",
        f"| v15 (not uploaded) | {v15_sum['A']:+,.0f} | {v15_sum['A'] / n['A']:+.2f} | {v15_sum['B']:+,.0f} | {v15_sum['B'] / n['B']:+.1f} |", "",
        f"v15 runs {v15_sum['A'] / n['A']:+.2f} s high on the average 2026 lane-A row and {v15_sum['B'] / n['B']:+.1f} s low on the average lane-B row. Removing the lane-B",
        f"level alone is worth about {gain:.2f} s of board composite, at no model cost; it is a change to a submitted file and rides the gate.", "",
        "## 3. The family the board leaves open, and what v15 scores on it", "",
        f"Name S1's board lane A and its board lane B follows exactly. S1's holdout lanes were {S1_HOLD['A']} and {S1_HOLD['B']:,}; v15's champions",
        f"are {V15_HOLD['A']} (MC2) and {V15_HOLD['B']:,} (L8c). Carrying v15's lanes across by the same ratio, or by the same additive shift,",
        "gives two projections per point. The point of the table is that both stay in a narrow band across the whole family.", "",
        "| S1 board lane A | forced S1 board lane B | ratio in A | ratio in B | v15 board, ratio carry | v15 board, additive carry |",
        "|---|---|---|---|---|---|", *rows, "",
        f"Across the whole plausible family, v15 projects to **{min(proj):.1f} to {max(proj):.1f}**. The equal-ratio point (r = {eq:.4f} in both lanes)",
        f"gives {mid:.2f}. The S4 gate carries two scalar anchors, 282.7 additive and 287.4 ratio. The additive one is reachable only under an",
        "additive carry AND a lane-B-heavy reading of the family (S1 board lane A near 200), so 287 is the honest single number to plan",
        "against and 283 is the optimistic corner of one corner.", "",
        "## 4. Where the leader's lead has to sit", "",
        f"The same identity at {LEADER}. Their split is equally unmeasurable, so this is the locus, not a claim about their model.", "",
        "| leader board lane A | forced leader board lane B |", "|---|---|", *lrows, "",
        f"Put v15 on the same board at the equal-ratio point: lane A {va:.1f}, lane B {vb:,.0f}, composite {mid:.2f}. Holding the leader's lane A equal",
        f"to ours forces their lane B to {lb_at_va:,.0f}, which is {100 * (1 - lb_at_va / vb):.1f}% better than ours. Holding their lane B equal to ours forces their lane A",
        f"to {la_at_vb:.1f}, which is {100 * (1 - la_at_vb / va):.1f}% better than ours. So the lead is about seven percent of one lane, or a mix of less in each. It is not",
        "a different kind of model, and it is not concentrated in lane B's artifact rows the way the drop-k arithmetic in `docs/LEV.md`",
        "suggests on 2025 data.", "",
    ]
    text = "\n".join(out) + "\n"
    with open("docs/BL.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
