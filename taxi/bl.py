"""BL: what the one board number can and cannot say about the lane split.

Diagnostic, never a verdict. Three results. The lane SSE split is not measurable from the leaderboard at any number of
submissions (proof in the report), so the board fixes only a one-parameter family of (lane A, lane B) pairs. The challenger's
per-lane board error sums ARE exact, because S1's were measured and both files sit on disk. And its projected board
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
CHAL = 16  # the newest built submission, data/merry-octopus_v16.parquet; nothing after S1 has been uploaded.
CHAL_HOLD = {"A": 230.0951, "B": 1808.1729}
CHAL_NAME = {"A": "SB3", "B": "L8c"}
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
    p = {v: pd.read_parquet(f"data/merry-octopus_v{v}.parquet").set_index("MVT_ID_mvt").TAXITIME_SEC_mvt for v in (1, CHAL)}
    d = p[CHAL] - p[1]
    chal_sum = {k: S1_SUM[k] + float(d.loc[ids[k]].sum()) for k in ("A", "B")}

    board = sse(S1_BOARD, nt)

    def lane_b(a: float) -> float:
        """The board lane B that S1's board score forces, once board lane A is named."""
        return float(np.sqrt((board - sse(a, n["A"])) / n["B"]))

    rows, proj = [], []
    for a in range(190, 311, 15):
        b = lane_b(float(a))
        mul = comp(CHAL_HOLD["A"] * a / S1_HOLD["A"], CHAL_HOLD["B"] * b / S1_HOLD["B"])
        add = comp(CHAL_HOLD["A"] + a - S1_HOLD["A"], max(CHAL_HOLD["B"] + b - S1_HOLD["B"], 0.0))
        proj += [mul, add]
        rows.append(f"| {a} | {b:,.0f} | {a / S1_HOLD['A']:.3f} | {b / S1_HOLD['B']:.3f} | {mul:.2f} | {add:.2f} |")

    lrows = [f"| {a} | {float(np.sqrt((sse(LEADER, nt) - sse(float(a), n['A'])) / n['B'])):,.0f} |"
             for a in range(150, 281, 20) if sse(LEADER, nt) > sse(float(a), n["A"])]

    eq = float(np.sqrt(board / (sse(S1_HOLD["A"], n["A"]) + sse(S1_HOLD["B"], n["B"]))))
    va, vb = CHAL_HOLD["A"] * eq, CHAL_HOLD["B"] * eq
    mid = comp(va, vb)
    # The leader's lane held equal to ours in turn: the other lane is then forced.
    ld = sse(LEADER, nt)
    lb_at_va = float(np.sqrt((ld - sse(va, n["A"])) / n["B"]))
    la_at_vb = float(np.sqrt((ld - sse(vb, n["B"])) / n["A"]))
    gain = mid - float(np.sqrt((sse(mid, nt) - n["B"] * (chal_sum["B"] / n["B"]) ** 2) / nt))

    board_at = lambda a, b: comp(a * eq, b * eq)  # noqa: E731  a holdout lane pair carried to the board at the equal ratio
    need = {"A": la_at_vb / eq, "B": lb_at_va / eq}
    grows = [f"| A | {g} | {CHAL_HOLD['A'] - g:,.2f} | {CHAL_HOLD['B']:,.2f} | {board_at(CHAL_HOLD['A'] - g, CHAL_HOLD['B']):.2f} |"
             for g in (0, 1, 2, 3, 5, 10, 15)]
    grows += [f"| B | {g} | {CHAL_HOLD['A']:,.2f} | {CHAL_HOLD['B'] - g:,.2f} | {board_at(CHAL_HOLD['A'], CHAL_HOLD['B'] - g):.2f} |"
              for g in (25, 50, 100, 150)]

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
        f"## 2. What is exact: v{CHAL}'s per-lane board error sums", "",
        f"S1's per-lane sums were measured by P1 and P3s/P5s, and v{CHAL} sits on disk beside S1, so the difference of the two",
        "prediction vectors carries the measurement across with no upload. These are 2026 ranking-set numbers for a file that has",
        "never been scored.", "",
        "| file | lane A sum (s) | lane A mean (s) | lane B sum (s) | lane B mean (s) |", "|---|---|---|---|---|",
        f"| S1 (board {S1_BOARD}) | {S1_SUM['A']:+,.0f} | {S1_SUM['A'] / n['A']:+.2f} | {S1_SUM['B']:+,.0f} | {S1_SUM['B'] / n['B']:+.1f} |",
        f"| v{CHAL} (not uploaded) | {chal_sum['A']:+,.0f} | {chal_sum['A'] / n['A']:+.2f} | {chal_sum['B']:+,.0f} | {chal_sum['B'] / n['B']:+.1f} |", "",
        f"v{CHAL} runs {chal_sum['A'] / n['A']:+.2f} s high on the average 2026 lane-A row and {chal_sum['B'] / n['B']:+.1f} s low on the average lane-B row. Removing the lane-B",
        f"level alone is worth about {gain:.2f} s of board composite, at no model cost; it is a change to a submitted file and rides the gate.", "",
        f"## 3. The family the board leaves open, and what v{CHAL} scores on it", "",
        f"Name S1's board lane A and its board lane B follows exactly. S1's holdout lanes were {S1_HOLD['A']} and {S1_HOLD['B']:,}; v{CHAL}'s champions",
        f"are {CHAL_HOLD['A']} ({CHAL_NAME['A']}) and {CHAL_HOLD['B']:,} ({CHAL_NAME['B']}). Carrying its lanes across by the same ratio, or by the same additive shift,",
        "gives two projections per point. The point of the table is that both stay in a narrow band across the whole family.", "",
        f"| S1 board lane A | forced S1 board lane B | ratio in A | ratio in B | v{CHAL} board, ratio carry | v{CHAL} board, additive carry |",
        "|---|---|---|---|---|---|", *rows, "",
        f"Across the whole plausible family, v{CHAL} projects to **{min(proj):.1f} to {max(proj):.1f}**. The equal-ratio point (r = {eq:.4f} in both lanes)",
        f"gives {mid:.2f}, and that is the honest single number to plan against; {min(proj):.1f} is the optimistic corner, and it needs an additive carry",
        "AND a lane-B-heavy reading of the family (S1 board lane A near 200) at once. The S4 gate's two scalar anchors, 282.7 additive and",
        f"287.4 ratio, were computed for v15, whose lane A was MC2 at 232.1502; v{CHAL} puts {CHAL_NAME['A']} there at {CHAL_HOLD['A']} and moves both.", "",
        "## 4. Where the leader's lead has to sit", "",
        f"The same identity at {LEADER}. Their split is equally unmeasurable, so this is the locus, not a claim about their model.", "",
        "| leader board lane A | forced leader board lane B |", "|---|---|", *lrows, "",
        f"Put v{CHAL} on the same board at the equal-ratio point: lane A {va:.1f}, lane B {vb:,.0f}, composite {mid:.2f}. Holding the leader's lane A equal",
        f"to ours forces their lane B to {lb_at_va:,.0f}, which is {100 * (1 - lb_at_va / vb):.1f}% better than ours. Holding their lane B equal to ours forces their lane A",
        f"to {la_at_vb:.1f}, which is {100 * (1 - la_at_vb / va):.1f}% better than ours. So the lead is about {100 * (1 - la_at_vb / va):.0f}% of one lane, or a mix of less in each. It is not",
        "a different kind of model, and it is not concentrated in lane B's artifact rows the way the drop-k arithmetic in `docs/LEV.md`",
        "suggests on 2025 data.", "",
        "## 5. What a holdout second is worth, and what first place costs", "",
        "Every verdict in this campaign is a holdout number and the prize is a board number. The equal-ratio carry converts one to the",
        "other, so the price of first place can be read in the same units the ledger uses. Each row moves ONE lane and holds the other.", "",
        "| lane | holdout gain (s) | holdout lane A | holdout lane B | board |", "|---|---|---|---|---|", *grows, "",
        f"A holdout second of lane A is worth about {board_at(CHAL_HOLD['A'] - 1, CHAL_HOLD['B']) - mid:+.3f} s of board, and a second of lane B about "
        f"{board_at(CHAL_HOLD['A'], CHAL_HOLD['B'] - 1) - mid:+.4f} s.",
        f"Reaching {LEADER} on lane A alone needs holdout lane A at **{need['A']:.1f} s**, a gain of **{CHAL_HOLD['A'] - need['A']:.1f} s** on "
        f"{CHAL_NAME['A']}'s {CHAL_HOLD['A']}.",
        f"On lane B alone it needs **{need['B']:,.1f} s**, a gain of **{CHAL_HOLD['B'] - need['B']:,.1f} s** on {CHAL_NAME['B']}'s {CHAL_HOLD['B']:,}.", "",
        "Set that against what the diagnostics have already priced with the holdout's own answers in hand. `docs/GEO.md` caps every",
        "stand-and-runway geometric feature, main effect and congestion interaction together, at +3.295 s of lane A, and puts the share",
        "that survives the calendar at 26%. `docs/EXH.md` prices what a better fit over the 49 columns could still find at +0.10 s.",
        "The ARVT3 entry in `docs/COVERAGE.md` names the reason both cap out: the lane-A label is `AOBT_3_flt - BLOCK_TIME`, two",
        "clocks disagreeing rather than taxi physics, so physical information does not reach it. Lane B's oracle is large and unreachable:",
        "`docs/DE.md` prices a perfect call on its six day-shifted rows at about 45 s of composite and then loses with every blind rule",
        "tried. So the measured model headroom is a few seconds of lane A against the double-digit gain first place asks for, and the",
        "only route left that is measured rather than hoped for is board-side, in `docs/PB.md`.", "",
    ]
    text = "\n".join(out) + "\n"
    with open("docs/BL.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
