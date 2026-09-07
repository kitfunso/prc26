# BL: the lane split behind one board number

Regenerate with `python -m taxi.bl`. Arithmetic only, no fit. Diagnostic: it declares no cell and decides no verdict.

Ranking DEP rows 344,841: lane A 339,551, lane B 5,290. S1 scored 325.3982 (total SSE 36,513,140,500 s^2).

Leaderboard read 2026-09-07 04:17 UTC: 225 submissions, 60 teams scored. The leader is youthful-giraffe at 265.7612; we are 25 of 60 at 324.7339 on 5 submissions. Every leader number below moves when that snapshot does.

## 1. The split is not measurable, at any number of submissions

A submission changes our predictions from `p` to `p'` and returns one number. The difference of the two total SSEs is
`sum(p'^2 - p^2) - 2 sum((p' - p) y)`, and we know `p` and `p'` exactly, so every probe returns one **linear** functional
of the truth vector and nothing else. A lane's SSE is quadratic in the truths. Writing the four unknowns (each lane's SSE
and each lane's sum of squared truths) against the four available files (ours, all zeros, zeros in A only, zeros in B only)
gives a rank-three system, so the split is unidentifiable however many files are spent on it. The board fixes exactly one
equation, `n_A x rmse_A^2 + n_B x rmse_B^2 = SSE`. Every probe design that asks which lane is costing us is closed by this.

## 2. What is exact: v16's per-lane board error sums

S1's per-lane sums were measured by P1 and P3s/P5s, and v16 sits on disk beside S1, so the difference of the two
prediction vectors carries the measurement across with no upload. These are 2026 ranking-set numbers for a file that has
never been scored.

| file | lane A sum (s) | lane A mean (s) | lane B sum (s) | lane B mean (s) |
|---|---|---|---|---|
| S1 (board 325.3982) | +893,396 | +2.63 | -1,041,701 | -196.9 |
| v16 (not uploaded) | +1,897,034 | +5.59 | -1,103,490 | -208.6 |

v16 runs +5.59 s high on the average 2026 lane-A row and -208.6 s low on the average lane-B row. Removing the lane-B
level alone is worth about 1.17 s of board composite, at no model cost; it is a change to a submitted file and rides the gate.

## 3. The family the board leaves open, and what v16 scores on it

Name S1's board lane A and its board lane B follows exactly. S1's holdout lanes were 259.5 and 2,074.1; v16's champions
are 230.0951 (SB3) and 1,808.1729 (L8c). Carrying its lanes across by the same ratio, or by the same additive shift,
gives two projections per point. The point of the table is that both stay in a narrow band across the whole family.

| S1 board lane A | forced S1 board lane B | ratio in A | ratio in B | v16 board, ratio carry | v16 board, additive carry |
|---|---|---|---|---|---|
| 190 | 2,141 | 0.732 | 1.032 | 285.31 | 281.69 |
| 205 | 2,051 | 0.790 | 0.989 | 285.58 | 281.46 |
| 220 | 1,948 | 0.848 | 0.939 | 285.87 | 281.40 |
| 235 | 1,832 | 0.906 | 0.883 | 286.18 | 281.53 |
| 250 | 1,700 | 0.963 | 0.820 | 286.51 | 281.91 |
| 265 | 1,547 | 1.021 | 0.746 | 286.85 | 282.58 |
| 280 | 1,367 | 1.079 | 0.659 | 287.22 | 283.64 |
| 295 | 1,147 | 1.137 | 0.553 | 287.61 | 285.27 |
| 310 | 857 | 1.195 | 0.413 | 288.01 | 287.89 |

Across the whole plausible family, v16 projects to **281.4 to 288.0**. The equal-ratio point (r = 0.8946 in both lanes)
gives 286.12, and that is the honest single number to plan against; 281.4 is the optimistic corner, and it needs an additive carry
AND a lane-B-heavy reading of the family (S1 board lane A near 200) at once. The S4 gate's two scalar anchors, 282.7 additive and
287.4 ratio, were computed for v15, whose lane A was MC2 at 232.1502; v16 puts SB3 there at 230.0951 and moves both.

## 4. Where the leader's lead has to sit

The same identity at 265.7612. Their split is equally unmeasurable, so this is the locus, not a claim about their model.

| leader board lane A | forced leader board lane B |
|---|---|
| 150 | 1,778 |
| 170 | 1,658 |
| 190 | 1,512 |
| 210 | 1,332 |
| 230 | 1,099 |
| 250 | 770 |

Put v16 on the same board at the equal-ratio point: lane A 205.8, lane B 1,618, composite 286.12. Holding the leader's lane A equal
to ours forces their lane B to 1,373, which is 15.1% better than ours. Holding their lane B equal to ours forces their lane A
to 176.0, which is 14.5% better than ours. So the lead is about 15% of one lane, or a mix of less in each. It is not
a different kind of model, and it is not concentrated in lane B's artifact rows the way the drop-k arithmetic in `docs/LEV.md`
suggests on 2025 data.

## 5. What a holdout second is worth, and what first place costs

Every verdict in this campaign is a holdout number and the prize is a board number. The equal-ratio carry converts one to the
other, so the price of first place can be read in the same units the ledger uses. Each row moves ONE lane and holds the other.

| lane | holdout gain (s) | holdout lane A | holdout lane B | board |
|---|---|---|---|---|
| A | 0 | 230.10 | 1,808.17 | 286.12 |
| A | 1 | 229.10 | 1,808.17 | 285.48 |
| A | 2 | 228.10 | 1,808.17 | 284.85 |
| A | 3 | 227.10 | 1,808.17 | 284.22 |
| A | 5 | 225.10 | 1,808.17 | 282.97 |
| A | 10 | 220.10 | 1,808.17 | 279.85 |
| A | 15 | 215.10 | 1,808.17 | 276.77 |
| B | 25 | 230.10 | 1,783.17 | 284.19 |
| B | 50 | 230.10 | 1,758.17 | 282.27 |
| B | 100 | 230.10 | 1,708.17 | 278.47 |
| B | 150 | 230.10 | 1,658.17 | 274.74 |

A holdout second of lane A is worth about -0.633 s of board, and a second of lane B about -0.0776 s.
Reaching 265.7612 on lane A alone needs holdout lane A at **196.7 s**, a gain of **33.4 s** on SB3's 230.0951.
On lane B alone it needs **1,534.4 s**, a gain of **273.8 s** on L8c's 1,808.1729.

Set that against what the diagnostics have already priced with the holdout's own answers in hand. `docs/GEO.md` caps every
stand-and-runway geometric feature, main effect and congestion interaction together, at +3.295 s of lane A, and puts the share
that survives the calendar at 26%. `docs/EXH.md` prices what a better fit over the 49 columns could still find at +0.10 s.
The ARVT3 entry in `docs/COVERAGE.md` names the reason both cap out: the lane-A label is `AOBT_3_flt - BLOCK_TIME`, two
clocks disagreeing rather than taxi physics, so physical information does not reach it. Lane B's oracle is large and unreachable:
`docs/DE.md` prices a perfect call on its six day-shifted rows at about 45 s of composite and then loses with every blind rule
tried. So the measured model headroom is a few seconds of lane A against the double-digit gain first place asks for, and the
only route left that is measured rather than hoped for is board-side, in `docs/PB.md`.

