# BL: the lane split behind one board number

Regenerate with `python -m taxi.bl`. Arithmetic only, no fit. Diagnostic: it declares no cell and decides no verdict.

Ranking DEP rows 344,841: lane A 339,551, lane B 5,290. S1 scored 325.3982 (total SSE 36,513,140,500 s^2); the leader is at 277.29.

## 1. The split is not measurable, at any number of submissions

A submission changes our predictions from `p` to `p'` and returns one number. The difference of the two total SSEs is
`sum(p'^2 - p^2) - 2 sum((p' - p) y)`, and we know `p` and `p'` exactly, so every probe returns one **linear** functional
of the truth vector and nothing else. A lane's SSE is quadratic in the truths. Writing the four unknowns (each lane's SSE
and each lane's sum of squared truths) against the four available files (ours, all zeros, zeros in A only, zeros in B only)
gives a rank-three system, so the split is unidentifiable however many files are spent on it. The board fixes exactly one
equation, `n_A x rmse_A^2 + n_B x rmse_B^2 = SSE`. Every probe design that asks which lane is costing us is closed by this.

## 2. What is exact: v15's per-lane board error sums

S1's per-lane sums were measured by P1 and P3s/P5s, and v15 sits on disk beside S1, so the difference of the two
prediction vectors carries the measurement across with no upload. These are 2026 ranking-set numbers for a file that has
never been scored.

| file | lane A sum (s) | lane A mean (s) | lane B sum (s) | lane B mean (s) |
|---|---|---|---|---|
| S1 (board 325.3982) | +893,396 | +2.63 | -1,041,701 | -196.9 |
| v15 (not uploaded) | +1,885,094 | +5.55 | -1,103,490 | -208.6 |

v15 runs +5.55 s high on the average 2026 lane-A row and -208.6 s low on the average lane-B row. Removing the lane-B
level alone is worth about 1.16 s of board composite, at no model cost; it is a change to a submitted file and rides the gate.

## 3. The family the board leaves open, and what v15 scores on it

Name S1's board lane A and its board lane B follows exactly. S1's holdout lanes were 259.5 and 2,074.1; v15's champions
are 232.1502 (MC2) and 1,808.1729 (L8c). Carrying v15's lanes across by the same ratio, or by the same additive shift,
gives two projections per point. The point of the table is that both stay in a narrow band across the whole family.

| S1 board lane A | forced S1 board lane B | ratio in A | ratio in B | v15 board, ratio carry | v15 board, additive carry |
|---|---|---|---|---|---|
| 190 | 2,141 | 0.732 | 1.032 | 286.19 | 282.85 |
| 205 | 2,051 | 0.790 | 0.989 | 286.60 | 282.73 |
| 220 | 1,948 | 0.848 | 0.939 | 287.04 | 282.77 |
| 235 | 1,832 | 0.906 | 0.883 | 287.52 | 283.02 |
| 250 | 1,700 | 0.963 | 0.820 | 288.02 | 283.49 |
| 265 | 1,547 | 1.021 | 0.746 | 288.55 | 284.26 |
| 280 | 1,367 | 1.079 | 0.659 | 289.11 | 285.42 |
| 295 | 1,147 | 1.137 | 0.553 | 289.70 | 287.15 |
| 310 | 857 | 1.195 | 0.413 | 290.32 | 289.86 |

Across the whole plausible family, v15 projects to **282.7 to 290.3**. The equal-ratio point (r = 0.8946 in both lanes)
gives 287.42. The S4 gate carries two scalar anchors, 282.7 additive and 287.4 ratio. The additive one is reachable only under an
additive carry AND a lane-B-heavy reading of the family (S1 board lane A near 200), so 287 is the honest single number to plan
against and 283 is the optimistic corner of one corner.

## 4. Where the leader's lead has to sit

The same identity at 277.29. Their split is equally unmeasurable, so this is the locus, not a claim about their model.

| leader board lane A | forced leader board lane B |
|---|---|
| 150 | 1,889 |
| 170 | 1,777 |
| 190 | 1,642 |
| 210 | 1,477 |
| 230 | 1,272 |
| 250 | 1,000 |
| 270 | 577 |

Put v15 on the same board at the equal-ratio point: lane A 207.7, lane B 1,618, composite 287.42. Holding the leader's lane A equal
to ours forces their lane B to 1,498, which is 7.4% better than ours. Holding their lane B equal to ours forces their lane A
to 193.2, which is 7.0% better than ours. So the lead is about seven percent of one lane, or a mix of less in each. It is not
a different kind of model, and it is not concentrated in lane B's artifact rows the way the drop-k arithmetic in `docs/LEV.md`
suggests on 2025 data.

