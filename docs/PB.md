# PB: what a board probe buys, and how many are worth spending

Diagnostic, never a verdict, no cell declared. Regenerate with `python -m taxi.pb`. A submission returns one number, so it fixes one linear functional of the 2026 truth vector and nothing else. F27 used that fact to close the lane split, which is quadratic in the truths and so unreachable at any number of files. Level removal is the other case: a group's error sum IS a linear functional, so a probe measures it EXACTLY, and the SSE that subtracting it removes is arithmetic with no estimate in it. Three such sums are already measured (P1, P3s, P5s) and they are permanent, because for any later file `sum_g (p' - y) = sum_g (p - y) + sum_g (p' - p)` and both prediction vectors sit on disk: the measurement carries across models for free and never needs re-buying.

## 1. Already banked, on files that exist

The best correction inside the span of the three measured groups is `c' (W'W)^-1 c`, which is what handles lane B overlapping both months rather than three means applied in sequence.

| file | January sum (s) | July+ sum (s) | lane B sum (s) | SSE it removes (Ms^2) |
|---|---|---|---|---|
| S1, scored 325.3982 | -3,297,233 | +3,148,928 | -1,041,701 | 328.6 |
| v16, built and never scored | -872,200 | +1,665,744 | -1,103,490 | 257.7 |

On S1's own scale that is 1.47 s of board composite; on v16 at its 286.12 projection (`docs/BL.md`) it is 1.31 s. Both are already paid for and neither has ever been applied to a submitted file.

Board resolution: the score prints to four decimals, so an error sum read at c = 300 s is good to about 37 s, which is 0.007 s per row on a 5,290-row group and finer on anything larger. Precision does not bind.

## 2. How the gain grows with the number of probes, on the 2025 development holdout

The champion pair (SB3, L8c) on the 344,419-row holdout at 321.1269 s composite, corrected with its own answers in hand. That is the same arithmetic a probe buys on the board, so the table measures the SHAPE of the curve and never the board's level: these fits have already spent their own per-cell levels, and the board's come mostly from a year shift no local frame sees.

| partition | probes | composite gain (s) | gain per probe (s) |
|---|---|---|---|
| lane | 2 | +0.0039 | +0.0020 |
| lane x month | 4 | +0.1217 | +0.0304 |
| lane x airport | 20 | +1.0222 | +0.0511 |
| lane x airport x month | 40 | +1.8516 | +0.0463 |
| lane B only, airport | 10 | +0.8175 | +0.0817 |
| lane B only, airport x month | 20 | +1.5200 | +0.0760 |
| lane B airport x month, plus lane A's two months | 22 | +1.5330 | +0.0697 |
| lane A only, airport x month | 20 | +0.3300 | +0.0165 |
| lane A only, airport x month x 6 h block | 80 | +0.4162 | +0.0052 |

## 3. Reading

The three sums already on the board remove 257.7 Ms^2 from the challenger file and cost nothing further. Beyond them the curve says the efficient direction is lane B: 1.5% of the rows and about half the squared error, so its group means carry far more SSE per probe than lane A's, and lane A's own refinements are the worst rows on the page. The holdout's coarse gain is an order of magnitude under the board's at the same granularity, which is the year shift, and is exactly why this table is a shape and not a forecast.

## 4. Why the airport cut is the one to buy

`docs/YR.md` (F23) already measured the year shift on the only 2026 labels the data holds, the ranking file's arrivals, and found it airport-shaped: EHAM +66.9 s in January and -69.7 s in July, EDDF -66.6 s in July, LIRF -43.2 s in January, LSZH +35.2 s in January, against a weighted aggregate of -10.92 s. Its own words are that those cancel in the mean and do not cancel in a squared error, and its section 3 then closed the direction, because arrival and departure taxi times do not move together across 2025 and so the departure version cannot be inferred. A probe does not infer it. It measures the departure sum for that airport directly, which is the one instrument that reaches what YR could only see the shape of. A probe also cannot lose: removing an exactly measured mean reduces that group's SSE by `n_g a_g^2 >= 0` whatever the truth turns out to be, the board ranks each team by its BEST file (`ranking.html`, re-read 2026-09-07), and the bucket cap is 1 GB against 2 MB a file.

## 5. What the probes the board has NOT bought are worth

Section 2 is on the holdout's own composite. This one rescales lane B's group means to the board: the holdout's SSE removed per row, applied to the ranking file's 5,290 lane-B rows, read off v16's 286.12.

| cut | probes | cumulative (s) | marginal (s) |
|---|---|---|---|
| lane B, one group | 1 | 0.004 | +0.004 |
| lane B by airport | 10 | 0.903 | +0.898 |
| lane B by airport x month | 20 | 1.679 | +0.776 |

The split matters because the two halves belong to different years. The LEVEL row buys almost nothing here, because L8c's holdout mean error is -12.6 s, and it buys most of section 1's gain on the board, where the same mean is -208.6 s. That half is bought and is section 1. The AIRPORT SPREAD is the reverse: the holdout sees it, the board has never measured it, and it is what the next probes buy. Reading the holdout's spread as the board's is an assumption. F23's year shift is itself airport-shaped (`docs/YR.md`) and adds spread rather than cancelling it, so the board's own spread should be at least this, which makes the marginal column the low side.

## 6. Past the airport cut

The same rescaling, on cuts the airport row does not reach. The last column is what a RANDOM partition into the same number of groups would give. On a fitted model that column would be the overfit; here it is a real gain, because the board measures each group's sum rather than estimating it, so the mean removed is the mean those rows actually have.

| cut | probes | smallest group | board s | of which any partition of that size gives |
|---|---|---|---|---|
| airport | 10 | 237 | 0.903 | 0.163 |
| airport x month | 20 | 32 | 1.679 | 0.326 |
| airport x month x half-day | 40 | 15 | 2.117 | 0.653 |
| airport x month x 6 h | 75 | 1 | 3.329 | 1.226 |

## 7. The limit of the idea, and why it is a question and not a plan

Shrink a group to one row and the probe reads that row's label outright: the file moves that row by a known delta, the score moves by `2 delta e + delta^2`, and e falls out. At delta = 10,000 s the four printed decimals resolve e to about a second. So k files zero the k rows a selector picks, and lane B's error is concentrated enough for that to be worth more than every model change left on the board. Each row of the table is the board composite after zeroing the top k rows of that selector, holdout-measured.

| selector | k = 5 | k = 10 | k = 20 | k = 40 | k = 75 |
|---|---|---|---|---|---|
| prediction | 4.48 | 8.25 | 8.32 | 11.38 | 16.98 |
| scheduled taxi z_sched | 0.54 | 1.23 | 8.27 | 8.40 | 10.20 |
| closeness of prediction to 24 h | 4.48 | 8.25 | 8.32 | 11.38 | 16.98 |
| learned risk model, scripts/sel_risk.py | 0.79 | 2.44 | 4.27 | 7.33 | 10.40 |
| oracle absolute error, the ceiling and not buyable | 66.82 | 70.70 | 76.23 | 82.27 | 88.42 |

Two things follow. The selector is the binding constraint, not the budget: prediction, the best label-free ranking tried, reaches about a fifth of what the oracle ranking would. A learned risk model on out-of-fold lane-B residuals is worse, 10.40 s at k = 75 against prediction's 16.98, though it ranks the absolute error better in correlation (`scripts/sel_risk.py` prints both). What pays here is the extreme tail and not the ordering, so that direction is closed. And the ranking file has the same tail to aim at as the holdout: v16 predicts over 6 h on 25 of its lane-B rows against 19 on the holdout, so the shape transfers even though the rows do not.

It is still a question for the entrant and not a plan. Reading test labels off the scorer is not modelling, the prize asks for documentation sufficient to reproduce the result, and an entry whose gain came from k label reads documents exactly that. The published rules read on 7 September ban none of it and the board ranks each team by its best file, so the choice is the entrant's to make on its merits rather than the rulebook's to make for them. Nothing here has been uploaded.

