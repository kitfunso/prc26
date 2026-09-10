# CEIL: the best board score still reachable, and where it ranks

Regenerate with `python -m taxi.ceil`. Arithmetic only: no fit, no cell declared, no verdict. Leaderboard snapshot `data/board.json` read 2026-09-10 05:41 UTC, 673 submissions across 95 scored teams. We are rank 30 at 294.9917 on 10 submissions; the leader is 245.0940. Our candidate file v16 was uploaded and scored: **296.2745**, rank **31**, 51.18 s behind the lead. That is a MEASURED number, so nothing on this page rests on a projection band any more. It also killed the old one: v16 was projected at 286.12 across a 281.4 to 288.0 family and scored 296.2745, above the whole band, because every board number in this repo was fitted from the single S1 anchor. 7 scored anchors now fix the line, board = 83.7534 + 0.661728 x holdout composite, and the per-lane rates below are differentiated from it at v16 instead of assumed: one holdout second of lane A is 0.8723 s of board, one of lane B is 0.0581 s. The campaign assumed 0.633 and 0.0776, so every gain priced before today was overstated by about 27%.

## The holdout and the ranking set disagree about which model is better

4 CONTROLLED lane-A pairs have now been scored: two files whose lane B is byte-identical, so the board delta is lane A alone and both ends are exact arithmetic on fixed rows, with no measurement noise in the ratio.

| pair | holdout s of lane A | board s | implied rate |
|---|---|---|---|
| v15/v16, MC2 to SB3 | 2.0551 | 1.7926 | 0.8723 |
| v17/v18, SB3+RC1 to SB3+NN3 | 2.1328 | 1.2573 | 0.5895 |
| v17/v19, SB3+RC1 to SB3+NN2 | 1.7437 | 0.7462 | 0.4279 |
| v17/v20, SB3+RC1 to SB3+LT2 | 0.9949 | 0.9533 | 0.9582 |

The rate spans 0.43 to 0.96, and the three files uploaded on 2026-09-10 are the sharp version of what that means. v19 beats v20 on the holdout by 0.75 s of lane A and LOSES to it on the board by 0.21 s, 295.5028 against 295.2957. That is an ORDER inversion, not a scale error, and no single rate can produce it: a rate rescales differences, it cannot reverse their sign. So a projection of the form board = V17 - gain x PER_A is not merely imprecise for a model it has not seen, it can rank two models the wrong way round, which is the one job this page is asked to do.

Everything below still multiplies by PER_A = 0.8723 and is now a DIAGNOSTIC in the protocol's sense: it cannot decide which model to build or upload. Two measured rows say why. The searched net projected 294.47 and scored 294.9917. Its own sibling NN2 projected 295.46 and scored 295.5028, close on the point and wrong on the rank.

The cause is PARTLY identified and `docs/MS1.md` holds the measurement. Splitting each pair's holdout gain by month ranks the 4 controlled pairs perfectly: the lane that put the least of its gain in January, 11%, kept the least on the board, 0.43, and the lane that put the most in January, 43%, kept 0.96, with no exception across all four (rank correlation +1.00, exact permutation p = 0.042). The magnitudes do not follow: two lanes that split at 35% and 33% carried at 0.87 and 0.59. So month share is a proxy for the mechanism and not the mechanism, and four points against a hypothesis written after the inversion was seen is a tie-break rule, not a model. It is enough to prefer a candidate whose gain spreads across the holdout year over one that concentrates it in July.

## Every measured gain, applied in order

| tier | gain | lane | holdout s | board s | board after | rank | source |
|---|---|---|---|---|---|---|---|
| built | the searched net blended onto SB3 at its out-of-fold weight | A | 2.07 | 1.81 | 294.47 | 29 | `docs/SW1.md` |
| reachable | stand and runway geometry, the share that survives the calendar | A | 0.60 | 0.52 | 293.95 | 29 | `docs/GEO.md` |
| reachable | a better fit over the 49 columns already read | A | 0.10 | 0.09 | 293.86 | 29 | `docs/EXH.md` |
| oracle | every other cached lane-A model stacked on top of that blend | A | 0.14 | 0.12 | 293.74 | 29 | `docs/BF.md` |
| oracle | perfect copy membership at the nine untouched airports | B | 25.40 | 1.48 | 292.26 | 26 | `docs/LBC.md` |
| oracle | a perfect call on the six day-shifted rows | B | 510.70 | 29.69 | 262.58 | 3 | `docs/DE.md` |

## What each tier buys

| tier | board s | board after | rank | clears the lead |
|---|---|---|---|---|
| built (run, measured on the untouched holdout, and held back only by the promotion floor) | 1.81 | 294.47 | 29 | no |
| reachable (measured headroom with a route, none of it yet run) | 2.41 | 293.86 | 29 | no |
| oracle (ceilings with the answers in hand, no route to any of them) | 33.70 | 262.58 | 3 | no |

## The board around us

| rank | team | score | submissions |
|---|---|---|---|
| 16 | vibrant-tornado | 284.8509 | 3 |
| 17 | dynamic-hamburger | 285.2168 | 12 |
| 18 | intelligent-ladder | 285.6822 | 11 |
| 19 | silly-net | 286.3414 | 13 |
| 20 | merry-quicksand | 288.1406 | 5 |
| 21 | dependable-quicksand | 288.2147 | 13 |
| 22 | victorious-quartz | 288.3508 | 17 |
| 23 | zestful-fountain | 288.3714 | 6 |
| 24 | playful-village | 291.0141 | 10 |
| 25 | delightful-ostrich | 291.5834 | 11 |
| 26 | lighthearted-ladder | 292.6878 | 3 |
| 27 | gentle-ostrich | 292.7011 | 6 |
| 28 | likable-eagle | 292.9926 | 6 |
| 29 | hardworking-kiwi | 294.5583 | 3 |

## Measured, real, and closed on the rules

| gain | board s | source |
|---|---|---|
| lane-B level removal, no model change | 1.17 | `docs/BL.md section 2` |

v16 runs 208.6 s low on the average 2026 lane-B row, but that level was measured by probes P1 and P3s/P5s. `docs/RULES.md` quotes the ranking page naming attempts to learn from the ranking process unfair, and the same objection that closes F38 closes this. Real, measured, and not ours to spend.

## Reading

The first row is not headroom. It is a file that could be written today: SB3 with NN3 blended in at the weight fitted out of fold takes v16 from 296.27 to 294.47, 1.81 s of board and rank 31 to 29. State the blocker exactly, because the campaign has stated it loosely: at +2.07 s the lane CLEARS the 2 s floor. It is held by the k = 40 optimism bound that `docs/NN3.md` adds on top, for a hurdle of 3.0 s.

That bound is now retired, by measurement rather than by argument. `docs/NC1.md` took NN3's winning cell as a FIXED k = 1 spec and read it on months 4, 5, 9 and 10, which `taxi.nn3.score` never ranked a cell on: against one base the blend gains +3.80 s on the months that chose the cell and +3.32 s on the months that did not, so the selection advantage is about 0.48 s, not the 1.0 s the bound assumed. The holdout never took part in choosing the cell at all, so the +2.07 needed no such correction in the first place. `docs/SW1.md` closes the other multiplicity, the campaign meeting one holdout 60 times: the lane survives Romano-Wolf step-down at 5% family-wise error over the eight cached lanes.

So every objection this campaign has instrumented is now answered, and `taxi.verdict` passes the lane on its own rule: +2.07 s against a 2 s floor with a lower bound of +1.1 s. Say the caveat as loudly as the result. The margin is 0.07 s, the floor sits INSIDE the interval [+1.1, +2.9], so these rows cannot rule out a true gain below the floor, and the 1.81 s of board above is a projection off the measured per-lane rate, not a scored file. This is a Stage 6 promotion decision and it is Keith's alone. Nothing here builds a file or uploads one.

Spending the reachable rows on top of it reaches 293.86, rank 29; every oracle as well reaches 262.58, and that column includes two that a real model has already been built for and lost. The lead is 51.18 s away, so first place is inside the oracle column and outside every route this campaign has found. `docs/BL.md` section 5 states it from the other end: first place on lane A alone needs holdout lane A at 171.4 s against SB3's 230.10, a gain of 58.7 s, against 2.91 s of lane-A seconds on this whole page.

The honest reading is unchanged in direction and smaller in size than it was. The remaining distance is not in the tuning and not in the features: `docs/COVERAGE.md` records that lane A's target is `AOBT_3_flt - BLOCK_TIME`, two clocks disagreeing rather than taxi physics, which is why GEO and EXH both cap under a second. Closing a 51 s gap needs a different reading of the target, not a better fit to this one.
