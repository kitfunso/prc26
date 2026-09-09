# CEIL: the best board score still reachable, and where it ranks

Regenerate with `python -m taxi.ceil`. Arithmetic only: no fit, no cell declared, no verdict. Leaderboard snapshot `data/board.json` read 2026-09-08 22:14 UTC, 473 submissions across 79 scored teams. We are rank 21 at 296.2490 on 8 submissions; the leader is 248.4818. Our candidate file v16 was uploaded and scored: **296.2745**, rank **22**, 47.79 s behind the lead. That is a MEASURED number, so nothing on this page rests on a projection band any more. It also killed the old one: v16 was projected at 286.12 across a 281.4 to 288.0 family and scored 296.2745, above the whole band, because every board number in this repo was fitted from the single S1 anchor. Two scored anchors now fix the line, board = 87.6084 + 0.651090 x holdout composite, and the per-lane rates below are differentiated from it at v16 instead of assumed: one holdout second of lane A is 0.8723 s of board, one of lane B is 0.0572 s. The campaign assumed 0.633 and 0.0776, so every gain priced before today was overstated by about 27%.

## Every measured gain, applied in order

| tier | gain | lane | holdout s | board s | board after | rank | source |
|---|---|---|---|---|---|---|---|
| built | the searched net blended onto SB3 at its out-of-fold weight | A | 2.07 | 1.81 | 294.47 | 19 | `docs/SW1.md` |
| reachable | stand and runway geometry, the share that survives the calendar | A | 0.60 | 0.52 | 293.95 | 19 | `docs/GEO.md` |
| reachable | a better fit over the 49 columns already read | A | 0.10 | 0.09 | 293.86 | 19 | `docs/EXH.md` |
| oracle | every other cached lane-A model stacked on top of that blend | A | 0.14 | 0.12 | 293.74 | 19 | `docs/BF.md` |
| oracle | perfect copy membership at the nine untouched airports | B | 25.40 | 1.45 | 292.29 | 17 | `docs/LBC.md` |
| oracle | a perfect call on the six day-shifted rows | B | 510.70 | 29.21 | 263.08 | 3 | `docs/DE.md` |

## What each tier buys

| tier | board s | board after | rank | clears the lead |
|---|---|---|---|---|
| built (run, measured on the untouched holdout, and held back only by the promotion floor) | 1.81 | 294.47 | 19 | no |
| reachable (measured headroom with a route, none of it yet run) | 2.41 | 293.86 | 19 | no |
| oracle (ceilings with the answers in hand, no route to any of them) | 33.20 | 263.08 | 3 | no |

## The board around us

| rank | team | score | submissions |
|---|---|---|---|
| 12 | vibrant-tornado | 284.8509 | 3 |
| 13 | dynamic-hamburger | 285.3884 | 9 |
| 14 | intelligent-ladder | 285.6822 | 10 |
| 15 | dependable-quicksand | 288.2147 | 12 |
| 16 | zestful-fountain | 291.4829 | 4 |
| 17 | lighthearted-ladder | 292.6878 | 3 |
| 18 | resourceful-horse | 293.2086 | 16 |
| 19 | hardworking-kiwi | 294.5583 | 3 |
| 20 | playful-village | 294.7626 | 7 |
| 21 | merry-octopus | 296.2490 | 8 |
| 22 | resilient-kiwi | 296.4736 | 6 |
| 23 | natural-kiwi | 298.8203 | 2 |
| 24 | victorious-quartz | 298.9287 | 7 |
| 25 | courageous-yarn | 300.1855 | 2 |

## Measured, real, and closed on the rules

| gain | board s | source |
|---|---|---|
| lane-B level removal, no model change | 1.17 | `docs/BL.md section 2` |

v16 runs 208.6 s low on the average 2026 lane-B row, but that level was measured by probes P1 and P3s/P5s. `docs/RULES.md` quotes the ranking page naming attempts to learn from the ranking process unfair, and the same objection that closes F38 closes this. Real, measured, and not ours to spend.

## Reading

The first row is not headroom. It is a file that could be written today: SB3 with NN3 blended in at the weight fitted out of fold takes v16 from 296.27 to 294.47, 1.81 s of board and rank 22 to 19. State the blocker exactly, because the campaign has stated it loosely: at +2.07 s the lane CLEARS the 2 s floor. It is held by the k = 40 optimism bound that `docs/NN3.md` adds on top, for a hurdle of 3.0 s.

Two things are now known about that bound and they do not agree. `docs/SW1.md` puts the lane against the six others that met these rows and it survives Romano-Wolf step-down at 5% family-wise error, so it is not an artifact of the campaign meeting one holdout 60 times. And `taxi.nn3.score` reads only the training frame, so the 40 cells were ranked on training folds and the +2.07 was measured once on rows the search never saw, which is the case an optimism bound is not usually written for. Neither fact retires the bound: step-down controls multiplicity ACROSS lanes and the bound controls selection WITHIN one, and no lane in this campaign has yet re-measured NN3's cell as a fixed k = 1 spec. That is the experiment that would settle it on the rule's own terms, and until it is run the rule stands and SB3 stays.

Spending the reachable rows on top of it reaches 293.86, rank 19; every oracle as well reaches 263.08, and that column includes two that a real model has already been built for and lost. The lead is 47.79 s away, so first place is inside the oracle column and outside every route this campaign has found. `docs/BL.md` section 5 states it from the other end: first place on lane A alone needs holdout lane A at 175.3 s against SB3's 230.10, a gain of 54.8 s, against 2.91 s of lane-A seconds on this whole page.

The honest reading is unchanged in direction and smaller in size than it was. The remaining distance is not in the tuning and not in the features: `docs/COVERAGE.md` records that lane A's target is `AOBT_3_flt - BLOCK_TIME`, two clocks disagreeing rather than taxi physics, which is why GEO and EXH both cap under a second. Closing a 48 s gap needs a different reading of the target, not a better fit to this one.
