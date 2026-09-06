# CA1: the lane-A chain judged end to end (holdout Jan+Jul 2025, floored)

Regenerate with `python -m taxi.ca1`. 339046 lane-A holdout rows, every prediction floored at 0 s (the file's C1 option; the floor was a tie in this lane, H21). Paired day-block bootstrap, 2,000 reps, seed 0.

## 1. The whole chain in one verdict

B4 is where lane A started. Each promotion since priced its own cell count; nothing priced the chain the champion actually rode.

| champion | challenger | RMSE champion | RMSE challenger | improvement [95%] |
|---|---|---|---|---|
| B4 floored | MC2 floored | 334.2 | 232.2 | +102.0 [+75.7, +144.8] |

Priced once for the whole lane at k = 39 declared and run cells: expected max 2.15 x SE gives a bound of 37.9 s against an observed +102.0 s. Verdict: clears the chain bound.

## 2. Does the MC2 step survive the extreme rows?

Rows removed in descending order of the champion's absolute error. The last row drops the top 1% of the lane, which `docs/M1.md` measures at 40.8% of the lane's squared error. If the lead collapses as those rows leave, the step is a level artifact of the tail.

| rows dropped | n | D1 | MC2 | improvement [95%] |
|---|---|---|---|---|
| 0 | 339046 | 241.3 | 232.2 | +9.2 [+8.0, +10.3] |
| 1 | 339045 | 241.0 | 232.0 | +8.9 [+7.7, +10.1] |
| 2 | 339044 | 240.7 | 231.8 | +8.9 [+7.7, +10.1] |
| 5 | 339041 | 240.2 | 231.3 | +8.9 [+7.7, +10.0] |
| 20 | 339026 | 238.5 | 229.6 | +8.9 [+7.8, +9.9] |
| 100 | 338946 | 233.2 | 225.4 | +7.8 [+6.8, +8.7] |
| 500 | 338546 | 221.0 | 214.2 | +6.8 [+5.8, +7.8] |
| 3390 | 335656 | 193.8 | 189.9 | +3.9 [+3.1, +4.6] |

The same table for the whole chain:

| rows dropped | n | B4 | MC2 | improvement [95%] |
|---|---|---|---|---|
| 0 | 339046 | 334.2 | 232.2 | +102.0 [+75.7, +144.8] |
| 1 | 339045 | 314.5 | 232.1 | +82.4 [+73.7, +93.2] |
| 2 | 339044 | 309.8 | 232.1 | +77.6 [+72.5, +83.1] |
| 5 | 339041 | 308.4 | 232.0 | +76.4 [+71.4, +81.6] |
| 20 | 339026 | 305.2 | 231.4 | +73.9 [+69.3, +78.6] |
| 100 | 338946 | 297.1 | 227.9 | +69.2 [+65.5, +72.8] |
| 500 | 338546 | 280.9 | 218.6 | +62.3 [+59.3, +65.2] |
| 3390 | 335656 | 248.1 | 198.1 | +50.1 [+48.0, +52.1] |

## 3. The baseline moved under A1

B4 was measured on 2026-09-04, before the A1 anchor guard. A1 was re-verdicted against the champions (H29 to H32) and never against B4, so the chain's starting point above is not the number the ledger records. Both fits below use the anchor-dependent columns `z_anchor, queue, dep_before, dep_after, arr_anchor, arr_off, push_anchor, occ, t_occ` restored from the snapshot `python -m taxi.a1 before` wrote.

| B4 floored on | RMSE | paired, new minus old [95%] |
|---|---|---|
| the pre-A1 anchors | 306.8 | |
| today's anchors | 334.2 | -27.4 [-71.8, -2.4] |

A1 changed z_anchor on 97 of 339,046 lane-A holdout rows and the queue count on 94, and it costs the linear baseline 27.4 s. Rows changed per column: {'z_anchor': 97, 'queue': 94, 'dep_before': 95, 'dep_after': 0, 'arr_anchor': 88, 'arr_off': 0, 'push_anchor': 4437, 'occ': 31, 't_occ': 31}. The guard is a data-honesty fix (F11: an anchor after takeoff cannot be known at prediction time), so it stays; the consequence for reporting is that every improvement quoted against B4 on today's data is about 27 s larger than the same improvement on the anchors B4 was measured on. Section 1's chain reads +102.0 s today and +74.7 s against the old baseline; both clear the 37.9 s bound.
