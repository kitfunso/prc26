# L6T: lane-B gated mixture threshold sweep, Jan+Jul 2025 development holdout

Train months (2, 3, 4, 5, 6, 8, 9, 10, 11, 12). Touched rows as L4 (['LIRF']); every other row keeps L4b. Regenerate with `python -m taxi.l6t`. L6a's rule (P(copy) x z_sched + (1 - P(copy)) x the band's non-copy mean, P(copy) from the shrunk airline x band table with 10 pseudo-rows) with the band's non-copy minimum at (10, 15, 20, 25) training rows; L6a itself is L6T20, so that cell is the champion under another name and a CV tie keeps L6a. Every cell and the champion carry the floor at 0 s (H22). k = 4; the CV winner alone meets the holdout.

## Month-pair CV (lane-B rows of the fold)

| fold | L6a RMSE | L6T10 RMSE | L6T15 RMSE | L6T20 RMSE | L6T25 RMSE |
|---|---|---|---|---|---|
| (2, 3) | 1650.6 | 1657.7 | 1650.6 | 1650.6 | 1650.6 |
| (4, 5) | 1956.4 | 1954.5 | 1956.4 | 1956.4 | 1956.4 |
| (6, 8) | 1633.9 | 1633.9 | 1633.9 | 1633.9 | 1633.9 |
| (9, 10) | 1103.9 | 1103.9 | 1103.9 | 1103.9 | 1103.9 |
| (11, 12) | 1159.9 | 1158.3 | 1159.9 | 1159.9 | 1159.9 |

CV RMSE over the five folds (every cell floored at 0 s): L6a 1501.5 s, L6T10 1501.9 s, L6T15 1501.5 s, L6T20 1501.5 s, L6T25 1501.5 s; best declared cell L6T15.

## Holdout verdict vs champion L6a with the floor (5373 rows; L6T15 differs from it on 0 rows)

| model | RMSE | Jan | Jul | min error | max error |
|---|---|---|---|---|---|
| L6a floored | 1861.8 | 2902.7 | 1288.0 | -83125 | 16549 |
| L6T15 floored | 1861.8 | 2902.7 | 1288.0 | -83125 | 16549 |

Improvement L6T15 vs L6a: +0.0 s [+0.0, +0.0]; paired SE 0.0 s; optimism bound for k = 4: 0.0 s; floor 2 s. Verdict: tie or lose, L6a stays.

## By airport (reported)

| airport | rows | L6a RMSE | L6T15 RMSE | share of B4 squared error |
|---|---|---|---|---|
| EDDF | 599 | 417 | 417 | 0.6% |
| EDDM | 237 | 679 | 679 | 0.6% |
| EGLL | 466 | 1162 | 1162 | 3.4% |
| EHAM | 808 | 719 | 719 | 2.2% |
| LEBL | 496 | 367 | 367 | 0.4% |
| LEMD | 309 | 346 | 346 | 0.2% |
| LFPG | 880 | 3449 | 3449 | 56.2% |
| LIRF | 397 | 3957 | 3957 | 33.4% |
| LSZH | 524 | 670 | 670 | 1.3% |
| LTFM | 657 | 720 | 720 | 1.8% |

## Touched rows by band (reported, 397 rows)

| band | rows | L6a RMSE | L6T15 RMSE |
|---|---|---|---|
| 0 | 5 | 1082 | 1082 |
| 1 | 17 | 2328 | 2328 |
| 2 | 194 | 1876 | 1876 |
| 3 | 136 | 4600 | 4600 |
| 4 | 29 | 5654 | 5654 |
| 5 | 5 | 230 | 230 |
| 6 | 11 | 12172 | 12172 |

All rows with L6a (reported, never decides): 327.7 s (lane A MC2 cached 232.7 s on 339046 rows, lane B 1861.8 s on 5373 rows); with L6T15: 327.7 s (lane A MC2 cached 232.7 s on 339046 rows, lane B 1861.8 s on 5373 rows).
