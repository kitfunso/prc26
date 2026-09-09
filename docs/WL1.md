# WL1: Weather trajectory on MC2's cell, Jan+Jul 2025 development holdout

Regenerate with `python -m taxi.lanes WL1`. Weather trajectory on MC2's cell: the five continuous METAR columns (wind, visibility, temperature, ceiling, precipitation) as they read 1 h and 3 h before the anchor, and the change since, twenty columns. W1 gave the model the level at the anchor and nothing else, so a front arriving and a front clearing are the same row to it. Cells: WL1. k = 1; the CV winner alone meets the holdout against MC2's cached holdout predictions. MC2 on the same folds and data (`taxi.champ.champion_oof`): CV 212.9 s.

## WL1: month-pair CV

| fold | WL1 RMSE (s) | WL1 trees (clf, tail, body) | fit s |
|---|---|---|---|
| (2, 3) | 215.2 | [1057, 723, 3291] | 310 |
| (4, 5) | 201.3 | [2766, 1267, 4582] | 503 |
| (6, 8) | 225.5 | [1482, 1426, 2115] | 298 |
| (9, 10) | 218.2 | [2638, 1772, 2754] | 433 |
| (11, 12) | 205.5 | [1591, 2147, 3486] | 423 |

CV RMSE over the five folds: WL1 213.5 s (MC2 on the same rows: see its report); final WL1 trees (1906, 1467, 3245).

CV RMSE by cell: WL1 213.5 s; MC2 212.9 s; CV winner WL1.

## Holdout verdict vs champion MC2 (339046 rows; tail rows 25898; WL1 fit at rounds (1906, 1467, 3245))

| model | RMSE | Jan | Jul | tail rows | body rows |
|---|---|---|---|---|---|
| MC2 | 232.7 | 206.8 | 251.9 | 591.0 | 172.5 |
| WL1 | 233.9 | 207.5 | 253.3 | 596.4 | 172.6 |

Improvement WL1 vs MC2: -1.1 s [-1.7, -0.6]; paired SE 0.3 s; optimism bound for k = 1: 0.0 s; floor 2 s. Verdict: tie or lose, MC2 stays.
