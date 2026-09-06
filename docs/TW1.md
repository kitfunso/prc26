# TW1: Traffic window sweep, Jan+Jul 2025 development holdout

Regenerate with `python -m taxi.lanes TW1`. Traffic window sweep: the six counts rebuilt at 300, 1,800 and 3,600 s (MC2 uses 900 s) on MC2's cell; the CV winner meets the holdout only if its CV beats MC2's own CV on the same folds and data. Cells: TW300, TW1800, TW3600. k = 4; the CV winner alone meets the holdout against MC2's cached holdout predictions. MC2 on the same folds and data (`taxi.champ.champion_oof`): CV 212.9 s.

## TW300: month-pair CV

| fold | TW300 RMSE (s) | TW300 trees (clf, tail, body) | fit s |
|---|---|---|---|
| (2, 3) | 214.5 | [1412, 640, 3128] | 303 |
| (4, 5) | 200.5 | [2278, 2356, 4872] | 511 |
| (6, 8) | 224.7 | [1710, 1494, 2777] | 336 |
| (9, 10) | 217.3 | [2229, 2034, 3176] | 402 |
| (11, 12) | 204.9 | [2011, 1615, 3641] | 400 |

CV RMSE over the five folds: TW300 212.8 s (MC2 on the same rows: see its report); final TW300 trees (1928, 1627, 3518).

## TW1800: month-pair CV

| fold | TW1800 RMSE (s) | TW1800 trees (clf, tail, body) | fit s |
|---|---|---|---|
| (2, 3) | 214.4 | [1500, 1610, 4064] | 389 |
| (4, 5) | 200.8 | [2299, 1986, 4527] | 473 |
| (6, 8) | 224.9 | [1680, 956, 3377] | 339 |
| (9, 10) | 217.3 | [2590, 1816, 4197] | 474 |
| (11, 12) | 205.6 | [1416, 1841, 4031] | 389 |

CV RMSE over the five folds: TW1800 213.0 s (MC2 on the same rows: see its report); final TW1800 trees (1897, 1641, 4039).

## TW3600: month-pair CV

| fold | TW3600 RMSE (s) | TW3600 trees (clf, tail, body) | fit s |
|---|---|---|---|
| (2, 3) | 215.0 | [783, 708, 2871] | 251 |
| (4, 5) | 201.5 | [2578, 2127, 4997] | 519 |
| (6, 8) | 225.5 | [1228, 1324, 2170] | 263 |
| (9, 10) | 217.7 | [2310, 1616, 3534] | 412 |
| (11, 12) | 206.4 | [1836, 1435, 2564] | 329 |

CV RMSE over the five folds: TW3600 213.6 s (MC2 on the same rows: see its report); final TW3600 trees (1747, 1442, 3227).

CV RMSE by cell: TW300 212.8 s, TW1800 213.0 s, TW3600 213.6 s; MC2 212.9 s; CV winner TW300.

## Holdout verdict vs champion MC2 (339046 rows; tail rows 25898; TW300 fit at rounds (1928, 1627, 3518))

| model | RMSE | Jan | Jul | tail rows | body rows |
|---|---|---|---|---|---|
| MC2 | 232.7 | 206.8 | 251.9 | 591.0 | 172.5 |
| TW300 | 233.2 | 207.0 | 252.5 | 593.5 | 172.5 |

Improvement TW300 vs MC2: -0.5 s [-0.9, -0.1]; paired SE 0.2 s; optimism bound for k = 4: 0.2 s; floor 2 s. Verdict: tie or lose, MC2 stays.
