# TH1: Tail split sweep, Jan+Jul 2025 development holdout

Regenerate with `python -m taxi.lanes TH1`. Tail split sweep: the |c| threshold that splits MC2's two regressors, at 400, 900 and 1,200 s (MC2 uses 600) on MC2's cell; the classifier target moves with it. The CV winner meets the holdout only if its CV beats MC2's own CV on the same folds and data. Cells: TH400, TH900, TH1200. k = 4; the CV winner alone meets the holdout against MC2's cached holdout predictions. MC2 on the same folds and data (`taxi.champ.champion_oof`): CV 212.9 s.

## TH400: month-pair CV

| fold | TH400 RMSE (s) | TH400 trees (clf, tail, body) | fit s |
|---|---|---|---|
| (2, 3) | 213.9 | [2028, 2337, 3764] | 438 |
| (4, 5) | 201.3 | [2251, 3817, 3960] | 516 |
| (6, 8) | 225.6 | [1690, 2199, 2577] | 375 |
| (9, 10) | 218.1 | [2753, 2900, 3921] | 532 |
| (11, 12) | 205.9 | [2198, 2804, 3608] | 460 |

CV RMSE over the five folds: TH400 213.4 s (MC2 on the same rows: see its report); final TH400 trees (2184, 2811, 3566).

## TH900: month-pair CV

| fold | TH900 RMSE (s) | TH900 trees (clf, tail, body) | fit s |
|---|---|---|---|
| (2, 3) | 212.7 | [854, 904, 3636] | 336 |
| (4, 5) | 199.5 | [1986, 1298, 5000] | 495 |
| (6, 8) | 224.5 | [1310, 2240, 2009] | 345 |
| (9, 10) | 216.6 | [2275, 1064, 4496] | 481 |
| (11, 12) | 204.9 | [1145, 1397, 4456] | 426 |

CV RMSE over the five folds: TH900 212.0 s (MC2 on the same rows: see its report); final TH900 trees (1514, 1380, 3919).

## TH1200: month-pair CV

| fold | TH1200 RMSE (s) | TH1200 trees (clf, tail, body) | fit s |
|---|---|---|---|
| (2, 3) | 213.7 | [434, 948, 3246] | 284 |
| (4, 5) | 199.9 | [1234, 2221, 4904] | 493 |
| (6, 8) | 224.1 | [1052, 2752, 1863] | 345 |
| (9, 10) | 216.4 | [851, 1263, 4993] | 418 |
| (11, 12) | 205.1 | [873, 284, 3627] | 303 |

CV RMSE over the five folds: TH1200 212.2 s (MC2 on the same rows: see its report); final TH1200 trees (888, 1493, 3726).

CV RMSE by cell: TH400 213.4 s, TH900 212.0 s, TH1200 212.2 s; MC2 212.9 s; CV winner TH900.

## Holdout verdict vs champion MC2 (339046 rows; tail rows 25898; TH900 fit at rounds (1514, 1380, 3919))

| model | RMSE | Jan | Jul | tail rows | body rows |
|---|---|---|---|---|---|
| MC2 | 232.7 | 206.8 | 251.9 | 591.0 | 172.5 |
| TH900 | 232.0 | 206.8 | 250.7 | 589.8 | 171.8 |

Improvement TH900 vs MC2: +0.7 s [+0.1, +1.4]; paired SE 0.3 s; optimism bound for k = 4: 0.3 s; floor 2 s. Verdict: tie or lose, MC2 stays.
