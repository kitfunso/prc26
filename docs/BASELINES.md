# Baselines B0-B4 on the Jan+Jul 2025 development holdout

Train months (2, 3, 4, 5, 6, 8, 9, 10, 11, 12); regenerate with `python -m taxi.baselines`.

| rows | training | holdout |
|---|---|---|
| lane A (flight plan matched) | 1723531 | 339046 |
| lane B (no flight plan) | 17097 | 5373 |

## Lane B: admission-cap sweep

| lane B cap | CV RMSE (s) |
|---|---|
| 3600 | 3207.5 |
| 7200 | 3046.0 |
| 14400 | 2932.9 |
| none | 2888.7 |

Chosen cap, lane B: none (least trimming among caps within 2 s of the best CV RMSE).

## Lane B: champion ladder (cap none)

| lane B | RMSE | Jan | Jul | improvement vs champion [95%] | verdict (floor 2 s) |
|---|---|---|---|---|---|
| B0 | 3959.4 | 4280.0 | 3835.9 | first rung | champion |
| B1 | 3709.9 | 4089.0 | 3561.8 | B0: +249.4 s [+135.5, +353.9] | WIN, promoted |
| B2 | 3807.5 | 4243.4 | 3635.7 | B1: -97.6 s [-232.9, +15.4] | tie or lose |
| B3s | 4889.7 | 5415.9 | 4683.2 | B1: -1179.7 s [-2262.0, -214.0] | tie or lose |
| B4 | 2074.1 | 2939.3 | 1649.7 | B1: +1635.8 s [+733.6, +2538.9] | WIN, promoted |

## Lane A: admission-cap sweep

| lane A cap | CV RMSE (s) |
|---|---|
| 3600 | 279.7 |
| 7200 | 279.2 |
| 14400 | 279.2 |
| none | 279.2 |

Chosen cap, lane A: none (least trimming among caps within 2 s of the best CV RMSE).

## Lane A: champion ladder (cap none)

| lane A | RMSE | Jan | Jul | improvement vs champion [95%] | verdict (floor 2 s) |
|---|---|---|---|---|---|
| B0 | 476.7 | 441.8 | 503.4 | first rung | champion |
| B1 | 440.7 | 413.8 | 461.4 | B0: +36.1 s [+31.6, +41.1] | WIN, promoted |
| B2 | 433.1 | 408.3 | 452.3 | B1: +7.6 s [+6.2, +8.9] | WIN, promoted |
| B3a | 413.6 | 355.7 | 455.5 | B2: +19.5 s [-17.9, +73.7] | tie or lose |
| B3b | 620.9 | 528.0 | 687.5 | B2: -187.8 s [-233.8, -132.0] | tie or lose |
| B3c | 709.7 | 611.5 | 780.6 | B2: -276.6 s [-323.6, -220.7] | tie or lose |
| B3d | 400.1 | 348.5 | 437.7 | B2: +33.0 s [-2.5, +86.3] | tie or lose |
| B4 | 311.2 | 279.9 | 334.5 | B2: +121.9 s [+88.7, +173.5] | WIN, promoted |

## All rows (reported, never decides): lane-A rung plus lane-B champion B4

| rung | all-row RMSE |
|---|---|
| B0 | 539.3 |
| B1 | 508.2 |
| B2 | 501.8 |
| B3a | 485.3 |
| B3b | 668.3 |
| B3c | 750.3 |
| B3d | 474.0 |
| B4 | 403.0 |

Champions: lane A B4, lane B B4.
