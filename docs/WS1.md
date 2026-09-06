# WS1: Weather join-shift stress, Jan+Jul 2025 development holdout

Regenerate with `python -m taxi.lanes WS1`. Weather join-shift stress: the as-of join reads the latest observation at or before the anchor minus 600 s; MC2's cell at its recorded rounds on the shifted columns, no CV. A loss says the join is fragile to observation latency, a tie says it is not. Cells: WS1. k = 1; the CV winner alone meets the holdout against MC2's cached holdout predictions. MC2 on the same folds and data (`taxi.champ.champion_oof`): CV 212.9 s.

## WS1: month-pair CV

No CV: fit at MC2's recorded rounds (1804, 1532, 3726).

## Holdout verdict vs champion MC2 (339046 rows; tail rows 25898; WS1 fit at rounds (1804, 1532, 3726))

| model | RMSE | Jan | Jul | tail rows | body rows |
|---|---|---|---|---|---|
| MC2 | 232.7 | 206.8 | 251.9 | 591.0 | 172.5 |
| WS1 | 233.2 | 206.8 | 252.7 | 593.4 | 172.5 |

Improvement WS1 vs MC2: -0.5 s [-0.9, -0.0]; paired SE 0.2 s; optimism bound for k = 1: 0.0 s; floor 2 s. Verdict: tie or lose, MC2 stays.
