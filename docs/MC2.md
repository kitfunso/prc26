# MC2: CatBoost (GPU) in the M2 recipe, Jan+Jul 2025 development holdout

Train months (2, 3, 4, 5, 6, 8, 9, 10, 11, 12); regenerate with `python -m taxi.mc1 MC2`. Cell {'depth': 8, 'learning_rate': 0.1, 'l2_leaf_reg': 3.0, 'border_count': 254, 'random_seed': 0, 'task_type': 'GPU'}, up to 5000 iterations, early stopping 100, on D1's 49 columns; k = 1. Champion D1 refit at its recorded rounds on the same rows. CV plus the final fit took 39.5 min of wall clock.

## Month-pair CV

| fold | MC2 RMSE (s) | MC2 trees (clf, tail, body) | fit s |
|---|---|---|---|
| (2, 3) | 214.2 | [1125, 720, 3912] | 340 |
| (4, 5) | 200.2 | [2772, 2549, 4989] | 561 |
| (6, 8) | 224.8 | [1537, 949, 2042] | 260 |
| (9, 10) | 216.6 | [2051, 2352, 4066] | 445 |
| (11, 12) | 205.5 | [1535, 1093, 3622] | 372 |

CV RMSE over the five folds: MC2 212.6 s (D1 on the same rows: see docs/D1.md); final MC2 trees (1804, 1532, 3726).

Wording note: the process that wrote this file started before the CV-summary fix in `taxi/mc1.py` (it named MC1 and W1-A here); the names were corrected by hand, the numbers are untouched.

## Holdout verdict vs champion D1 (339046 rows; tail rows 25898)

| model | RMSE | Jan | Jul | tail rows | body rows |
|---|---|---|---|---|---|
| D1 | 242.2 | 217.9 | 260.3 | 627.0 | 176.0 |
| MC2 | 232.6 | 206.9 | 251.6 | 590.6 | 172.4 |

Improvement MC2 vs D1: +9.6 s [+8.2, +10.9]; paired SE 0.7 s; optimism bound for k = 1: 0.0 s; floor 2 s. Verdict: WIN, clears bound plus floor, MC2 promoted.
