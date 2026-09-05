# MC1: CatBoost (GPU) in the M2 recipe, Jan+Jul 2025 development holdout

Train months (2, 3, 4, 5, 6, 8, 9, 10, 11, 12); regenerate with `python -m taxi.mc1`. Cell {'depth': 8, 'learning_rate': 0.1, 'l2_leaf_reg': 3.0, 'border_count': 254, 'random_seed': 0, 'task_type': 'GPU'}, up to 5000 iterations, early stopping 100, on W1-A's 36 columns; k = 1. Champion W1-A refit at its recorded rounds on the same rows. CV plus the final fit took 30.8 min of wall clock.

## Month-pair CV

| fold | MC1 RMSE (s) | MC1 trees (clf, tail, body) | fit s |
|---|---|---|---|
| (2, 3) | 225.9 | [1170, 1024, 2275] | 307 |
| (4, 5) | 209.8 | [2745, 1708, 4994] | 424 |
| (6, 8) | 236.4 | [1137, 1549, 2513] | 233 |
| (9, 10) | 230.8 | [2539, 1280, 3774] | 347 |
| (11, 12) | 214.1 | [1528, 1349, 2765] | 251 |

CV RMSE over the five folds: MC1 223.9 s (W1-A on the same rows: see docs/W1.md); final MC1 trees (1823, 1382, 3264).

## Holdout verdict vs champion W1-A (339046 rows; tail rows 25898)

| model | RMSE | Jan | Jul | tail rows | body rows |
|---|---|---|---|---|---|
| W1-A | 253.8 | 224.1 | 275.6 | 680.5 | 177.3 |
| MC1 | 248.2 | 217.9 | 270.3 | 658.0 | 175.7 |

Improvement MC1 vs W1-A: +5.6 s [+4.4, +6.8]; paired SE 0.6 s; optimism bound for k = 1: 0.0 s; floor 2 s. Verdict: WIN, clears bound plus floor, MC1 promoted.
