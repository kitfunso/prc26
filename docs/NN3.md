# NN3: the neural net searched on every parameter, lane A, Jan + Jul 2025 holdout

Regenerate with `python -m taxi.nn3`. Declared in protocol section 4 before the code was written. Optuna TPE, seed 0, 13 knobs, 40 completed trials of 40 budgeted in 0.5 h. The objective is the pooled BLEND RMSE over folds ((2, 3), (6, 8), (11, 12)) at a weight fitted on those rows, never the net's own level: `docs/BF.md` shows the gain is set by level and agreement together, and NN1 to NN2 improved the level 5.16 s while agreement moved the wrong way.

## Trials

| trial | blend RMSE | net alone | agreement |
|---|---|---|---|
| 0 | 213.46 | 220.54 | 0.9250 |
| 1 | 215.19 | 229.57 | 0.9212 |
| 2 | 215.08 | 228.78 | 0.9197 |
| 3 | 213.38 | 221.50 | 0.9173 |
| 4 | 214.93 | 226.70 | 0.9244 |
| 5 | 213.80 | 222.07 | 0.9221 |
| 6 | 214.03 | 223.33 | 0.9193 |
| 7 | 213.94 | 225.36 | 0.9061 |
| 8 | 213.85 | 223.25 | 0.9163 |
| 10 | 213.68 | 221.48 | 0.9231 |
| 11 | 213.09 | 219.38 | 0.9257 |
| 12 | 213.19 | 219.59 | 0.9262 |
| 13 | 212.99 | 219.11 | 0.9257 |
| 14 | 212.86 | 218.96 | 0.9241 |
| 15 | 213.22 | 221.63 | 0.9131 |
| 16 | 213.05 | 218.43 | 0.9315 |
| 17 | 214.18 | 222.46 | 0.9282 |
| 18 | 213.08 | 219.72 | 0.9230 |
| 19 | 213.43 | 220.46 | 0.9248 |
| 20 | 213.78 | 220.64 | 0.9305 |
| 21 | 212.55 | 217.10 | 0.9324 |
| 22 | 212.62 | 218.65 | 0.9217 |
| 23 | 212.71 | 218.71 | 0.9228 |
| 24 | 212.71 | 219.12 | 0.9199 |
| 25 | 212.33 | 217.68 | 0.9236 |
| 27 | 213.04 | 219.42 | 0.9241 |
| 28 | 212.69 | 220.23 | 0.9117 |
| 29 | 212.61 | 217.06 | 0.9337 |
| 30 | 212.69 | 217.59 | 0.9310 |
| 31 | 212.26 | 216.49 | 0.9320 |
| 32 | 212.83 | 218.38 | 0.9274 |
| 33 | 212.72 | 217.78 | 0.9301 |
| 34 | 212.65 | 217.42 | 0.9318 |
| 38 | 212.31 | 218.28 | 0.9185 |
| 39 | 212.04 | 217.39 | 0.9204 |
| 41 | 212.13 | 217.90 | 0.9181 |
| 42 | 212.48 | 218.49 | 0.9203 |
| 43 | 211.96 | 217.02 | 0.9218 |
| 45 | 212.36 | 218.48 | 0.9183 |
| 47 | 212.74 | 218.96 | 0.9214 |

Best cell: `{'hidden': (1024, 1024, 1024), 'dropout': 0.25552142953876766, 'lr': 0.001499640363737197, 'weight_decay': 0.00016460068843348537, 'batch': 1024, 'max_epochs': 150, 't_max': 60, 'patience': 5, 'emb_cap': 16, 'act': 'silu', 'norm': 'batch', 'loss': 'huber', 'huber': 0.4048945158649465, 'cols': 'd2'}`.

## The best cell refit over five folds, bagged over five seeds

| cached | | | |

Weight on NN3, fitted out of fold on 1,723,531 rows: **0.5189**. The holdout is never read to choose it.

## Holdout (339,046 rows)

| model | RMSE |
|---|---|
| SB3 (champion) | 230.10 |
| NN3 alone (diagnostic) | 233.48 |
| SB3 + NN3 at w=0.5189 (challenger) | 228.03 |

NN3 reaches 233.48 s at agreement **0.9344** with SB3, against NN2's 234.22 s at 0.9403 and NN1's 239.38 s at 0.9169. The oracle weight puts 0.611 on SB3 and reaches 227.76 s, +2.33 s.

Improvement SB3 + NN3 vs SB3: +2.1 s [+1.1, +2.9]; paired SE 0.5 s; optimism bound for k = 40: 1.0 s; floor 2 s. Verdict: WIN on the interval but inside bound plus floor, SB3 stays.

