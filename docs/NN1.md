# NN1: a neural net as a blend partner for SB3, Jan+Jul 2025 development holdout

Regenerate with `python -m taxi.nn1`. Declared in protocol section 4 before the code was written; k = 1, one cell, floor 2 s. Cell {'hidden': (512, 256, 128), 'dropout': 0.1, 'lr': 0.001, 'weight_decay': 0.0001, 'batch': 4096, 'max_epochs': 60, 'patience': 5}, 10 embedded categorical columns and 39 numeric on D1's 49 columns, fitting the correction `c` directly with no tail/body split. Device cuda. CV plus the final fit took 4.3 min.

## Out-of-fold folds

| fold | NN1 RMSE (s) | epochs | fit s |
|---|---|---|---|
| (2, 3) | 223.6 | 20 | 48 |
| (4, 5) | 206.4 | 17 | 38 |
| (6, 8) | 231.0 | 21 | 46 |
| (9, 10) | 224.1 | 17 | 38 |
| (11, 12) | 215.0 | 17 | 38 |

Weight on NN1, fitted out of fold against MC2's cached out-of-fold predictions on 1,723,531 rows: **0.2862**. The holdout is never read to choose it.

## Holdout (339,046 rows)

| model | RMSE | Jan | Jul |
|---|---|---|---|
| SB3 (champion) | 230.10 | 204.37 | 249.11 |
| NN1 alone (diagnostic) | 239.38 | 216.77 | 256.34 |
| SB3 + NN1 at w=0.2862 (challenger) | 228.70 | 203.87 | 247.10 |

NN1's holdout error agrees **0.9169** with SB3's (uncentred correlation, `taxi.bf.agreement`), at 239.38 s against SB3's 230.10 s. The oracle weight, read off the holdout itself and so unavailable in practice, puts 0.736 on SB3 and reaches 228.69 s, a gain of +1.40 s.

Improvement SB3 + NN1 vs SB3: +1.4 s [+0.9, +1.9]; paired SE 0.2 s; optimism bound for k = 1: 0.0 s; floor 2 s. Verdict: tie or lose, SB3 stays.
