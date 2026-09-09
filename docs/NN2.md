# NN2: NN1 bagged and trained out, Jan + Jul 2025 development holdout

Regenerate with `python -m taxi.nn2`. Declared in protocol section 4 before the code was written; k = 1, no search. NN1's cell unchanged ({'hidden': (512, 256, 128), 'dropout': 0.1, 'lr': 0.001, 'weight_decay': 0.0001, 'batch': 4096, 'max_epochs': 60, 'patience': 5}) except patience 5 to 15, bagged over seeds [0, 1, 2, 3, 4] at both stages. Wall 41.4 min.

## Out-of-fold folds

| fold | bagged RMSE (s) | epochs per seed | fit s |
|---|---|---|---|
| (2, 3) | 218.9 | [30, 29, 33, 38, 40] | 456 |
| (4, 5) | 201.8 | [33, 29, 33, 31, 28] | 397 |
| (6, 8) | 225.7 | [38, 28, 27, 27, 26] | 373 |
| (9, 10) | 218.0 | [27, 29, 30, 30, 27] | 368 |
| (11, 12) | 208.3 | [35, 37, 31, 29, 30] | 424 |

Weight on NN2, fitted out of fold against MC2's cached out-of-fold predictions on 1,723,531 rows: **0.4160**. The holdout is never read to choose it. The base is MC2 and the challenger blends against SB3, an inherited mis-specification from NN1 priced there at 0.01 s (fitted 0.2862 against an oracle 0.264).

## Holdout (339,046 rows)

| model | RMSE | Jan | Jul |
|---|---|---|---|
| SB3 (champion) | 230.10 | 204.37 | 249.11 |
| NN2 alone (diagnostic) | 234.22 | 211.59 | 251.17 |
| SB3 + NN2 at w=0.4160 (challenger) | 228.41 | 203.94 | 246.58 |

NN2 reaches 234.22 s at error agreement **0.9403** with SB3, against NN1's 239.38 s at 0.9169. The frontier in `docs/BF.md` asks for agreement under 0.931 near this level. Oracle weight 0.648 on SB3 reaching 228.36 s, a gain of +1.74 s.

Improvement SB3 + NN2 vs SB3: +1.7 s [+1.1, +2.3]; paired SE 0.3 s; optimism bound for k = 1: 0.0 s; floor 2 s. Verdict: tie or lose, SB3 stays.
