# RC1: ridge regression searched with Optuna, lane A, Jan + Jul 2025 holdout

Regenerate with `python -m taxi.rc1`. Declared in protocol section 4 before the code was written. Optuna TPE, seed 0, six knobs, 40 completed trials of 40 budgeted, 2.9 h. The campaign's FIRST linear model on lane A: every other lane-A model is a boosted tree or a neural net, so this is both the Stage-3 linear baseline the campaign never had and a blend partner from the most distant family it can reach. The objective is the pooled BLEND RMSE over all five `CV_FOLDS` month pairs against MC2's cached out-of-fold predictions, never ridge's own level, for NN3's reason: what a partner adds is set by level and agreement together.

## Trials

| trial | blend RMSE | ridge alone | agreement |
|---|---|---|---|
| 0 | 212.65 | 261.05 | 0.8180 |
| 1 | 212.65 | 286.78 | 0.7566 |
| 2 | 212.66 | 255.24 | 0.8320 |
| 3 | 212.65 | 265.20 | 0.8130 |
| 4 | 212.65 | 256.83 | 0.8341 |
| 5 | 212.65 | 281.77 | 0.7743 |
| 6 | 212.65 | 255.72 | 0.8346 |
| 7 | 212.65 | 271.17 | 0.7913 |
| 8 | 212.65 | 267.11 | 0.8073 |
| 9 | 212.65 | 277.30 | 0.7765 |
| 10 | 212.65 | 261.94 | 0.8155 |
| 11 | 212.65 | 287.69 | 0.7543 |
| 12 | 212.65 | 260.34 | 0.8222 |
| 13 | 212.65 | 267.77 | 0.8044 |
| 14 | 212.65 | 271.35 | 0.7907 |
| 15 | 212.65 | 256.75 | 0.8238 |
| 16 | 212.65 | 264.26 | 0.8110 |
| 17 | 212.65 | 261.03 | 0.8179 |
| 18 | 212.65 | 256.79 | 0.8328 |
| 19 | 212.65 | 261.09 | 0.8181 |
| 20 | 212.65 | 256.57 | 0.8307 |
| 21 | 212.65 | 279.76 | 0.7744 |
| 22 | 212.65 | 271.35 | 0.7907 |
| 23 | 212.65 | 256.66 | 0.8244 |
| 24 | 212.65 | 256.62 | 0.8248 |
| 25 | 212.65 | 256.66 | 0.8244 |
| 26 | 212.65 | 257.53 | 0.8265 |
| 27 | 212.65 | 257.05 | 0.8279 |
| 28 | 212.66 | 259.55 | 0.8185 |
| 29 | 212.65 | 256.75 | 0.8238 |
| 30 | 212.65 | 256.75 | 0.8238 |
| 31 | 212.65 | 256.75 | 0.8238 |
| 32 | 212.65 | 256.75 | 0.8238 |
| 33 | 212.65 | 256.75 | 0.8238 |
| 34 | 212.65 | 264.30 | 0.8117 |
| 35 | 212.65 | 256.75 | 0.8238 |
| 36 | 212.65 | 255.76 | 0.8328 |
| 37 | 212.65 | 256.73 | 0.8304 |
| 38 | 212.65 | 265.39 | 0.8121 |
| 39 | 212.65 | 256.75 | 0.8238 |

Best cell on the blend objective: `{'alpha': 0.02153437905139501, 'min_freq': 20, 'knots': 4, 'degree': 3, 'wins': None, 'cols': 'd2'}`.

## The Stage-3 linear baseline

The best RIDGE-ALONE trial is selected on a different criterion from the same 40: trial 2 at **255.24 s** of cross-validated RMSE, against D2's recorded 211.8 s on the same five folds. That gap is the answer Stage 4 asks for, that a richer class must beat the simple one, and it is reported here whichever way it falls.

Weight on RC1, fitted out of fold on 1,723,531 rows: **0.0136**. The holdout is never read to choose it.

## Holdout (339,046 rows)

| model | RMSE |
|---|---|
| SB3 (champion) | 230.10 |
| RC1 alone (diagnostic) | 485.68 |
| SB3 + RC1 at w=0.0136 (challenger) | 230.16 |

RC1 reaches 485.68 s at agreement **0.4722** with SB3, against NN3's 233.48 s. The oracle weight puts 0.999 on SB3 and reaches 230.09 s, +0.00 s, which upper-bounds any weight an out-of-fold choice could have picked.

Improvement SB3 + RC1 vs SB3: -0.1 s [-0.1, -0.0]; paired SE 0.0 s; optimism bound for k = 40: 0.0 s; floor 2 s. Verdict: tie or lose, SB3 stays.

