# FAM2: the model-class rows the register left thin, lane A screen

Regenerate with `python -m taxi.fam2`. Declared in protocol section 4 before the code was written. 8 arms, one fixed cell each, NO search inside any arm, k = 8. Folds are the three declared month pairs ((2, 3), (6, 8), (11, 12)) that FAM1, O2 and NN3 use, and the comparator is the one FAM1 used: MC2's cached out-of-fold vector on those rows. Total 1.5 h of a 8 h budget.

FAM1 emptied the four register rows that stood blank. These are the rows that were thin rather than blank, which a register cannot distinguish: penalised linear named lasso and elastic net but instantiated only ridge; robust linear read 'lane B only'; and three classes had no row at all. Every linear arm shares RC1's winning encoder ({'min_freq': 20, 'knots': 4, 'degree': 3, 'wins': None}), so an arm differs from RC1 only by its model class and the comparison is not confounded by the basis.

## The screen

Gate declared in advance, identical to FAM1: an arm meets the untouched holdout only if its blend with MC2 beats MC2 alone by 1.0 s on these rows. An arm that fails is a closed register row and costs no holdout fit.

| arm | family | own RMSE | blend with MC2 | gain over MC2 | unclipped weight | agreement | outcome |
|---|---|---|---|---|---|---|---|
| FAM2-lt | piecewise-linear boosted trees (LightGBM linear_tree) | 219.36 | 214.23 | +1.15 | +0.3207 | 0.9492 | PASS |
| FAM2-pls | latent-factor regression (partial least squares) | 295.98 | 215.38 | +0.00 | -0.0249 | 0.7431 | closed at this cell |
| FAM2-brr | Bayesian linear (evidence-maximised ridge) | 295.57 | 215.38 | +0.00 | -0.0249 | 0.7439 | closed at this cell |
| FAM2-twd | non-Gaussian GLM (Tweedie, log link, on the level) | 272.42 | 215.38 | +0.00 | -0.0202 | 0.7999 | closed at this cell |
| FAM2-en | sparse linear (elastic net) | 295.87 | 215.38 | +0.00 | -0.0136 | 0.7365 | closed at this cell |
| FAM2-svr | linear support-vector regression, epsilon-insensitive loss | 274.98 | 215.38 | +0.00 | -0.0429 | 0.8027 | closed at this cell |
| FAM2-hub | robust linear (Huber), the lane-A row the register leaves to lane B | 269.42 | 215.38 | +0.00 | -0.0352 | 0.8143 | closed at this cell |
| FAM2-ada | adaptive boosting (AdaBoost.R2 on depth-6 trees) | 428.12 | 215.38 | +0.00 | +0.0007 | 0.5020 | closed at this cell |

## Verdicts against SB3

- **FAM2-lt + SB3** at w=0.3207, holdout 237.50 s alone: Improvement FAM2-lt vs SB3: +0.1 s [-0.4, +0.6]; paired SE 0.3 s; optimism bound for k = 8: 0.4 s; floor 2 s. Verdict: tie or lose, SB3 stays.

## Not done, and why

- **Explainable boosting machine (GA2M).** `interpret` is not installed on this box. It is the one remaining named family with a mechanism story this lane does not cover, so it is a NOT-DONE row rather than a closure by argument.

