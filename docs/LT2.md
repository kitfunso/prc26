# LT2: a tuned piecewise-linear tree, searched on the blend

Regenerate with `python -m taxi.lt2`. Declared in `docs/EXPERIMENT-PROTOCOL.md` section 4 before this module existed. 40 completed trials of a declared 40, TPE seed 0, 2.5 h of a 14 h clock.

## The winning cell

| axis | value |
|---|---|
| colsample_bytree | 0.5395426398695684 |
| learning_rate | 0.028036194537419727 |
| linear_lambda | 0.03132539945794413 |
| min_child_samples | 1097 |
| n_estimators | 2140 |
| num_leaves | 200 |
| reg_lambda | 18.82437023447042 |

## Read 1: the months no cell was ever ranked on

The weight is fitted on the 1,000,485 rows of the search months and never re-fitted. The verdict is read on the 723,046 rows of months 4, 5, 9 and 10, which no cell in this campaign was chosen on, so the cell and the weight are both fixed by rows disjoint from every judged row and the optimism bound is 0.0 s.

| rows | MC2 out of fold | blend | gain |
|---|---|---|---|
| search months, 1,000,485 | 215.38 | 212.43 | +2.96 |
| held months, 723,046 | 208.82 | 206.81 | +2.01 |

The gap between those two gains is the SELECTION advantage this cell bought from the 40 trials, and it is measured here rather than bounded.

**PASS**: +2.01 s [+1.69, +2.33] against a 2 s floor and a 0.0 s bound, weight 0.5307.

## Read 2: the untouched holdout

Holdout, LT2 alone 232.97 s, blended onto SB3 at the SAME weight 0.5307 fitted out of fold on the search months: Improvement LT2+SB3 vs SB3: +0.9 s [+0.3, +1.6]; paired SE 0.3 s; optimism bound for k = 40: 0.7 s; floor 2 s. Verdict: tie or lose, SB3 stays.

