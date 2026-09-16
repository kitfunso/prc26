# Lane B mechanism model: last `laneb.py cv` run (the analysis is docs/LANEB.md)

`PYTHONPATH=/c/Users/skf_s/prc26 python laneb.py cv mix_cb_wx` then `python laneb.py fit 117 238`.

Result: **FAIL** vs donor mix_cb_wx (1900.6). New 1736.9, donor 1900.6, bootstrap delta +163.7 [+50.2, +341.9]. Copy 2834.9 vs donor 2707.4 (worse).

Shift rule: LIRF z_sched > 57600s, n=7 train rows, empirical rate 0.714.

Day-feature multiclass log loss on CV folds: with 0.1211, without 0.1212 (day features help).

Donor blend: P(copy) > 0.12 rows take 0.10*ours + 0.90*donor (tau misjudges the remainder there).

Hull: LIRF rows with z_sched >= 14400s are projected onto the convex hull of their feasible labels (`hull()`).

Hull prior: before `hull()` those rows take P(copy) z + (1 - P(copy)) (1100 + 86400 kn), P(copy) the Laplace-smoothed copy share of their cell among the training rows (`hull_prior()`).

## By airport (holdout lane B)

| apt   |   n |   rmse_new |   rmse_donor |
|:------|----:|-----------:|-------------:|
| EDDF  | 599 |      379.1 |        366.1 |
| EDDM  | 237 |      616.2 |        613.9 |
| EGLL  | 466 |     1072.9 |       1062.5 |
| EHAM  | 808 |      535.7 |        565.8 |
| LEBL  | 496 |      315.6 |        321.9 |
| LEMD  | 309 |      276.2 |        337.8 |
| LFPG  | 880 |     3448.9 |       3436.9 |
| LIRF  | 397 |     3299.7 |       4362.9 |
| LSZH  | 524 |      608.9 |        607.8 |
| LTFM  | 657 |      561.7 |        582   |

## By mechanism class (holdout lane B)

| mech   |    n |   rmse_new |   rmse_donor |
|:-------|-----:|-----------:|-------------:|
| copy   |  253 |     2834.9 |       2707.4 |
| near   |   39 |     4640.2 |       4752.6 |
| phys   | 5075 |     1099.8 |       1153   |
| shift  |    6 |    34636.4 |      40674   |

## Ablation (holdout lane B RMSE, one piece removed)

| variant | rmse |
|---|---|
| full | 1736.9 |
| donor_only | 1900.6 |
| no_blend | 1740.5 |
| no_prior | 1798.7 |
| no_shift | 1736.9 |
| no_day | 1733.9 |
| no_hist | 1742.5 |

Overall holdout (344,419 rows, lane A unchanged): donor blend 326.99 -> lane B swapped 312.46.
