# Lane B mechanism model: last `laneb.py cv` run (the analysis is docs/LANEB.md)

`PYTHONPATH=/c/Users/skf_s/prc26 python laneb.py cv mix_cb_wx` then `python laneb.py fit 100 449`.

Result: **FAIL** vs donor mix_cb_wx (1900.6). New 1816.9, donor 1900.6, bootstrap delta +83.7 [+13.1, +194.4]. Copy 2710.1 vs donor 2707.4 (worse).

Shift rule: LIRF z_sched > 57600s, n=7 train rows, empirical rate 0.714.

Day-feature multiclass log loss on CV folds: with 0.1231, without 0.1227 (day features do not help).

Donor blend: P(copy) > 0.12 rows take 0.10*ours + 0.90*donor (tau misjudges the remainder there).

Hull: LIRF rows with z_sched >= 14400s are projected onto the convex hull of their feasible labels (`hull()`).

## By airport (holdout lane B)

| apt   |   n |   rmse_new |   rmse_donor |
|:------|----:|-----------:|-------------:|
| EDDF  | 599 |      386.3 |        366.1 |
| EDDM  | 237 |      631.5 |        613.9 |
| EGLL  | 466 |     1090.4 |       1062.5 |
| EHAM  | 808 |      553.7 |        565.8 |
| LEBL  | 496 |      337.3 |        321.9 |
| LEMD  | 309 |      317   |        337.8 |
| LFPG  | 880 |     3451.1 |       3436.9 |
| LIRF  | 397 |     3809.8 |       4362.9 |
| LSZH  | 524 |      604.5 |        607.8 |
| LTFM  | 657 |      591   |        582   |

## By mechanism class (holdout lane B)

| mech   |    n |   rmse_new |   rmse_donor |
|:-------|-----:|-----------:|-------------:|
| copy   |  253 |     2710.1 |       2707.4 |
| near   |   39 |     4649.1 |       4752.6 |
| phys   | 5075 |     1137.1 |       1153   |
| shift  |    6 |    37580.9 |      40674   |

## Ablation (holdout lane B RMSE, one piece removed)

| variant | rmse |
|---|---|
| full | 1816.9 |
| donor_only | 1900.6 |
| no_shift | 1829.1 |
| no_blend | 1783.9 |
| no_day | 1815.3 |

Overall holdout (344,419 rows, lane A unchanged): donor blend 326.99 -> lane B swapped 319.48.
