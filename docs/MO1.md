# MO1: monotone constraints on lane A, the register's last EMPTY row

Regenerate with `python -m taxi.mo1`. Declared in protocol section 4 before the code was written. Two arms, both a FIXED cell with no search, k = 1: M1's recorded lane-A cell {'num_leaves': 31, 'min_data_in_leaf': 200, 'objective': 'l2'} at 1016 rounds, once free and once with `monotone_constraints_method="advanced"`. Folds are the three declared month pairs ((2, 3), (6, 8), (11, 12)); the comparator is MC2's cached out-of-fold vector on those rows, the same one FAM1, FAM2 and FAM3 used. Total 3 min.

`docs/MONO.md` left this rung open on two conditions. The first is met here by construction: the sign is a Spearman rank correlation against `c` on each fold's OWN training rows, so no judged row ever chose the constraint imposed on it, and a column whose |rho| falls under 0.02 is left free. The second condition is the question this file answers, because a constraint is regularisation and lane A fits 1,723,531 rows.

## The sign each fold read, and imposed

| fold | rows | queue | dep_before | dep_after | occ | t_occ | arr_off | dur | w_sknt | w_gust | w_vsby | w_ceil |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (2, 3) | 1,417,542 | -1 | -1 | -1 | +1 | +1 | -1 | +1 | -1 | 0 | -1 | -1 |
| (6, 8) | 1,354,039 | -1 | -1 | -1 | +1 | +1 | -1 | +1 | -1 | 0 | -1 | -1 |
| (11, 12) | 1,398,527 | -1 | -1 | -1 | +1 | +1 | -1 | +1 | -1 | 0 | 0 | 0 |

## The screen

Gate declared in advance and identical to FAM1, FAM2 and FAM3: an arm meets the untouched holdout only if its blend with MC2 beats MC2 alone by 1.0 s on these rows. A failure is a closed register row and costs no holdout fit.

| arm | what it is | own RMSE | blend with MC2 | gain over MC2 | unclipped weight | agreement | outcome |
|---|---|---|---|---|---|---|---|
| MO1-free | M1 lane-A cell, unconstrained (the control) | 221.16 | 214.35 | +1.03 | +0.2790 | 0.9405 | PASS |
| MO1-mono | the same cell, monotone on the measured sign | 221.71 | 214.43 | +0.96 | +0.2646 | 0.9389 | closed at this cell |

## What the constraint itself cost

The two arms differ in one parameter, so the constraint's own effect is the difference between them and not a comparison across families: **+0.55 s** of out-of-fold lane A, 221.16 free against 221.71 constrained.

## The arm failed the declared gate

The constrained blend with MC2 buys +0.96 s against a 1.0 s gate, so the arm never met the holdout. The register row closes on a measured constrained fit rather than on the argument that a constraint would be imposed backwards.
