# FAM3: explainable boosting, the register's last EMPTY row, lane A screen

Regenerate with `python -m taxi.fam3`. Declared in protocol section 4 before the code was written. One arm, one FIXED cell {'interactions': 32, 'outer_bags': 1, 'max_bins': 256, 'random_state': 0}, no search, k = 1. Folds are the three declared month pairs ((2, 3), (6, 8), (11, 12)); the comparator is MC2's cached out-of-fold vector on those rows, the same one FAM1 and FAM2 used. Total 175 min.

`interpret` 0.7.8 is now installed, which is the only thing that ever kept this row empty. A GA2M is additive in every single feature plus a fixed budget of pairwise terms, so it is the one named family whose function class the champion's depth-8 boosting cannot contain.

## The screen

Gate declared in advance and identical to FAM1 and FAM2: the arm meets the untouched holdout only if its blend with MC2 beats MC2 alone by 1.0 s on these rows. A failure is a closed register row and costs no holdout fit.

| arm | family | own RMSE | blend with MC2 | gain over MC2 | unclipped weight | agreement | outcome |
|---|---|---|---|---|---|---|---|
| FAM3-ebm | explainable boosting (GA2M) | 237.10 | 215.26 | +0.13 | +0.0689 | 0.8930 | closed at this cell |

## The arm failed the declared gate

The blend with MC2 buys +0.13 s against a 1.0 s gate, so the arm never met the holdout and the register row closes by measurement rather than by a missing library.
