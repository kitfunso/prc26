# FAM1: the four empty model-class rows, lane A screen

Regenerate with `python -m taxi.fam1`. Declared in protocol section 4 before the code was written, and declared as COVERAGE rather than as a champion hunt. Four arms, one fixed cell each, NO search inside any arm, k = 4. Folds are the three declared month pairs ((2, 3), (6, 8), (11, 12)) that O2 and NN3 use, so the comparator needs no new fit: MC2's cached out-of-fold vector on the same rows. Total 0.1 h of a 8 h budget.

These four families were the empty rows of the model-class register in `docs/COVERAGE.md` section 4. That register exists because this campaign lost a whole algorithm family for 60 ledger rows: section 4 answered "what model" as prose indexed by lane id, so a class nobody tried produced no row at all and the section read as complete. Each arm below now closes its row with a number instead of an argument.

## The screen

Gate declared in advance: an arm meets the untouched holdout only if its blend with MC2 beats MC2 alone by 1.0 s on these rows. An arm that fails is a closed register row and costs no holdout fit.

| arm | family | own RMSE | blend with MC2 | gain over MC2 | unclipped weight | agreement | outcome |
|---|---|---|---|---|---|---|---|
| FAM1-nys | kernel (RBF via Nystroem, then ridge) | 282.80 | 215.38 | +0.00 | -0.0230 | 0.7737 | closed at this cell |
| FAM1-et | randomised trees | 264.62 | 215.38 | +0.00 | -0.0411 | 0.8297 | closed at this cell |
| FAM1-rf | bagged trees (random forest) | 240.05 | 215.38 | +0.00 | -0.0355 | 0.9044 | closed at this cell |
| FAM1-knn | k-nearest neighbours | 290.20 | 215.38 | +0.00 | -0.0182 | 0.7528 | closed at this cell |

## No arm passed the gate

Every arm failed the declared 1.0 s gate, so none met the holdout. That is the lane's result and it was the pre-registered prediction for three of the four. The four register rows are now closed by measurement rather than by argument, which is what this lane was declared to do.

