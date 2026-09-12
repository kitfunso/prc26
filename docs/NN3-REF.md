# NN3-REF: the incumbent NN cell on NN3's own instrument (diagnostic, never a verdict)

Regenerate with `python -m taxi.nn3ref`. NN2's cell, one seed, the same three folds ((2, 3), (6, 8), (11, 12)) and the same pooled out-of-fold blend against MC2 that every NN3 trial is scored on. It is the reading NN3's trial table needs to be interpretable: the declared grid cannot express this cell, so it is measured beside the search rather than inside it, and it does not count toward k.

| cell | blend RMSE | net alone | agreement |
|---|---|---|---|
| NN2 incumbent | 214.05 | 223.40 | 0.9196 |

Any NN3 trial under this blend RMSE is a cell the search found and the campaign did not have.

