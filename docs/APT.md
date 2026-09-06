# APT: per-airport structure on MC2's out-of-fold lane-A predictions

Diagnostic, never a verdict (protocol section 2). Regenerate with `python -m taxi.apt`. Reads the cached out-of-fold predictions
`data/cache/mc2_oof.parquet` over the ten training months, 1,723,531 lane-A rows, OOF RMSE 212.86 s. It asks one question: does the
global lane-A model leave per-airport structure on the table that a per-airport model would pick up?

## Per airport (out of fold)

| airport | rows | RMSE | bias | share of lane-A squared error |
|---|---|---|---|---|
| LIRF | 133,085 | 341.4 | -3.0 | 19.9% |
| LTFM | 223,438 | 252.3 | -8.7 | 18.2% |
| LFPG | 197,047 | 230.3 | +1.7 | 13.4% |
| EGLL | 198,388 | 221.0 | +1.0 | 12.4% |
| LEMD | 176,276 | 178.1 | +1.5 | 7.2% |
| EDDF | 191,954 | 170.4 | -0.6 | 7.1% |
| LEBL | 149,353 | 193.1 | -0.8 | 7.1% |
| EHAM | 203,726 | 156.7 | -1.6 | 6.4% |
| EDDM | 139,470 | 159.8 | -0.5 | 4.6% |
| LSZH | 110,794 | 162.3 | +0.6 | 3.7% |

RMSE spreads more than two to one across airports, but no airport is biased by more than 8.7 s against its own 156 to 341 s RMSE,
so the global model already carries each airport's level.

## Cross-fitted recalibration (RMSE, lower is better)

A fold's rows are corrected by a per-key shift or a per-key least-squares line fitted on the other folds' out-of-fold rows only,
so the correction never sees the rows it grades. A shift key needs 50 training rows, a line 200; a key below its minimum is left alone.

| correction | RMSE |
|---|---|
| none | 212.86 |
| airport shift | 213.07 |
| airport line | 213.37 |
| airport x runway shift | 213.38 |
| airport x runway line | 213.53 |
| airport x hour shift | 213.29 |
| airport x hour line | 213.86 |
| airport x stand shift | 213.18 |
| airport x stand line | 213.92 |

Every correction is worse than none, by 0.2 to 1.1 s. The four keys are the ones a per-airport model would specialise on, and each is
already a categorical column MC2 splits on. Nothing is left for a per-airport head to recover.

## Error concentration (lane A, out of fold)

| rows | share of lane-A squared error |
|---|---|
| top 10 | 0.58% |
| top 100 | 2.45% |
| top 1,000 | 10.41% |
| top 10,000 | 28.08% |

Lane A is the mirror image of lane B: the error is spread over the whole sample, so outlier work buys nothing here and only a broad
gain (model class, tuning, a feature every row uses) moves the number.

## What this closes

A per-airport lane-A model is not worth its GPU hours. Recorded in `docs/COVERAGE.md` section 9; no lane declared, no run queued.
