# SE: is lane A's error seasonal, Jan + Jul 2025 development holdout

Diagnostic, never a verdict, no cell declared. Regenerate with `python -m taxi.se`. `docs/HB.md` section 2 records that no lane-A column carries the season, and then prices seasonal WEIGHTS (arm D, -0.9444 s) and the judged months' own ROWS (arm B, +3.9552 s). Neither is a season COLUMN, which is a different mechanism: a weight changes which rows the loss cares about, a column lets the trees fit a different function inside each season.

MC2's out-of-fold residuals on 1,723,531 lane-A training rows, RMSE 212.86 s. The holdout months are absent here by construction, so the seasonal shape below is read off the ten months that ARE fit on.

## 1. The error by month

| month | rows | mean residual (s) | RMSE (s) |
|---|---|---|---|
| 2 | 142,622 | -10.9 | 230.3 |
| 3 | 163,367 | -10.1 | 199.9 |
| 4 | 174,203 | -7.0 | 198.1 |
| 5 | 183,597 | -6.0 | 203.1 |
| 6 | 180,640 | +1.4 | 224.9 |
| 8 | 188,852 | +2.2 | 225.2 |
| 9 | 181,485 | +5.7 | 218.6 |
| 10 | 183,761 | +3.8 | 215.6 |
| 11 | 160,976 | -1.0 | 202.0 |
| 12 | 164,028 | +7.3 | 208.0 |

Spread of the monthly bias: 18.2 s across a 213 s error. Spread of the monthly RMSE: 32.2 s.

## 2. What removing a seasonal cell's own mean residual is worth

Each row's residual loses its cell's mean, fit on the very rows it is judged on, so this is a CEILING. Cells free error from pure noise too, so the honest column subtracts what the same cell count frees from shuffled residuals.

| cells | count | in-sample gain (s) | minus the shuffle null (s) |
|---|---|---|---|
| month | 10 | +0.092 | +0.088 |
| month + airport | 100 | +0.576 | +0.567 |
| month + airport + hour | 2,233 | +1.554 | +1.411 |

## 3. The buildable version: a day-of-year harmonic

A month category cannot be cross-validated here at all: the folds ARE month pairs, so a fold's months never appear in its own training rows and every judged row would meet an unseen category. That is the structural reason this campaign never carried one. Sin/cos harmonics of day-of-year interpolate instead, so they survive the same folds.

| harmonics | cross-fitted gain (s) |
|---|---|
| 1 | +0.036 |
| 2 | +0.004 |
| 3 | -0.446 |
| 4 | -0.043 |

Carried to the holdout: the 1-harmonic correction fit on all ten training months moves SB3 from 230.10 s to 230.14 s, **-0.042 s**, against a 2 s floor. Jan 204.4 s and Jul 249.1 s on the holdout, a 44.7 s difference the correction does not touch, because it is variance and not bias.

