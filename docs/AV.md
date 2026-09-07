# AV: does the 2026 row set sit where the 2025 holdout sits?

Diagnostic, never a verdict, no cell declared. Regenerate with `python -m taxi.av`. A CatBoost classifier separates the 339,046 lane-A holdout rows from an equal sample of the 339,551 lane-A ranking rows on the champion's own 52 columns, fit on half of each side and scored on the other half. Both sides are January and July, one year apart, so the calendar is held and only the year moves. If the two row sets were exchangeable the classifier would read 0.5.

## Separability

Out-of-sample AUC **0.8686** on 339,046 scored rows, and **0.7664** on the same rows with the 16 weather columns removed, which separates a different weather realisation from a different operation.

| column | importance |
|---|---|
| w_tmpf | 22.99 |
| w_sknt | 9.95 |
| runway | 9.79 |
| apt | 8.38 |
| wday | 7.90 |
| w_ceil | 5.87 |
| operator | 4.92 |
| actype | 3.89 |

## What the shift is worth on the champion's own errors

The importance weight `p / (1 - p)` turns the holdout into an unbiased sample of the ranking set under covariate shift, so the weighted RMSE below is the estimate of what lane A scores on the board, against HB's airport-and-month figure of 233.3014 s.

| weighting | lane-A RMSE (s) | effective rows | shift | 95% day-block interval |
|---|---|---|---|---|
| unweighted (the number every verdict rides) | 231.4851 | 169,523 | | |
| importance weighted, raw | 229.7303 | 62,431 | -1.7548 | [-9.25, +6.08] |
| importance weighted, capped at the 99% weight (3.79) | 228.0069 | 87,770 | -3.4781 | [-9.91, +3.08] |
