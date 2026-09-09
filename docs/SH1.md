# SH1: the selection-honest re-judgement, Jan + Jul 2025 development holdout

Diagnostic, never a verdict, no cell chosen. Regenerate with `python -m taxi.sh1`. Declared in protocol section 4 before the code was written. Stage 5.3 of the protocol asks that a spec be nominated on early rows and judged on untouched late rows, with the haircut reported as its own number, and this campaign had never run it: the whole repo returned zero matches for "selection-honest". The development holdout is two whole months, so the split needs no new fit. Nomination reads ONE month and picks the `argmin` of its RMSE; the judged number is that model's RMSE on the other month, which the nomination never saw. The as-run column is the champion this campaign actually promoted while reading both months, scored on the same judged rows. A positive haircut means the campaign's reported numbers are optimistic by that much.

## Lane A: 7 models, 339,046 holdout rows, promoted champion SB3

| model | January RMSE | July RMSE | both months |
|---|---|---|---|
| SB3 | 204.37 | 249.11 | 230.10 |
| MC2 | 206.18 | 251.34 | 232.15 |
| D1 | 216.38 | 259.93 | 241.35 |
| M3 | 240.99 | 281.83 | 264.27 |
| W1-A | 221.04 | 272.83 | 250.90 |
| SB1 | 205.24 | 250.33 | 231.17 |
| SB3+NN2 | 203.94 | 246.58 | 228.41 |

| direction | nominated on one month alone | its RMSE on the judged month | promoted while reading both | its RMSE on the same rows | haircut |
|---|---|---|---|---|---|
| nominate on January, judge on July | SB3+NN2 | 246.58 | SB3 | 249.11 | -2.53 |
| nominate on July, judge on January | SB3+NN2 | 203.94 | SB3 | 204.37 | -0.43 |

- January nominates **SB3+NN2**, NOT the promoted SB3; on July that costs -2.53 s against the promoted champion.
- July nominates **SB3+NN2**, NOT the promoted SB3; on January that costs -0.43 s against the promoted champion.

## Lane B: 8 models, 5,373 holdout rows, promoted champion L8c

| model | January RMSE | July RMSE | both months |
|---|---|---|---|
| B4 | 2,932.87 | 1,461.92 | 1,965.64 |
| L4a | 3,312.19 | 2,106.33 | 2,486.30 |
| L4b | 2,929.56 | 1,416.27 | 1,939.65 |
| L6a | 2,902.73 | 1,287.97 | 1,861.76 |
| L6b | 2,903.75 | 1,312.40 | 1,874.65 |
| L8a | 2,926.08 | 1,310.74 | 1,883.05 |
| L8b | 2,912.13 | 1,297.99 | 1,870.76 |
| L8c | 2,897.61 | 1,183.78 | 1,808.17 |

| direction | nominated on one month alone | its RMSE on the judged month | promoted while reading both | its RMSE on the same rows | haircut |
|---|---|---|---|---|---|
| nominate on January, judge on July | L8c | 1,183.78 | L8c | 1,183.78 | +0.00 |
| nominate on July, judge on January | L8c | 2,897.61 | L8c | 2,897.61 | +0.00 |

- January nominates **L8c**; on July that costs +0.00 s against the promoted champion.
- July nominates **L8c**; on January that costs +0.00 s against the promoted champion.

