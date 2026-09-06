# PV: the four unread plan columns, priced on the Jan + Jul 2025 development holdout

Regenerate with `python -m taxi.pv`. Champion MC2 on 339,046 lane-A rows, 232.7269 s. A flag is 1 where a flight-plan field disagrees with the movement field it twins; that disagreement is the only part of these columns the model cannot already compute. No fit is run and no cell is declared.

## 1. Mass, difficulty and bias

| flag | rows | share | MC2 on flagged | MC2 on rest | mean residual flagged | rest | not a function of `operator` |
|---|---|---|---|---|---|---|---|
| actype | 1,234 | 0.364% | 376.7 s | 232.0 s | -21.1 s | -0.1 s | 1,147 rows |
| ades | 4,703 | 1.387% | 283.2 s | 231.9 s | -7.6 s | -0.1 s | 4,542 rows |
| frule | 233 | 0.069% | 216.2 s | 232.7 s | +41.6 s | -0.2 s | 232 rows |
| callsign | 90,847 | 26.795% | 184.3 s | 248.1 s | +0.1 s | -0.3 s | 52,774 rows |

## 2. What a perfect recalibration on each flag is worth

The champion's residual is re-centred on the judged rows themselves, first inside 10 deciles of its own prediction and then inside each decile split by the flag. Fitting the correction on the rows it is judged on makes this a ceiling for the level mechanism rather than a fit, and the decile-only row is its noise floor.

| correction | ceiling [95%] |
|---|---|
| deciles alone, no flag | +0.04 s |
| deciles x actype | +0.14 [+0.04, +0.24] s |
| deciles x ades | +0.05 [-0.01, +0.12] s |
| deciles x frule | +0.05 [-0.01, +0.11] s |
| deciles x callsign | +0.05 [-0.01, +0.12] s |

## 3. Where the callsign flag looked like something

It carries real mass, it is not a function of the operator, and MC2's error on it is 60 s below the rest of the lane, so it is the one flag that had to be taken apart rather than dismissed on rate. Inside a decile of MC2's own prediction the two groups differ in spread but not in level, and squared error pays for the level.

| MC2 prediction decile | mean prediction | flagged rows | RMSE flagged | rest rows | RMSE rest | mean residual gap |
|---|---|---|---|---|---|---|
| 0 | 506 s | 20,632 | 140.2 s | 13,273 | 133.3 s | -3.7 s |
| 1 | 670 s | 14,821 | 133.2 s | 19,084 | 134.5 s | +2.3 s |
| 2 | 757 s | 12,104 | 148.0 s | 21,800 | 154.5 s | +1.8 s |
| 3 | 832 s | 9,818 | 158.9 s | 24,087 | 169.9 s | +5.0 s |
| 4 | 907 s | 8,104 | 177.8 s | 25,800 | 193.6 s | +5.7 s |
| 5 | 986 s | 7,351 | 182.6 s | 26,554 | 207.5 s | +1.1 s |
| 6 | 1076 s | 6,411 | 190.9 s | 27,493 | 222.2 s | +0.4 s |
| 7 | 1185 s | 5,427 | 224.7 s | 28,478 | 238.2 s | -5.9 s |
| 8 | 1336 s | 3,492 | 296.1 s | 30,412 | 286.5 s | -12.7 s |
| 9 | 1793 s | 2,687 | 468.0 s | 31,218 | 434.8 s | +1.5 s |

The mismatch is an encoding difference before it is anything else: `FLIGHT_mvt` carries the IATA commercial number and `CALLSIGN_flt` the ICAO ATC callsign, so `LH1158` and `DLH73U` are the same flight. Its 1175 distinct prefixes sit against 567 operators and 1533 airline codes the model already reads.

