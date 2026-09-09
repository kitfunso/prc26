# AP1: one CatBoost per airport against the pooled model, Jan + Jul 2025 development holdout

Regenerate with `python -m taxi.ap1`. Declared in protocol section 4 before the code ran; k = 1, one cell, no search. D2's cell and recipe unchanged (`taxi.mc1.CELL`, the M2 three-stage split at 600 s, early stopping on the fold); the only change is that ten models are fitted, one per airport, on 51 columns, D2's 52 less the airport itself, which is constant inside every fit. The base is D2's own five-fold CV of 211.8 s (`docs/D2.md`), already measured and so the reference rather than a trial. Wall 69.4 min.

## Month-pair CV, pooled over the ten airports

| fold | rows | AP1 RMSE (s) | fit min |
|---|---|---|---|
| (2, 3) | 305,989 | 215.2 | 14 |
| (4, 5) | 357,800 | 200.2 | 16 |
| (6, 8) | 369,492 | 225.1 | 12 |
| (9, 10) | 365,246 | 217.6 | 15 |
| (11, 12) | 325,004 | 206.0 | 12 |

CV RMSE over 1,723,531 out-of-fold rows: **213.2 s** against D2's 211.8 s (+1.4).

## By airport

| airport | out-of-fold rows | AP1 RMSE (s) | trees (clf, tail, body) |
|---|---|---|---|
| EDDF | 191,954 | 171.6 | [291, 436, 824] |
| EDDM | 139,470 | 163.1 | [326, 284, 871] |
| EGLL | 198,388 | 217.9 | [457, 796, 683] |
| EHAM | 203,726 | 155.6 | [316, 741, 793] |
| LEBL | 149,353 | 194.6 | [287, 323, 557] |
| LEMD | 176,276 | 177.6 | [293, 494, 616] |
| LFPG | 197,047 | 231.3 | [432, 866, 754] |
| LIRF | 133,085 | 340.2 | [402, 801, 591] |
| LSZH | 110,794 | 162.4 | [282, 283, 609] |
| LTFM | 223,438 | 254.9 | [594, 918, 570] |

AP1 does not beat D2's 211.8 s on CV, so it never meets the holdout and SB3 stands. Reading: the pooled model was not paying a real price for reading the airport as a categorical. Ten fits see a tenth of the rows each and lose more to that than they win from a free per-airport structure.
