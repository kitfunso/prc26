# MIX: what the 2026 ranking file's own rows say (diagnostic, never a verdict)

Regenerate with `python -m taxi.mix`. Lane-B champion L8c, lane-A champion MC2. Nothing here decides a verdict; two of the four questions come back negative and are recorded so they are not asked again.

## 1. The ranking set's lane-B rows sit in different cells from the holdout's

The lane split itself matches: 5,290 of 344,841 ranking departures have no flight plan (1.534%) against 5,373 of 344,419 in the holdout (1.560%). The mix inside the lane does not. Lane B was July-heavy in 2025 and is balanced in 2026, and the January cells that grew are the ones the champion handles worst.

| airport | month | holdout rows | holdout RMSE | ranking rows | ranking / holdout |
|---|---|---|---|---|---|
| EDDF | 1 | 164 | 397.5 | 212 | 1.29 |
| EDDF | 7 | 435 | 423.6 | 209 | 0.48 |
| EDDM | 1 | 65 | 709.1 | 119 | 1.83 |
| EDDM | 7 | 172 | 667.3 | 119 | 0.69 |
| EGLL | 1 | 128 | 801.6 | 112 | 0.88 |
| EGLL | 7 | 338 | 1272.7 | 136 | 0.40 |
| EHAM | 1 | 293 | 1075.2 | 805 | 2.75 |
| EHAM | 7 | 515 | 391.7 | 401 | 0.78 |
| LEBL | 1 | 66 | 341.1 | 77 | 1.17 |
| LEBL | 7 | 430 | 371.1 | 365 | 0.85 |
| LEMD | 1 | 32 | 321.4 | 63 | 1.97 |
| LEMD | 7 | 277 | 349.3 | 175 | 0.63 |
| LFPG | 1 | 216 | 6908.6 | 325 | 1.50 |
| LFPG | 7 | 664 | 489.4 | 409 | 0.62 |
| LIRF | 1 | 60 | 4187.1 | 107 | 1.78 |
| LIRF | 7 | 337 | 3490.8 | 276 | 0.82 |
| LSZH | 1 | 179 | 789.4 | 300 | 1.68 |
| LSZH | 7 | 345 | 598.4 | 359 | 1.04 |
| LTFM | 1 | 232 | 605.6 | 333 | 1.44 |
| LTFM | 7 | 425 | 775.0 | 388 | 0.91 |

## 2. Reweighting the holdout to that mix, and whether it survives the rows F17 names

Each cell's holdout mean squared error, re-averaged with the ranking set's row counts as weights. The first row is the headline and it is misleading on its own; the rest of the table is why.

| worst lane-B rows dropped | lane B at the holdout mix | lane B at the ranking mix | shift | composite, holdout mix | composite, ranking mix |
|---|---|---|---|---|---|
| 0 | 1808.2 | 2088.5 | +280.4 | 321.28 | 346.38 |
| 1 | 1409.4 | 1549.6 | +140.2 | 289.03 | 299.84 |
| 2 | 1173.3 | 1212.2 | +38.8 | 272.37 | 274.97 |
| 5 | 1035.9 | 1048.8 | +12.9 | 263.68 | 264.46 |
| 20 | 906.6 | 920.4 | +13.7 | 256.27 | 257.02 |

Read the last four rows, not the first. The headline shift is the two LFPG January rows flag F17 already names, arriving again through the weights. With them set aside the mix shift is worth roughly +15 to +40 s of lane-B RMSE, about +1 to +3 s of composite. Small, real, and against us: the ranking mix is tilted toward the cells we predict worst. The board anchors on `docs/gate-s4.html` assume the holdout mix and do not carry it.

## 3. Arrival taxi times: available for 2026, and worth nothing (negative)

The ranking file blanks the departure taxi time but keeps the arrival one on every one of its arrivals. An arrival that has already reached a stand when the anchor passes has a measured taxi-in, in seconds rather than the counts every existing traffic column uses. It measures the same surface. It is also already captured.

| statistic | value |
|---|---|
| coverage on lane-A holdout rows | 1.000 |
| correlation with the label | +0.132 |
| correlation with the MC2 residual | +0.024 |

A residual correlation that small caps the gain near 0.2 s, an order of magnitude under the 2 s floor. No cell was declared.

## 4. Did the recording process shift between the years?

| year | month | arrivals | mean taxi-in | over an hour | rate |
|---|---|---|---|---|---|
| 2025 | 1 | 153,551 | 544.8 s | 28 | 0.00018 |
| 2025 | 6 | 1 | 727.0 s | 0 | 0.00000 |
| 2025 | 7 | 190,447 | 550.9 s | 52 | 0.00027 |
| 2026 | 1 | 152,871 | 553.8 s | 116 | 0.00076 |
| 2026 | 7 | 191,822 | 523.7 s | 29 | 0.00015 |

Mean arrival taxi-in is stable across the years, so the surface has not moved and a large level correction is not warranted. Arrivals carry the over-an-hour artifact at under 0.1% in every month above, against 3.8% on lane-B departures, so it is a property of off-block recording and arrivals cannot be used to estimate its 2026 rate.

## 5. Is the artifact separable beyond what the champion already uses? (negative)

Lane B carries 881 taxi times over an hour in 17,097 training rows (5.15%) and 206 in 5,373 holdout rows (3.83%). A classifier on the champion's own columns reaches AUC 0.898, which looks like a lane. It is not: outside LIRF the AUC is 0.719 on 58 positives, and the rest is the no-type and schedule-gap structure the champion's gate already rides.

Adding a shrunk artifact term, the prediction plus lambda x P(artifact) x (mean artifact magnitude 9443 s minus the prediction), floored:

| lambda | lane B RMSE | paired against the champion [95%] |
|---|---|---|
| 0.25 | 1836.0 | -27.8 [-49.1, -16.7] |
| 0.50 | 1908.4 | -100.3 [-168.5, -65.4] |
| 1.00 | 2166.9 | -358.7 [-582.7, -228.8] |

Every weight loses, and the loss grows with the weight. The champion already predicts high where the artifacts are: in the top decile of P(artifact) it over-predicts on a truth mean near 4,900 s. The level is not the missing piece. No cell was declared.
