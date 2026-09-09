# HZ: where the remaining headroom is not, both lanes, Jan + Jul 2025 development holdout

Diagnostic, never a verdict, no cell declared. Regenerate with `python -m taxi.hz`. Three questions the campaign had left open, each answered against the standing champions SB3 (lane A) and L8c (lane B, 1,808.17 s).

## 1. Lane A: what a perfect model reading only the discrete keys would score

Rows are matched on each key set, and the pooled within-group spread of `c` is the RMSE of a model that conditions on those keys and nothing else. That is a CEILING for the key set: no feature engineering over those columns can beat it. Groups of one would read as zero error, so they are dropped and SB3 is scored on the same covered rows. `zb` is the plan gap in 2-minute bins, `qb` the departure queue capped at 40, `hr` the hour of day.

| keys | groups | rows covered | perfect on these keys | SB3, same rows | headroom |
|---|---|---|---|---|---|
| apt | 10 | 100.0% | 408.8 | 230.1 | -178.7 |
| apt + runway | 59 | 100.0% | 404.8 | 230.1 | -174.7 |
| apt + stand | 2,420 | 100.0% | 397.5 | 230.0 | -167.5 |
| apt + stand + runway | 9,058 | 99.6% | 392.2 | 229.8 | -162.4 |
| apt + stand + runway + actype | 44,009 | 95.5% | 385.9 | 228.6 | -157.3 |
| apt + stand + runway + actype + hr | 165,586 | 70.9% | 369.4 | 223.3 | -146.0 |
| apt + zb | 479 | 100.0% | 338.3 | 229.1 | -109.2 |
| apt + runway + zb | 1,326 | 99.9% | 328.4 | 228.7 | -99.7 |
| apt + stand + runway + zb | 40,298 | 95.2% | 292.4 | 220.6 | -71.8 |
| apt + stand + runway + actype + zb | 101,520 | 83.3% | 279.0 | 216.2 | -62.8 |
| apt + stand + runway + actype + zb + qb | 234,847 | 45.8% | 261.0 | 208.8 | -52.2 |

Every rung is NEGATIVE. SB3 beats a perfect conditioner on the discrete keys at every one of them. Two flights sharing airport, stand, runway, aircraft type, plan-gap bin and queue bin still differ in `c` by more than SB3's own error, so matching on those keys throws away more than it gains. The categorical axis is not where the remaining signal is: the continuous context columns are carrying the model past anything key-matching could reach.

## 2. Lane A: the source columns the view never reads

`taxi.data.VIEW` selects 22 of the 30 columns in the movement files. The unread ones are probed six ways below. `d_arv3` is the flight's arrival gap between source 3 and source 1, from the same source-3 system as the `AOBT_3_flt` that forms half of the lane-A label, so it is the one with a mechanism behind it. Correlation with `c` says whether a probe knows anything at all; correlation with SB3's RESIDUAL says whether it knows anything SB3 does not already hold.

| probe | non-null | corr with c | corr with SB3 residual |
|---|---|---|---|
| d_arv3 | 100.0% | +0.3158 | -0.0033 |
| air3 | 100.0% | +0.0588 | +0.0123 |
| cs_mm | 100.0% | -0.0818 | +0.0016 |
| typ_mm | 100.0% | +0.0423 | -0.0018 |
| ades_mm | 100.0% | -0.0135 | -0.0032 |
| rule_mm | 100.0% | -0.0059 | +0.0021 |

`d_arv3` is the strongest of them against `c` and among the weakest against the residual. Removing its 50-bin mean from SB3's own residual, fit on the very rows it is judged on and therefore an upper bound no honest feature could reach, moves 230.10 s to 230.07 s: **+0.03 s**. The champion already holds what these columns carry, through `d_ae`. The unread-column axis is closed.

## 3. Lane B: the copy gate is calibrated

96% of lane-B rows are ordinary flights while a handful of huge ones hold most of the squared error, so the obvious suspicion is that the copy gate leaks onto ordinary flights and the many pay for the few. Where the error sits, and what perfect knowledge of one band alone would be worth:

| band | rows | share of lane-B squared error | band RMSE | lane B if perfect here | gain |
|---|---|---|---|---|---|
| normal, under 1 h | 5,167 | 20.7% | 838.5 | 1,610.4 | +197.8 |
| stale, 1 to 12 h | 193 | 12.1% | 3,324.2 | 1,694.9 | +113.3 |
| over 12 h | 13 | 67.2% | 30,129.5 | 1,035.9 | +772.3 |

The gate weight implied by each prediction, against the rate of a big taxi actually realised in that bin. The body level is read off the lane's own predictions (880 s) and the weight from the prediction and the schedule gap, so both sides use ranking-visible values only: this is a blind calibration curve, not an oracle.

| implied gate weight | rows | mean weight | realised rate of y over 1 h | mean prediction | mean truth |
|---|---|---|---|---|---|
| 0.00 to 0.01 | 2,086 | 0.0006 | 0.0134 | 817 | 925 |
| 0.01 to 0.05 | 1,802 | 0.0285 | 0.0050 | 1,066 | 1,115 |
| 0.05 to 0.10 | 787 | 0.0670 | 0.0216 | 1,251 | 1,227 |
| 0.10 to 0.25 | 428 | 0.1379 | 0.0514 | 1,627 | 1,597 |
| 0.25 to 0.50 | 99 | 0.3463 | 0.1212 | 3,178 | 1,786 |
| 0.50 to 1.01 | 171 | 0.7954 | 0.6901 | 12,509 | 12,059 |

Mildly over-confident in the middle bins and close everywhere else. Shrinking or stretching every implied weight by one scalar is the only blind correction a calibration error of that shape admits:

| scalar on the implied weight | lane B RMSE | vs L8c |
|---|---|---|
| 0.00 | 3,987.2 | -2179.1 |
| 0.50 | 2,565.9 | -757.7 |
| 0.75 | 2,043.7 | -235.5 |
| 1.00 | 1,808.2 | +0.0 |
| 1.25 | 1,965.3 | -157.2 |
| 1.50 | 2,440.4 | -632.2 |
| 2.00 | 3,826.4 | -2018.2 |

The sweep peaks exactly at 1.00, the champion as it stands, and falls away hard in both directions: the same scalar that would fix the middle bins wrecks the top one. Two further blind rules were priced beside it and both land far under the floor. Replacing every prediction below a cut with a per-(airport, runway) mean of the training rows is worth at most +3.2 s at a 5,400 s cut and LOSES 16.6 s at 7,200 s. Swapping the whole body for that group mean wherever the gate is silent, which is 63.6% of the lane's squared error, is worth +3.1 s. Lane B's median detectable effect is 32 s (`docs/MDE.md`), so none of these is measurable, let alone bankable. The gate is not the problem and the body is not the problem. The 13 rows are, and `docs/DE.md` closed them.

## What this closes

Lane A's categorical axis (section 1), lane A's unread-column axis (section 2) and lane B's calibration axis (section 3). Taken with `docs/EXH.md` on the read columns, `docs/HP1.md` on the champion's own hyperparameter cell, `docs/BF.md` on every stack of every cached model and `docs/DE.md` on the artifact rows, no mechanism this campaign can reach is worth the 2 s floor on either lane.

