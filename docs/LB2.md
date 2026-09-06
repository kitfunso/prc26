# LB2: four lane-B hypotheses, measured and closed (diagnostic, never a verdict)

Regenerate with `python -m taxi.lb2`. Champion L8c, floored, holdout Jan + Jul 2025. No cell is declared. Lane B holds 49% of the composite squared error (`docs/LEV.md`) and its champion reads two things, so these four are the obvious places to look. All four lose, and the point of writing them down is that they stay lost.

## 1. Lane A's local pushback delay does not transfer

On the holdout, a lane-B flight's own pushback delay has median 4,441 s, mean 4,556, sd 4,909, against lane A's `d_as` at median 480, mean 1,007, sd 2,308. Different populations, so the transfer is already unlikely. Measured anyway.

| window | covered | corr(local lane-A delay, the lane-B flight's own delay) | RMSE of z_sched minus that level |
|---|---|---|---|
| plus or minus 30 min | 99.1% | +0.207 | 5,379 s |
| plus or minus 60 min | 99.7% | +0.213 | 5,363 s |
| plus or minus 120 min | 99.8% | +0.216 | 5,374 s |

The champion scores 1,808 s on these rows. The best of those is several times worse and the correlation never reaches +0.25, so lane A's delay level does not carry lane B's delay.

## 2. The arrival side separates nothing

Arrivals over the same two months: 343,999 rows, block null 0.00%, schedule null 0.00%, taxi-in null 0.00%. The arrival recorder is complete where the departure one is not, and an arrival block equals its schedule within 60 s on 5.06% of rows, so the copy behaviour exists on that side too.

Two local arrival features on the 5,373 lane-B holdout rows (883 of them copies), each the mean over that airport's arrivals in the hour before the departure's anchor:

| feature | on copy rows | on non-copy rows |
|---|---|---|
| local arrival copy rate | 0.0500 | 0.0418 |
| local arrival taxi-in | 666 s | 564 s |

Neither separates the copies. In lane A the same recent arrival taxi-in correlates -0.091 with the label against `rec_taxi`'s -0.115, so it is a weaker copy of a column the champion already reads. No cell declared on either side.

## 3. The copy gate is already calibrated

L8c's copy probability, fitted out of fold over the five month pairs: 1,090 gated rows, base copy rate 0.5468, mean P 0.5480.

| P bin | rows | mean P | actual copy rate | median z_sched |
|---|---|---|---|---|
| (-0.001, 0.05] | 52 | 0.038 | 0.077 | 5,252 s |
| (0.05, 0.2] | 210 | 0.108 | 0.095 | 5,698 s |
| (0.2, 0.4] | 158 | 0.285 | 0.222 | 5,883 s |
| (0.4, 0.6] | 118 | 0.499 | 0.602 | 7,080 s |
| (0.6, 0.8] | 177 | 0.706 | 0.734 | 6,600 s |
| (0.8, 0.95] | 244 | 0.887 | 0.861 | 7,832 s |
| (0.95, 1.0] | 131 | 0.971 | 0.962 | 8,765 s |

Every bin tracks. Shrinking P toward the base rate is the one free knob the mixture has:

| shrink lambda | RMSE on the gated rows |
|---|---|
| 1.0 | 5,683.5 s |
| 0.9 | 5,732.0 s |
| 0.8 | 5,795.7 s |
| 0.7 | 5,874.2 s |
| 0.6 | 5,966.9 s |
| 0.5 | 6,073.2 s |

Monotonically worse. The gate is calibrated and recalibration is closed.

## 4. The rows the gate never touches hold no gain either

The gate touches 397 of 5,373 holdout rows (7.4%). Split the lane on it and compare the champion with a per-airport median constant, which reads no features at all:

| rows | champion | per-airport median | share of lane-B squared error |
|---|---|---|---|
| gate touched (397) | 3,604.6 s | 12,501.7 s | 29.4% |
| untouched (4,976) | 1,579.1 s | 1,588.2 s | 70.6% |

The gate earns its keep where it fires, and on the other rows the champion is level with a constant. That looks like the place to model, until the error inside it is counted:

- The worst 1 untouched row holds 55.7% of the untouched squared error.
- The worst 5 untouched rows hold 84.2% of the untouched squared error.
- The worst 20 untouched rows hold 88.5% of the untouched squared error.
- The 2 untouched rows over 12 h hold 82.0%, and without them the untouched rows score 670.7 s on 4,974 rows.

Those rows are the day-shift artifact of `docs/DE.md`, which flag F21 prices at zero through every route. What is left on the untouched rows is an ordinary population the champion already fits to within a constant's distance, so no large modelling gain is hiding under the gate's 93%.
