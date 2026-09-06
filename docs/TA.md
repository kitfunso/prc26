# TA: lane A's tail is variance, not bias (diagnostic, never a verdict)

Regenerate with `python -m taxi.ta`. Champion MC2, floored, holdout Jan + Jul 2025, 339,046 lane-A rows at 232.1502 s. No cell is declared here. The point of the run is to say which KINDS of work still have room, and the answer removes two of them.

## 1. Where the error is, and which way it points

| rows, worst first by squared error | share of the lane's squared error |
|---|---|
| top 10 | 1.4% |
| top 100 | 6.7% |
| top 1,000 | 22.4% |
| top 3,390 | 36.1% |

The tail is under-prediction. Of the top 3,390 rows, 2,567 are predicted too low and carry 83.3% of the tail's squared error, against 823 predicted too high carrying 16.7%. A model that misses long taxis, then. The next section asks whether that is a bias worth correcting.

## 2. It is not a bias. It is regression to the mean, and the two tables that look alike are not alike

Read these two tables together, in this order, because the second is the trap.

**Conditioned on what the model says.** This is the honest calibration question, because the prediction is what we would correct:

| predicted decile | rows | mean prediction | mean truth | bias |
|---|---|---|---|---|
| 1 | 33,905 | 507 s | 509 s | -2.5 s |
| 2 | 33,905 | 670 s | 670 s | +0.2 s |
| 3 | 33,904 | 757 s | 755 s | +1.9 s |
| 4 | 33,905 | 832 s | 832 s | +0.1 s |
| 5 | 33,904 | 907 s | 906 s | +0.4 s |
| 6 | 33,905 | 986 s | 985 s | +1.9 s |
| 7 | 33,904 | 1,076 s | 1,072 s | +4.1 s |
| 8 | 33,905 | 1,185 s | 1,181 s | +3.7 s |
| 9 | 33,904 | 1,336 s | 1,335 s | +1.4 s |
| 10 | 33,905 | 1,793 s | 1,805 s | -12.5 s |

**Conditioned on the truth.** This is the same model on the same rows, and it looks like massive shrinkage:

| true decile | rows | mean prediction | mean truth | bias |
|---|---|---|---|---|
| 1 | 34,024 | 563 s | 452 s | +110.4 s |
| 2 | 34,111 | 713 s | 628 s | +85.1 s |
| 3 | 34,134 | 782 s | 722 s | +60.0 s |
| 4 | 34,075 | 859 s | 805 s | +54.9 s |
| 5 | 33,275 | 922 s | 883 s | +38.7 s |
| 6 | 34,155 | 993 s | 973 s | +20.1 s |
| 7 | 33,926 | 1,073 s | 1,071 s | +2.4 s |
| 8 | 33,861 | 1,171 s | 1,195 s | -24.3 s |
| 9 | 33,777 | 1,300 s | 1,376 s | -75.8 s |
| 10 | 33,708 | 1,682 s | 1,958 s | -275.8 s |

The second table is what an unbiased predictor of a noisy quantity ALWAYS looks like when you sort by the noisy quantity: the rows in the top true decile are there partly because their noise was positive, and no predictor can follow that. It is not evidence of shrinkage and must never be quoted as a reason to expand the tail. The first table is the one that carries information, and it says the champion is calibrated.

The ceiling on any additive recalibration, that is, the RMSE freed by removing each group's mean error exactly:

| grouping | rows | ceiling on the gain |
|---|---|---|
| predicted decile | 339,046 | +0.043 s |
| schedule gap band | 339,046 | +0.040 s |
| airport | 339,046 | +0.307 s |
| airport and runway | 339,046 | +0.419 s |

Every one of those is far under the 2 s floor, and the worst single band is only a few seconds out:

| schedule gap | rows | bias |
|---|---|---|
| under 20 m | 125,906 | -0.1 s |
| 20 to 40 m | 128,625 | +3.2 s |
| 40 to 60 m | 46,349 | -0.8 s |
| 1 to 1.5 h | 23,150 | -8.3 s |
| 1.5 to 2 h | 7,344 | -13.5 s |
| over 2 h | 7,672 | -16.3 s |

So isotonic calibration, tail expansion, a per-airport offset and a per-band offset are all closed together, by arithmetic rather than by a run. There is no bias to collect.

## 3. What the worst rows are

The ten worst, with the columns the champion reads:

| airport | runway | truth | schedule gap | EOBT gap | queue | prediction | error |
|---|---|---|---|---|---|---|---|
| LFPG | 26R | 8,563 | 10,301 | 4,961 | 6 | 2,568 | -5,995 |
| EHAM | 24 | 10,972 | 12,776 | 8,876 | 98 | 5,360 | -5,612 |
| LTFM | 35L | 6,908 | 8,584 | 4,204 | 11 | 1,371 | -5,537 |
| EHAM | 24 | 6,711 | 6,775 | 4,975 | 11 | 1,412 | -5,299 |
| LFPG | 27L | 15,059 | 16,858 | 6,658 | 6 | 10,298 | -4,761 |
| LTFM | 35L | 6,291 | 7,855 | 3,955 | 16 | 1,537 | -4,754 |
| LIRF | 25 | 5,950 | 5,945 | 1,445 | 7 | 1,208 | -4,742 |
| LIRF | 25 | 7,992 | 7,986 | 2,586 | 7 | 3,280 | -4,712 |
| LIRF | 25 | 6,960 | 6,960 | 4,560 | 10 | 2,317 | -4,643 |
| LSZH | 16 | 5,777 | 6,296 | 2,336 | 5 | 1,296 | -4,481 |

Every one is a flight that took off far later than the model expected, and in each the schedule already said so. Nothing here is a recording artifact of the kind lane B carries (`docs/DE.md`): lane A has exactly one day-shifted row in 339,046. What the tail rows look like, against the whole lane:

| marker | all lane A | in the top 1% | their share of the tail's squared error |
|---|---|---|---|
| a taxi over 1 h | 775 | 473 | 34.7% |
| a taxi over 12 h | 1 | 1 | 0.1% |
| within an hour of a full day | 1 | 1 | 0.1% |
| the truth equals the schedule | 181,195 | 1,360 | 47.3% |
| the truth equals EOBT | 237,886 | 1,389 | 32.8% |

By airport, against each airport's share of the lane:

| airport | rows | share of the lane's rows | share of its squared error |
|---|---|---|---|
| LIRF | 26,131 | 7.7% | 21.1% |
| LTFM | 45,882 | 13.5% | 17.1% |
| LFPG | 38,732 | 11.4% | 13.5% |
| EGLL | 39,744 | 11.7% | 13.3% |
| LEBL | 28,490 | 8.4% | 7.8% |
| EDDF | 36,231 | 10.7% | 6.7% |
| EHAM | 40,141 | 11.8% | 6.3% |
| LEMD | 35,019 | 10.3% | 6.0% |
| EDDM | 26,854 | 7.9% | 4.1% |
| LSZH | 21,822 | 6.4% | 3.9% |

## 4. Two model families miss on the same rows

The question section 2 leaves open is whether a DIFFERENT model would find what this one misses. Three champions are cached on these exact rows, two families and two column sets, so the question is answerable without fitting anything.

| pair | error correlation, all rows | error correlation, MC2's top 1% |
|---|---|---|
| MC2 against D1 | 0.946 | 0.960 |
| MC2 against W1-A | 0.918 | 0.915 |
| D1 against W1-A | 0.943 | 0.941 |

MC2 is CatBoost with ordered boosting and categorical combinations; D1 is LightGBM on the same 49 columns. Different families, different splits, and their errors correlate 0.946 over the lane and 0.960 in the tail. Their worst-1% row sets share 2,642 of 3,390 rows, 78%. The mean of the two scores 233.5 s, WORSE than MC2 alone at 232.2, which is why the declared E2 blend tied. Even an oracle allowed to pick the better family row by row reaches only 214.3 s, and it is an oracle.

Two families given the same information make the same mistakes on the same flights. That is the signature of an information limit, not a model-class limit.

## 5. What this licenses

Nothing about a model, and no cell is declared. What it changes is where the remaining effort is worth spending.

- **Closed by arithmetic:** post-hoc calibration, tail expansion, per-airport and per-band offsets. Ceiling under 0.1 s against a 2 s floor.

- **Low value on this evidence:** more of the same model class. A different family on the same columns lands within 9 s and misses the same rows, blending them is worse than the better one, and hyper-parameter search (the queued O1) moves inside that same space.

- **The only direction with room:** new information about the specific flight that is about to taxi a long way. The queued TH1 and Q1 cells are that kind of work, and this diagnostic raises their priority above O1's.

- **For the gate:** the lane-A tail is not a defect waiting to be fixed. Combined with F21 in lane B, a board gap of a few tens of seconds is not by itself proof that a better recipe exists.
