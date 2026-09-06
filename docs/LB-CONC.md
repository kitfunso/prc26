# Lane-B error concentration, Jan+Jul 2025 development holdout

Diagnostic, not a declared search (protocol section 2): it cannot decide any verdict. Regenerate the tables with the scripts quoted below.

## Where the composite squared error sits

Lane A is MC2's cached holdout predictions (339,046 rows), lane B the named model on 5,373 rows.

| pair | composite RMSE | lane-B share of squared error | lane-B share of rows |
|---|---|---|---|
| MC2 + B4 | 337.0 | 53.1% | 1.56% |
| MC2 + L6a | 327.7 | 50.4% | 1.56% |

Lane B carries about half the squared error on about one row in sixty-four.

## The concentration is two rows

| model | RMSE | top 1 row | top 2 | top 5 | top 10 | top 20 |
|---|---|---|---|---|---|---|
| lane B B4 | 1965.6 | 33.3% | 49.0% | 56.4% | 59.9% | 65.3% |
| lane B L6a | 1861.8 | 37.1% | 54.6% | 63.5% | 66.6% | 70.7% |
| lane A MC2 | 232.7 | 0.2% | 0.4% | 0.8% | 1.4% | 2.4% |

The two rows are LFPG, January, truths 84,240 s and 58,206 s against predictions of about 1,100 s. They are 27.5% of the whole
composite squared error. The training months hold no LFPG lane-B row above 9,182 s, so no honest model can see them (F3). Dropping
them takes the MC2 + L6a composite from 327.7 to 279.0 s; that number is a measurement of how much of the local level is unmodellable
junk, never a score.

Lane A shows no such concentration: its worst fifty rows are 4.4% of its squared error.

## The lane-B champion survives the extreme rows

Champion B4, challenger L6a, the same paired day-block bootstrap every verdict uses.

| rows dropped | n | B4 | L6a | improvement [95%] |
|---|---|---|---|---|
| 0 | 5373 | 1965.6 | 1861.8 | +103.9 [+57.0, +178.1] |
| 1 | 5372 | 1605.7 | 1476.7 | +129.0 [+71.7, +191.4] |
| 2 | 5371 | 1404.1 | 1254.5 | +149.6 [+89.1, +206.1] |
| 5 | 5368 | 1298.8 | 1125.1 | +173.7 [+118.3, +225.1] |
| 20 | 5353 | 1164.6 | 1025.8 | +138.8 [+99.5, +174.8] |

L6a's lead grows when the worst rows go, so H26's verdict is not an artifact of them. This is evidence about F13, and it does not
touch F16, which is a different fold.

## What is left to model at LIRF

LIRF is the only gated airport (397 holdout rows, 33.4% of lane-B squared error). L6a's gated predictions run high:

| L6a prediction band | rows | mean prediction | mean truth |
|---|---|---|---|
| 3,000 to 6,000 s | 136 | 4,349.5 | 3,307.8 |
| 6,000 to 12,000 s | 100 | 8,011.0 | 5,692.6 |
| over 12,000 s | 33 | 35,660.1 | 38,132.6 |

A correct per-airport level alone would remove 3.917e8 of lane B's 1.862e10 squared error, and LIRF is 3.307e8 of that.

The cause is in the probability, not the level. On training months alone, LIRF's no-type copy rate is seasonal:

| months | rows | copy rate |
|---|---|---|
| Nov to Apr | 326 | 0.718 |
| May to Oct | 764 | 0.474 |

The difference is 0.244 with SE 0.031, z = 7.9. By month it runs 0.76, 0.75, 0.67 (Feb, Mar, Apr), 0.48, 0.50, 0.46, 0.44, 0.50
(May to Oct), 0.64, 0.76 (Nov, Dec). `taxi.l5.shrunk` pools every month, so one static rate is too low for January and too high
for July at the same time. L7 is the declared lane that tests a seasonal probability.
