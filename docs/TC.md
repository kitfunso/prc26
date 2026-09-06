# TC: the label's own clock, and whether any of its structure pays

Regenerate with `python -m taxi.tc`. Diagnostic: no cell declared, no champion moves, no verdict decided.
MC2 holdout RMSE 232.7269 s on 339,046 lane-A rows, against the 2.0 s improvement floor.

## 1. The recording grid

The target is `y = takeoff - BLOCK_TIME` and the takeoff time is given on every ranking row, so any lattice the
block time is recorded on is a lattice the answer sits on. That would be free accuracy, needing no model.

 apt      n  blk_min  blk_5s  aobt_min  rounded  pred_5s
EGLL 239546    8.380  24.968    99.995    6.827   25.462
LEMD 212242    8.366  24.939    99.718    6.813   25.450
LIRF 160704    8.349  25.017    99.572    6.795   25.436
EDDM 167334    8.344  25.059    99.943    6.791   25.433
LTFM 272965    8.302  24.973    98.911    6.748   25.398
LEBL 179705    8.300  24.826    99.573    6.745   25.396
LFPG 239552    8.280  25.024    99.718    6.726   25.381
LSZH 134907    1.980  20.372    99.990    0.319   20.255
EDDF 230141    1.694  20.225    99.043    0.028   20.022
EHAM 247951    1.687  19.960    83.800    0.021   20.017

`AOBT_3` is written to the whole minute nearly everywhere, which the build already reads through `aobt_sec` and
`aobt_m5`. The block time is not. Reading its 8.3% minute rate at seven airports as a 5-second recording grid was
wrong, and the second unit is what caught it: a 5-second grid would put `blk_5s` at 100% and it is 25%. The rate
that does fit both columns is a MINORITY of rounded rows, a share q written to the minute and the rest recorded
to the second, which implies q = 6.8% at the top airport and predicts `blk_5s` of 25.5% against the
25.1% observed. The three airports at the bottom of the table show 1.7% and 20%, which is chance in both
columns: no rounding at all.

Either way the idea is closed by arithmetic, and by a wider margin than the mistaken reading would have given.
A lattice only buys accuracy when the residual is comparable to its spacing. Snapping onto a 60 s lattice removes
at most its quantisation variance, 3600/12 = 300 against MC2's 54,162, which is 0.65 s on a row known to be
rounded; at 6.8% of rows at seven of ten airports, and with no way to identify which rows those are, the
expected return is about 0.04 s. Membership is the binding problem, not the size of the lattice.

## 2. Is the block time a copy of another clock rather than a measurement?

The lane-B champion is built around exactly this: L4 to L8 model rows whose block time was copied from the
schedule, and L8c is a CatBoost classifier choosing between the copy branch and the regression. F28 then showed
why the shape matters, that a linear branch emits `y = z_sched` exactly while a tree emits piecewise constants.
Lane A carries 51.4% of board squared error and has never been asked the question.

 lane_a       n   sched    eobt    lobt    aobt
  False   22470   0.445     NaN     NaN     NaN
   True 2062577   0.389   0.476   0.439   0.653

## 3. The oracle: perfect knowledge of which lane-A rows are copies, and the exact value on them

| copy branch | rows | share | MC2 RMSE on them | share of lane SSE | oracle gain (s) |
|---|---|---|---|---|---|
| y = z_aobt (AOBT copy, c = 0) | 2,188 | 0.645% | 144 | 0.25% | +0.29 |
| y = z_sched (schedule copy) | 1,202 | 0.355% | 394 | 1.01% | +1.18 |
| y = z_eobt (EOBT copy) | 1,510 | 0.445% | 237 | 0.46% | +0.54 |
| y = z_lobt (LOBT copy) | 1,380 | 0.407% | 290 | 0.63% | +0.73 |
| **any of the four** | 3,878 | 1.144% | 259 | 1.42% | +1.66 |

## 4. Answer: both channels are closed, and the second is the one worth recording

Copy rows are genuinely bad rows. MC2 is at 394 s on schedule copies against 233 s over the lane, so they are
mispriced by 69%. They are also rare, 1.144% of the lane, and rarity wins: perfect classification plus
the exact value on every one of the four branches is worth +1.66 s, under the 2.0 s floor. Lane B's classifier is
nowhere near perfect, so a port of it to lane A would collect a fraction of a number that is already too small.

This is the closure that matters, because F28's mechanism makes the port look obvious. A tree cannot emit
`y = z_sched` exactly in lane A either, and the reflex is to give lane A the branch lane B has. The oracle says
not to: the mechanism is real in both lanes and the mass is not. Lane B's copies carry a lane where 13 rows hold
67% of the squared error; lane A's carry 1.42% of theirs.

Consequence: do not declare a lane-A copy branch, a copy classifier, or a mixture ported from L4 to L8c, and do
not treat the label's recording grid as free accuracy.

