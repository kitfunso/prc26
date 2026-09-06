# SB1: five-seed bag of MC2's cell, Jan+Jul 2025 development holdout

Regenerate with `python -m taxi.sb1`. MC2's cell and recorded rounds (1804, 1532, 3726) on its 49 columns, fit once per seed on the ten training months with no early stopping; seed 0 is the cached MC2 refit. The challenger is the mean of the five predictions; k = 1.

## Seeds (the spread is the GPU reproducibility yardstick)

| seed | holdout RMSE (s) | paired delta vs seed 0 (s) |
|---|---|---|
| 0 | 232.73 | +0.00 |
| 1 | 232.66 | +0.07 |
| 2 | 232.89 | -0.16 |
| 3 | 232.84 | -0.11 |
| 4 | 232.76 | -0.04 |

Seed-to-seed RMSE spread: 0.23 s.

## Holdout verdict vs champion MC2 (339046 rows; tail rows 25898)

| model | RMSE | Jan | Jul | tail rows | body rows |
|---|---|---|---|---|---|
| MC2 | 232.7 | 206.8 | 251.9 | 591.0 | 172.5 |
| SB1 | 231.7 | 205.9 | 250.9 | 588.7 | 171.7 |

Improvement SB1 vs MC2: +1.0 s [+0.7, +1.3]; paired SE 0.1 s; optimism bound for k = 1: 0.0 s; floor 2 s. Verdict: tie or lose, MC2 stays.
