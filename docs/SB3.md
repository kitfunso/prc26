# SB3: the five-seed D2 bag, Jan + Jul 2025 development holdout

Regenerate with `python -m taxi.sb3`. D2's cell and recorded trees (2068, 1987, 3901) on its 52 columns, fit once per seed on the ten training months with no early stopping, then C1's floor at 0 s. Every constant was fixed by an earlier declared run, so k = 1. ST1a's gate was part of SB3 as promoted at H52 and is withdrawn: it set each row's weight from `is_tail(te)`, which reads `c = y - z_aobt`, that row's own label. Its line is kept below to price what the withdrawal costs. The verdict line reads NOT promoted because SB3 already holds lane A: this is H52's promotion re-scored without the gate, not a new one.

## Seeds (unfloored RMSE; the delta is each seed floored, against the champion)

| seed | holdout RMSE (s) | paired delta vs MC2 floored (s) |
|---|---|---|
| 0 | 231.99 | +0.64 |
| 1 | 231.57 | +1.07 |
| 2 | 231.82 | +0.86 |
| 3 | 231.48 | +1.18 |
| 4 | 231.41 | +1.16 |

Seed-to-seed RMSE spread: 0.58 s.

## Each step added, on the same 339,046 rows

| model | RMSE | Jan | Jul | tail rows | body rows |
|---|---|---|---|---|---|
| MC2 floored (predecessor) | 232.15 | 206.2 | 251.3 | 588.1 | 172.5 |
| D2 seed 0 floored | 231.51 | 205.7 | 250.6 | 587.8 | 171.6 |
| D2 seed bag floored (SB3) | 230.10 | 204.4 | 249.1 | 583.3 | 170.8 |
| the same bag under ST1a's withdrawn label gate | 229.78 | 204.2 | 248.7 | 583.3 | 170.4 |

Improvement SB3 vs MC2: +2.1 s [+1.7, +2.4]; paired SE 0.2 s; optimism bound for k = 1: 0.0 s; floor 2 s. Verdict: WIN, clears bound plus floor against MC2, NOT promoted: the current champions are {'A': 'SB3', 'B': 'L8c'}.
