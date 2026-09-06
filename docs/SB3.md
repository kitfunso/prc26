# SB3: SB2's composition on D2's base, Jan + Jul 2025 development holdout

Regenerate with `python -m taxi.sb3`. D2's cell and recorded trees (2068, 1987, 3901) on its 52 columns, fit once per seed on the ten training months with no early stopping, then ST1a's out-of-fold weights (1.0 on tail rows, 0.7 on body rows, blending toward the cached D1 holdout prediction) and C1's floor at 0 s. Every constant was fixed by an earlier declared run, so k = 1.

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
| MC2 floored (champion) | 232.15 | 206.2 | 251.3 | 588.1 | 172.5 |
| D2 seed 0 floored | 231.51 | 205.7 | 250.6 | 587.8 | 171.6 |
| D2 seed bag | 230.10 | 204.4 | 249.1 | 583.3 | 170.8 |
| D2 seed bag + ST1a gate (SB3) | 229.78 | 204.2 | 248.7 | 583.3 | 170.4 |

Improvement SB3 vs MC2: +2.4 s [+2.0, +2.7]; paired SE 0.2 s; optimism bound for k = 1: 0.0 s; floor 2 s. Verdict: WIN, clears bound plus floor, SB3 promoted.
