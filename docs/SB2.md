# SB2: three parked ties composed, Jan + Jul 2025 development holdout

Regenerate with `python -m taxi.sb2`. Every constant was fixed by an earlier declared run (SB1's five seeds, ST1a's out-of-fold weights 1.0 on tail rows and 0.7 on body rows, C1's floor at 0 s), so nothing is chosen here and k = 1. All predictions are cached; no GPU.

## Each step added, on the same 339,046 rows

| model | RMSE | Jan | Jul | tail rows | body rows |
|---|---|---|---|---|---|
| MC2 floored (champion) | 232.15 | 206.2 | 251.3 | 588.1 | 172.5 |
| seed bag | 231.17 | 205.2 | 250.3 | 585.7 | 171.7 |
| seed bag + ST1a gate | 230.84 | 205.2 | 249.8 | 585.7 | 171.2 |

Improvement SB2 vs MC2: +1.3 s [+1.0, +1.7]; paired SE 0.2 s; optimism bound for k = 1: 0.0 s; floor 2 s. Verdict: tie or lose, MC2 stays.
