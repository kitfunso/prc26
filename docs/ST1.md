# ST1: a row-dependent blend of MC2 and D1, Jan + Jul 2025 development holdout

> **WITHDRAWN 2026-09-07 (flag F36).** The ST1a cell below calls `taxi.m2.is_tail`, which is `abs(c) > 600` with `c = y - z_aobt`, so it sets each predicted row's weight from that row's own label. ST1b and ST1c are honest (a meta-learner on features, a per-airport weight) and their rows stand; ST1a's row is withdrawn and decides nothing. `python -m taxi.st1` raises, because `main` walks all three cells and ST1a is one of them. The honest re-scoring of what ST1a's gate was worth lives in `docs/SB3.md`.

Regenerate with `python -m taxi.st1`. Both base models are cached, so this run touches no GPU. The weight is chosen out of fold over the ten training months (1,723,531 rows) and only the OOF winner meets the holdout, priced at k = 3.

MC2 OOF 212.86 s, D1 OOF 221.08 s on the same rows.

| cell | out-of-fold RMSE (s) | fit s |
|---|---|---|
| ST1a | 212.27 | 2 |
| ST1b | 216.14 | 23 |
| ST1c | 212.73 | 3 |

OOF winner: **ST1a**, refit on all ten training months. What it learned: weight on MC2: tail rows 1.0, body rows 0.7.

## Holdout verdict vs champion MC2 (339,046 rows; tail rows 25,898)

| model | RMSE | Jan | Jul | tail rows | body rows |
|---|---|---|---|---|---|
| MC2 | 232.7 | 206.8 | 251.9 | 591.0 | 172.5 |
| D1 | 242.4 | 217.7 | 260.8 | 627.7 | 176.1 |
| ST1a | 232.1 | 206.4 | 251.1 | 591.0 | 171.6 |

Improvement ST1a vs MC2: +0.6 s [+0.5, +0.8]; paired SE 0.1 s; optimism bound for k = 3: 0.1 s; floor 2 s. Verdict: tie or lose, MC2 stays.
