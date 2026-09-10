# MDE: the smallest gain each lane's verdict can see (diagnostic, never a verdict)

Regenerate with `python -m taxi.mde`. Champions SB3+NN3 and L8c, floored, holdout Jan + Jul 2025. No cell is declared and nothing here decides a verdict. Protocol Stage 0 asks for the detectable-effect floor before any modelling and this campaign never wrote one down. The decision rule needs BOTH halves: the 95% paired day-block interval above zero AND a point gain of at least the lane's own floor, 2 s on lane A and 30 s on lane B, which are the same 1.74 board seconds priced at each lane's measured carry. This prices what a challenger must actually deliver to satisfy them, by shrinking the champion's own errors on a named set of rows until the rule fires. Shrinking real errors is the most generous challenger there is: it never mis-fires on a row it does not improve, so every number below is a LOWER bound on what a real model has to do.

## 1. What the rule demands

### Lane A, champion SB3+NN3 at 228.03 s

| rows improved | n | their RMSE now | interval clears zero at | 2 s floor met at | binding | their RMSE then | composite (s) |
|---|---|---|---|---|---|---|---|
| every row | 339,046 | 228.0 s | 0.00% (+0.00 s) | 0.88% (+2.00 s) | 2 s floor | 226.0 s | +1.401 |
| body rows only (|c| under 600 s) | 313,148 | 167.1 s | 0.00% (+0.00 s) | 1.78% (+2.00 s) | 2 s floor | 164.1 s | +1.401 |
| tail rows only | 25,898 | 585.9 s | 0.00% (+0.00 s) | 1.75% (+2.00 s) | 2 s floor | 575.7 s | +1.401 |

### Lane B, champion L8c at 1,808.17 s

| rows improved | n | their RMSE now | interval clears zero at | 30 s floor met at | binding | their RMSE then | composite (s) |
|---|---|---|---|---|---|---|---|
| every row | 5,373 | 1,808.2 s | 0.00% (+0.00 s) | 1.66% (+30.01 s) | 30 s floor | 1,778.2 s | +2.637 |
| ordinary rows (truth under 1 h) | 5,167 | 838.5 s | 0.00% (+0.00 s) | 8.30% (+30.01 s) | 30 s floor | 768.9 s | +2.637 |
| the rest (truth over 1 h) | 206 | 8,224.4 s | 0.00% (+0.00 s) | 2.10% (+30.01 s) | 30 s floor | 8,051.9 s | +2.637 |


Every interval clears zero at the first fraction the bisection tries, so section 1 measures the FLOOR half of the rule and nothing else. A proportional shrink improves every row at once, so the paired difference is positive on every day and the bootstrap has no disagreement to price. Real challengers move rows both ways, and section 2 reads their noise off the campaign's own verdicts instead of inventing it.

## 2. What the interval actually cost, on this campaign's real challengers

Every holdout verdict in `docs/TRIAL-LEDGER.md` that recorded a paired standard error, grouped by lane. The minimum detectable effect is 1.96 SE: below it a challenger's interval contains zero however good its point estimate is.

| lane | verdicts | smallest MDE | median MDE | largest MDE | promotion floor | what binds |
|---|---|---|---|---|---|---|
| A | 22 | 0.2 s | 0.4 s | 9.4 s | 2 s | 2 s floor |
| B | 11 | 2.9 s | 31.6 s | 465.1 s | 30 s | the interval, by 1.1x |

Lane A's machinery is not the constraint: its median challenger had to beat 0.4 s of lane RMSE to clear zero, and the rule asks for 2. On lane B the two halves are now close: its median challenger had to beat 32 s against a floor of 30 s, so the interval binds. Under the flat 2 s floor this was never a contest, because 2 s of lane B is 0.12 board seconds and the interval asked for 1.83.

## 3. Why lane B's interval is so wide

| rows | share of lane-B squared error |
|---|---|
| worst 1 rows | 39.3% |
| worst 2 rows | 57.9% |
| worst 5 rows | 67.3% |
| worst 20 rows | 75.4% |
| worst 100 rows | 87.7% |

Lane B is 5,373 rows against lane A's 339,046, and its squared error is concentrated in a handful of them (`docs/DE.md`, flag F21). A day-block bootstrap resamples whole days, so a day carrying one of those rows moves the whole lane's RMSE by hundreds of seconds, and the interval inherits that spread whatever the challenger does elsewhere.

