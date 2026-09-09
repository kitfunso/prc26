# MDE: the smallest gain each lane's verdict can see (diagnostic, never a verdict)

Regenerate with `python -m taxi.mde`. Champions SB3 and L8c, floored, holdout Jan + Jul 2025. No cell is declared and nothing here decides a verdict. Protocol Stage 0 asks for the detectable-effect floor before any modelling and this campaign never wrote one down. The decision rule needs BOTH halves: the 95% paired day-block interval above zero AND a point gain of at least 2 s of lane RMSE. This prices what a challenger must actually deliver to satisfy them, by shrinking the champion's own errors on a named set of rows until the rule fires. Shrinking real errors is the most generous challenger there is: it never mis-fires on a row it does not improve, so every number below is a LOWER bound on what a real model has to do.

## 1. What the rule demands

### Lane A, champion SB3 at 230.10 s

| rows improved | n | their RMSE now | interval clears zero at | 2 s floor met at | binding | their RMSE then | composite (s) |
|---|---|---|---|---|---|---|---|
| every row | 339,046 | 230.1 s | 0.00% (+0.00 s) | 0.87% (+2.00 s) | 2 s floor | 228.1 s | +1.408 |
| body rows only (|c| under 600 s) | 313,148 | 170.8 s | 0.00% (+0.00 s) | 1.71% (+2.00 s) | 2 s floor | 167.9 s | +1.408 |
| tail rows only | 25,898 | 583.3 s | 0.00% (+0.00 s) | 1.78% (+2.00 s) | 2 s floor | 572.9 s | +1.408 |

### Lane B, champion L8c at 1,808.17 s

| rows improved | n | their RMSE now | interval clears zero at | 2 s floor met at | binding | their RMSE then | composite (s) |
|---|---|---|---|---|---|---|---|
| every row | 5,373 | 1,808.2 s | 0.00% (+0.00 s) | 0.11% (+2.00 s) | 2 s floor | 1,806.2 s | +0.176 |
| ordinary rows (truth under 1 h) | 5,167 | 838.5 s | 0.00% (+0.00 s) | 0.54% (+2.00 s) | 2 s floor | 834.0 s | +0.176 |
| the rest (truth over 1 h) | 206 | 8,224.4 s | 0.00% (+0.00 s) | 0.14% (+2.00 s) | 2 s floor | 8,212.9 s | +0.176 |


Every interval clears zero at the first fraction the bisection tries, so section 1 measures the FLOOR half of the rule and nothing else. A proportional shrink improves every row at once, so the paired difference is positive on every day and the bootstrap has no disagreement to price. Real challengers move rows both ways, and section 2 reads their noise off the campaign's own verdicts instead of inventing it.

## 2. What the interval actually cost, on this campaign's real challengers

Every holdout verdict in `docs/TRIAL-LEDGER.md` that recorded a paired standard error, grouped by lane. The minimum detectable effect is 1.96 SE: below it a challenger's interval contains zero however good its point estimate is.

| lane | verdicts | smallest MDE | median MDE | largest MDE | what binds |
|---|---|---|---|---|---|
| A | 18 | 0.2 s | 0.4 s | 9.4 s | the 2 s floor |
| B | 10 | 2.9 s | 32.0 s | 465.1 s | the interval, by 16x |

Lane A's machinery is not the constraint: its median challenger had to beat 0.4 s of lane RMSE to clear zero, and the rule asks for 2. Lane B's is: its median challenger had to beat 32 s, 16 times the nominal floor and worth 2.8 s of composite. The 2 s floor has never been the binding constraint on a lane-B lane; the interval has, every time.

## 3. Why lane B's interval is so wide

| rows | share of lane-B squared error |
|---|---|
| worst 1 rows | 39.3% |
| worst 2 rows | 57.9% |
| worst 5 rows | 67.3% |
| worst 20 rows | 75.4% |
| worst 100 rows | 87.7% |

Lane B is 5,373 rows against lane A's 339,046, and its squared error is concentrated in a handful of them (`docs/DE.md`, flag F21). A day-block bootstrap resamples whole days, so a day carrying one of those rows moves the whole lane's RMSE by hundreds of seconds, and the interval inherits that spread whatever the challenger does elsewhere.

