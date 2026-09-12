# DQ1: the conditional distribution instead of the hand-set 600 s split, lane A, Jan + Jul 2025 holdout

Regenerate with `python -m taxi.dq1`. Three arms, k = 3, protocol section 4. All three fit `c` in ONE stage on D2's 52 columns at `taxi.mc1.CELL` unchanged, up to 5000 iterations, early-stopped on the last 10% of training DAYS, with the plan gap added back and the C1 floor applied. They differ in the LOSS alone. Comparator SB3, 230.0951 s. The arms are single-seed against a five-seed champion bag, which `docs/SEEDS.md` prices at 1.07 s; any arm inside that of SB3 is re-run bagged rather than called a tie.

## Arms

| arm | holdout RMSE | mean predicted c | mean realised c | midpoint-rule bias |
|---|---|---|---|---|
| DQ1c (RMSE, no split) | 233.23 | -7.1 | -8.9 | +1.9 |
| DQ1g (Gaussian head) | 240.47 | -5.8 | -8.9 | +3.1 |
| DQ1q (20 quantiles) | 373.07 | -21.5 | -8.9 | -12.6 |

## The loss, isolated

DQ1c carries both the removed split and the changed stopping, so it is not a clean reading of either. The arm-to-arm contrasts are, because the three share everything but the loss:

| contrast | RMSE change (negative is better) |
|---|---|
| DQ1g (Gaussian head) minus DQ1c | +7.24 |
| DQ1q (20 quantiles) minus DQ1c | +139.84 |

## Verdicts against SB3

- **DQ1c (RMSE, no split)**: Improvement DQ1c vs SB3: -3.1 s [-4.1, -2.3]; paired SE 0.5 s; optimism bound for k = 3: 0.4 s; floor 2 s. Verdict: tie or lose, SB3 stays.
- **DQ1g (Gaussian head)**: Improvement DQ1g vs SB3: -10.4 s [-11.3, -9.5]; paired SE 0.5 s; optimism bound for k = 3: 0.4 s; floor 2 s. Verdict: tie or lose, SB3 stays.
- **DQ1q (20 quantiles)**: Improvement DQ1q vs SB3: -143.0 s [-152.2, -133.5]; paired SE 4.8 s; optimism bound for k = 3: 4.1 s; floor 2 s. Verdict: tie or lose, SB3 stays.


