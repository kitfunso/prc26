# SW1: Romano-Wolf step-down over the lane-A family, Jan + Jul 2025 development holdout

Regenerate with `python -m taxi.sw1`. Declared in protocol section 4 before the run. Champion SB3 floored at 230.0951 s on 339,046 rows; 7 challengers, each tested against it on identical rows. 2,000 day-block resamples, seed 0, ONE index set shared by every lane so the dependence between lanes is preserved; statistics are studentised by their own bootstrap SD and the null is the centred distribution. Family-wise error rate 5%. This lane promotes nothing: `taxi.verdict.FLOOR` is still the promotion rule.

## The family

| lane | holdout RMSE | improvement on SB3 | unadjusted 95% | studentised t | RW p | at 5% FWER |
|---|---|---|---|---|---|---|
| SB3+NN3 | 228.03 | +2.07 | [+1.12, +2.92] | +4.48 | 0.0000 | SURVIVES |
| SB3+NN2 | 228.41 | +1.68 | [+1.06, +2.28] | +5.26 | 0.0000 | SURVIVES |
| SB1 | 231.17 | -1.08 | [-1.31, -0.83] | -8.81 | 1.0000 | no |
| MC2 | 232.15 | -2.06 | [-2.37, -1.71] | -12.23 | 1.0000 | no |
| D1 | 241.35 | -11.25 | [-12.43, -9.98] | -18.13 | 1.0000 | no |
| W1-A | 250.90 | -20.80 | [-23.15, -18.42] | -17.28 | 1.0000 | no |
| M3 | 264.27 | -34.18 | [-41.89, -27.16] | -9.08 | 1.0000 | no |

Survives step-down: SB3+NN3, SB3+NN2.

## What this does and does not say

The family is the 7 lanes whose per-row holdout errors are cached on disk, not the 60 in the ledger, because the other lanes' vectors were never written. A larger family makes step-down STRICTER, so surviving here is necessary and not sufficient. What it does settle is the direction the campaign has never tested: the 2 s floor is a constant chosen once, while step-down is a measurement of what the shared holdout can support. A lane that survives at 5% FWER is not a multiple-comparisons artifact, whatever the floor says about it.

