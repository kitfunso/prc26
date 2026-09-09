# SEEDS: what more seeds in the bag are worth (diagnostic, never a verdict)

Regenerate with `python -m taxi.seeds`. No cell is declared and no champion moves. The champion SB3 averages five seeds of D2's
cell; "bag more seeds" is the cheapest-looking knob left on lane A, so it is priced here rather than guessed at.

Read from the five cached seed predictions in `data/cache/sb3_seed*_holdout.parquet`, floored by C1, on the same 339,046 lane-A
holdout rows. Each bag size averages every subset of that size, so the row is the expected bag and not one lucky draw.

| bag size | mean RMSE over all subsets (s) | best subset | worst subset |
|---|---|---|---|
| 1 | 231.1679 | 230.9751 | 231.5067 |
| 2 | 230.4983 | 230.3161 | 230.7321 |
| 3 | 230.2745 | 230.1187 | 230.4054 |
| 4 | 230.1625 | 230.0764 | 230.2156 |
| 5 | 230.0951 | 230.0951 | 230.0951 |

Seed noise is independent across seeds, so squared error should fall as `MSE(n) = M + s2/n`. Fitting that on the five points:

    MSE(n) = 52820.1 + 618.5 / n      seed-noise sd 24.87 s

The largest residual of the fit is 0.08 MSE units against values near 52,900, so the law holds and the extrapolation is arithmetic
rather than hope.

| bag size | predicted RMSE (s) | gain vs the five-seed bag |
|---|---|---|
| 5 | 230.0953 | +0.0000 |
| 10 | 229.9608 | +0.1344 |
| 15 | 229.9160 | +0.1793 |
| 20 | 229.8936 | +0.2017 |
| 40 | 229.8600 | +0.2353 |
| infinite | 229.8263 | +0.2689 |

## What this closes

An infinite bag of D2 seeds is worth **+0.269 s** of lane A against a 2 s floor, and about **0.17 s of board** at the equal-ratio
carry in `docs/BL.md` section 5. Twenty seeds, three more GPU hours than the champion already spends, reach +0.202 s. The knob is
exhausted at five: `M = 52,820` is 99.99% of the five-seed MSE and no amount of averaging touches it. Anything that moves lane A
has to move `M`, which is the model and not the seed.
