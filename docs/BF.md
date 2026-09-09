# BF: what a partner model must be to move lane A, Jan + Jul 2025 development holdout

Diagnostic, never a verdict, no cell declared. Regenerate with `python -m taxi.bf`. Lane-A champion SB3, the five-seed D2 bag, at 230.10 s on 339,046 rows. For two predictors the blend RMSE is closed form in their two levels and their error correlation, so the best weight and the gain it buys need no fit. Agreement is the UNCENTRED error correlation, mean(e0 x ex) over the two RMSEs, which is the form that keeps the identity exact when the errors are not zero mean. The weight below is read off the holdout itself and is therefore unavailable in practice: it upper-bounds what any out-of-fold choice could reach.

## Every lane-A model this campaign has cached, blended with the champion

| partner | its RMSE (s) | error agreement with SB3 | best blend (s) | its weight | gain (s) |
|---|---|---|---|---|---|
| MC2, CatBoost, the champion's own family | 232.15 | 0.9892 | 230.08 | 0.912 | +0.02 |
| D1, LightGBM, a different library | 241.35 | 0.9446 | 230.01 | 0.923 | +0.08 |
| M3, TabPFN per airport, a different architecture | 264.27 | 0.8429 | 229.79 | 0.917 | +0.31 |
| NN2, a bagged neural net | 234.22 | 0.9403 | 228.36 | 0.648 | +1.74 |

The three partners span the whole model-class axis, from the champion's own library to a transformer, and the gain rises +0.02 to +1.74 s against a 2 s floor.

## The frontier: error agreement a partner needs for a 2 s gain

| partner RMSE (s) | agreement it must be under |
|---|---|
| 230 | 0.966 |
| 240 | 0.901 |
| 250 | 0.851 |
| 265 | 0.786 |
| 280 | 0.731 |
| 300 | 0.668 |

The least agreeing partner is TabPFN at 0.940, and at its 234 s level the frontier asks for 0.935. A partner at the champion's own level would have to stay under 0.965, and the two GBM libraries measured here read 0.9892 and 0.9446. Reading: a further model class does not reach the floor by blending, whatever library it comes from. It reaches it only by being better than SB3 outright, which is a different question and needs no blend.

## Every subset at once: the 4-partner stack

The pairwise table above answers one partner at a time, which is not the question a stack asks. These weights are affine (they sum to one) and solved on the holdout's own errors for every subset that contains the champion, so each row is a bound no out-of-fold stack can beat. Weights are listed champion first, then the partners in the row's own order.

| stack | RMSE (s) | gain (s) | weights |
|---|---|---|---|
| SB3 + MC2 + D1 + M3 + NN2 | 228.22 | +1.88 | 0.71, -0.08, -0.04, 0.05, 0.36 |
| SB3 + D1 + M3 + NN2 | 228.23 | +1.86 | 0.65, -0.05, 0.05, 0.35 |
| SB3 + MC2 + M3 + NN2 | 228.24 | +1.86 | 0.70, -0.10, 0.05, 0.34 |
| SB3 + M3 + NN2 | 228.26 | +1.84 | 0.62, 0.05, 0.34 |
| SB3 + MC2 + D1 + NN2 | 228.32 | +1.77 | 0.74, -0.07, -0.04, 0.37 |
| SB3 + D1 + NN2 | 228.33 | +1.76 | 0.68, -0.04, 0.37 |
| SB3 + MC2 + NN2 | 228.34 | +1.76 | 0.73, -0.09, 0.36 |
| SB3 + NN2 | 228.36 | +1.74 | 0.65, 0.35 |
| SB3 + MC2 + D1 + M3 | 229.74 | +0.36 | 0.83, 0.03, 0.06, 0.08 |
| SB3 + D1 + M3 | 229.74 | +0.36 | 0.86, 0.06, 0.08 |
| SB3 + MC2 + M3 | 229.78 | +0.32 | 0.85, 0.07, 0.08 |
| SB3 + M3 | 229.79 | +0.31 | 0.92, 0.08 |
| SB3 + MC2 + D1 | 230.01 | +0.09 | 0.88, 0.05, 0.07 |
| SB3 + D1 | 230.01 | +0.08 | 0.92, 0.08 |
| SB3 + MC2 | 230.08 | +0.02 | 0.91, 0.09 |

The best of the 15 subsets is **SB3 + MC2 + D1 + M3 + NN2** at 228.22 s, +1.88 s against a 2 s floor. The best single partner buys +1.74 s, so putting every cached model in one stack adds +0.14 s on top of the best pair.

Every lane-A model whose holdout predictions are on disk is in the table, NN2 included. NN1 is not: it was superseded by NN2 and its own predictions were never written to a file.

The bound is under the 2 s floor with the weights read off the answers, so no out-of-fold stack of these models can reach it and the direction is closed by arithmetic. The partners agree with each other about as much as they agree with the champion, so their errors leave little for a wider combination to remove.
