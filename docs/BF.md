# BF: what a partner model must be to move lane A, Jan + Jul 2025 development holdout

Diagnostic, never a verdict, no cell declared. Regenerate with `python -m taxi.bf`. Champion SB3 at 229.78 s on 339,046 rows. For two predictors the blend RMSE is closed form in their two levels and their error correlation, so the best weight and the gain it buys need no fit. The weight below is read off the holdout itself and is therefore unavailable in practice: it upper-bounds what any out-of-fold choice could reach.

## Every lane-A model this campaign has cached, blended with the champion

| partner | its RMSE (s) | error correlation with SB3 | best blend (s) | its weight | gain (s) |
|---|---|---|---|---|---|
| MC2, CatBoost, the champion's own family | 232.15 | 0.9889 | 229.77 | 0.959 | +0.00 |
| D1, LightGBM, a different library | 241.35 | 0.9589 | 229.71 | 1.081 | +0.07 |
| M3, TabPFN per airport, a different architecture | 264.27 | 0.8462 | 229.56 | 0.929 | +0.22 |

The three partners span the whole model-class axis, from the champion's own library to a transformer, and the gain rises +0.00 to +0.22 s against a 2 s floor.

## The frontier: error correlation a partner needs for a 2 s gain

| partner RMSE (s) | correlation it must be under |
|---|---|
| 230 | 0.963 |
| 240 | 0.899 |
| 250 | 0.849 |
| 265 | 0.785 |
| 280 | 0.730 |
| 300 | 0.667 |

TabPFN, the least correlated model available at 0.8462, sits at 265 s where the frontier asks for 0.785. A partner at the champion's own level would need to stay under 0.963, and the two GBM libraries measured here read 0.9889 and 0.9589. Reading: a further model class does not reach the floor by blending, whatever library it comes from. It reaches it only by being better than SB3 outright, which is a different question and needs no blend.
