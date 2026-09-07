# BF: what a partner model must be to move lane A, Jan + Jul 2025 development holdout

Diagnostic, never a verdict, no cell declared. Regenerate with `python -m taxi.bf`. Lane-A champion SB3, the five-seed D2 bag, at 230.10 s on 339,046 rows. For two predictors the blend RMSE is closed form in their two levels and their error correlation, so the best weight and the gain it buys need no fit. Agreement is the UNCENTRED error correlation, mean(e0 x ex) over the two RMSEs, which is the form that keeps the identity exact when the errors are not zero mean. The weight below is read off the holdout itself and is therefore unavailable in practice: it upper-bounds what any out-of-fold choice could reach.

## Every lane-A model this campaign has cached, blended with the champion

| partner | its RMSE (s) | error agreement with SB3 | best blend (s) | its weight | gain (s) |
|---|---|---|---|---|---|
| MC2, CatBoost, the champion's own family | 232.15 | 0.9892 | 230.08 | 0.912 | +0.02 |
| D1, LightGBM, a different library | 241.35 | 0.9446 | 230.01 | 0.923 | +0.08 |
| M3, TabPFN per airport, a different architecture | 264.27 | 0.8429 | 229.79 | 0.917 | +0.31 |

The three partners span the whole model-class axis, from the champion's own library to a transformer, and the gain rises +0.02 to +0.31 s against a 2 s floor.

## The frontier: error agreement a partner needs for a 2 s gain

| partner RMSE (s) | agreement it must be under |
|---|---|
| 230 | 0.966 |
| 240 | 0.901 |
| 250 | 0.851 |
| 265 | 0.786 |
| 280 | 0.731 |
| 300 | 0.668 |

The least agreeing partner is TabPFN at 0.843, and at its 264 s level the frontier asks for 0.789. A partner at the champion's own level would have to stay under 0.965, and the two GBM libraries measured here read 0.9892 and 0.9446. Reading: a further model class does not reach the floor by blending, whatever library it comes from. It reaches it only by being better than SB3 outright, which is a different question and needs no blend.
