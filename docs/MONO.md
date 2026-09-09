# MONO: is there a monotone sign to impose on lane A?

Regenerate with `python -m taxi.mono`. Spearman rank correlation over all 1,723,531 lane-A training rows; the holdout is not read. Diagnostic only: no model is fitted here and nothing is promoted.

Stage 4.1 rung 6 asks for monotone constraints where a driver's sign is domain knowledge. Lane A does not predict taxi time. It predicts `c = y - z_aobt`, and `z_aobt` is the flight plan's own taxi estimate, so `c` is the disagreement between two clocks. The question this file answers is whether the physics sign survives that subtraction.

| column | physics sign | rho vs level `y` | rho vs target `c` | level agrees | on the target |
|---|---|---|---|---|---|
| queue | + | +0.4189 | -0.2103 | yes | FLIPPED |
| dep_before | + | +0.1087 | -0.0631 | yes | FLIPPED |
| dep_after | + | +0.1185 | -0.0498 | yes | FLIPPED |
| occ | + | +0.1513 | +0.0277 | yes |  |
| t_occ | + | +0.4478 | +0.0760 | yes |  |
| arr_off | + | +0.1108 | -0.0669 | yes | FLIPPED |
| dur | + | +0.3351 | +0.0543 | yes |  |
| w_sknt | + | +0.0779 | -0.0964 | yes | FLIPPED |
| w_gust | + | +0.0208 | +0.0117 | yes |  |
| w_vsby | - | +0.0107 | -0.0331 | no |  |
| w_ceil | - | +0.0072 | -0.0111 | no |  |

## What it says

**Physics is right about the level.** 9 of 11 columns carry the sign physics predicts against `y`, and the three largest are the congestion and duration measures.

**Physics is backwards on the target.** 5 columns flip sign between `y` and `c`: `queue` +0.4189 to -0.2103, `dep_before` +0.1087 to -0.0631, `dep_after` +0.1185 to -0.0498, `arr_off` +0.1108 to -0.0669, `w_sknt` +0.0779 to -0.0964. The plan's estimate over-absorbs congestion, so when the queue is long `z_aobt` reads high and the correction runs the other way. A constraint taken from physics would be imposed with the wrong sign and would cost, not help.

## What that leaves

The rung is NOT closed by this, because a sign measured on the target is still a sign. It is closed as a lane on two conditions that a declaration would have to meet. First, the sign has to come from rows the verdict does not judge, or the constraint is fitted rather than imposed. Second, and the reason it is not queued ahead of the tuned piecewise-linear lane, a monotone constraint is regularisation, and it binds when data is thin; lane A fits 1,723,531 rows, where a rank correlation of 0.2103 is not something a depth-8 ensemble mistakes for noise.

