# M1 per-feature knockout (protocol section 0, complexity control)

Month-pair CV at the chosen cell and rounds, one feature dropped at a time; regenerate with `python -m taxi.m1 knockout`.

## Lane A

Lane A, full model CV RMSE 232.9 s at {'num_leaves': 31, 'min_data_in_leaf': 200, 'objective': 'l2'}, 1016 rounds.

| dropped feature | CV RMSE (s) | loss vs full (s) | clears floor |
|---|---|---|---|
| apt | 233.7 | +0.7 | no |
| hour | 233.6 | +0.7 | no |
| wday | 232.5 | -0.4 | no |
| has_fpl | 232.9 | +0.0 | no |
| z_aobt | 236.1 | +3.2 | yes |
| z_eobt | 237.3 | +4.4 | yes |
| z_lobt | 234.5 | +1.6 | no |
| z_sched | 238.4 | +5.5 | yes |
| z_anchor | 232.4 | -0.5 | no |
| runway | 239.0 | +6.1 | yes |
| stand | 235.9 | +3.0 | yes |
| actype | 233.5 | +0.6 | no |
| wake | 232.5 | -0.4 | no |
| segment | 233.2 | +0.3 | no |
| queue | 233.8 | +0.9 | no |
| dep_before | 232.9 | +0.0 | no |
| dep_after | 233.1 | +0.2 | no |
| arr_anchor | 232.8 | -0.1 | no |
| arr_off | 232.8 | -0.1 | no |
| push_anchor | 232.7 | -0.2 | no |

## Lane B

Lane B, full model CV RMSE 1857.2 s at {'num_leaves': 7, 'min_data_in_leaf': 50, 'objective': 'l2'}, 385 rounds.

| dropped feature | CV RMSE (s) | loss vs full (s) | clears floor |
|---|---|---|---|
| apt | 1931.3 | +74.1 | yes |
| hour | 1866.5 | +9.3 | yes |
| wday | 1854.8 | -2.3 | no |
| has_fpl | 1857.2 | +0.0 | no |
| z_aobt | 1857.2 | +0.0 | no |
| z_eobt | 1857.2 | +0.0 | no |
| z_lobt | 1857.2 | +0.0 | no |
| z_sched | 1856.3 | -0.9 | no |
| z_anchor | 1856.3 | -0.9 | no |
| runway | 1920.4 | +63.2 | yes |
| stand | 1781.0 | -76.2 | no |
| actype | 1861.2 | +4.0 | yes |
| wake | 1857.2 | +0.0 | no |
| segment | 1857.2 | +0.0 | no |
| queue | 1891.4 | +34.2 | yes |
| dep_before | 1873.4 | +16.3 | yes |
| dep_after | 1872.5 | +15.3 | yes |
| arr_anchor | 1870.9 | +13.7 | yes |
| arr_off | 1869.4 | +12.3 | yes |
| push_anchor | 1870.4 | +13.2 | yes |

