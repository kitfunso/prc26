# M3 pilot: TabPFN 6.3.1 on LIRF lane A (R12)

Train months (2, 3, 4, 5, 6, 8, 9, 10, 11, 12), month-pair folds; regenerate with `python -m taxi.m3`. Features ['hour', 'wday', 'has_fpl', 'z_aobt', 'z_eobt', 'z_lobt', 'z_sched', 'z_anchor', 'runway', 'stand', 'actype', 'wake', 'segment', 'queue', 'dep_before', 'dep_after', 'arr_anchor', 'arr_off', 'push_anchor'] (M1's set without the constant airport). TabPFN sees a seeded subsample of 50,000 rows per fold (its pretraining limit); M1 is the champion fit on all airports' lane-A rows outside the fold, scored on this airport's fold rows. n_estimators 8 (default), device cuda.

## Folds

| fold | fit rows (of the airport's rows outside the fold) | predicted rows | M1 RMSE (s) | M3 RMSE (s) | fit + predict s | peak VRAM GB |
|---|---|---|---|---|---|---|
| (2, 3) | 50000 of 109955 | 23130 | 334.2 | 328.1 | 399 | 3.76 | seeded from the printed line of a killed run (RMSE rounded to 0.1 s)
| (4, 5) | 50000 of 104685 | 28400 | 369.5 | 357.6 | 469 | 3.76 | seeded from the printed line of a killed run (RMSE rounded to 0.1 s)
| (6, 8) | 50000 of 103312 | 29773 | 462.0 | 447.4 | 554 | 3.76 | seeded from the printed line of a killed run (RMSE rounded to 0.1 s)
| (9, 10) | 50000 of 104559 | 28526 | 428.4 | 411.4 | 461 | 3.76 | seeded from the printed line of a killed run (RMSE rounded to 0.1 s)
| (11, 12) | 50000 of 109829 | 23256 | 337.5 | 333.5 | 403 | 3.76 |

Pooled OOF over 133,085 rows: M1 394.4 s, M3 382.7 s, gain +11.6 s.

## Gates

| gate | measured | limit | result |
|---|---|---|---|
| fit plus predict under 30 min (all five folds) | 38.1 | 30 | FAIL |
| peak VRAM under 16 GB | 3.8 | 16 | PASS |
| OOF gain over M1 on this airport of at least 2 s | 11.6 | 2 | PASS |

Pilot verdict: a gate fails; M3 stops here unless the declaration is amended.

