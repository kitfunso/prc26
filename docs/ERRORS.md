# Holdout error breakdown (v23 blend, Jan+Jul 2025)

`python errors.py "blend_ra_cb+ra_cb_s1+ra_lgb+nn3"` on the cached holdout predictions. Total RMSE 327.08 over 344,419 rows;
"share" is the row group's part of the total squared error.

| group | rows | RMSE | share | note |
|---|---|---|---|---|
| lane B (no plan), all airports | 5,373 | ~1900 | 0.53 | 1.6% of rows |
| LFPG lane B | 880 | 3437 | 0.28 | two rows with y about 71,000 s (2025-01-19, 2025-01-23) hold 0.28 by themselves; the rest of LFPG lane B is RMSE ~445 in July |
| LIRF lane B | 397 | 4363 | 0.21 | bimodal label: median 1447 s, top quartile above 6960 s (schedule copies); the mixture hedges (p median 3231), which squared loss asks for when the copy class is not separable; `laneb.py` gated cell lost |
| lane A tail, abs(residual) > 600 s | 25,898 | 575 | 0.23 | mean bias -159 s (under-predicted), the gate's expected shrinkage |
| lane A tail, abs(residual) > 1800 s | 2,398 | 1236 | 0.10 | bias -684 s |
| lane A, all | 339,046 | 226.8 | 0.47 | the part the model can move; the v22 to v23 lane-A gain carried to the board about 1:1 |

Per-airport lane A RMSE: LIRF 375, LTFM 255 (bias -26 s, the only airport with a systematic offset), LFPG 246, EGLL 240,
LEBL 220, EDDF 180, LSZH 177, LEMD 174, EHAM 166, EDDM 166. Bias contributes under 0.1% of squared error anywhere,
so per-airport offsets are not a lever.

Lane B rows carry no AOBT/EOBT/LOBT/IOBT stamp; only the schedule stamp and the anchor exist. The two LFPG monsters
cannot be predicted from that, and shifting the group mean toward them costs the other 878 rows more than it saves.
The holdout "all" figure is therefore dominated by a handful of rows; compare candidates on lane A and on the paired
day-block bootstrap, which is what the ledger does.
