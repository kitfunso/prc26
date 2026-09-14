# Holdout error breakdown (v23 blend, Jan+Jul 2025)

`python errors.py "blend_ra_cb+ra_cb_s1+ra_lgb+nn3"` on the cached holdout predictions (mechanism = which stamp the label copies, if any; day_shift = y above 72,000 s). Total RMSE 327.08 over 344,419 rows;
"share" is the row group's part of the total squared error.

| group | rows | RMSE | share | note |
|---|---|---|---|---|
| lane B (no plan), all airports | 5,373 | ~1900 | 0.53 | 1.6% of rows |
| lane B day-shift rows (y > 72,000 s) | 6 | 40674 | 0.27 | one LFPG row at 83,066 s (2025-01-19), five LIRF rows; a second LFPG row at 59,380 s sits in the 21,600 to 72,000 bucket (17 rows, 0.10) |
| LFPG lane B, physics rows | 874 | 2000 | 0.18 | includes the 59,380 s row; July LFPG lane B is RMSE ~445 |
| LIRF lane B, physics + schedule copies | 392 | 3226 / 3712 | 0.23 | bimodal label: median 1447 s, top quartile above 6960 s; physics rows over-predicted by 1779 s, schedule copies under by 2283 s: the mixture hedges, which squared loss asks for when the copy class is not separable; `laneb.py` gated cell lost |
| lane A schedule copies (y within 60 s of the schedule stamp) | 5,190 | 563 | 0.045 | bias -140 s; the only lane-A mechanism with a large offset |
| lane A tail, abs(residual) > 600 s | 25,898 | 575 | 0.23 | mean bias -159 s (under-predicted), the gate's expected shrinkage |
| lane A tail, abs(residual) > 1800 s | 2,398 | 1236 | 0.10 | bias -684 s |
| lane A, all | 339,046 | 226.8 | 0.47 | the part the model can move; the v22 to v23 lane-A gain carried to the board about 1:1 |

Per-airport lane A RMSE: LIRF 375, LTFM 255 (bias -26 s, the only airport with a systematic offset), LFPG 246, EGLL 240,
LEBL 220, EDDF 180, LSZH 177, LEMD 174, EHAM 166, EDDM 166. Bias contributes under 0.1% of squared error anywhere,
so per-airport offsets are not a lever.

Lane B rows carry no AOBT/EOBT/LOBT/IOBT stamp; only the schedule stamp and the anchor exist. The day-shift rows
cannot be predicted from that, and shifting a group mean toward them costs the other rows more than it saves.
The holdout "all" figure is therefore dominated by a handful of rows; compare candidates on lane A and on the paired
day-block bootstrap, which is what the ledger does.

Midnight crossing does not explain them either (session scratchpad `shift_mid.py`, 2025 train). Of 5,903 rows under
4 h late whose schedule falls on the local date before the off-block, one is a day-shift row (LIRF lane B, 23:59
schedule, 03:13 takeoff). The two day-shift rows off LIRF sit on one date with no crossing: LFPG 2025-01-19 (y 84,240 s,
29 min late, stamp 20:30 the evening before) and LSZH 2025-05-14 (y 87,341 s, `z_sched` 939 s: the schedule time on
the previous date). The one row over 21,600 s off LIRF, LFPG 2025-01-23 (y 58,206 s), carries a 23:59 stamp from the
evening before. The other twelve LIRF lane B day-shift rows are 4 h or more late (`docs/LANEB.md` sections 4 and 5).

Postponed-flight traces do not find them either (session scratchpad `tow_a.py`, `tow_b.py`, 2025 train). The two LFPG
rows read as flights first pushed back the evening before and flown the next day: the same flight number has no
departure on the day before, and their stands saw no other movement for 32.5 h and 40.6 h before takeoff. Both traces
exist at test time, and neither selects anything: off LIRF, 913 lane B rows had their stand idle 20 h or more before
takeoff, 2 of them over 6 h (the two LFPG rows), mean label 1,101 s; 1,519 rows had no flight the day before after 4
or more of the 7 days before, none over 6 h. The recorded off-block is an arrival's in-block on the same stand (within
1 min) on 2 of 33 lane B rows over 12 h, 10 of 901 lane B rows between 1 and 3 h, and 0.03% of lane A rows under 30 min.
