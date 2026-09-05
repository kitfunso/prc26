# Trial ledger (append-only)

Contract: `docs/EXPERIMENT-PROTOCOL.md` section 8. Sign convention: improvement = champion RMSE minus challenger RMSE, positive is good.
Rows are never edited; a superseded row gets a note in a later row. Old-spec rows stay, labelled.

Counters: N (board submissions) = 1 model (S1) + 2 scored probes (P1, P2); 4 probe files rejected by the scorer's daily cap on 2026-09-05. N_h (holdout verdicts) = 19.

## A. Board submissions

| id | date | file | spec (lane A + lane B) | git | holdout lane A | holdout lane B | holdout all rows | holdout improvement | board RMSE | board improvement | haircut | usedPairs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 2026-09-05 | `merry-octopus_v1.parquet` (sha256 c79d20d7b691ef1696acdf93537e092fa8faf405c62a08aea15f5a55488f39de, 344,841 rows, uploaded 09:5x UTC, scored 09:58:03 UTC) | lane A M1 (31 leaves, min_data 200, l2, 1,016 rounds) + lane B B4 [z_sched, queue], both fit on all 12 months of 2025 | 256b4c1 | 259.5 | 2,074.1 | 365.3 | lane A +51.6 [+48.4, +55.1] vs B4 (H13); all rows +37.7 vs B4 in both lanes (reported) | 325.3982 (rank 11 of 29 teams; leader 277.29) | N/A, first submission | N/A | 344,841 |
| S2 | 2026-09-05 | `merry-octopus_v10.parquet` (sha256 1076653a3bcce1c554daa5609a22f529f1078a31206d95bbef90d9b76e5fca1c, 344,841 rows, BUILT, NOT uploaded, WITHDRAWN as the candidate 2026-09-05 when D1 superseded W1-A (H18): Keith gate `docs/gate-s2.html`, replaced by `docs/gate-s3.html`) | lane A W1-A (M2 recipe: classifier plus tail and body LightGBM, 31 leaves, min_data 200, l2, rounds 577/378/1724, 36 columns incl. 16 METAR) + lane B B4 [z_sched, queue], both fit on all 12 months of 2025 | 2afe8fa | 253.8 | 2,074.1 | 361.3 | lane A +2.9 [+1.8, +3.9] vs M2 (H16) and +5.7 vs the S1 model M1 (H14 + H16, point estimates); lane B unchanged | (after upload) | (after upload) | (after upload) | 344,841 |

## B. Holdout verdicts (Jan + Jul 2025, paired day-block bootstrap, 2,000 reps, seed 0; floor 2 s)

Source: `python -m taxi.baselines` on 2026-09-04, output in `docs/BASELINES.md`; `python -m taxi.m1` on 2026-09-05, output in `docs/M1.md`. Caps: none in both lanes (CAP sweep, protocol section 4).

| id | date | lane | champion | challenger | challenger RMSE | improvement [95%] | verdict |
|---|---|---|---|---|---|---|---|
| H1 | 2026-09-04 | B | B0 (3,959.4) | B1 | 3,709.9 | +249.4 [+135.5, +353.9] | WIN, B1 promoted |
| H2 | 2026-09-04 | B | B1 | B2 | 3,807.5 | -97.6 [-232.9, +15.4] | lose |
| H3 | 2026-09-04 | B | B1 | B3s | 4,889.7 | -1,179.7 [-2,262.0, -214.0] | lose |
| H4 | 2026-09-04 | B | B1 | B4 | 2,074.1 | +1,635.8 [+733.6, +2,538.9] | WIN, B4 promoted (lane-B champion) |
| H5 | 2026-09-04 | A | B0 (476.7) | B1 | 440.7 | +36.1 [+31.6, +41.1] | WIN, B1 promoted |
| H6 | 2026-09-04 | A | B1 | B2 | 433.1 | +7.6 [+6.2, +8.9] | WIN, B2 promoted |
| H7 | 2026-09-04 | A | B2 | B3a | 413.6 | +19.5 [-17.9, +73.7] | tie (interval crosses zero) |
| H8 | 2026-09-04 | A | B2 | B3b | 620.9 | -187.8 [-233.8, -132.0] | lose |
| H9 | 2026-09-04 | A | B2 | B3c | 709.7 | -276.6 [-323.6, -220.7] | lose |
| H10 | 2026-09-04 | A | B2 | B3d | 400.1 | +33.0 [-2.5, +86.3] | tie (interval crosses zero) |
| H11 | 2026-09-04 | A | B2 | B4 | 311.2 | +121.9 [+88.7, +173.5] | WIN, B4 promoted (lane-A champion) |
| H12 | 2026-09-05 | B | B4 | M1 | 2,386.3 | -312.1 [-643.8, -12.1] | lose; B4 stays lane-B champion (`docs/M1.md`) |
| H13 | 2026-09-05 | A | B4 | M1 | 259.5 | +51.6 [+48.4, +55.1] | WIN, clears bound 2.8 s plus floor 2 s; M1 promoted (lane-A champion) |
| H14 | 2026-09-05 | A | M1 | M2 | 256.6 | +2.9 [+1.8, +4.1] | WIN, clears bound 0 s (k = 1) plus floor 2 s on the point estimate, low end of the interval 1.8 s; Jan +2.3, Jul +3.3; M2 promoted (lane-A champion, `docs/M2.md`) |
| H15 | 2026-09-05 | B | B4 | W1-B (B4 + w_sknt, w_vsby, w_p3h, w_ts3h, w_sn) | 2,064.2 | +9.9 [-2.7, +22.6] | tie (interval crosses zero); B4 stays (`docs/W1.md`) |
| H16 | 2026-09-05 | A | M2 | W1-A (M2 recipe on M1's columns plus 16 weather columns) | 253.8 | +2.9 [+1.8, +3.9] | WIN, clears bound 0 s (k = 1) plus floor 2 s on the point estimate, low end of the interval 1.8 s under the floor (same shape as H14); Jan +4.8, Jul +1.6; W1-A promoted (lane-A champion, `docs/W1.md`) |
| H17 | 2026-09-05 | B | B4 | L1a (B4 plus the VFR flag; CV winner of the four declared L1 cells: L1a 1,590.1 s, B4 1,592.8, L1b log target 2,158.5, L1c over-1-h classifier 2,750.5, L1d pooled fit 2,178.4) | 2,072.1 | +2.1 [-0.2, +5.3] | tie (interval crosses zero; bound 1.5 s at k = 4 plus floor 2 s not cleared); B4 stays (`docs/L1.md`) |
| H18 | 2026-09-05 | A | W1-A | D1 (M2 recipe on W1-A's 36 columns plus 13 plan-provenance columns) | 242.2 | +11.6 [+9.2, +14.0] | WIN, the whole interval clears bound 0 s (k = 1) plus floor 2 s (the first lane-A win not at the floor edge, F10); Jan +6.2, Jul +15.3; tail rows 627.0 vs 680.5, body 176.0 vs 177.3; the gain sits in AOBT_3 minus EOBT_1 (d_ae), AOBT_3 minus LOBT (d_al), operator and destination, the rounding and second-of-minute flags carry none; D1 promoted (lane-A champion, `docs/D1.md`) |
| H19 | 2026-09-05 | A | W1-A | MC1 (CatBoost 1.2.8 on the GPU, M2 recipe on W1-A's 36 columns, depth 8, learning rate 0.1) | 248.2 | +5.6 [+4.4, +6.8] | WIN against the declared champion W1-A, both ends clear bound 0 s (k = 1) plus floor 2 s; Jan +6.2, Jul +5.3; tail rows 658.0 vs 680.5, body 175.7 vs 177.3; NOT promoted: D1 (H18, 242.2 s) had already superseded W1-A and is 6.0 s better than MC1 on the same rows by point estimate (Jan identical at 217.9, Jul 260.3 vs 270.3); no paired D1 vs MC1 interval exists (MC2 is the declared lane that closes it); the model-class read on identical columns is LightGBM 253.8 vs CatBoost 248.2 (`docs/MC1.md`) |

Declared searches so far (not holdout verdicts; judged on month-pair CV): CAP lane A (k = 4, bound 1.03 x SE), CAP lane B (k = 4); M1 grid lane A (k = 12, bound 1.63 x paired SE 1.7 s = 2.8 s), M1 grid lane B (k = 8, bound 1.42 x paired SE 161.1 s = 229.4 s); M2 lane A (k = 1, bound 0 x SE); W1 lane A (k = 1) and W1 lane B (k = 1), declared 2026-09-05 before the run; M3 pilot LIRF lane A (k = 1), run 2026-09-05: OOF gain +11.6 s and VRAM pass, the 30 min time gate fails at 38.1 min, M3 stopped pending Keith; D1 lane A (k = 1), MC1 lane A (k = 1) and L1 lane B (k = 4, bound 1.03 x paired SE), declared 2026-09-05 before the run (protocol section 4); D1, L1 and MC1 RUN 2026-09-05 (H17, H18, H19); MC2 lane A (k = 1, MC1's CatBoost cell on D1's 49 columns, judged against D1) declared 2026-09-05 after H19, before its run.

## C. Board probes (P1-P6, after S1; protocol section 3)

For group G with perturbation c = 300 s on the S1 file: m_G = (SSE+ + SSE- - 2 SSE0) / (2 c^2); sum of S1 errors over the scored rows of G = (SSE+ - SSE-) / (4 c); SSE = 215,876 x RMSE^2.

| id | date | group G | sign | file | board RMSE | derived m_G | derived error sum | note |
|---|---|---|---|---|---|---|---|---|
| P1 | 2026-09-05 | lane B (AOBT_3 null, 5,290 ranking rows) | + | merry-octopus_v2.parquet (sha256 f1b1688b0e97dc701e84fa8d5159f7ef2b3ac411fde2e3f4d1837f4f233f42da) | 324.7339 | 5,289.9 from the P1/P2 pair (ranking count 5,290) | -1,041,701 s (prediction minus truth), mean -196.9 s per lane-B row | scored 10:04:57 UTC, usedPairs 344,841. m matches the row count, so the scorer uses exactly the 344,841 DEP rows and the SSE arithmetic holds. S1 runs 197 s low on the 2026 lane-B rows; the best constant lane-B shift alone would score 324.48 s (0.9 s), so lane B's level is not the gap to the leader |
| P2 | 2026-09-05 | lane B | - | merry-octopus_v3.parquet (sha256 d73de348cd46dd09ef2ae249fc7e4748c6a073a36cb8f61a8d1c97f09078098a) | 330.2682 | (pair with P1) | (pair with P1) | scored 10:05:03 UTC, usedPairs 344,841 |
| P3 | 2026-09-05 | January (month 1, 152,719 rows) | + | merry-octopus_v4.parquet (sha256 c6e5a22c466e09c539d857a932c8a3a2d91c53acbff27e6d9ee523e968af1150) | REJECTED | | | scorer result file: DAILY_LIMIT_REACHED, 3 of 3 used today, resets 00:00 UTC. The site states no cap; the scorer enforces 3 per team per day. Re-run design gated on Keith |
| P4 | 2026-09-05 | January | - | merry-octopus_v5.parquet (sha256 f613342bb446fd6a09c0462cc593c08194c327e1aaa1e532be034a7d63f00e26) | REJECTED | | | as P3 |
| P5 | 2026-09-05 | July and later (month >= 7, 192,122 rows, 11 of them on 1 August UTC) | + | merry-octopus_v6.parquet (sha256 792381558270b9b6af667672f849fd622f8fa85db4fd2e125f6d41acae534301) | REJECTED | | | as P3 |
| P6 | 2026-09-05 | July and later | - | merry-octopus_v7.parquet (sha256 2926066cc83d202777191d7bc0cad6c4ebc400e29ad9103574c480fafb893de7) | REJECTED | | | as P3 |
| P3s | 2026-09-05 | January | + | merry-octopus_v8.parquet (sha256 c6e5a22c466e09c539d857a932c8a3a2d91c53acbff27e6d9ee523e968af1150, byte-identical to v4) | BUILT, upload scheduled 2026-09-06 00:05 UTC (Windows task `prc26-probes-20260906`) | 152,719 (known, not derived) | (after scoring) | single-sided re-run of P3: sum = (SSE+ - SSE0 - m c^2) / (2 c). Option A put to Keith 2026-09-05, run after his "Keep going please". UNKNOWN whether the scorer re-tries the rejected v4-v7 after the reset; `python -m taxi.probes derive` reads whichever files scored |
| P5s | 2026-09-05 | July and later | + | merry-octopus_v9.parquet (sha256 792381558270b9b6af667672f849fd622f8fa85db4fd2e125f6d41acae534301, byte-identical to v6) | BUILT, upload scheduled as P3s | 192,122 (known) | (after scoring) | single-sided re-run of P5 |
