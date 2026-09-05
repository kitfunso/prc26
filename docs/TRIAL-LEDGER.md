# Trial ledger (append-only)

Contract: `docs/EXPERIMENT-PROTOCOL.md` section 8. Sign convention: improvement = champion RMSE minus challenger RMSE, positive is good.
Rows are never edited; a superseded row gets a note in a later row. Old-spec rows stay, labelled.

Counters: N (board submissions) = 1 model (S1) + 2 scored probes (P1, P2); 4 probe files rejected by the scorer's daily cap on 2026-09-05. N_h (holdout verdicts) = 13.

## A. Board submissions

| id | date | file | spec (lane A + lane B) | git | holdout lane A | holdout lane B | holdout all rows | holdout improvement | board RMSE | board improvement | haircut | usedPairs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 2026-09-05 | `merry-octopus_v1.parquet` (sha256 c79d20d7b691ef1696acdf93537e092fa8faf405c62a08aea15f5a55488f39de, 344,841 rows, uploaded 09:5x UTC, scored 09:58:03 UTC) | lane A M1 (31 leaves, min_data 200, l2, 1,016 rounds) + lane B B4 [z_sched, queue], both fit on all 12 months of 2025 | 256b4c1 | 259.5 | 2,074.1 | 365.3 | lane A +51.6 [+48.4, +55.1] vs B4 (H13); all rows +37.7 vs B4 in both lanes (reported) | 325.3982 (rank 11 of 29 teams; leader 277.29) | N/A, first submission | N/A | 344,841 |

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

Declared searches so far (not holdout verdicts; judged on month-pair CV): CAP lane A (k = 4, bound 1.03 x SE), CAP lane B (k = 4); M1 grid lane A (k = 12, bound 1.63 x paired SE 1.7 s = 2.8 s), M1 grid lane B (k = 8, bound 1.42 x paired SE 161.1 s = 229.4 s).

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
