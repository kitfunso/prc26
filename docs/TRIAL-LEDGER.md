# Trial ledger (append-only)

Contract: `docs/EXPERIMENT-PROTOCOL.md` section 8. Sign convention: improvement = champion RMSE minus challenger RMSE, positive is good.
Rows are never edited; a superseded row gets a note in a later row. Old-spec rows stay, labelled.

Counters: N (board submissions) = 0. N_h (holdout verdicts) = 13.

## A. Board submissions

| id | date | file | spec (lane A + lane B) | git | holdout lane A | holdout lane B | holdout all rows | holdout improvement | board RMSE | board improvement | haircut | usedPairs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

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
