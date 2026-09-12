# Board ledger

Best file per team counts. Scorer cap: 3 files a day, reset 00:00 UTC. Board baseline: `data/archive-v1/merry-octopus_v18.parquet` = 294.9917.
Upload rule: expected board gain > ~3 s over the current best, holdout CI excludes zero, `python board.py check` passes.

| file | date (UTC) | config | holdout Jan+Jul 2025 (A / B / all) | expected board | actual board | note |
|---|---|---|---|---|---|---|
| v18 (campaign 1) | 2026-09-10 | v1 blend | ~230 / ~1808 / n.a. | n.a. | 294.9917 | baseline carried over |
| v21 | 2026-09-12 | 0.648 mix_cb_wx + 0.352 mix_lgb_wxb (NNLS on OOF) | 246.74 / 1891.47 / 340.21 | unknown, calibration point | 310.891 | first upload of campaign 2; LOSES to v18 by 15.9 s; v1's holdout was better (~230 / ~1808) but holdout order does not carry to the board |
| v22 | 2026-09-12 | 0.732 ra_cb + 0.268 ra_lgb (residual-to-AOBT lane A, tail gate 600 s; lane B from mix_cb_wx; NNLS on OOF) | 229.10 / 1900.60 / 328.67 | beat v21 (310.9) and v18 (295.0) | 292.887 | NEW BEST, rank 34/123; holdout gain over v21 (+11.5) came through on the board (+18.0); stamp mixture hedging was the loss |
| v23 | 2026-09-12 | 0.304 ra_cb + 0.328 ra_cb_s1 + 0.084 ra_lgb + 0.285 nn3 (seed bag + MLP residual partner; lane B from mix_cb_wx; NNLS on OOF) | 226.79 / 1900.60 / 327.08 | beat v22 (292.9); holdout +1.66 [+1.18, +2.28] vs v22 blend | 290.983 | NEW BEST, rank 30/123; holdout +1.66 became +1.90 on the board; seed bag and MLP partner both carry |
