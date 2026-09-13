# Board ledger

Best file per team counts. Scorer cap: 3 files a day, reset 00:00 UTC. Board baseline: `data/archive-v1/merry-octopus_v18.parquet` = 294.9917.
Upload rule: expected board gain > ~3 s over the current best, holdout CI excludes zero, `python board.py check` passes.

| file | date (UTC) | config | holdout Jan+Jul 2025 (A / B / all) | expected board | actual board | note |
|---|---|---|---|---|---|---|
| v18 (campaign 1) | 2026-09-10 | v1 blend | ~230 / ~1808 / n.a. | n.a. | 294.9917 | baseline carried over |
| v21 | 2026-09-12 | 0.648 mix_cb_wx + 0.352 mix_lgb_wxb (NNLS on OOF) | 246.74 / 1891.47 / 340.21 | unknown, calibration point | 310.891 | first upload of campaign 2; LOSES to v18 by 15.9 s; v1's holdout was better (~230 / ~1808) but holdout order does not carry to the board |
| v22 | 2026-09-12 | 0.732 ra_cb + 0.268 ra_lgb (residual-to-AOBT lane A, tail gate 600 s; lane B from mix_cb_wx; NNLS on OOF) | 229.10 / 1900.60 / 328.67 | beat v21 (310.9) and v18 (295.0) | 292.887 | NEW BEST, rank 34/123; holdout gain over v21 (+11.5) came through on the board (+18.0); stamp mixture hedging was the loss |
| v23 | 2026-09-12 | 0.304 ra_cb + 0.328 ra_cb_s1 + 0.084 ra_lgb + 0.285 nn3 (seed bag + MLP residual partner; lane B from mix_cb_wx; NNLS on OOF) | 226.79 / 1900.60 / 327.08 | beat v22 (292.9); holdout +1.66 [+1.18, +2.28] vs v22 blend | 290.983 | NEW BEST, rank 30/123; holdout +1.66 became +1.90 on the board; seed bag and MLP partner both carry |
| v24 | 2026-09-13 | 0.218 ra_cb + 0.247 ra_cb_s1 + 0.181 ra_cb_s2 + 0.074 ra_lgb + 0.28 nn3 (three-seed CatBoost bag + LightGBM + MLP residual partner; lane B from mix_cb_wx; NNLS on OOF) | 226.65 / 1900.60 / 326.99 | holdout +0.09 [+0.04, +0.15] all, +0.13 [+0.06, +0.20] lane A vs v23 blend (paired day-block bootstrap) | 290.7995 | NEW BEST, rank 33/127; holdout +0.09 became +0.18 on the board; third CatBoost seed carries a small further gain |
| v25 | 2026-09-13 | 0.5824 ra_cb_arr + 0.1399 ra_lgb_arr + 0.2777 nn3_arr (v24's lane A designs plus 16 arrival recorder-mode features; lane B rows overridden by the laneb.py mechanism model via combine.py --over; NNLS on OOF) | 223.65 / 1874.62 / 322.59 | holdout +4.40 [+2.18, +6.84] all, +3.00 [+2.38, +3.64] lane A, +25.98 [+5.24, +57.48] lane B vs v24 blend (paired day-block bootstrap) | 288.6796 | NEW BEST, rank 33/131; holdout +4.40 became +2.12 on the board, about half; 60% of the lane B holdout gain sat on 6 day-shift rows, so that part depends on how many the scored set holds |
