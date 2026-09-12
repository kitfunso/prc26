# Board ledger

Best file per team counts. Scorer cap: 3 files a day, reset 00:00 UTC. Board baseline: `data/archive-v1/merry-octopus_v18.parquet` = 294.9917.
Upload rule: expected board gain > ~3 s over the current best, holdout CI excludes zero, `python board.py check` passes.

| file | date (UTC) | config | holdout Jan+Jul 2025 (A / B / all) | expected board | actual board | note |
|---|---|---|---|---|---|---|
| v18 (campaign 1) | 2026-09-10 | v1 blend | ~230 / ~1808 / n.a. | n.a. | 294.9917 | baseline carried over |
| v21 | 2026-09-12 | 0.648 mix_cb_wx + 0.352 mix_lgb_wxb (NNLS on OOF) | 246.74 / 1891.47 / 340.21 | unknown, calibration point | 310.891 | first upload of campaign 2; LOSES to v18 by 15.9 s; v1's holdout was better (~230 / ~1808) but holdout order does not carry to the board |
