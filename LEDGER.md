# Board ledger

Best file per team counts. Scorer cap: 3 files a day, reset 00:00 UTC. Board baseline: `data/archive-v1/merry-octopus_v18.parquet` = 294.9917.
Upload rule: expected board gain > ~3 s over the current best, holdout CI excludes zero, `python board.py check` passes.

| file | date (UTC) | config | holdout Jan+Jul 2025 (A / B / all) | expected board | actual board | note |
|---|---|---|---|---|---|---|
| v18 (campaign 1) | 2026-09-10 | v1 blend | ~230 / ~1808 / n.a. | n.a. | 294.9917 | baseline carried over |
