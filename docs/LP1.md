# LP1: TabPFN on lane B, Jan + Jul 2025 development holdout

Regenerate with `python -m taxi.lp1`. All 17,097 training rows in one fit on raw y, 24 columns, predicted in 4,096-row chunks and floored at 0 s. Built with PriorLabs-TabPFN (Apache 2.0 plus attribution). Holdout fit 25 s, peak VRAM 2.3 GB.

## Month-pair folds

| fold | rows | L8c RMSE (s) | LP1 RMSE (s) | fit s | VRAM GB |
|---|---|---|---|---|---|
| (2, 3) | 2,192 | 1626.0 | 2025.3 | 12 | 1.6 |
| (4, 5) | 2,690 | 1919.5 | 1930.6 | 10 | 1.6 |
| (6, 8) | 4,832 | 1600.1 | 1675.8 | 15 | 2.3 |
| (9, 10) | 4,378 | 1055.8 | 1226.6 | 15 | 2.3 |
| (11, 12) | 3,005 | 1111.3 | 1518.3 | 10 | 1.7 |

## Holdout verdict vs champion L8c (5,373 rows)

| model | RMSE | Jan | Jul |
|---|---|---|---|
| L8c | 1808.2 | 2897.6 | 1183.8 |
| LP1 | 2094.2 | 2977.3 | 1659.3 |

Improvement LP1 vs L8c: -286.0 s [-562.0, -68.7]; paired SE 125.8 s; optimism bound for k = 1: 0.0 s; floor 2 s. Verdict: tie or lose, L8c stays.
