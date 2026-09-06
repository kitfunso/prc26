# Forward fold (reported, never decides): train Jan-Nov 2025, predict Dec 2025

Regenerate with `python -m taxi.fwd`. Every model at the rounds its report recorded; the M1 numbers are in `docs/M1.md`.

| lane | model | Dec rows | RMSE (s) | fit s |
|---|---|---|---|---|
| A | M2 | 164028 | 220.5 | 35 |
| A | W1-A | 164028 | 217.6 | 38 |
| A | D1 | 164028 | 212.1 | 20 |
| A | MC1 | 164028 | 213.5 | 286 |
| A | MC2 | 164028 | 204.3 | 389 |
| B | B4 | 1649 | 916.9 | 0 |
| B | L4b | 1649 | 1002.1 | 0 |
| B | L6a | 1649 | 949.7 | 0 |
| B | L8c | 1649 | 906.0 | 13 |
