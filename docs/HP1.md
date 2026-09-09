# HP1: tuning the champion's cell, Jan + Jul 2025 development holdout

Regenerate with `python -m taxi.hp1`. Eleven cells, each ONE knob off D2's cell on its 52 columns, plus a twelfth composing whatever beat the base; k = 12. The base is D2's own CV of 211.8 s (`docs/D2.md`), already measured and so the reference rather than a trial. Every CatBoost model in this campaign shares `taxi.mc1.CELL` and no knob in it had been moved.

## Month-pair CV by arm

| arm | knob | CV RMSE (s) | vs D2 | trees (clf, tail, body) | min |
|---|---|---|---|---|---|
| depth6 | `{"depth": 6}` | 212.2 | +0.4 | [3308, 2035, 4340] | 34 |
| depth10 | `{"depth": 10}` | 212.1 | +0.3 | [1085, 1114, 2002] | 31 |
| lr05 | `{"learning_rate": 0.05}` | 211.6 | -0.2 | [4175, 2968, 4996] | 57 |
| l2reg1 | `{"l2_leaf_reg": 1.0}` | 212.1 | +0.3 | [1978, 1489, 3638] | 45 |
| l2reg10 | `{"l2_leaf_reg": 10.0}` | 211.8 | -0.0 | [2455, 1749, 3873] | 41 |
| l2reg30 | `{"l2_leaf_reg": 30.0}` | 211.7 | -0.1 | [2486, 2880, 4331] | 47 |
| rstr4 | `{"random_strength": 4.0}` | 211.8 | -0.0 | [1936, 1655, 3843] | 38 |
| bagt0 | `{"bagging_temperature": 0.0}` | 211.3 | -0.5 | [1880, 1446, 3512] | 36 |
| bagt3 | `{"bagging_temperature": 3.0}` | 214.5 | +2.7 | [2859, 2245, 4359] | 50 |
| onehot64 | `{"one_hot_max_size": 64}` | 211.9 | +0.1 | [1803, 1779, 3774] | 26 |
| border128 | `{"border_count": 128}` | 212.1 | +0.3 | [1845, 1446, 3704] | 37 |
| composed | `{"learning_rate": 0.05, "l2_leaf_reg": 30.0, "random_strength": 4.0, "bagging_temperature": 0.0}` | 211.5 | -0.3 | [4599, 4015, 4994] | 66 |

CV winner: bagt0 at 211.3 s.

## Holdout verdict vs SB3 (339,046 rows), the winner as its own five-seed bag

SB3 is a five-seed bag and `docs/SEEDS.md` prices the bag at 1.07 s over one seed, so the challenger is bagged the same way at cell `{"bagging_temperature": 0.0}` and trees (1880, 1446, 3512).

| model | RMSE | Jan | Jul | tail rows | body rows |
|---|---|---|---|---|---|
| SB3 | 230.10 | 204.4 | 249.1 | 583.3 | 170.8 |
| HP1-bagt0 bag | 229.70 | 204.2 | 248.6 | 582.2 | 170.6 |

Improvement HP1-bagt0 vs SB3: +0.4 s [+0.2, +0.6]; paired SE 0.1 s; optimism bound for k = 12: 0.2 s; floor 2 s. Verdict: tie or lose, SB3 stays.
