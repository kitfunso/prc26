# LBC: the copy mixture at the nine airports the gate never touches (diagnostic, never a verdict)

Regenerate with `python -m taxi.lbc`. No cell is declared here and no champion moves.
Champion L8c floored, 5,373 lane-B holdout rows, RMSE 1808.1729 s, composite 322.5796 s.

## 1. Where the error is, and where the branch is allowed to run

`taxi.l4.touched` admits a row only if it has no aircraft type AND its airport has at least 100 such rows in training.
That is a gate on data availability, not on where the copies are, so it is worth checking what it excludes.

      rows            sse  exact  near  touched  notyp  fit_notyp     share      rmse
apt                                                                                  
LFPG   880 10468401511.15      1    23        0      1          4     59.59   3449.04
LIRF   397  5158392396.97     17   137      397    397       1090     29.36   3604.64
EGLL   466   629742049.26      0     5        0      0          0      3.58   1162.49
EHAM   808   417740517.26      1   146        0      0          1      2.38    719.03
LTFM   657   340361113.84      2   389        0      1          6      1.94    719.76
LSZH   524   235100872.89      1   148        0      3          4      1.34    669.83
EDDM   237   109261251.09      0     5        0      0          0      0.62    678.98
EDDF   599   103954940.30      0    15        0      0          0      0.59    416.59
LEBL   496    66914134.31      0    14        0      0          0      0.38    367.30
LEMD   309    37097624.39      0     1        0      0          0      0.21    346.49

## 2. The oracle: perfect membership and the exact value on every copy row

| copy definition | rows | share of lane | share of lane SSE | branch: predict z_sched, lane B | composite | ceiling: know y, lane B | composite |
|---|---|---|---|---|---|---|---|
| exact copies, y = z_sched | 22 | 0.409% | 0.45% | +4.0 | +0.353 | +4.0 | +0.353 |
| near copies, |y - z_sched| <= 600 s | 883 | 16.434% | 8.02% | +70.3 | +6.081 | +74.0 | +6.401 |

## 3. The decision: what EXTENDING the branch to the other nine airports would buy

| oracle | rows | share of lane SSE | lane-B gain (s) | composite gain (s) |
|---|---|---|---|---|
| exact copies, already touched (LIRF) | 17 | 0.44% | +4.0 | +0.347 |
| exact copies, the nine untouched airports | 5 | 0.01% | +0.1 | +0.006 |
| near copies, already touched (LIRF) | 137 | 4.83% | +44.2 | +3.844 |
| near copies, the nine untouched airports | 746 | 3.18% | +25.4 | +2.211 |

## 4. Reading

The gate is not a statement about where copies are. `touched` needs a row with no aircraft type at an airport carrying 100 such rows in training, and only LIRF clears it (1090 no-type fit rows against 0 to 6 everywhere else). No-type is a data-availability marker, and the copy behaviour plainly is not confined to it: LTFM has 389 near copies in 657 holdout rows with 6 no-type fit rows, EHAM 146 in 808, LSZH 148 in 524.

Two answers, and they point opposite ways.

The EXACT-copy extension is dead. LFPG is the obvious target because it carries 59.6% of the lane's squared error, and it holds 1 exact copy in 880 rows. Across all nine untouched airports there are 5 of them, worth +0.006 s of composite with the answer in hand. Where the error is, the copies are not; where the copies are, the error is not. This closes the reflex that F28 and `docs/L8.md` invite, that the LIRF-only mixture should simply be given to the other airports.

The NEAR-copy extension is not dead and should not be written off with it. At the nine untouched airports 746 rows sit within 600 s of `z_sched`, holding 3.18% of the lane's squared error, and predicting `z_sched` on exactly those rows is worth +25.4 s of lane B and +2.211 s of composite. That is a perfect-membership ceiling and a real classifier collects a fraction of it, but the ceiling is above the 2 s floor's composite equivalent for this lane (0.175 s, `docs/LEV.md`) by more than an order of magnitude, and those nine airports have no copy handling of any kind today. The honest conclusion is a declared lane, not a closure.

