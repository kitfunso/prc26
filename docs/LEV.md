# LEV: a lane second is not a composite second (diagnostic, never a verdict)

Regenerate with `python -m taxi.lev`. Champions SB3 and L8c, floored, holdout Jan + Jul 2025. No cell is declared. The point is that the floor and every verdict interval are quoted in lane RMSE while the board is scored in composite RMSE, and the two differ by a factor of eight between the lanes.

## 1. The conversion

| lane | rows | RMSE | share of composite squared error | one second of it, in composite seconds |
|---|---|---|---|---|
| A | 339,046 | 230.10 s | 50.5% | 0.705 |
| B | 5,373 | 1,808.17 s | 49.5% | 0.088 |
| both | 344,419 | 321.13 s | 100% | 1.000 |

One second of lane-A RMSE is worth 8.0 seconds of lane-B RMSE, because lane A carries 63 times the rows. The shares of squared error are near-equal, so a given PERCENTAGE gain pays about the same either side: 1% of lane A is 1.62 s of composite, 1% of lane B is 1.58 s. Lane B looks like half the problem and is, but its seconds are cheap and its percentages are not reachable, for the reason section 3 gives.

## 2. Every parked tie, priced in the unit the board scores

Effects and intervals are quoted from `docs/TRIAL-LEDGER.md`; only the conversion is computed here.

| ledger | lane | cell | lane effect | interval | composite | counted |
|---|---|---|---|---|---|---|
| H34 | A | E2, the fixed-weight MC2/D1 blend | +0.2 s | [+0.1, +0.3] | +0.14 s | overlaps ST1a's blend |
| H35 | A | SB1, the five-seed bag | +1.0 s | [+0.7, +1.3] | +0.70 s | inside H48 |
| H47 | A | ST1a, the row-dependent MC2/D1 gate | +0.6 s | [+0.5, +0.8] | +0.42 s | inside H48 |
| H48 | A | SB2, SB1 then ST1a then the C1 floor | +1.3 s | [+1.0, +1.7] | +0.92 s | yes |
| H38 | A | G1, target encoding of airport/stand/runway | +0.2 s | [-0.2, +0.6] | +0.14 s | interval crosses zero |
| H7 | A | B3a | +19.5 s | [-17.9, +73.7] | +13.45 s | interval crosses zero |
| H10 | A | B3d | +33.0 s | [-2.5, +86.3] | +22.39 s | interval crosses zero |
| H15 | B | W1-B, the five weather columns | +9.9 s | [-2.7, +22.6] | +0.87 s | interval crosses zero |
| H17 | B | L1a | +2.1 s | [-0.2, +5.3] | +0.18 s | interval crosses zero |

**The whole parked inventory is worth +0.92 s of composite.** SB1 and ST1a are not separate gains; SB2 IS their composition and H48 measures the stack. Of the rest, five intervals cross zero and are noise, and E2's fixed blend reaches for the same D1 information ST1a already uses inside SB2.

## 3. Why lane B's half of the error is not half of the opportunity

- Deleting lane B's worst 1 row takes the composite to 288.20 s, a gain of 32.92 s.
- Deleting lane B's worst 2 rows takes the composite to 271.28 s, a gain of 49.85 s.
- Deleting lane B's worst 5 rows takes the composite to 262.26 s, a gain of 58.87 s.

Those rows are the day-shift artifact of `docs/DE.md` and flag F21 prices every route to them at zero. So lane B's 49% share of squared error is mostly unreachable, and the lane-B percentage that section 1 costs at par cannot actually be bought. What is left in lane B after the artifacts is a few seconds of composite at best.

## 4. What this changes

- **The 2 s floor is not one floor.** In composite seconds it is 1.41 s on lane A and 0.176 s on lane B, an 8-fold difference in what the same rule demands. Flag F18 already records that the floor is about 10 paired SE in lane A and about 0.12 in lane B; this is the same fact in the board's own unit. Changing it is a decision-rule change and stays with Keith.

- **The parked inventory is small.** 0.92 s of composite against a board gap of 58.97 s: our scored file is at 324.7339 and the leader at 265.7612 (leaderboard read 2026-09-07 04:17 UTC, `python lb.py`). The unuploaded champion closes most of that on paper, and `docs/BL.md` sections 3 and 5 carry the current band and what the remainder costs. Releasing every parked tie moves a small fraction of either.

- **Lane A is where the reachable seconds are**, not because it holds more error, but because its error is spread over 339,046 rows that a model can move, while half of lane B's sits on two rows that no ranking-visible column separates (LK1, F17, F21).
