# State oracle: ceiling on lane A gains from knowing the recorder state

Lane A = departures with `has_plan==1`. About 31% of these rows are a recorder copy of a
known stamp instead of the physical off-block (`mix.label`: 0 AOBT copy, 1 EOBT copy, 2
schedule copy, 3 physical), confirmed on the full lane A train set (2,062,577 rows):
19.8% class 0, 9.4% class 1, 1.6% class 2, 69.2% class 3 (30.8% combined copy share).
Copy rates cluster by airport-day. This asks: if a model knew each day's, and each +-1h
window's, realised state exactly, how much lane A RMSE would it gain over `ra_lgb_arr`?
That gain is the ceiling for any label-free estimator of the state.

Script: `analysis/state_oracle.py`. Oracle columns are computed once on all lane A train
rows (leave-one-out per group, using `mix.label()` and `c = y - z_aobt`), then attached
only in memory, never written to `data/cache/feat.parquet`, never added to `features.py`
or `lanea.py`. Days never straddle folds, so a single global leave-one-out pass is
equivalent to recomputing per fold. Six variants copy `lanea.py`'s `ra_lgb_arr` config
(`CONFIGS["ra_lgb_arr"]` deep-copied, extra columns appended to `cfg["num"]` and
`cfg["reg"]["num"]`) through `lanea.predict()`, 5 month-pair folds (`train.FOLDS`) then
holdout (Jan+Jul) at `iters = 1.1x mean fold iters`, predictions floored at 0, the same
loop as `lanea.cv()`, without the donor merge or the `docs/RESULTS.md`/cache-name side
effects.

## Oracle columns

Two group types, both leave-one-out (the row itself excluded from its own group stat):

- **Day group**: same (`apt`, `day`).
- **Window group**: same (`apt`, `day`) and `|t_off - t_off_i| <= 3600s` (does not cross
  midnight even if physically within an hour, since it is day-scoped first).

Two column sets per group type:

- **cls set** (5 cols): share class 0, share class 1, share class 1 with
  `|z_eobt - z_aobt| > 300`, share class 2, group count (leave-one-out).
- **tail set** (2 cols): share `|c| > 600` (`lanea.TAIL`); mean of `clip(c, -1800, 1800)`
  over class-3 (physical) rows only, NaN when the group has none.

Variants: A0 none (`ra_lgb_arr` as shipped) &middot; A1 day cls &middot; A2 window cls
&middot; A3 day tail &middot; A4 window tail &middot; A5 all four sets (14 columns).

## Harness check

A0 (no extra columns) reproduces the cached `data/cache/ra_lgb_arr_oof.parquet` and
`ra_lgb_arr_holdout.parquet` on lane A rows exactly: max abs prediction difference
`0.000e+00` on both oof and holdout, holdout lane A RMSE `231.64` (reference `231.64`).
**PASS.**

## Results

Lane A rows: 1,723,531 non-holdout, 339,046 holdout. RMSE by fold, pooled CV (all-fold
OOF), and holdout (floored at 0):

| variant | fold0 (2,3) | fold1 (4,5) | fold2 (6,8) | fold3 (9,10) | fold4 (11,12) | CV lane A (OOF) | holdout lane A | refit iters (clf, tail, body) |
|---|---|---|---|---|---|---|---|---|
| A0 (none) | 213.60 | 200.16 | 224.34 | 217.25 | 204.81 | 212.41 | 231.64 | (254, 249, 414) |
| A1 (day cls) | 213.59 | 199.96 | 224.05 | 216.89 | 204.91 | 212.24 | 230.85 | (234, 275, 419) |
| A2 (window cls) | 212.48 | 198.18 | 222.69 | 215.54 | 203.56 | 210.85 | 229.36 | (232, 286, 449) |
| A3 (day tail) | 212.81 | 199.46 | 223.65 | 216.92 | 204.20 | 211.80 | 230.52 | (229, 305, 392) |
| A4 (window tail) | 204.75 | 193.98 | 219.18 | 212.05 | 199.77 | 206.42 | 223.67 | (233, 310, 438) |
| A5 (all four) | 205.46 | 194.02 | 219.09 | 211.55 | 199.88 | 206.44 | 224.03 | (238, 252, 466) |

`train.boot_delta(y, p_A0, p_Ak, day)` on holdout lane A rows only (positive = Ak beats
A0), n=400 paired day-block bootstrap resamples:

| variant | delta vs A0 | 95% CI |
|---|---|---|
| A1 (day cls) | +0.79 | [+0.42, +1.19] |
| A2 (window cls) | +2.28 | [+1.54, +3.13] |
| A3 (day tail) | +1.12 | [+0.51, +1.82] |
| A4 (window tail) | +7.98 | [+5.99, +10.25] |
| A5 (all four) | +7.61 | [+5.54, +9.90] |

All five lower bounds clear zero: every oracle variant is a real, non-noise gain over
`ra_lgb_arr`. A4 and A5 clear +3s with the CI lower bound alone; A1-A3 do not.

## What each variant's gain means as a ceiling

- **A0**: baseline, `ra_lgb_arr` unchanged; zero by construction, the harness anchor.
- **A1 (day cls)**: knowing each day's exact AOBT/EOBT/schedule-copy mix buys +0.79s
  [+0.42, +1.19]. A perfect day-level "which stamp got copied" classifier is a small win;
  not worth building for this alone.
- **A2 (window cls)**: the same classification, narrowed to the +-1h window, buys +2.28s
  [+1.54, +3.13], bigger than day-level (finer grain matters) but still under +3s on the
  point estimate.
- **A3 (day tail)**: knowing each day's exact tail share and physical-residual level buys
  +1.12s [+0.51, +1.82]: a day is too coarse a grain for this signal.
- **A4 (window tail)**: knowing the +-1h tail share and physical-residual level is the
  standout, +7.98s [+5.99, +10.25], roughly 3.4% of the 231.64s baseline holdout RMSE, and
  the only variant whose ceiling is unambiguously large. The useful signal is not which
  stamp got copied, but whether the surrounding hour runs tail-heavy and what its physical
  residual looks like: that is the target worth chasing if a label-free estimator gets built.
- **A5 (all four)**: +7.61s [+5.54, +9.90], fractionally *below* A4 alone (both CV and
  holdout) despite having 12 more columns. The four sets are not additive: window-tail
  (A4) already carries essentially the whole ceiling, and the classification-style
  columns (day/window cls, day tail) add noise rather than lift once it is present.

## Do-not compliance

No edits to `features.py`, `lanea.py`, `nn.py`, `laneb.py`, `combine.py`,
`docs/RESULTS.md`, `LEDGER.md`, `README.md`, or `data/cache/feat.parquet` (mtimes
unchanged, confirmed after the run). No GPU, no git, no scorer, no upload.

## Files

- `analysis/state_oracle.py`: build, harness check, six-variant CV/holdout loop.
- `data/analysis_state_oracle_summary.csv`: per-fold, CV, holdout, and bootstrap numbers.
- `data/analysis_state_oracle.log`: run log (fold lines, harness check, DONE).
- `data/cache/analysis_state_oracle_{A0..A5}_{oof,holdout}.parquet`: columns `id`, `p`.
- `docs/STATE_ORACLE.md`: this file.
