# Fingerprint / norm-gap / mechanism-encoding features

Do exact-stamp fingerprints (group 1), norm-relative gaps (group 2), and out-of-fold
mechanism target encodings (group 3) help the lane-A stamp classifier on top of the new
baseline CAT+NUM+DAY+ARR (16 arrmode columns, since arrmode passed docs/ARRMODE.md)?

Build: `analysis/fe.py` (`build_fp`, `flight_ids`, `fit_apply`, `oof_apply`), cache at
`data/cache/fe_fp.parquet` (group 1 only; groups 2/3 are fit-row dependent, computed fresh
per fold, never cached). Test: `analysis/fe_test.py`, copying the CV/holdout harness from
`analysis/arr_feat_test.py`.

Commands:

```
PYTHONPATH=/c/Users/skf_s/prc26 python analysis/fe.py        # not runnable standalone, imported by fe_test.py
PYTHONPATH=/c/Users/skf_s/prc26 python analysis/fe_test.py
```

## Feature groups

- **Group 1** (`fp`, 8 cols): row-level exact-equality fingerprints from the row's own raw
  stamps only, no label: `lobt_eq_eobt`, `iobt_eq_eobt`, `aobt_eq_eobt`, `d_ae_m5`, `d_ae_m1`,
  `eobt_sec`, `sched_eq_eobt`, `lobt_min5`.
- **Group 2** (`g2`, 4 cols): gaps relative to `norm_taxi` = median y of physical-class
  (`mix.label`==3) lane-A rows per (apt, runway, hour), fit on FIT rows only: `za_rel`,
  `ze_rel`, `zs_rel`, `ze_over_za` (z_eobt/z_aobt, NaN-safe, clipped to [0, 10]).
- **Group 3** (`g3`, 12 cols): per-FLIGHT_mvt, per-(apt,stand), per-(apt,airline) rates of
  class 0/1/2 among FIT lane-A rows, smoothed toward the airport rate (prior n=20); unseen
  keys fall back to the airport rate with n=0: `fl_aobt_rate`, `fl_eobt_rate`,
  `fl_sched_rate`, `fl_n`, `st_*`, `al_*`.
- **`all`**: fp + g2 + g3 together.

Groups 2/3 are computed by `fit_apply(fit_df, apply_df)`: for each outer CV fold, fit on the
fold's training rows and apply to its validation rows; for holdout, fit on all non-holdout
training rows. Within the training rows used to fit the classifier itself, `oof_apply` runs
`fit_apply` 5-fold out-of-fold, blocked by day (not the outer month-fold), so a training row's
own label never reaches its own encoding.

## Leak check

`fl_eobt_rate` vs. the row's own true-eobt label (`mix.label`==1), on the full non-holdout
training set (n=1,723,531 lane-A rows):

| | correlation |
|---|---|
| in-fold (`fit_apply(tr_all, tr_all)`, a row's own label counted in its own group mean) | 0.1933 |
| out-of-fold (`oof_apply(tr_all)`, day-blocked) | 0.1434 |

In-fold runs a little hot, as expected (each row nudges its own flight's rate), but the two
are the same order of magnitude, not a large leak spike; `oof_apply` is what the classifier
actually trains on.

## Results

CV over `train.FOLDS`, then holdout (Jan+Jul), lane A only (has_plan==1). Full numbers also in
`data/analysis_fe_test.csv`.

| fold | months | n | rmse base | rmse fp | rmse g2 | rmse g3 | rmse all |
|---|---|---|---|---|---|---|---|
| fold0 | (2, 3) | 305,989 | 233.35 | 233.47 | 232.70 | 233.76 | 233.52 |
| fold1 | (4, 5) | 357,800 | 216.50 | 216.60 | 215.37 | 216.15 | 215.20 |
| fold2 | (6, 8) | 369,492 | 243.06 | 242.57 | 241.85 | 242.47 | 241.12 |
| fold3 | (9, 10) | 365,246 | 236.64 | 236.72 | 235.71 | 236.30 | 235.07 |
| fold4 | (11, 12) | 325,004 | 224.37 | 224.53 | 224.16 | 224.24 | 223.82 |
| holdout | (1, 7) | 339,046 | 261.34 | 260.65 | 262.98 | 260.55 | 259.88 |

Log loss, holdout: base 1.0267, fp 1.0271, g2 1.0250, g3 1.0259, all 1.0233.

## Per-fold verdicts

Pass mark: better mixture RMSE than base on holdout **and** in every fold.

- **fp**: FAIL. Holdout delta +0.70 (fp better), but worse than base in 3/5 folds (fold0,
  fold3, fold4) — a wash, not a one-directional signal.
- **g2**: FAIL. Beats base in every single CV fold (232.70/215.37/241.85/235.71/224.16, all
  below base), but reverses hard on holdout: 262.98 vs base 261.34, delta -1.64. The norm-taxi
  gaps look like a real in-distribution signal that does not transfer to Jan+Jul (LIRF's
  persistent EOBT-copy baseline, docs/LANEA_CLF.md section 3) — classic CV-holdout
  disagreement, not a keepable feature as specified.
- **g3**: FAIL. Holdout delta +0.79 (g3 better), but worse than base in fold0 (233.76 vs
  233.35) — fails the every-fold bar.
- **all**: FAIL. Holdout delta +1.46 (best of the four variants on holdout), but worse than
  base in fold0 (233.52 vs 233.35) — same single-fold failure as g3, since g3 is the
  weak link inside `all`.

## Bottom line

None of the four variants pass. `g2` (norm-relative gaps) is the strongest single group on
CV (consistent across all 5 folds) but its holdout reversal is the most informative failure:
whatever the (apt, runway, hour) physical-noise median is encoding, it does not describe
Jan+Jul, the same LIRF-heavy months where day_feat_test.py's airport-day signature and
arrmode both found their biggest holdout gains. `fp` and `g3` are each a wash (one bad fold
apiece), and `all` inherits g3's fold0 failure. Read this as a null result for all four
variants as specified, not as "close" or "promising": none clears the bar of beating base
everywhere.

## Plan against airport record (2026-09-14)

Six raw columns no model reads say whether the flight plan (`_flt`) and the airport record (`_mvt`) agree: callsign
against flight number, aircraft type, departure and destination airport, filed against actual destination, flight
rule (session scratchpad `mm_d.py`, log `mm_d.log`, v28 predictions). Lane B rows carry no plan column at all, so
there is nothing to read there. On lane A train rows the callsign differs from the flight number on 27%, and those
rows are easier (RMSE 160 against 218 on the rest) with a mean residual of +0.05 s; the type (0.3%), destination
(1.4%), filed destination (0.1%) and rule (0.07%) mismatches run harder (RMSE 234 to 305 against 203) with mean
residuals between -5.7 and +4.1 s. No flag moves the conditional mean, so none is a feature. Not used.

## Files

- `analysis/fe.py` — feature build (`build_fp`, `flight_ids`, `fit_apply`, `oof_apply`),
  cache `data/cache/fe_fp.parquet`
- `analysis/fe_test.py` — test harness, `data/analysis_fe_test.csv`
