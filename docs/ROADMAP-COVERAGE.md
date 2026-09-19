# Coverage against the 44-step time-series roadmap

Keith pasted a general roadmap for building an advanced time-series model (11 phases, 44 steps,
a 21-item checklist). This file grades prc26 against it, step by step, with a pointer per row.

Verdicts: DONE, PARTIAL, NOT DONE, N/A (does not apply to this task), DIVERGENT (we do the
opposite on purpose, reason given).

The roadmap targets a forecasting system: a horizon, a decision, a live pipeline. prc26 is a
cross-sectional regression on a single event per row, scored once on a fixed set. Phases 9 and 10
mostly fall away for that reason, and saying so is not the same as skipping them.

## Phase 1: define the problem (steps 1-4)

| Step | Verdict | Evidence |
|---|---|---|
| 1 business problem and decision | DONE | `README.md:1-4`; the task, metric and deadline are set by the challenge |
| 2 target defined mathematically | DONE | `TAXITIME_SEC_mvt` = takeoff minus recorded off-block; `docs/LANEB.md` |
| 3 horizon and observation frequency | N/A | no horizon: one row, one event, no multi-step path |
| 4 metrics fixed before training | DONE | RMSE, fixed by the challenge before any code |

## Phase 2: data (steps 5-10)

| Step | Verdict | Evidence |
|---|---|---|
| 5 identify and collect datasets | DONE | movements, METAR (`weather.py`), ATFM (`data/atfm/2026-09-17/`) |
| 6 point-in-time dataset | DIVERGENT | `PLAN.md:25-26`: features may use any movement in the same file, before or after the takeoff, because the challenge rules allow it. `dep_after` and `arr_off` look forward on purpose. This is the roadmap's single most emphasised rule and we break it knowingly. |
| 7 clean and validate raw data | PARTIAL | `audit.py` runs adversarial validation and a drift table; no standing data-quality report |
| 8 align frequencies and calendars | DONE | `weather.py:46-66` as-of join at the anchor, with look-back windows |
| 9 missing values | DONE | NaN passed through to LightGBM and CatBoost, which split on it natively |
| 10 version the dataset | DONE | dated immutable raw dirs (`data/weather/2026-09-05`, `data/atfm/2026-09-17`), `data/cache/feat.parquet`, `data/archive-v1/` |

## Phase 3: exploration (steps 11-14)

| Step | Verdict | Evidence |
|---|---|---|
| 11 visualise target and predictors | PARTIAL | `explore.py` prints the numbers; no charts were ever produced |
| 12 stationarity (ADF, KPSS) | N/A | no single series to difference; the target is cross-sectional |
| 13 seasonality and cyclicality | DONE | DAY group; `PLAN.md:27` hour and weekday are features, month and year never are |
| 14 regime changes | DONE | `regime.py`, `docs/REGIME.md`; disruption ratio r, hot band at r >= 1.35 |

## Phase 4: features (steps 15-20)

| Step | Verdict | Evidence |
|---|---|---|
| 15 lagged features | DONE | stamp gaps and differences; `docs/FE.md` |
| 16 rolling statistics | DONE | OCC occupancy counts, `weather.py:21-23` look-back sums over 3h to 24h |
| 17 domain features | DONE | the stamp-mixture model itself; `mix.py`, `docs/LANEB.md` |
| 18 interactions | DONE | WZ and PW groups; trees learn the rest |
| 19 select and reduce | PARTIAL | group-level ablation is the method, but no SHAP, no permutation importance, no knockoffs or stated FDR |
| 20 audit the final dataset | DONE | `audit.py` covers drift and adversarial separation. All three roadmap tests written 2026-09-17 (R5): `tests/test_label_timing.py`, `tests/test_feature_leakage.py`, `tests/test_reproducibility.py`, 11 tests, `python -m pytest tests/ -q`. Label reconstructs to 0.0 s max error over all 2,085,047 train rows; `y` null on all 344,841 rank rows; worst feature-to-label correlation is `z_eobt` at 0.556; `z_aobt` stays 98.5% populated on rank rows where the recorded off-block is 100% null, so it is a reconstruction and not a leak. |

## Phase 5: validation (steps 21-25)

| Step | Verdict | Evidence |
|---|---|---|
| 21 chronological split | PARTIAL | HOLDOUT = months (1, 7), FOLDS = 5 month-pairs (`train.py`, used at `lanea.py:64-70`). Held out by month, not by a moving origin. |
| 22 walk-forward validation | DONE, NULL | closed 2026-09-17 by R4 (`window.py`). Two forward slices, training rows strictly before the test months. 26 weeks loses on both: test Nov+Dec -10.59 s [-14.39, -7.40] (1.09M rows vs 1.74M), test Sep+Oct -1.34 s [-2.15, -0.57] (1.07M vs 1.37M). Expanding wins; the production folds stay month-pair. |
| 23 purging and gap | N/A | labels do not overlap forward; nothing to purge |
| 24 nested validation | NOT DONE | `tune.py` tunes on the same folds used to pick features and configs. `PLAN.md:28` keeps the holdout clean, which is the honest half of the answer, not the whole one. |
| 25 transforms fitted inside folds | DONE | `lanea.py:43-51`, every fit happens per fold |

## Phase 6: models (steps 26-31)

| Step | Verdict | Evidence |
|---|---|---|
| 26 simple baselines | DONE | `docs/RESULTS.md:4-7`: constant mean 686.62, airport x lane mean 637.07, anchor offset 698.91, raw takeoff minus AOBT_3 on lane A 428.17 |
| 27 regularised linear models | DONE, NULL | closed 2026-09-17 by R3. `ra_ridge` in `lanea.py` (Ridge on the same residual, `train.py` family `ridge`). Holdout lane A 273.79 against the v32 blend's 218.69; NNLS blend weight 0.000. Most decorrelated member we have and still worth nothing. |
| 28 tree models | DONE | `lanea.py:20-35`, LightGBM and CatBoost across 14 configs |
| 29 deep learning | DONE | `nn.py`, members `nn3_*` |
| 30 train and regularise NNs | DONE | `nn.py`; seed variants `s1`, `s2` exist for the tree families too |
| 31 ensembles | DONE | `blend.py` NNLS, then `recal.py` LightGBM second stage |

## Phase 7: optimisation (steps 32-33)

| Step | Verdict | Evidence |
|---|---|---|
| 32 systematic hyperparameters | **NOT DONE** (regraded 2026-09-18, was PARTIAL on a wrong reading of the code) | `tune.py` *describes* a study `ra_reg`, but `data/optuna.db` has never held one. The database contains a single study, `mix_reg`: 3 completed trials and 1 left RUNNING on 2026-09-12, on the mixture regressor, not the live lane A path. Trial 0, the pinned hand-set baseline, won it (283.31 vs 295.04 and 286.25). So no live model has ever been searched; all of them run on round-number defaults (`train.py:25-28`, `mix.py:19-22`). Lane R6 opens the real search. The earlier PARTIAL grade read the source and not the database, which is the mistake this audit exists to catch. |
| 33 feature ablation | PARTIAL | each config adds one feature group, so the ladder exists in `docs/RESULTS.md`, but never written up as a declared A/B/C/D/E table with per-group deltas |

## Phase 8: evaluation (steps 34-38)

| Step | Verdict | Evidence |
|---|---|---|
| 34 errors and residuals | DONE | residual bias by band, per lane, per month, per regime; `analysis/error_budget.py` |
| 35 uncertainty and calibration | PARTIAL | the mixture emits P(class), but no reliability diagram, no Brier score, no conformal interval. `recal.py` recalibrates predictions; it does not assess probability calibration. |
| 36 statistical significance | DONE | paired day-block bootstrap on every result line (`docs/RESULTS.md:11,13,15,17`), plus the measured holdout-to-board transfer sd of 0.253 s and the 1.5 s gate in `docs/EXPERIMENT-PROTOCOL.md` |
| 37 regime and stress testing | PARTIAL | regime slice rides every result line; seed sensitivity covered by `s1`/`s2`. Missing-feature and delayed-data stress not run. |
| 38 final untouched test | DIVERGENT | the board is the test set and it is not untouched: it has been read on every upload. Inherent to a competition. Mitigation is the measured transfer noise and the 1.5 s promise gate, not an untouched set. |

## Phase 9: decisions (steps 39-40)

| Step | Verdict | Evidence |
|---|---|---|
| 39 decision framework | N/A | the prediction is the deliverable; there is no position to take |
| 40 backtesting engine | N/A | no costs, no execution, no P&L |

## Phase 10: production (steps 41-44)

| Step | Verdict | Evidence |
|---|---|---|
| 41 inference pipeline | DONE | `submit.py` builds, checks and uploads through the same `build()` used in research |
| 42 version models and experiments | PARTIAL | git plus `LEDGER.md` plus `docs/RESULTS.md`; no MLflow, and not every run is tied to a commit |
| 43 monitor performance | N/A | 24-day competition, no live serving |
| 44 retraining and rollback | N/A | same reason |

## The 21-item checklist

Pass 11, partial 5, fail 2, not applicable 3.

Fails: "every feature has a verified availability timestamp" and "feature engineering uses only
past information". Both are the deliberate divergence at step 6, allowed by the challenge rules
and stated in `PLAN.md:25-26` and the README.

Partial: missing-data and outlier policy, chronological splits, probability calibration, multiple
testing, reproducible experiment versions.

Not applicable: horizon matching a decision horizon, decision costs, monitoring and rollback.

## What this audit changed

Three gaps were named on 2026-09-17 and all three were closed the same day. Two are nulls.
Nothing here moved the board; the value is that three open questions are now shut.

1. **Regularised linear model** (step 27), lane R3. Built and run. `ra_ridge` gets NNLS blend
   weight **0.000**. The useful by-product is a rule that prices every future member:

   > A candidate with residual sd `s_new` earns blend weight against a champion with residual sd
   > `s_champ` only if their residual correlation is below `s_champ / s_new`.

   Ridge needed below 0.8423 against `ra_cb_wz` and had 0.8479; below 0.8590 against `ra_lgb_pw`
   and had 0.8630. It misses by 0.006 and 0.004. Chasing that margin is not worth a run: even a
   ridge improved from 242.29 to 230 blends to 203.57 against 204.07, half a second on lane A OOF
   against a 1.5 s transfer gate.

2. **Expanding versus rolling windows** (step 22), lane R4. Built and run on two forward slices.
   Shorter loses on both. Matched-n controls show the loss is part sample size (-2.42 and -2.22,
   consistent) and part season coverage (-8.17 on a winter test set the window cannot reach,
   +0.88 on an autumn one it can). The recency term flips sign between slices, so only the
   verdict travels, not that term. The constraint worth recording: training is 52.0 weeks
   (2025-01-01 to 2025-12-31), so 26-week increments give only {26, 52} and 52 is already the
   production refit. Neither can be judged on the Jan+Jul holdout, because January is the first
   month in the file and nothing precedes it.

3. **Automated tests** (step 20), lane R5. All three written, 11 tests, 10 passing on the first
   run. The one failure was a missing seed on the R3 config written the same hour, now fixed.

Everything else on the list is done, deliberately divergent, or does not apply to a
fixed-set regression competition.
