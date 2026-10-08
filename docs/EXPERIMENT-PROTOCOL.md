# EXPERIMENT PROTOCOL - prc26 (v8, 2026-09-23)

> Instantiated from the quant-ml-protocol skill. This file governs every run and every scorer slot.
> Amendments are appended with a version bump and a date, never edited in place. An experiment not
> declared here is a diagnostic, not evidence.
>
> Written after v33. Campaign 2 shipped 11 files; the last five were all sub-noise bets
> (section 0), and `PLAN.md:20` already forbade four of them.

## 0. Sample-size math

The verdict statistic is a paired day-block delta, so it rides airport-days, not rows.

| | train | ranking (the board) |
|---|---|---|
| lane A rows | 2,062,577 | 339,551 |
| airport-days | 3,650 | 620 |
| calendar days | 365 | 62 |

**The noise yardstick is not the holdout CI.** The holdout bootstrap answers "would this repeat on
another sample of Jan+Jul 2025 days". The board is Jan+Jul **2026**. The transfer error between the
two is measured from our own ledger, one point per upload (`LEDGER.md`, regenerate with the snippet
in section 9):

| holdout promise | n | board delta |
|---|---|---|
| 1.5 s or more | 5 | mean +7.79, min +1.90, **every one positive** |
| under 1.5 s | 9 | mean **+0.04**, sd 0.407, **5 of 9 negative**; v34's +0.99 is the only one above +0.18 (n = 7 after v34: mean +0.08, sd 0.463; n = 6 before it: mean -0.08, sd 0.253) |
| a loss (below 0) | 2 | both lost, as promised: v38 -4.12 on -2.53, v39 -2.86 on -3.92; the size carried 1.63 and 0.73 times |

- Transfer sd: **0.407 s** of board, 95% CI [0.275, 0.780] (n = 9, chi-square), re-estimated after the v36 and
  v37 uploads per section 8; it was 0.463 [0.298, 1.020] at n = 7 and 0.253 [0.158, 0.620] at n = 6.
- **Detectable-effect floor: 0.81 s of board** at the point estimate, **1.56 s** at the sd's upper
  bound (0.93 and 2.04 at n = 7, 0.51 and 1.24 before v34). Any holdout gain that projects below this is noise-sized and must be labeled so.
- Feature cap: row-level features ride 2.06M rows and are uncapped in practice. Day-level and
  window-level features (`DAY` 7, `WZ` 20, `PW` 5 = 32) ride 620 ranking airport-days, so the
  n_indep/8 cap is **77**. We are under it; the next day-level family must count against it.

## 1. Target

- Target: `TAXITIME_SEC_mvt` = takeoff minus the recorded off-block. Takeoff is given; the off-block
  is blanked, so the unknown is one timestamp. Metric: RMSE on 344,841 rows, Jan+Jul 2026.
- Never normalised: RMSE is scored on raw seconds, so the conditional mean is the optimal
  prediction and every model targets it.
- Naive-baseline check: the airport mean scores RMSE 417 on lane A and 3,398 on lane B
  (`docs/AUDIT.md`, "Where the label variance sits"). Plain CatBoost on fold 0 scores 285 against
  the stamp mixture's 253 (`PLAN.md:31`). Current: v34, 216.42 lane A holdout, board 270.5655 (v32 was 217.18 and 271.5521).
- Carry / roll / basis decomposition: **N/A**, this is not a traded target.

## 2. Data register

| source | vintaged? | revision risk | staleness cap | PIT rule |
|---|---|---|---|---|
| challenge `training_*.parquet`, `ranking.parquet` | static files | none | n/a | as-of rule below |
| IEM ASOS METAR (`data/weather/2026-09-05/`) | dated pull dir | archive, stable | pull ends 2026-08-01, after every anchor | joined as-of the estimated off-block (`weather.join`) |
| EUROCONTROL PRU monthly taxi-out xlsx | published 2026-08-14 | monthly restatement | covers to Jun 2026 | **not used**, see section 7 |
| EUROCONTROL daily airport ATFM delay (`data/atfm/2026-09-17/apt_dly_{2025,2026}.csv.bz2`) | dated pull dir | monthly restatement, `ATFM_VERSION` column | 2025-01-01 to 2026-07-31, covers both ranking months | day-level, joined on (apt, day). **Tested and null**, R2. Read with `encoding='latin-1'` |
| open ADS-B (adsb.lol) | release-tagged | none | n/a | **closed null**, `docs/ADSB.md` |

As-of rule: a feature may use any movement in the same file, before or after the takeoff (the rules
allow it). Point-in-time vintages are **N/A**: the files are static. The one real point-in-time bug
found so far was an end-of-data artefact in stand occupancy (`docs/FE.md`, "Stand occupancy at the
data end"), fixed at `ab92caf`.

## 3. The decision rule (attack this, not just the numbers)

A change reaches a scorer slot only if **all** of these hold:

1. **Size.** Holdout gain (paired day-block bootstrap, all rows) is **1.5 s or more**. Below that the
   ledger says the expected board move is zero. `PLAN.md:20`'s "~3 s" stays the target; 1.5 s is the
   hard floor, and anything between 1.5 and 3 s is reported with its projected board move beside it.
2. **Direction on both regimes.** The `hot` slice delta (r >= 1.35, `regime.py`) is not negative
   beyond its own interval. A change that pays on calm rows and loses on hot rows fails: hot rows
   are 6.77% of the ranking lane A rows and a projected **27.3%** of its lane A SSE (`docs/REGIME.md`).
3. **Trial ledger.** The pass is reported beside N x alpha, N = the declared verdict lanes to date
   (section 4). Lanes share the same holdout, so the Bonferroni line over-punishes; a joint bootstrap
   across the family rides beside it once more than three lanes are open at once.
4. **Sign-off.** Promotion and every scorer slot need the Stage-6 gate, explicitly signed off.

Nulls are **vetoes** on the change and **caution flags** that ride every later report.

Primary metrics, both with paired intervals, never one alone: **all-rows RMSE delta** and the
**hot-slice RMSE delta**. Tie-break: parsimony (fewer members, fewer columns).

**This verdict is only as good as this rule. Attack the rule, not just the numbers.**

## 4. Declared experiments (registry)

N counts verdict lanes, including peeks used to choose the next build. Campaign 2 before this file:
**61 holdout result lines** (`docs/RESULTS.md`), 11 of them spent as scorer slots. At alpha = 0.05
that is about 3 expected false passes; v29 and v33 both passed a CI-excludes-zero gate and lost on
the board, which is what 3 expected false passes looks like.

| id | declared | design | null bar | status | verdict |
|---|---|---|---|---|---|
| R0 | 2026-09-17 | regime map frozen at `data/cache/regime.parquet` | n/a, a definition | DONE | 51 of 620 ranking airport-days are hot vs 49 of 3,650 in training |
| R1 | 2026-09-17 | disruption-day specialist, lane A hot rows | 1.5 s all-rows, hot slice positive | SUPERSEDED by R1a | bias check: holdout mean residual by band is ~0 in all six bands, worst +44.4 on r>1.75 at 0.6% of that band's SSE. Hot-day error is variance, not bias, so a second-stage hot correction has nothing to fit |
| R1a | 2026-09-17 | weather history (`SNOW`, 6 cols) added to `ra_lgb_pw` -> `ra_lgb_sn` | 1.5 s all-rows, hot slice positive | **NULL** | holdout lane A 225.91 vs 225.75, OOF all rows -0.07 [-0.18, +0.05]. By snow band: sn>=25 +3.89 [+1.21, +6.72] but sn 15-24 is 7.3 s worse. A sign flip between adjacent bands of a monotone feature is noise; 1 of 5 band looks. `ra_cb_sn` not run, the screen killed it |
| R1b | 2026-09-17 | does the off-block stamp mixture shift by regime? | n/a, a measurement | DONE | yes, monotone: AOBT-copy share 20.3% on r<=1.1 falls to 7.2% on r>1.75, physical 68.6% -> 84.2%. Lane A's residual-to-AOBT anchor is 3x rarer where the error lives |
| R2 | 2026-09-17 | EUROCONTROL daily airport ATFM delay (`apt_dly`, `data/atfm/2026-09-17/`), 6 day-level columns incl. the cause split | non-zero corr with the airport-day mean residual | **NULL** | data is complete (2025-01-01 to 2026-07-31, all 10 airports, both ranking months). Corr with the airport-day **mean residual**: total +0.023, capacity +0.012, aerodrome -0.000, staffing +0.000, weather +0.030, on 3,650 airport-days (SE 0.017). Corr with **raw y** is real (aerodrome -0.231, capacity +0.101), so the model already extracts all of it. The de-icing cause column `DLY_APT_ARR_D_1` is **identically zero** at all 10 airports. A day-level feature can only move day-level means, so a zero residual correlation at that aggregate closes the feature class, not just these columns |
| R1c | 2026-09-17 | does the existing stamp-mixture model (`mix_cb_wx`, the lane B donor) beat the residual architecture on hot lane A rows? | mixture wins a hot band | **NULL** | worse in all 6 bands by 8 to 35 s. One positive 50/50 blend band (1.5-1.75, +6.27, n=4,862) sits below a negative in the band above. Limit: this rejects the lane-B-tuned mixture on lane A, not every mixture model; a lane-A-tuned one is untested and now has a poor prior |
| R3 | 2026-09-17 | regularised linear member `ra_ridge` (Ridge on the same residual c = y - z_aobt, same tail/body split, `NUM+DAY+OCC+WX` numeric, CAT one-hot capped at `min_frequency=500`), roadmap step 27 | **not** 1.5 s alone: a blend member is judged by what it adds to the NNLS blend, so the bar is a non-zero NNLS weight AND the reblended holdout beating v32's blend by a CI that excludes zero | **NULL** | NNLS weight **0.000**. Holdout lane A 273.79 vs the v32 blend's 218.69, OOF lane A 242.29 vs `ra_cb_wz` 204.07. It is the most decorrelated member we have (OOF residual corr 0.878 with `ra_lgb_wz` on a one-month probe, against 0.906 to 0.982 for every live pair on 1,723,531 lane A OOF rows) and NNLS still takes none of it: decorrelation only pays when the member is competitive, and a 25% larger residual sd swamps a 0.03 correlation edge. Closes the linear family; the roadmap's "start with a regularised linear model" is advice for the start, not for a tuned tree stack |
| R4 | 2026-09-17 | rolling vs expanding training window, `window.py`. Keith asked for 26-week increments; training is 52.0 weeks (2025-01-01..2025-12-31), so the increments give exactly two lengths, 26 and 52, and 52 is the current setup. Causal design: evaluate on a month pair, train only on rows before it. Two slices, (9,10) and (11,12), each with the 26-week window against everything-before. Slice (6,8) is impossible: only 21 weeks precede June. Month 7 is holdout and is never used as an evaluation slice here | 26 weeks beats everything-before on BOTH slices with a CI excluding zero. One slice out of two is a diagnostic, not a pass | **NULL, and the reverse is true** | `ra_lgb_wz`. Test Nov+Dec: 26w 210.25 (n=1,093,306) vs all-before 199.66 (n=1,737,573), **-10.59 [-14.39, -7.40]**. Test Sep+Oct: 26w 213.59 (n=1,067,069) vs all-before 212.25 (n=1,372,327), **-1.34 [-2.15, -0.57]**. Both slices lose, both CIs exclude zero. Matched-n controls (the same row count drawn at random from all history) split each loss into a sample-size part and a recency part. Nov+Dec: rand-n 202.08, so -2.42 is sample size and **-8.17 is recency**. Sep+Oct: rand-n 214.47, so -2.22 is sample size and recency is **+0.88, the other sign**. The sample-size cost is consistent (-2.42, -2.22); the recency effect is **not** and one slice cannot carry it. Mechanism consistent with both: the 26 weeks before November is May-October, no winter at all against a winter test set, while the 26 weeks before September is March-August, adjacent seasons to an early-autumn test set. Season coverage, not age, is what the window destroys. **Not measured here:** exponential recency weighting (Stage 4.1 rung 3). The same mechanism predicts it hurts our actual task, because the ranking months are January and July and recency weighting would downweight January and July 2025, the two best-matched months we own. That is a prediction from mechanism, not a result, and it is cheap to run via the `cfg["weight"]` hook at `train.py:52` if it is ever worth settling |
| R5 | 2026-09-17 | automated tests: `test_label_timing.py`, `test_feature_leakage.py`, `test_reproducibility.py` (roadmap step 20) | n/a, an audit not an experiment. A failing test is a finding and stops the lane | **DONE, 11 pass** | `python -m pytest tests/ -q`, 48 s. Label reconstructs (takeoff minus recorded off-block) to 0.0 s max error over all 2,085,047 train rows; the 369 negative-y rows are pinned exactly; `y` is null on all 344,841 rank rows. Worst feature-to-label correlation is `z_eobt` at 0.556. `z_aobt` is built only from `AOBT_3_flt`, never from `BLOCK_TIME_UTC_mvt`, and stays 98.5% populated on rank rows where the recorded off-block is 100% null, so it is a reconstruction and not a leak. Forward-looking features are **not** treated as failures: `PLAN.md:25-26` allows them by the challenge rules. One real finding on the first run: `ra_ridge` (R3, written the same hour) had no seed, caught by `test_every_model_config_has_a_fixed_seed` and fixed |

| R6 | 2026-09-18 | **Hyperparameter search, declared before the first trial.** Finding that opened it: `data/optuna.db` holds exactly one study, `mix_reg`, 3 completed trials and 1 left RUNNING, all 2026-09-12, all on the *mixture* regressor, which is not the live lane A path. The `ra_reg` study `tune.py` creates has never existed. Trial 0, the pinned hand-set baseline, **won** it: 283.31 against 295.04 and 286.25. So every live model runs on round-number defaults (`train.py:25-28`, `mix.py:19-22`) and roadmap step 32 is NOT DONE, not PARTIAL. **Design:** Optuna TPE, seed 0, k = 40 trials, over the 7 LightGBM regressor params of `ra_lgb_pw` (lr, num_leaves, min_child_samples, feature_fraction, bagging_fraction, l2, l1); the classifier stays shipped and is precomputed once per fold. Tune on folds (2,3) and (4,5); **judge on (6,8), (9,10), (11,12), which the search never sees**, so the winner is re-scored without its own selection bias. That disjoint split is roadmap step 24 (nested validation) for this lane; the Jan+Jul holdout is not touched at all. LightGBM first because it is about 2 min a trial against CatBoost's about 10, so 40 trials price the whole question in about 1.5 h. Known bias, equal on both arms: early stopping sees the scored fold, exactly as `lanea.cv` does, so both numbers are optimistic by the same amount and only their difference is read | **Tuned beats shipped on the three judge folds.** A win on the tune folds alone is the search fitting fold noise, which is what k = 40 buys at alpha = 0.05. If it passes, the member then has to earn a non-zero NNLS weight and the reblend has to beat v32 with a CI excluding zero; the slot gate stays section 3's 1.5 s all-rows. If the judge-fold gain is under about 1 s, tuning is priced as dead for this task and CatBoost is not searched | **PASS, +2.188 s on the judge folds** | `python tune.py 40 ra_lgb_pw 2-3,4-5 6-8,9-10,11-12`, log `data/r6.log`, study `ra_lgb_pw_reg_23_45`. Tune folds: shipped (trial 0) 202.558, best 200.306 at trial 22 of 40, gain +2.252. **Judge folds (6,8), (9,10), (11,12), never seen by the search: shipped 211.250, tuned 209.062, gain +2.188.** The judge gain is 97% of the tune gain, so selection inflation on this search is about 0.06 s and effectively nil. Winning region: `lr` 0.0255 against the shipped 0.05, `num_leaves` 88 against 255, `feature_fraction` 0.524 against 0.8. Trials 18 to 22 all land 200.3 to 200.8 in that same region, so it is a basin and not a lucky draw. `l2` (1.4 to 29.9), `l1` (0.7 to 2.9) and `ff` (0.50 to 0.92) vary freely among the winners, so the gain is carried by the learning-rate and leaf-count pair, not by regularisation. Not isolated: lr and leaves were never varied one at a time. **Priced, not shipped:** this is a member holding 0.175 of the blend, and `LEDGER.md:18-22` shows the last four lane A gates carrying 0.4, 0.15, negative, negative of their holdout promise. Superseded as a search by R7, which opens all 17 LightGBM parameters; kept as the control that says a narrow search does transfer |
| R6 space | 2026-09-18 | **The exact search space, recorded so it cannot be widened after seeing a result.** `lr` 0.02-0.12 log, `num_leaves` 63-511 log, `min_child_samples` 10-200 log, `feature_fraction` 0.5-1.0, `bagging_fraction` 0.5-1.0, `lambda_l2` 0.3-30 log, `lambda_l1` 0.0-3.0 linear. The shipped values (0.05, 255, 50, 0.8, 0.8, 3.0, 0.0) sit inside every range and are enqueued as trial 0, so the search is always judged against the thing it must beat. **Amendment, made before any trial scored:** `lambda_l1` was declared log 1e-3 to 3; the shipped value is 0.0, which no log distribution can represent, so the smoke test failed at trial 0 and the range became linear from 0.0. Nothing had been measured when this changed. Second fix from the same smoke: the study is keyed `<CFG>_reg_<TUNEFOLDS>`, because an objective averaged over one fold set is not comparable with another, and the enqueue is gated on a COMPLETE trial so a crashed run cannot silently cost the baseline anchor | n/a, a record | DONE | reproduce: `python tune.py 40 ra_lgb_pw 2-3,4-5 6-8,9-10,11-12` |
| R7 | 2026-09-18 | every tunable parameter of all three v32 members, 200 trials each, same tune/judge fold split as R6. Full spaces and rationale in the R7 section below | tuned beats shipped on the three judge folds the search never saw | **PASS on all three members** | `ra_lgb_pw` +4.235 (211.250 to 207.015), `ra_cb_wz` +1.142 (207.375 to 206.233), `nn3_pw` +3.481 (212.053 to 208.572), 0 failed trials of 600. Counts, transfer ratios and the R6 comparison in the R7 results table below. Priced, not shipped: no scorer slot without a non-zero NNLS weight, a reblend beating v32 with a CI excluding zero, and section 3's 1.5 s gate |

### R7: every hyperparameter of all three v32 members, 200 trials each

Declared 2026-09-18, before the first trial, at Keith's instruction. R6 searched 7 LightGBM params on one member.
R7 searches **every** tunable parameter on **all three** members of the shipped v32 blend. Same honesty design as R6:
tune on folds (2,3) and (4,5), judge on (6,8), (9,10) and (11,12), which the search never sees. The Jan+Jul holdout is
never touched. Studies are `<CFG>_full_<TUNEFOLDS>` in `data/optuna.db`, separate from R6's `<CFG>_reg_<TUNEFOLDS>`
because the parameter distributions differ and Optuna cannot mix two distributions under one name.

Trial 0 of every study is the shipped configuration, enqueued, so the search is always judged against the thing it must
beat. Verified before launch: each space reconstructs its shipped values exactly (`lgb` 17 params, `cb` 13, `nn` 12).

**Not searched, on purpose:** `n_estimators` / `iterations` / `max_epochs`. Early stopping already picks them per fold,
so searching them would fight the callback that sets them. `rsm` (CatBoost column sampling) is excluded because
CatBoost ignores it under `task_type="GPU"`. `max_depth` is excluded from LightGBM because growth is leaf-wise and
`num_leaves` is the binding constraint.

**`ra_lgb_pw`, LightGBM** (shipped values from `train.py:27-28` plus the library defaults the config never set)

| parameter | shipped | range | scale |
|---|---|---|---|
| `learning_rate` | 0.05 | 0.01 to 0.15 | log |
| `num_leaves` | 255 | 31 to 1023 | log |
| `min_child_samples` | 50 | 5 to 300 | log |
| `min_child_weight` | 0.001 | 1e-5 to 10 | log |
| `feature_fraction` | 0.8 | 0.4 to 1.0 | linear |
| `bagging_fraction` | 0.8 | 0.4 to 1.0 | linear |
| `bagging_freq` | 1 | 0 to 7 | integer, 0 disables |
| `lambda_l1` | 0.0 | 0.0 to 10.0 | linear, must hold 0 |
| `lambda_l2` | 3.0 | 0.0 to 50.0 | linear, must hold 0 |
| `min_split_gain` | 0.0 | 0.0 to 1.0 | linear |
| `max_bin` | 255 | 63 to 511 | log |
| `path_smooth` | 0.0 | 0.0 to 100.0 | linear |
| `extra_trees` | False | False, True | categorical |
| `cat_smooth` | 10.0 | 1 to 200 | log |
| `cat_l2` | 10.0 | 1 to 200 | log |
| `max_cat_threshold` | 32 | 8 to 128 | log |
| `min_data_per_group` | 100 | 10 to 500 | log |

The four `cat_*` parameters are live because `train.py:63` passes `categorical_feature=cfg["cat"]`, so LightGBM does its
own categorical handling on 11 columns and has never had those knobs touched.

**`ra_cb_wz`, CatBoost** (shipped from `train.py:25-26` `BASE` and `lanea.py:18` `REG`)

| parameter | shipped | range | scale |
|---|---|---|---|
| `depth` | 8 | 4 to 10 | integer |
| `learning_rate` | 0.1 | 0.01 to 0.2 | log |
| `l2_leaf_reg` | 3.0 | 0.5 to 30 | log |
| `random_strength` | 1.0 | 0.0 to 10.0 | linear |
| `border_count` | 254 | 32 to 254 | integer |
| `max_ctr_complexity` | 4 | 1 to 4 | integer |
| `one_hot_max_size` | 2 | 2 to 255 | log |
| `model_size_reg` | 0.5 | 0.0 to 2.0 | linear |
| `leaf_estimation_iterations` | 1 | 1 to 10 | integer |
| `leaf_estimation_method` | Newton | Newton, Gradient | categorical |
| `grow_policy` | SymmetricTree | SymmetricTree, Depthwise, Lossguide | categorical |
| `bootstrap_type` | Bayesian | Bayesian, Bernoulli, No | categorical |
| `bagging_temperature` | 1.0 | 0.0 to 5.0 | only under Bayesian |
| `subsample` | n/a | 0.3 to 1.0 | only under Bernoulli |
| `min_data_in_leaf` | n/a | 1 to 200 log | only under Depthwise or Lossguide |
| `max_leaves` | n/a | 16 to 128 log | only under Lossguide |

`grow_policy` is the one that could matter most and has never been moved: the shipped model only ever grew symmetric
(oblivious) trees, which is CatBoost's speed default, not its accuracy default.

**`nn3_pw`, MLP** (shipped from `nn.py:23-24` `CELL`)

| parameter | shipped | range | scale |
|---|---|---|---|
| depth, `hidden` layers | 3 | 1 to 5 | integer |
| width per layer | 1024 | 128 to 2048 | log |
| `dropout` | 0.2555 | 0.0 to 0.6 | linear |
| `lr` | 0.0015 | 1e-4 to 1e-2 | log |
| `weight_decay` | 0.000165 | 1e-7 to 1e-2 | log |
| `batch` | 1024 | 256, 512, 1024, 2048, 4096 | categorical |
| `t_max`, cosine period | 60 | 10 to 150 | log |
| `emb_cap` | 16 | 4 to 64 | log |
| `huber` delta | 0.405 | 0.05 to 3.0 | log |
| `patience` | 5 | 3 to 15 | integer |
| activation | SiLU | SiLU, ReLU, GELU | categorical |
| batchnorm | on | on, off | categorical |

Activation and batchnorm needed a two-line change in `nn.py:57-60` to become reachable; both default to the shipped
behaviour when the key is absent, so every earlier result stands.

**Pruning.** `MedianPruner(n_startup_trials=15)`. A trial reports after its first tune fold and is killed if it is worse
than the median at that point. The first 15 trials never prune, so trial 0 (the shipped baseline) always runs to
completion. This is what makes 200 CatBoost trials affordable; it does not narrow the space.

**Measured cost per trial** on 2 tune folds, from the recorded run times in `docs/RESULTS.md`: LightGBM 3.6 min
(confirmed against R6's live log), CatBoost about 16 min on GPU, the MLP about 60 min on CPU and unmeasured on GPU.

**Null bar, same as R6 and stated before the first trial:** the tuned configuration must beat the shipped one on the
three judge folds it never saw. A win on the tune folds alone is the search fitting fold noise, which is exactly what
200 trials over 17 dimensions buys. Passing that, the member must still earn a non-zero NNLS weight and the reblend must
beat v32 with a confidence interval excluding zero, and the slot gate stays section 3's 1.5 s.

**Standing caution that rides this lane.** `LEDGER.md:18-22`: the last four gated lane A changes carried 0.4, then 0.15,
then negative, then negative of their holdout promise. Two of them passed with a CI excluding zero and still lost board
seconds. A judge-fold win here is necessary, not sufficient, and nothing from R7 goes to a scorer slot on the strength of
the judge folds alone.

**R7 results, all three searches complete (2026-09-22).** Logs `data/r7_lgb.log`, `data/r7_cb.log` and `data/r7_nn.log`;
regenerate the trial counts with `grep -c "Trial [0-9]* \(finished\|pruned\)" data/r7_*.log`.

| member | trials | COMPLETE / PRUNED / FAIL | tune folds: shipped to best (trial) | judge folds: shipped to tuned | judge gain |
|---|---|---|---|---|---|
| `ra_lgb_pw` | 200 | 128 / 72 / 0 | 202.558 to 198.956 (t183) | 211.250 to 207.015 | **+4.235** |
| `ra_cb_wz` | 200 | 113 / 87 / 0 | 198.123 to 196.709 (t194) | 207.375 to 206.233 | **+1.142** |
| `nn3_pw` | 200 | 115 / 85 / 0 | 203.276 to 198.548 (t112) | 212.053 to 208.572 | **+3.481** |

All three pass the declared null bar, 0 failed trials of 600. Three readings:

1. **No selection inflation on LightGBM.** The judge gain (+4.235) is larger than the tune gain (+3.602), so the winning
   region is not fitted fold noise. R6's narrow 7-parameter search transferred at 97%; this 17-parameter one transfers at
   118%. CatBoost transfers at 81% (+1.142 judged against +1.414 tuned), still positive.
2. **The wide search beats the narrow one on the same folds.** R6 and R7 share the judge folds and the same shipped
   baseline of 211.250, so they compare directly: R6 (40 trials, 7 params) reached 209.062, R7 (200 trials, 17 params)
   reached 207.015. The extra 10 parameters and 160 trials bought a further 2.047 s.
3. **`grow_policy` did not move.** The protocol flagged it as the CatBoost knob most likely to matter, and the winner
   kept `SymmetricTree`. The CatBoost gain comes from `depth` 8 to 10, `learning_rate` 0.1 to 0.0417 and `l2_leaf_reg`
   3.0 to 21.6, the same lower-rate, higher-capacity, more-regularised direction LightGBM took.
4. **The MLP won by shrinking.** Width 1024 to 509 and `emb_cap` 16 to 5, against a higher rate (0.0015 to 0.0045),
   more dropout and a much larger Huber delta (0.405 to 0.452). Transfer 74% (+3.481 judged against +4.728 tuned).
   The one architectural change is `SiLU` to `GELU`; batch norm stayed on.

CatBoost bought a quarter of what LightGBM bought from the same trial budget, which prices further CatBoost tuning as
close to dead. Nothing here is a scorer slot: the standing caution above still applies, and each member must still earn
a non-zero NNLS weight and a reblend that beats v32 with a CI excluding zero.

**R7 re-fits on the full CV, judged on the real holdout.** The judge folds score lane A residuals inside `tune.py`;
these rows are the member's own `lanea.py cv` output, which is what the blend consumes. Regenerate with
`grep -E "^\| (ra_lgb_pw|ra_lgb_pw_t7|nn3_pw|nn3_pw_t7) " docs/RESULTS.md`.

| member | CV all | CV laneA | holdout all | holdout laneA | holdout laneA delta |
|---|---|---|---|---|---|
| `ra_lgb_pw` | 253.44 | 208.12 | 326.37 | 225.75 | baseline |
| `ra_lgb_pw_t7` | 250.20 | 204.12 | 323.67 | 221.75 | **-4.00** |
| `nn3_pw` | 253.80 | 208.56 | 327.13 | 226.86 | baseline |
| `nn3_pw_t7` | 251.35 | 205.55 | 325.92 | 225.08 | **-1.78** |
| `ra_cb_wz_t7` | stopped, see below | | | | |

The LightGBM holdout lane A gain (4.00) lands within 0.24 of the judge-fold gain (4.235), on rows neither the search nor
the judge folds touched. That is the strongest transfer evidence in the campaign so far. Two cautions: the tuned member
alone (323.67) is still worse than the v32 blend (321.61), so the gain only matters if it survives the reblend; and the
low learning rate pushed the body regressor from 293 to 6938 iterations, a 12x fit-time cost that the `fit` step pays
again.

The tuned MLP won all five CV folds, and gained more on the three judge folds the search never saw (lane A 211.639 to
208.181, +3.458) than on the two it was searched on (+2.233); per-fold RMSE of each member's OOF, session scratchpad
`fold_rmse.py`. The CPU re-fit's judge mean, 208.181, is below the GPU search's 208.572, which closes the device flag
in section 7. On the holdout it kept 1.78 s of lane A, about half its judge gain, and the member interval crosses zero:
+1.786 [-3.500, +4.666] lane A, `python gate.py nn3_pw_t7_holdout.parquet nn3_pw_holdout.parquet`.

**Tuned CatBoost stopped.** The GPU re-fit of `ra_cb_wz_t7` ran at about 219 s per iteration while fifty's training
chain held the card, and its own estimate was 18.6 days. It was stopped. It re-runs when the card is free, as a v35
candidate; v34 keeps the shipped `ra_cb_wz`.

**v34, declared before the tuned MLP's CV landed (2026-09-22).** Members: shipped `ra_cb_wz`, tuned `ra_lgb_pw_t7`,
tuned `nn3_pw_t7`.

- **Weights.** NNLS on all non-holdout OOF rows, `blend.py`'s rule, unchanged since v21. The two tuned members were
  searched on folds (2,3) and (4,5), so their OOF on those months could be optimistic and pull weight toward them.
  Checked on the interim set (tuned LightGBM only): fitting on the judge months alone moves the tuned LightGBM weight
  from 0.311 to 0.345, up, not down, and the untuned v32 set moves the same way, 0.175 to 0.194. The tune months do
  not inflate the tuned weight, so the all-months rule stays and the judge-month weights ride the report as a
  diagnostic. The same check runs on the final set.
- **Slot rule.** Upload if the file-level holdout gain over v32 (all rows, paired day-block bootstrap, `gate.py`) has
  an interval excluding zero and the lane A hot slice is not negative beyond its interval. Keith asked for the upload
  ("Submit them and check live result", 2026-09-22), so a gain under section 3's 1.5 s floor is uploaded on that
  instruction and recorded in `LEDGER.md` as a below-floor upload, a section 8 breach. Anything else holds the slot.

**v34 result (2026-09-22): held, not uploaded.** NNLS on all non-holdout OOF gave 0.345484 `ra_cb_wz` + 0.265653
`ra_lgb_pw_t7` + 0.388863 `nn3_pw_t7`, every weight non-zero. The judge months alone gave 0.30697 / 0.279992 /
0.413038: both tuned weights rise when the search months leave, so the all-months rule stands. Blend vs the v32 blend
+0.55 [-0.25, +1.07] (`blend.py`); second stage lane A 217.87 to 216.42, +1.45 [+0.85, +2.08]. File-level gate,
`python gate.py recal_holdout_v34.parquet recal_holdout_v32.parquet`: holdout all 305.24 vs 305.77, **+0.526 [-0.123,
+1.043]**, lane A +0.754 [-0.181, +1.396], SSE +110.8M; hot lane A -0.52 [-3.56, +3.20], calm +0.80 [-0.15, +1.45].
The all-rows interval includes zero, so the slot rule holds the slot: no file built, no slot spent. The 1.5 s floor
needed lane A at 215.03; v34 reached 216.42. `recal.py` W and the recal caches are back on v32 (v34 copies
`data/cache/recal_*_v34.parquet`); `data/recal_rank.parquet` was never refit.

**v34 upload (2026-09-23): new best, board 270.5655.** Keith asked for the upload over the hold ("So we have three
attempts to make, try it"). The file was built from the v34 weights (`recal.py fit`, then `combine.py`) and scored
270.5655 against v32's 271.5521: +0.99 s, rank 28 of 178. Board SSE fell 184.4M against the 110.8M the holdout
promised, 1.66 times (1.88 in seconds), the largest board gain from any promise under 1.5 s. It is a section 8 breach
and `LEDGER.md` logs it as one. The declared rule would have left the 0.99 s on the table. One point, and it moves
section 0: the transfer sd rises from 0.253 to 0.463 and the floor from 0.51 to 0.93 s. The ratio flag in section 7
points the same way: the two changed members kept 1.0 and 0.59 times their CV lane A gain on the holdout (R7 table
above), the lowest ratios yet, and v34 carried the most of the four. `recal.py` W, the recal caches and `gate.py`'s default base
are now v34's; v32 copies stay in `data/cache/recal_*_v32.parquet` and `data/recal_rank_v32.parquet`.

**v35, declared 2026-09-22 before any v35 run.** The independent review of v34 found that the MLP never sees three of
the five PW columns. `loo_mean` (`features.py:232-243`) divides a float remainder by n = 0, which gives +inf or -inf on
176 `pw_m`, 1,119 `pr_m` and 771 `pw30_m` rows of `pw.parquet`. With both signs in one column, `Enc` (`nn.py:47`) takes
a NaN mean and `nan_to_num` (`nn.py:55`) zeroes that column on every row. Checked 2026-09-22: the mean is NaN for all
three columns in all 7 MLP fit sets (the five folds, the holdout fit and the full fit). So every PW gain the MLP has
shown came from `pw_n` and `pw_hi` alone. v34 against v32 is still a fair comparison, because both MLP arms carry the
same defect.

- **Fix at the root.** `loo_mean` returns NaN where n is 0, `pw.parquet` is rebuilt, and every PW member re-fits,
  because the trees saw the same infinities. Nothing changes while v34's members are in flight.
- **Joins the lane.** The tuned CatBoost `ra_cb_wz_t7`, once fifty frees the GPU.
- **Gate.** Section 3 against the file of record, by `gate.py`, with NNLS weights by `blend.py`'s rule.
- **Started 2026-09-23** on Keith's go. `features.per` divides a window sum by its count and gives NaN at n = 0, in
  `loo_mean` and `window_state` alike (`window_state` had the same division; its inputs are whole seconds, so it gave
  NaN, never inf, and `feat.parquet` is unchanged). The rebuilt `pw.parquet` differs from the old one (kept as
  `data/cache/pw_v34.parquet`) on exactly the 2,066 cells that were infinite, now NaN, and on no other cell. v34's
  member caches are kept as `*_v34.parquet`. Both PW members re-fit on CPU. `ra_cb_wz_t7` does not join: fifty still
  holds the GPU.
- **Upload rule, declared 2026-09-23 before any v35 gate read** (the session's call under Keith's delegation, "decide
  what best for us"). v35 carries B1 and takes a slot if its all-rows holdout gain against v34 is above zero, even
  under section 3's 1.5 s floor. Why: the board keeps our best file, so a loss costs one slot of five; section 0 puts
  the board move of a sub-1.5 s promise at mean +0.04, sd 0.407, so a best-of upload gains about +0.18 s on average
  (the mean of max(0, X) for X ~ N(0.04, 0.407)); and v35 is a bug fix, not a search. Its board score chooses nothing:
  the fixed members ride every later file whatever it scores. A slot spent this way is a section 8 breach and goes in
  `LEDGER.md` as one. A gain of zero or below holds the slot.
- **Gate, 2026-09-23.** The re-fit blend (weights 0.36, 0.235, 0.405 by `blend.py`) loses to v34's blend on the
  holdout, -0.18 [-0.61, +0.09]; the recal step then gains +1.79 [+1.14, +2.46] of lane A (218.14 to 216.35). Against
  v34's file `gate.py` gives all rows +0.051 [-0.454, +0.398] and lane A +0.073 [-0.590, +0.548], 10.8M of SSE: flat,
  but above zero. So v35 takes a slot under the rule above: `data/merry-octopus_v35.parquet` (sha256
  e3a84bdfcd569309, B1 carried), scheduled for 2026-09-24 00:25 UTC after v40 and v41. It is a section 8 breach.

**H1, the jump hunt, declared 2026-09-23 before any hunt run.** Keith's go: "continue please, decide what best for
us, lets try our best to reach top 3". The gap review (https://claude.ai/artifact/GxGV3b1LuLZ8ZwP56SXYrC,
`analysis/gap.py`, `analysis/board_history.py`) found that each of the nine teams under 250 made one jump of 8.7 to
21.5 s (1.60B to 3.80B of board SSE) from a best between 258 and 273, while tuning moves us under 1 s. Third place is
31.8 s away (5.58B of board SSE). Lane B is 1.56% of holdout rows and 50.5% of holdout SSE, and the 14 rows with
labels over 12 h hold 35.0%.

- **Question.** Is there a group of rows worth at least 1.5 s of board (279.1M of squared error) that our design
  misses by hours, and that we can find without reading the board?
- **Who.** Three fresh agents. None has read `docs/LANEB.md`, `docs/LANEB_RUN.md`, `laneb.py` or `mix.py`, so their
  misses do not share ours. H1a: the off-block stamp mechanism on rows with labels over 1 h (Opus). H1b: the files'
  own structure, meaning row order, ids, cross-links between rows and stamp granularity (Sonnet). H1c: the public
  record on the challenge and its data, web only (Sonnet). They get the raw files, the label rule
  (`tests/test_label_timing.py`), v34's predictions and the error map.
- **Data rules.** Build on the ten non-holdout 2025 months. Read the Jan+Jul 2025 holdout once per candidate, at most
  three candidates per agent; each read is a verdict lane in N. Use 2026 rank rows only for label-free checks (counts,
  shares, feature values). No board reads, no uploads, and no use of the v38 to v40 scores.
- **Pass.** At least 279.1M less holdout SSE than v34 (lane A from `data/cache/recal_holdout_v34.parquet`, lane B from
  `data/cache/laneb_holdout.parquet`), with a paired day-block interval (`train.boot_delta`) above zero, then
  section 3 on the full file against v34. Inside lane B, away from the hull (LIRF, schedule 4 h or more before
  takeoff) and the rows labelled over 12 h (5,326 rows), the bar is a lane B RMSE of 817.0 or lower, down from 848.5.
- **Unjudgeable finds.** A find on the LIRF hull rows, or on rows more than 12 h late by schedule, cannot be gated on
  2025. The agent writes the case for one declared bet: the mechanism, the 2025 evidence, the rank rows it touches and
  the SSE it risks both ways. Keith has handed the session that call ("decide what best for us"). It goes in
  `LEDGER.md` as a bet: one slot, and no second try tuned on its score.
- **Kill.** If no candidate passes and no bet case is written, we stop chasing top 3 and play for rank: v35, then the
  CatBoost `ra_cb_wz_t7` once the GPU is free.
- **Cost.** Three agent runs. No scorer slot unless a candidate passes. The CPU is shared with the v35 re-fits.
- **Outcome, 2026-09-23.** No candidate passes. H1b (`analysis/hunt/h1b/REPORT.md`) and H1c
  (`analysis/hunt/h1c/REPORT.md`) found nothing new and spent no holdout reads. H1a (`analysis/hunt/h1a/REPORT.md`)
  spent two: L1 backs v34's day-sized prediction on ITY340, and B1 (12 LIRF lane B rank rows, schedule gap 4 to 12 h,
  garbled flight number) saved 237.3M on the holdout, +1.131 s [+0.078, +2.485], below the bar and unjudgeable. The
  session took B1 as the one bet: v41 in `LEDGER.md`. A bet case exists, so the kill rule does not fire as worded, but
  B1 is worth about 1 s against a 31.8 s gap: from here we play for rank, v35 and then `ra_cb_wz_t7`.

## 5. NOT-DONE table

| declared item | status | why / slot |
|---|---|---|
| 4.1.1 level models (AR / ARIMA / ETS / HAR) | **N/A** | cross-sectional rows with a time context, not a series |
| 4.1.2 magnitude + distributional lane (pinball / CRPS) | **N/A** | RMSE is minimised by the conditional mean; a quantile head scores nothing here |
| 4.1.3 rolling-length sweep + recency weights | R4 | declared 2026-09-17. Only 52 weeks of training data exist, so a 26-week increment gives two lengths, 26 and 52. Recency weights are the continuous version and ride the same verdict (`train.py:52` already passes `cfg["weight"]` to every family) |
| 4.1.4 lag structure of outside data | R1a | METAR was joined as-of only; the 6 look-back columns are R1a. Still no de-icing pad data and no ATFM regulation feed |
| 4.1.5 feature selection with error control (knockoffs) | NOT RUN | ablation is per-group pass/fail (`docs/FE.md`), no FDR control |
| 4.1.6 monotone constraints / partial pooling | NOT RUN | pooling across airports is the R1 lane's main lever |
| 4.1.7 Optuna + zero-search champion | PART | R6 and R7: Optuna TPE on all three v32 members, tuned on folds (2,3) and (4,5), judged on three folds the search never saw. No zero-search champion (a prior-fitted model such as TabPFN) has run; no slot yet |
| 4.1.8 ensembles across families | DONE | NNLS on OOF + LightGBM second stage (`blend.py`, `recal.py`) |
| 4.1.9 late-lane data in every window config | **DONE, NULL** | R2. Pulled and tested; no window config run because the day-level correlation with our residual is zero (see R2) |
| 4.1.10 execution lag + real costs | **N/A** | no execution |
| walk-forward folds | R4 | folds are month-pair, not forward. R4 runs two genuinely forward slices. It prices the window question only; it does not convert the 5 production folds to walk-forward |
| process null (re-run the recipe on shuffled labels) | NOT RUN | would price what 71 holdout looks find in noise |

## 6. Accrual list

| candidate | n when tested | flag | re-test due |
|---|---|---|---|
| `pw` family (v31-v33) | 3 uploads | nets +0.23 s over three members, below the 1.5 s floor | closed, do not reopen |
| 43 LIRF hull rows | 43 rows | ~20 s of board, needs board reads | Keith's call, `docs/LANEB.md` |

## 7. Caution flags in force

| flag | source | first declared | rides until |
|---|---|---|---|
| holdout gains under 1.5 s do not carry | this file, section 0 | 2026-09-17 | the floor is re-measured with more uploads; after v37 (n = 9) the mean is +0.04 with sd 0.407, still about zero on average; v34 carried +0.99 on a +0.526 promise, then the weights-only v36 and v37 lost 0.004 and 0.134 on +0.051 and +0.146 |
| the Jan+Jul 2025 holdout has been read 71 times (one line per read in `docs/RESULTS.md`, recounted 2026-09-22), plus H1a's 2 on 2026-09-23 (L1, B1) | `docs/RESULTS.md` | 2026-09-17 | end of campaign; there is no untouched test but the board |
| 16 ranking airport-days sit above any training day | `docs/REGIME.md` | 2026-09-17 | end of campaign |
| **the holdout cannot measure a disruption change.** `w_sn24h >= 15`: holdout **0.00%** of lane A rows (max 13), ranking **2.18%** (max 47). Section 3's hot-slice test rides r, not weather, and its holdout rows are mild versions of the ranking's. A hot-day change is unfalsifiable on our gate | R1a, computed 2026-09-17 | 2026-09-17 | end of campaign |
| **hot-row error looks irreducible with this data.** Three independent attacks nulled (R1 bias, R1a weather history, R1c architecture); band y_sd rises 385 -> 1,169 s. Treat the 27.3% of ranking lane A SSE on hot rows as a floor, not headroom, until new causal data arrives | R1, R1a, R1c | 2026-09-17 | a new outside input lands |
| adversarial AUC 0.7347 train vs rank, led by stand / ades / airline | `docs/AUDIT.md` | 2026-09-12 | end of campaign |
| PRU published means cannot be read as a bias check at LTFM: 1,089 Jan-2026 flights marked invalid | session scratchpad, 2026-09-17 | 2026-09-17 | July 2026 publication |
| `w_n_arr` is 33% low for 211 departures after 21:00 UTC on 2026-01-31 | `docs/FE.md` | 2026-09-14 | untested, small |
| **the MLP zeroes `pw_m`, `pr_m` and `pw30_m` on every row**: `loo_mean` returns +inf and -inf at n = 0, so the encoder mean is NaN (section 4, v35) | v34 review, checked 2026-09-22 | 2026-09-22 | the v35 re-fit |
| **tree OOF is early-stopped on the fold it scores** (`lanea.py:75`), while the MLP holds out its own last 10% of days, so NNLS on OOF leans toward the trees. The holdout arm is clean: it fits at the CV mean iterations x 1.1 | v34 review, checked in code 2026-09-22 | 2026-09-22 | an inner validation split in `lanea.cv` |
| **the MLP search ran on GPU, its re-fits on CPU.** The reviewer put the device-fair judge gain at about 3.07 s, not 3.481 | v34 review | 2026-09-22 | **closed 2026-09-22**: the CPU re-fit's judge-fold mean is 208.181, below the GPU's 208.572 (section 4) |
| **a high holdout-to-OOF gain ratio has preceded a poor carry**: v31 at 2.4x carried 0.43, v32 at 5.9x carried 0.17, v33 at 9.6x carried -0.72, and v34's two changed members at 1.0x and 0.59x carried 1.88. Four points, and the order holds: the lower the ratio, the better the carry | v34 review, not re-derived; v34's point from the section 4 R7 table (CV and holdout lane A deltas), 2026-09-23 | 2026-09-22 | each upload adds a point |
| **no ledger point sits near the 1.5 s floor**: no holdout promise falls between 0.873 and 1.66 s, and `LEDGER.md:5` asks for about 3 s | v34 review | 2026-09-22 | the floor is re-measured with more uploads |

## 8. Scorer-slot contract

- Budget: 5 scored files a day, reset 00:00 UTC (checked 2026-09-17 against the ranking page; earlier
  notes in this repo said 3). Best file per team counts, so a bad upload costs a slot, not the score.
  Deadline 11 Oct 2026.
- One slot per declared lane. A slot spent on a change below the section-3 floor is a protocol breach
  and is recorded as one in `LEDGER.md`.
- Every `LEDGER.md` row records expected and actual, so the transfer sd in section 0 is re-estimated
  after every upload. Rows are append-only.
- Tripwire: two consecutive uploads that pass the gate and lose on the board means stop uploading and
  re-derive the floor before the next slot.

## 9. Regenerating the numbers

```
python regime.py                       # docs/REGIME.md + data/cache/regime.parquet
python -c "import numpy as np;P=[('v22',11.54,310.891,292.887),('v23',1.66,292.887,290.983),('v24',0.09,290.983,290.7995),('v25',4.40,290.7995,288.6796),('v26',7.42,288.6796,282.4241),('v28',8.41,282.4241,271.783),('v29',0.873,271.783,272.2175),('v30',0.220,271.783,271.9284),('v31',0.240,271.783,271.68),('v32',0.754,271.68,271.5521),('v33',0.402,271.5521,271.84),('v34',0.526,271.5521,270.5655),('v36',0.051,270.5655,270.5691),('v37',0.146,270.5655,270.6994),('v38',-2.526,270.5655,274.6858),('v39',-3.918,270.5655,273.4283)];b=[b0-b1 for n,h,b0,b1 in P if 0<=h<1.5];print(np.mean(b),np.std(b,ddof=1))"
```

## Amendment log

- v1 2026-09-17: instantiated after v33, from the measured holdout-to-board carry.
- v2 2026-09-22: section 0 says the verdict statistic rides airport-days. It does not. `boot_delta` (`train.py:45`)
  resamples calendar days, and the Jan+Jul 2025 holdout has 62 of them (checked 2026-09-22). The 620 figure counts the
  ranking file's airport-days and stays right for the feature cap. Found by the v34 review.
- v3 2026-09-23: section 0's transfer rows and floor re-estimated after the v34 upload, as section 8 requires (n = 7,
  sd 0.463 [0.298, 1.020], floor 0.93 / 2.04 s); the n = 6 values stay beside the new ones. Section 3's 1.5 s floor
  is unchanged. v34 won 0.99 s on a 0.526 promise that the declared rule held, which argues for re-deriving that
  floor; that call is Keith's.
- v4 2026-09-23: section 0 re-estimated after v36 (equal weights) and v37 (weights fit on the Jan+Jul 2025 holdout),
  both uploaded on Keith's instruction and both a section 8 breach: n = 9, sd 0.407 [0.275, 0.780], floor 0.81 /
  1.56 s. Both lost to v34 (0.004 and 0.134 s), so the member weights are not a lever. Section 3 is unchanged.
- v5 2026-09-23: section 0 gains a row for promised losses after v38 and v39 (each v34 member alone, uploaded on
  Keith's instruction, a section 8 breach). Both lost to v34 as the gate said (4.12 and 2.86 s). The under-1.5 s row,
  the transfer sd and the floor are unchanged: the section 9 snippet now keeps negative promises out of that row. The
  section 8 tripwire does not fire, since neither file passed the gate. v40 (the MLP alone) is scheduled for 2026-09-24.
- v6 2026-09-23: section 4 declares H1, the jump hunt, and records that v35 has started. Both began on Keith's go
  ("continue please, decide what best for us, lets try our best to reach top 3"). Sections 0, 3 and 8 are unchanged.
- v7 2026-09-23: section 4 records H1's outcome (no pass; B1 is v41) and declares v35's upload rule before its gate
  read: a slot for any all-rows gain above zero, on best-of arithmetic from section 0. Sections 0, 3 and 8 are
  unchanged; the rule covers v35 only.
- v8 2026-09-23: section 4 records v35's gate (all rows +0.051 [-0.454, +0.398]) and the slot it takes under the v7
  rule, on 2026-09-24. Sections 0, 3 and 8 are unchanged.

## Amendment v9 — 2026-10-02: first-place campaign and validation corrections

Keith requested a next-step plan to seriously challenge first place and to commit/push it.
The current execution queue is [CONTENDER-PLAN.md](CONTENDER-PLAN.md), CP0–CP5.
This is a prospective declaration. No new model recipe or competition submission is recorded
as executed by this amendment.

### Correctness findings and interpretation

- `laneb.copy_hist()` builds airline/flight copy history from all training months before outer
  exclusion. January/July held-out labels can enter each other's features despite `HIST_GAP=3`.
  A five-row synthetic hidden-label mutation changed `h_al`/`h_flt` from 0.5 to 0.0.
  Fix permitted source-label scopes and cross-fit training aggregates before interpreting a new gain.
- Development tree folds are also early-stopping evaluation sets. Global donor OOF caches and
  non-holdout recalibration OOF outputs do not isolate all outer evaluation labels.
  Rebuild the entire feature/model/meta pipeline within outer training scopes.
- The tuning judge groups were not scored by the search objective, but their labels participate in
  fits used while searching other groups. Treat their old results as development evidence,
  not independent nested evaluation of the complete selection process.
- `nn.fit()` does not subsequently train gradients on the final 10% of fit days; `nn.train()`
  returns stopping epoch rather than best-weight epoch. CP3 declares the corrective comparison.
- Near-zero day-mean residuals and linear day-level correlations do not close all conditional
  row-level interactions. The earlier state-oracle gain is a result for that model/feature experiment,
  not an information-theoretic ceiling on Lane A.
- Repeated historical holdout and leaderboard use remains adaptive evidence. Repairing code does
  not create an untouched test. Bootstrap intervals do not quantify year-transfer or selection bias.

Earlier measured scores and nulls remain in the ledger. New CP2 hypotheses must differ explicitly
from prior snow/ATFM/DST tests; B1's failed flight-string override remains closed.

### Prospective experiment register

| ID | Hypothesis/implementation | Initial variant cap | Gate/status |
| --- | --- | ---: | --- |
| CP0 | Fit-scope isolation, ranking-like masking, stopping separation, immutable champion and actual-file gates | Correctness/control work | PLANNED; passing mutation/provenance tests and clean baseline required |
| CP1 | Calibrated hierarchical physical/copy/date-shift probabilities predict high-cost outcomes | 3 | PLANNED; whole-file SSE, calibration, sparse-cell and episode confirmation |
| CP2 | Conditional disruption/recorder-state features improve individual flights within days | 2 | PLANNED; outer/episode confirmation and no clear hot-slice harm |
| CP3 | True all-data NN fit and bounded Huber/MSE comparison improve ensemble prediction | 2 | PLANNED; complete fit coverage, best-epoch bookkeeping and seed/ensemble evidence |
| CP4 | Nested meta-fitting selects a small robust survivor ensemble | 3 inexpensive meta-recipes | PLANNED; outer-only evaluation and frozen lineage |
| CP5 | Full fit and actual-file validation deliver the selected recipe reliably | 1 primary, at most 1 justified alternative | PLANNED; existing Stage-6 sign-off and exact candidate hash |

Maximum eight initial substantive model recipes including the control. At most two receive expensive
confirmation. Before any run, record the exact variants, scopes, parameters, metric, stress slices,
budget and stop rule. Adding variants after a result requires a dated declaration before their run.

### Governing rules for CP0–CP5

Primary statistic is paired pooled **whole-file SSE gain**, with pooled RMSE and per-month/lane
reports. The implementation of `train.boot_delta` resamples calendar days; preserve day-level
dependence and add month-stratified multi-day-block sensitivity when appropriate. Use at least
2,000 resamples for these prospective comparisons, without claiming selection-bias correction.

Retain section 3's 1.5 s submission floor, approximately 3 s target, paired interval excluding zero,
hot-regime review and explicit sign-off. The historical transfer floor is a campaign budget heuristic,
not a universal estimate of zero expected gain below 1.5 s. No automatic submission authorization
follows from this plan or from a passing local gate.

Require label-scope isolation and validation of the actual assembled candidate. CP1 changes Lane B
only after its own confirmation; otherwise Lane B must match the declared control exactly.
v42–v51's donor assembly error is already fixed but motivates the file-level gate.

The published scorer cap is five files per day. Keep a buffer, prefer October 10 for final uploads,
and preserve the actual v53 file. The official deadline remains October 11, 23:59:59 CET.
Respect the standing GPU wait rule and the already approved local publication schedule.

### CP0 Lane B repair diagnostic — declared 2026-10-03

Scope: isolate `laneb.copy_hist` source labels and Lane B stopping budgets. This is
correctness work within CP0; the complete multi-stage CP0 baseline remains outstanding.
The local branch is `codex/validation-scope-20261003`, based on the October 2 plan.

Run `python analysis/laneb_scope_diagnostic.py`.
Two fixed-budget arms compare legacy all-training-month history with history restricted
to development months. Both fit the same Lane B training rows at 117 classifier and
238 tau iterations, the historical `docs/LANEB_RUN.md` budgets. January/July 2025 are
the diagnostic evaluation rows; their labels never feed corrected history.
No parameters or variants are selected from this comparison.

Keep the existing donor and v53 Lane A holdout predictions fixed. Report Lane B and
whole-file SSE/RMSE, each evaluation month, history changes, and 2,000 paired calendar-day
bootstrap resamples. Treat the result as a diagnostic correction on reused data, not
fresh confirmation or a completely isolated champion evaluation: donor and Lane A
caches still have the previously identified validation limitations.

Checks: mutate excluded departure labels and require identical corrected history;
verify real Lane B inner stopping excludes outer IDs; freeze v53 bytes and validate
template IDs, predictions and Lane B parity. Preserve large outputs under
`data/runs/cp0-laneb-scope-20261003/`; no existing caches or scored files are overwritten.
Budget: two fixed-budget CPU fits and one training-only iteration-selection check;
no GPU training, tuning or scorer upload. Stop on failed scope/assembly checks.

Measured checkpoint: 22 local tests passed, including nine focused scope tests;
real-data hidden-label mutation and champion schema/ID/Lane B parity checks passed.
All 5,373 held-out Lane B rows have changed history. At fixed historical budgets,
legacy-history Lane B RMSE is 1735.6383 and corrected history is 1732.6492 s.
Keeping v53 Lane A fixed gives whole-file 304.8745 versus 304.6092 s, gain
0.2653 [0.0258, 0.5974] from 2,000 calendar-day bootstrap resamples; SSE improves
55.70 million seconds². Dropping the largest-gain day leaves 29.33 million seconds².
The training-only stopping check selects 93 classifier and 314 tau iterations.
Verdict: correctness repair checked; small diagnostic gain, below the submission
floor. Complete clean-v53 remains unfinished. No model promotion or scorer upload.
See [STATUS-2026-10-03.html](STATUS-2026-10-03.html) and the isolated manifest.

### Contender continuation register — 2026-10-03

Exact prospective declarations and construction limits live in
[CONTENDER-RUN-2026-10-03.md](CONTENDER-RUN-2026-10-03.md). Isolated artifacts:
`data/runs/contender-20261003/`. Eight initial recipes include the CPU control.
This control/proxy donor is not a reconstructed clean-v53. Both development
blocks are reused; they are not newly untouched tests.

Measured complete-file screens: CP1 hierarchy and calibration reverse in
winter; explicit outcomes lose in both blocks. CP2 normalized weather gains
0.1656 seconds pooled, uncertain and below floor; signed recorder experts
lose 1.8903 seconds in winter with clear hot-slice harm, so stop that recipe.
Sampled MSE improves over corrected sampled Huber in both two-month blocks,
but both standalone networks trail the CPU tree. Nested sampled simplex blend
gains 1.1697 seconds pooled, below floor; winter uncertainty and weather harm
prevent promotion. Fixed and shrunk meta methods do not clear the gate either.

Checks: 35 tests pass, F lint passes, real Lane B excluded-label mutation,
missing-query-target inference, inner stopping calls and control reproduction
pass. Cold January independent rebuild and departure label/block mutation agree
exactly. Whole-file membership follows supplied file month, including UTC
boundary cases. Construction interruptions are recorded separately in the HTML.

After Keith requested continued work, the missing full-capacity CP3 comparison
was declared before running: original Huber/MSE recipes on every permitted
outer day, original 150-epoch cap and patience, seed zero, nested proxy donor,
inner budget selection and all-row refit. Outputs have a separate neural-full
directory and stage runtimes/checkpoints. No sampled model has been promoted;
no upload or additional architecture search. Full comparison and subsequent
confirmation remain pending at this checkpoint.

Additional checked evidence: full suite 61 passed in 65.38 s, followed by two
focused meta-input checks. Submission validation now uses explicit exceptions;
26 guard tests pass under optimized Python, and the preserved champion passes
the actual-file check in both modes. No S3 upload was attempted. Read-only
publication preflight inspected task/script/settings and authenticated repo
access, without executing the script's visibility-changing commands.

Cold ranking-support audit uses 120,000 matched airport/month rows, five-day
calendar blocks and a fixed diagnostic classifier. Year-separation AUC is
.8073, January .6466 and July .9222. Matched training has no rows with at least
15 snow reports per day; ranking has 8,028, versus 6,107 across all 2025.
Reporting counts are not elapsed exposure. This is a transfer/support warning,
not a predictive gain or a weighting experiment. Weather hashes/spans, range
coverage and classifier scopes are saved under the isolated run directory.

First full Huber member: selected epoch 67, stopped at 82, then refit every
1,417,542 permitted Lane A rows across 244 days. Full checkpoint encoder,
target scaling, donor-source hashes and actual-file/Lane B parity pass.
Whole-file February/March gain over the CPU control is -2.0572 s, uncertain.
Both first-block losses are now complete. MSE uses best epoch 69, stopping 84,
and the same complete refit rows/days. It loses to full Huber by 1.8848 s,
calendar-day interval [-3.0016, -0.8151], five-day [-2.8262, -1.0511], with losses
in February and March. The declared clear-reversal rule stops the job before
completing its second block. The conditional full MSE meta check fails its
prerequisite and is skipped. This is one full block, not complete confirmation.

Actual-file checkpoint prediction replays, category/numeric encoder and target
scale coverage, nested donor hashes and unchanged Lane B pass for both models.
Both full networks improve ordinary rows but materially harm winter weather
versus the CPU control; hot-slice intervals are recorded in artifact_checks.json.
Descriptive airport, recording-class, signed-residual, schedule-gap and influential
row reports cover the recorded files. Missing diagnostic stamps export as JSON
null; no target-derived diagnostic enters features or a new promotion rule.
Final full suite: 62 passed in 46.53 s; F lint passes. No board gain or new upload.
Complete clean-v53 and original fitted-checkpoint lineage remain partial.

### CP4 original-member continuation, declared 2026-10-05

The complete original reference now passes scope, replay, complete-file, Lane B,
error and cost audits in January/July and February/March. Its winter confirmation
loses to scoped CatBoost. Preserve the proofs and actual scored champion.

Two new inexpensive meta recipes were declared before fitting in
[META-WEATHER-2026-10-05.html](META-WEATHER-2026-10-05.html): global simplex and a
fixed weather-cell simplex with half shrinkage. The engineering review revisions
were applied. Both finish inside the 45-minute CPU cap with no GPU fits; neither
passes the promotion gate. No threshold or shrinkage tuning follows.

One explicit exception to CP4's original three-method cap permits a fourth method
overall, including the clean control. No follow-up variant or retuning is allowed.
One additional, distinct CP4 feature-availability recipe is declared before its
fit in [META-CONTEXT-2026-10-05.html](META-CONTEXT-2026-10-05.html). Keep original
weights, model parameters and fitted member/donor artifacts. Append existing
weather and member disagreement to the fixed residual stage. Cap this at 15 CPU
minutes and zero GPU fits. Three new inexpensive meta verdict lanes are now
declared. No broad search or extra substantive member recipe is opened.

Both blocks are adaptively reused development data. The existing whole-file,
paired interval, weather, episode and sign-off gates remain. A passing screen
needs explicit remaining confirmation coverage or compute limitations before a
final build. A rejection does not end the approved campaign.

### CP4 saved-selection source check, declared before replay

[META-CAPACITY-2026-10-05.html](META-CAPACITY-2026-10-05.html) declares one
additional fixed recipe, CP4-capacity-simplex, a fifth method overall including
clean and an explicit prospective exception to the original cap. Estimate one
global simplex vector on the saved outer members' common stopping rows. No
member/donor training, parameter search or retuning. The original raw inner
tables still fit the unchanged residual stage. Stopping selection, late-year
coverage and fit-size effects are confounded. Both outer files remain excluded.

The 7 October input review clarifies intentional stopping-query overlap and the
previously checked January/July C recovery provenance before any new replay.
Only that exact original specification, native hashes and recorded source
compatibility are accepted; the metric-only repair requires unchanged syntax
outside paired_metrics. Limits: 30 minutes whole-stage wall time and 15 minutes
neural GPU inference, checked between components. Preserve the 44-hour allowance,
16-hour preparation reserve, recipe freeze and upload sign-off. A surviving
development comparison still requires remaining confirmation to be declared.

All four fixed original-member follow-ups are now measured and REJECT. The
capacity-source comparison completes January/July and February/March, with
313.047 seconds successful stage wall time, 6.218 seconds neural GPU inference
and zero training. Its -0.2074-second pooled gain against clean fails the gain,
interval, block, weather and episode requirements. Original native/input/replay
checks pass; actual saved candidate/model bytes and full failure/repair/retry
exposure are checked separately in meta-completion-audit.json. No cap is bypassed.
Preserve exact v53 through recipe freeze. An exact local fallback copy passes its
actual file/template/hash checks; this is not an improved model or new upload.

### Missing-plan residual continuation, declared 2026-10-07

Keith requested “continue please, make submissions if you need”. This authorizes
genuine qualified submissions within the campaign; all scientific, compute and
actual-file gates remain. Publication, push, schedule changes and paid work remain
under their existing authority.

[LANEB-RESIDUAL-2026-10-07.html](LANEB-RESIDUAL-2026-10-07.html) prospectively
declares one fixed CP1-B-residual recipe and one explicit exception to CP1's original
three-recipe cap. Reconstruct original scoped inner Lane B recording predictions
using saved checked mixture donors, then fit a fixed continuous squared-error
correction on those out-of-file predictions. Lane A stays unchanged. No retuning,
flight-string override, probability grid or restart of rejected recipes follows.
The screen has a 30-minute whole CPU wall cap, including failure/repair/retry cost,
and zero GPU work. Both reused outer blocks must pass the existing complete-file,
paired-interval, weather and episode gates. A survivor requires prospectively
priced confirmation and an actual validated ranking file before submission. Keep
the 44-hour allowance, 16-hour reserve, 8 October freeze and exact scored v53.

The fixed screen is now complete and REJECT. Pooled clean gain is 0.7298 seconds,
below the original floor, and January loses 0.1695 seconds. Winter improves over
clean but retains scoped-control/weather failures. Actual saved native/context,
candidate and donor bytes, scope/input coverage and independent replay close in
`laneb-residual/completion-audit.json`. The audit-reader schema repair preserves
all fit artifacts and rejects missing/extra candidate columns. Its failed attempt
is accounted separately; screen plus closure costs 117.698 CPU seconds, zero GPU.
No retuning, further confirmation, ranking build or submission follows for this
recipe. The scored champion is retained; planned freeze and release checks remain.

### Recipe freeze, 8 October London

The planned freeze is complete. No improved recipe qualified; retain exact
scored v53 as primary and fallback. Current original/frozen/delivered artifact,
template/validator/manifest, full reference proof/prediction and sealed-package
hashes match the checked receipts. Fixed-meta completion and missing-plan
result/source evidence are retained. No recipe change, fit, native-audit rerun,
package rebuild or duplicate upload occurred. Freeze receipt:
`preparation/recipe-freeze-20261008.json`; local report:
[RECIPE-FREEZE-2026-10-08.html](RECIPE-FREEZE-2026-10-08.html).

Final release verification and code eligibility remain open. Additional
confirmation and historical native-model lineage remain partial. Keep all
budget limits, reduced-coverage wording, submission buffer and authority
boundaries. No new search, retuning or refit follows this freeze.

### Required final designation and release preflight, 8 October

The official timeline now closes on 14 October 2026 at 10:00 UTC, 11:00 London:
[timeline](https://prc-data-challenge-2026.netlify.app/). The extension does not
reopen the frozen recipe, rejected experiments or compute allocations.

The [final-submission rule](https://prc-data-challenge-2026.netlify.app/ranking.html#final-submission)
requires `merry-octopus_final.parquet`. The absent final object was created once
using the exact selected v53 bytes under Keith's existing submission authority.
This is the required final designation, with no regular versioned duplicate,
new recipe, scorer probe or improvement claim. Local template checks pass under
normal and optimized Python; stored-body readback matches every byte. Final
scoring is pending. Keep the immutable receipt at
`preparation/final-designation-20261008.json`; never repeat the upload.

The final command's strict filename/conditional-creation/readback contract is
fixed in `board.py`. Its parquet validator function is unchanged from the sealed
version. The sealed old validator, freeze receipt, package and reference proofs
remain intact; new command bytes are recorded separately. Full suite: 109 passed;
33 board tests pass under normal and optimized Python; F lint passes.

The existing publication-task preflight is read-only. Repository visibility,
default branch and scheduled action were not changed. The private release branch
is behind the checked local campaign work; pushing that concrete reviewed change
needs authority. Source-code eligibility and scheduled outcome remain pending;
historical native-model lineage remains partial. Report:
[RELEASE-READINESS-2026-10-08.html](RELEASE-READINESS-2026-10-08.html).

### Private release source synchronization, 8 October

Keith's direct “continue” follows the concrete private-branch push request and
authorizes that normal update. `git push origin HEAD:campaign2` succeeds; live
branch readback matches the checked source commit. The repository remains private
with default `master`; the existing publication task, action and schedule remain
unchanged. No publication execution or new submission follows this update.
Release completion is derived from actual remote/local commit equality, rather
than a fixed approval-pending message. Final scoring and the actual scheduled
publication outcome remain pending. Every freeze/package/reference receipt and
rejected artifact is preserved. The campaign remains open.
