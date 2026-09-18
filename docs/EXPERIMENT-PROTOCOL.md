# EXPERIMENT PROTOCOL - prc26 (v1, 2026-09-17)

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
| under 1.5 s | 6 | mean **-0.08**, sd 0.253, **3 of 6 negative** |

- Transfer sd: **0.253 s** of board, 95% CI [0.158, 0.620] (n = 6, chi-square).
- **Detectable-effect floor: 0.51 s of board** at the point estimate, **1.24 s** at the sd's upper
  bound. Any holdout gain that projects below this is noise-sized and must be labeled so.
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
  the stamp mixture's 253 (`PLAN.md:31`). Current: 217.18 lane A holdout, board 271.5521.
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

## 5. NOT-DONE table

| declared item | status | why / slot |
|---|---|---|
| 4.1.1 level models (AR / ARIMA / ETS / HAR) | **N/A** | cross-sectional rows with a time context, not a series |
| 4.1.2 magnitude + distributional lane (pinball / CRPS) | **N/A** | RMSE is minimised by the conditional mean; a quantile head scores nothing here |
| 4.1.3 rolling-length sweep + recency weights | R4 | declared 2026-09-17. Only 52 weeks of training data exist, so a 26-week increment gives two lengths, 26 and 52. Recency weights are the continuous version and ride the same verdict (`train.py:52` already passes `cfg["weight"]` to every family) |
| 4.1.4 lag structure of outside data | R1a | METAR was joined as-of only; the 6 look-back columns are R1a. Still no de-icing pad data and no ATFM regulation feed |
| 4.1.5 feature selection with error control (knockoffs) | NOT RUN | ablation is per-group pass/fail (`docs/FE.md`), no FDR control |
| 4.1.6 monotone constraints / partial pooling | NOT RUN | pooling across airports is the R1 lane's main lever |
| 4.1.7 Optuna + zero-search champion | PART | `tune.py` on CV folds only. No zero-search champion |
| 4.1.8 ensembles across families | DONE | NNLS on OOF + LightGBM second stage (`blend.py`, `recal.py`) |
| 4.1.9 late-lane data in every window config | **DONE, NULL** | R2. Pulled and tested; no window config run because the day-level correlation with our residual is zero (see R2) |
| 4.1.10 execution lag + real costs | **N/A** | no execution |
| walk-forward folds | R4 | folds are month-pair, not forward. R4 runs two genuinely forward slices. It prices the window question only; it does not convert the 5 production folds to walk-forward |
| process null (re-run the recipe on shuffled labels) | NOT RUN | would price what 61 holdout looks find in noise |

## 6. Accrual list

| candidate | n when tested | flag | re-test due |
|---|---|---|---|
| `pw` family (v31-v33) | 3 uploads | nets +0.23 s over three members, below the 1.5 s floor | closed, do not reopen |
| 43 LIRF hull rows | 43 rows | ~20 s of board, needs board reads | Keith's call, `docs/LANEB.md` |

## 7. Caution flags in force

| flag | source | first declared | rides until |
|---|---|---|---|
| holdout gains under 1.5 s do not carry | this file, section 0 | 2026-09-17 | the floor is re-measured with more uploads |
| the Jan+Jul 2025 holdout has been read 61 times | `docs/RESULTS.md` | 2026-09-17 | end of campaign; there is no untouched test but the board |
| 16 ranking airport-days sit above any training day | `docs/REGIME.md` | 2026-09-17 | end of campaign |
| **the holdout cannot measure a disruption change.** `w_sn24h >= 15`: holdout **0.00%** of lane A rows (max 13), ranking **2.18%** (max 47). Section 3's hot-slice test rides r, not weather, and its holdout rows are mild versions of the ranking's. A hot-day change is unfalsifiable on our gate | R1a, computed 2026-09-17 | 2026-09-17 | end of campaign |
| **hot-row error looks irreducible with this data.** Three independent attacks nulled (R1 bias, R1a weather history, R1c architecture); band y_sd rises 385 -> 1,169 s. Treat the 27.3% of ranking lane A SSE on hot rows as a floor, not headroom, until new causal data arrives | R1, R1a, R1c | 2026-09-17 | a new outside input lands |
| adversarial AUC 0.7347 train vs rank, led by stand / ades / airline | `docs/AUDIT.md` | 2026-09-12 | end of campaign |
| PRU published means cannot be read as a bias check at LTFM: 1,089 Jan-2026 flights marked invalid | session scratchpad, 2026-09-17 | 2026-09-17 | July 2026 publication |
| `w_n_arr` is 33% low for 211 departures after 21:00 UTC on 2026-01-31 | `docs/FE.md` | 2026-09-14 | untested, small |

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
python -c "import numpy as np;P=[('v22',11.54,310.891,292.887),('v23',1.66,292.887,290.983),('v24',0.09,290.983,290.7995),('v25',4.40,290.7995,288.6796),('v26',7.42,288.6796,282.4241),('v28',8.41,282.4241,271.783),('v29',0.873,271.783,272.2175),('v30',0.220,271.783,271.9284),('v31',0.240,271.783,271.68),('v32',0.754,271.68,271.5521),('v33',0.402,271.5521,271.84)];b=[b0-b1 for n,h,b0,b1 in P if h<1.5];print(np.mean(b),np.std(b,ddof=1))"
```

## Amendment log

- v1 2026-09-17: instantiated after v33, from the measured holdout-to-board carry.
