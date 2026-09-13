# Lane A as a classification problem (holdout Jan+Jul 2025, has_plan==1, n=339,046)

Scripts: `analysis/lanea_clf.py` (oracle, confusion, dayclust, noisefloor subcommands, cheap,
cache-only) and `analysis/day_feat_test.py` (the feature test, trains LightGBM). Run with
`PYTHONPATH=/c/Users/skf_s/prc26 python analysis/<script>.py [subcommand]`.

## 1. Oracle: how much headroom does a perfect stamp classifier buy

mu backed out from `data/cache/mix_cb_wx_holdout.parquet` as
`mu = (p - p_aobt*z_aobt - p_eobt*z_eobt - p_sched*z_sched) / (1 - p_aobt - p_eobt - p_sched)`
where the denominator exceeds 0.02 (0 rows fell below that cutoff on this holdout, so no fallback
was needed).

Command: `python analysis/lanea_clf.py oracle`

| model | actual RMSE | oracle classifier RMSE | oracle AOBT-only | oracle EOBT-only |
|---|---|---|---|---|
| mix_cb_wx | 249.26 | 201.72 | 242.74 | 224.77 |

| model | actual RMSE | RMSE with copy rows replaced by the exact stamp |
|---|---|---|
| blend_ra_cb+ra_cb_s1+ra_cb_s2+ra_lgb+nn3 | 226.65 | 199.44 |

A perfect stamp classifier (holding the physical regressor's quality fixed) would take the shipped
blend from 226.65 to about 199.4: roughly 12% off lane A RMSE, and about 27 RMS points. Most of
that headroom sits in EOBT: fixing only the AOBT-vs-rest decision buys 6.5 points (249.26 to
242.74); fixing only EOBT-vs-rest buys 24.5 points (249.26 to 224.77). The classifier is doing real
work, but EOBT-copy detection is the lever, not AOBT.

## 2. Classifier quality (mix_cb_wx, argmax vs true class)

Command: `python analysis/lanea_clf.py confusion`

Confusion matrix (rows = true class from `mix.label()`, cols = argmax of the 4-way probability):

| true \ pred | aobt | eobt | sched | other |
|---|---|---|---|---|
| aobt | 4,267 | 1,364 | 139 | 59,893 |
| eobt | 1,630 | 3,634 | 90 | 24,968 |
| sched | 191 | 69 | 589 | 4,341 |
| other | 3,331 | 2,540 | 334 | 231,666 |

Multiclass log loss: 0.6870.

Per-class recall: aobt 0.0650, eobt 0.1198, sched 0.1135, other 0.9739.

The classifier almost never picks a copy class outright (recall 6-12% on aobt/eobt/sched); it
mostly hedges via the probability weights rather than committing, which is why the oracle gap
above is so large relative to how good the log loss looks.

Squared-error share per (true, pred) cell, percent of total lane-A SSE:

| true \ pred | aobt | eobt | other | sched |
|---|---|---|---|---|
| aobt | 0.05 | 0.09 | 5.33 | 0.13 |
| eobt | 0.07 | 0.07 | 18.70 | 0.03 |
| other | 0.38 | 0.45 | 66.03 | 0.48 |
| sched | 0.06 | 0.06 | 7.42 | 0.66 |

Three numbers explain almost all of the error: true-other/pred-other (66.03%, the physical noise
floor, not a classifier problem), true-eobt/pred-other (18.70%, the single largest classifier-fixable
chunk), true-sched/pred-other (7.42%) and true-aobt/pred-other (5.33%). Together the three
"copy row scored as physical" cells are 31.4% of total lane-A squared error: that is the ceiling on
what a better classifier can recover, and it lines up with the oracle numbers in section 1.

## 3. Day clustering of the copy mechanism

Command: `python analysis/lanea_clf.py dayclust`

Between-day dispersion vs a binomial with the airport's own mean rate (airport-days with n>=20):

| class | observed between-day var | binomial-expected var | dispersion ratio |
|---|---|---|---|
| aobt | 0.00073 | 0.00029 | 2.6x |
| eobt | 0.00068 | 0.00015 | 4.6x |
| sched | 0.00005 | 0.00003 | 1.9x |

All three are overdispersed relative to a stable per-airport binomial rate, EOBT most of all
(4.6x): the copy mechanism is not just airport-specific noise, whole days really do move together.

Top 15 airport-days by max(eobt_shr, sched_shr):

| apt | day | n | aobt_shr | eobt_shr | sched_shr | nB | bshare | d_ae med | aobt0 shr | aobt_m5 shr | eobt_m5 shr | d_ae=0 shr | LOBT=EOBT shr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LIRF | 2025-01-29 | 327 | 0.199 | 0.266 | 0.028 | 0 | 0.0 | 120.0 | 0.991 | 0.177 | 0.713 | 0.083 | 0.520 |
| LIRF | 2025-01-22 | 328 | 0.192 | 0.265 | 0.064 | 0 | 0.0 | 60.0 | 1.000 | 0.186 | 0.716 | 0.079 | 0.494 |
| LIRF | 2025-01-20 | 379 | 0.153 | 0.261 | 0.047 | 0 | 0.0 | 180.0 | 1.000 | 0.216 | 0.681 | 0.061 | 0.544 |
| LIRF | 2025-01-28 | 314 | 0.169 | 0.261 | 0.025 | 0 | 0.0 | 240.0 | 1.000 | 0.229 | 0.707 | 0.080 | 0.535 |
| LIRF | 2025-01-14 | 308 | 0.123 | 0.256 | 0.071 | 0 | 0.0 | 240.0 | 0.997 | 0.211 | 0.669 | 0.045 | 0.481 |
| LIRF | 2025-01-07 | 373 | 0.188 | 0.252 | 0.035 | 0 | 0.0 | 120.0 | 1.000 | 0.177 | 0.694 | 0.083 | 0.566 |
| LIRF | 2025-01-18 | 319 | 0.197 | 0.251 | 0.053 | 0 | 0.0 | 60.0 | 1.000 | 0.179 | 0.677 | 0.075 | 0.502 |
| LIRF | 2025-01-13 | 378 | 0.124 | 0.246 | 0.042 | 0 | 0.0 | 450.0 | 1.000 | 0.177 | 0.653 | 0.040 | 0.519 |
| LIRF | 2025-01-05 | 421 | 0.178 | 0.245 | 0.055 | 0 | 0.0 | 240.0 | 0.998 | 0.171 | 0.708 | 0.055 | 0.572 |
| LIRF | 2025-01-01 | 385 | 0.179 | 0.244 | 0.073 | 0 | 0.0 | 180.0 | 1.000 | 0.213 | 0.631 | 0.049 | 0.522 |
| LIRF | 2025-01-25 | 328 | 0.186 | 0.244 | 0.034 | 0 | 0.0 | 120.0 | 1.000 | 0.216 | 0.671 | 0.082 | 0.479 |
| LIRF | 2025-01-08 | 337 | 0.160 | 0.240 | 0.053 | 0 | 0.0 | 120.0 | 0.997 | 0.220 | 0.671 | 0.065 | 0.490 |
| LIRF | 2025-01-19 | 369 | 0.117 | 0.238 | 0.046 | 0 | 0.0 | 300.0 | 0.995 | 0.190 | 0.661 | 0.043 | 0.534 |
| LIRF | 2025-01-23 | 337 | 0.181 | 0.237 | 0.068 | 0 | 0.0 | 180.0 | 0.997 | 0.163 | 0.700 | 0.050 | 0.490 |
| LIRF | 2025-01-31 | 389 | 0.165 | 0.234 | 0.039 | 0 | 0.0 | 180.0 | 0.997 | 0.211 | 0.638 | 0.057 | 0.553 |

The raw top-15 is every January day at LIRF: this airport's baseline EOBT-copy rate (18.4%
overall, see below) dwarfs everyone else's, so ranking by raw share just finds LIRF. Re-ranking by
excess over each airport's *own* mean (`eobt_shr - apt_mean`) separates the airport effect from a
day effect:

| apt | day | n | eobt_shr | apt mean | excess |
|---|---|---|---|---|---|
| EDDF | 2025-01-28 | 468 | 0.154 | 0.072 | 0.082 |
| LIRF | 2025-01-29 | 327 | 0.266 | 0.184 | 0.082 |
| LIRF | 2025-01-22 | 328 | 0.265 | 0.184 | 0.081 |
| LIRF | 2025-01-20 | 379 | 0.261 | 0.184 | 0.077 |
| LIRF | 2025-01-28 | 314 | 0.261 | 0.184 | 0.077 |
| LSZH | 2025-01-28 | 280 | 0.129 | 0.052 | 0.077 |
| LIRF | 2025-01-14 | 308 | 0.256 | 0.184 | 0.072 |
| EDDM | 2025-01-23 | 376 | 0.165 | 0.095 | 0.070 |
| LSZH | 2025-01-27 | 305 | 0.121 | 0.052 | 0.069 |
| EDDF | 2025-01-29 | 498 | 0.141 | 0.072 | 0.069 |

Two findings, both real:
- LIRF has a persistently high EOBT-copy baseline (18.4% of lane A, vs single digits at most other
  airports) that holds across essentially every January day, not one outage. That matches
  ERRORS.md's flag of LIRF as the worst lane-A airport (RMSE 375).
- On top of that, 2025-01-28 shows an excess EOBT-copy spike at three *different* airports (EDDF,
  LSZH, LIRF) simultaneously, suggesting a shared, date-specific driver (e.g. a network-wide
  ops/weather event) rather than a purely local one.

Day-level signature of a high-eobt-copy LIRF day: essentially 100% AOBT-on-the-second
(`aobt0_shr` ~0.99-1.0, vs the sample-wide mean of 0.965), an elevated EOBT-on-5-min-grid share
(0.63-0.72 vs sample-wide mean 0.84 -- actually *below* average here, so 5-min rounding is not the
tell), and a high LOBT==EOBT share (0.48-0.57). These days carry zero lane-B rows (`bshare` = 0),
so they are not "the whole day had no plan" outages; they are specifically an off-block *value*
that leans on EOBT more than usual while a flight plan still exists.

## 4. Does the signature help a classifier

Command: `python analysis/day_feat_test.py`

New features (from visible AOBT/EOBT/LOBT stamps only, never from `y`): per airport-day and per
airport-day-hour (+-2h window, no day-boundary wraparound) share of AOBT on-the-second, AOBT on a
5-min grid, EOBT on a 5-min grid, AOBT==EOBT, and LOBT==EOBT. Baseline = CAT+NUM+DAY (feature
lists in `features.py`); test = baseline + the 10 new columns. LightGBM multiclass, num_leaves=63,
lr=0.05, early stopping 100 rounds, CPU (n_jobs=24). Physical (class 3) rows capped at 200,000 per
training fold for CPU time (both variants use the identical cap and seed, so the comparison is
fair); the mixture RMSE uses one shared physical regressor (LightGBM, same NUM+DAY features, same
cap) per fold so the only thing that differs between variants is the classifier.

CV over `train.FOLDS` (train months, non-holdout) then holdout (Jan+Jul); full numbers also in
`data/analysis_day_feat_test.csv`:

| fold | months | n | logloss base | logloss new | rmse base | rmse new | rmse delta |
|---|---|---|---|---|---|---|---|
| fold0 | (2, 3) | 305,989 | 1.0059 | 1.0063 | 234.60 | 234.17 | -0.43 |
| fold1 | (4, 5) | 357,800 | 0.9835 | 0.9836 | 217.82 | 217.72 | -0.10 |
| fold2 | (6, 8) | 369,492 | 0.9497 | 0.9487 | 244.50 | 244.30 | -0.20 |
| fold3 | (9, 10) | 365,246 | 0.9451 | 0.9441 | 237.91 | 237.68 | -0.23 |
| fold4 | (11, 12) | 325,004 | 0.9810 | 0.9806 | 226.01 | 225.71 | -0.31 |
| holdout | (1, 7) | 339,046 | 1.0297 | 1.0291 | 265.15 | 262.99 | -2.16 |

("delta" = new minus base; negative means the new-features model is better.)

Result: real, small, one-directional win on RMSE, not a strong one. RMSE improves with the new
features in every single fold and on the holdout (0.10 to 2.16 points, 0.05% to 0.8% relative),
and the biggest gain by far is exactly on the holdout, which is Jan+Jul, the months carrying
LIRF's persistent EOBT-copy baseline from section 3. Log loss is the falsifying number: it is
flat to mixed, improving in 3 of 6 evaluations (fold2, fold3, fold4, holdout) and getting very
slightly worse in 2 of 6 (fold0, fold1). The new features shift probability mass in a way that
helps the mixture's RMSE a little without meaningfully sharpening the classifier's overall
calibration. This model (LightGBM, num_leaves=63, no OCC/WX, physical class capped at 200k) also
runs at logloss ~0.95-1.03, well behind mix_cb_wx's shipped 0.6870 (CatBoost, num_leaves=127,
GPU, +OCC+WX, no physical-class cap) -- the two are not the same model, so only the base-vs-new
delta within this script is a fair comparison, not the absolute level.

Bottom line: the airport-day/hour visible-stamp signature is a real signal (consistently helps
RMSE, never hurts it) but it recovers only about 2 points of the roughly 25-45 point oracle gap
found in section 1 -- it narrows the headroom, it does not close it. Do not read this as "solved";
read it as one small, keepable feature addition, with EOBT-vs-other classification quality still
the open, unsolved lever.

## 5. Physical-row (class 3) noise floor

Command: `python analysis/lanea_clf.py noisefloor`

n = 237,871 (70.1% of lane A holdout rows are physical, no plan-stamp copy).

| stat | value |
|---|---|
| median residual (y - z_aobt) | -102.0 s |
| IQR | [-240.0, 179.0] s (width 419.0 s) |
| share \|residual\| > 600 s | 9.28% |
| z_aobt alone, RMSE | 464.80 |
| best constant (mean shift), RMSE | 464.41 |
| best constant (median shift), RMSE | 471.75 (worse: RMSE is minimized by the mean, not the median) |
| airport-hour mean, RMSE (in-sample) | 417.95 |
| airport-hour median-residual + z_aobt, RMSE (in-sample) | 443.31 |

z_aobt is already close to unbiased (median -102s, mean -19s) but has a wide spread: even the
best in-sample airport-hour grouping only gets physical-row RMSE down to about 418, a ~10%
improvement over z_aobt alone and the practical noise floor for the 70% of lane A that is not a
stamp copy. This is the ceiling the "other" cell (66% of total lane-A SSE, section 2) is bounded
by; it is a much bigger term than the classifier headroom and is not fixable by better
classification.

## Bottom line

Lane A error splits roughly into two regimes: about 31% of total squared error (aobt+eobt+sched
rows scored as "other") is classifier-fixable, worth an oracle 25-47 RMSE points depending on
which blend it is layered onto, with EOBT the dominant single lever (24.5 of that 47); the day/hour
visible-stamp signature recovers a real but small slice of it (about 2 RMSE points on holdout, more
consistent than large). The other ~66% of squared error is physical-row regression noise with an
in-sample floor around 418 RMSE (vs 465 for z_aobt alone) that no stamp classifier touches. Lane A
is not "essentially a classification problem": classification headroom is real and the current
classifier is leaving a meaningful amount on the table (recall 6-12% on the three copy classes),
but the physical-noise term is the larger of the two and caps how far classification alone can move
the total.
