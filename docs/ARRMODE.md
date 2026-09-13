# Arrival recorder-mode features

Arrivals' block times are visible for every arrival in both splits, and arrivals show the same
copy-vs-physical recorder behaviour as departures. `analysis/arrmode.py` turns that into features keyed
by departure `id`, cached at `data/cache/arrmode.parquet`. `analysis/arr_feat_test.py` (lane A) and
`analysis/arr_feat_test_b.py` (lane B) test whether those features help the stamp/mechanism classifiers.

## Build

`PYTHONPATH=/c/Users/skf_s/prc26 python analysis/arrmode.py`

16 columns, all from arrivals of the same `split` only:

- Airport-day shares (`ad_arr_a1`, `ad_arr_sc`, `ad_arr_tail`, `ad_n_arr`): block time within 60s of
  ARVT_1 / SCHED, taxi-in over 1800s, arrival count, grouped by (split, apt, day).
- +-2h window around the departure's takeoff (`w_arr_a1`, `w_arr_sc`, `w_arr_tail`, `w_n_arr`), same-apt
  arrivals, via the searchsorted + cumsum pattern from `features.traffic`.
- Previous arrival on the same stand (`pa_a1`, `pa_sc`, `pa_taxi_in`, `pa_gap`, `pa_missing`): last arrival
  that blocked on that stand before the departure's anchor (`t_off - z_anchor`), capped at a 24h lookback,
  via the per-(split,apt,stand) dict lookup pattern from `features.occupancy`.
- Airline-day shares (`al_arr_a1`, `al_arr_sc`, `al_n_arr`): same as the airport-day shares, restricted to
  the departure's own airline (first 3 chars of FLIGHT_mvt) that day.

## Sanity checks

Non-null share by split: airport-day and window columns are 100% populated (every departure's airport has
arrivals that day). Previous-stand-arrival columns are ~98% populated (98.0% train, 97.5% rank; the rest
have no prior arrival on that stand within 24h). Airline-day columns are ~98.9% populated (rare airlines
with no arrivals that day get NaN, as specified).

Descriptive stats reproduced exactly from `analysis/arrmode._arrivals`: arrival ARVT_1-copy share 3.9%
train / 3.9% rank, schedule-copy share 5.2% train / 4.9% rank, LIRF schedule-copy share 0.118 train.

Within-airport (airport-mean removed) correlation across 3,649 train airport-days with >=20 arrivals and
>=20 departures, arrival share vs. **independent** per-stamp departure match share (not the mutually-
exclusive `mix.label` class; that reproduces the reference table, `mix.label`'s priority-ordered class
does not):

| arrival share (block time within 60s of ...) | corr with dep AOBT-copy share | dep EOBT-copy share | dep sched-copy share |
|---|---|---|---|
| ARVT_1_flt (planned arrival) | 0.10 (ref 0.12) | 0.47 (ref 0.47) | 0.44 (ref 0.44) |
| SCHED_TIME_UTC_mvt | 0.14 (ref 0.16) | 0.20 (ref 0.20) | 0.29 (ref 0.29) |
| ARVT_3_flt | -0.03 (ref -0.02) | 0.01 (ref 0.02) | 0.01 (ref 0.01) |

7 of 9 cells match to two decimals; the other 2 (both in the dep-AOBT-copy column) are off by 0.02 and
0.02. Close enough to confirm the arrival-side pipeline is right; not an exact match, reported as-is.

## Test A: lane A stamp classifier

`PYTHONPATH=/c/Users/skf_s/prc26 python analysis/arr_feat_test.py`

Baseline CAT+NUM+DAY vs. +NEWDAY (the 10 stamp-signature columns from `day_feat_test.py`), +arrmode (all
16 columns above), +both. Same harness as `day_feat_test.py`: CPU LightGBM, physical class capped at
200k rows/fold, holdout = months 1+7.

Holdout (339,046 lane-A rows):

| variant | logloss | mixture RMSE | recall eobt | recall sched | SSE share true-eobt/pred-other |
|---|---|---|---|---|---|
| base | 1.0297 | 265.15 | 0.524 | 0.430 | 0.0221 |
| newday | 1.0291 | 262.99 | 0.523 | 0.430 | 0.0230 |
| arr | 1.0267 | **261.34** | 0.522 | 0.423 | 0.0231 |
| both | 1.0275 | 263.41 | 0.524 | 0.424 | 0.0227 |

Per-fold RMSE, base vs. arr: fold0 234.60->233.35, fold1 217.82->216.50, fold2 244.50->243.06,
fold3 237.91->236.64, fold4 226.01->224.37. Arr beats base in every fold and on holdout. Full table:
`data/analysis_arr_feat_test.csv`.

**Verdict**
- variant `arr`: holdout delta = base - arr = +3.80 (need > 2.16), beats base every fold: **PASS**
- variant `both`: holdout delta = +1.74 (need > 2.16), beats base every fold: **FAIL** (arrmode's gain does
  not stack additively with NEWDAY's; `both` sits between `arr` alone and `newday` alone)

## Test B: lane B mechanism classifier

`PYTHONPATH=/c/Users/skf_s/prc26 python analysis/arr_feat_test_b.py`

Lane B rows (has_plan==0, train split), copy/near/phys 3-class LightGBM (`laneb.classes` labels, shift rows
dropped), features `B_CAT + B_NUM + ["cross", "sched_hour"]` with and without the 16 arrmode columns.
Holdout n_copy = 253 (matches the shipped lane-B holdout copy-class count).

| metric | base | +arrmode |
|---|---|---|
| CV log loss (n-weighted, 5 folds) | 0.1841 | 0.1803 |
| holdout log loss | 0.1251 | 0.1247 |
| holdout copy-class Brier | 0.0259 | 0.0261 |
| holdout mean P(copy) on true-copy rows | 0.4351 | 0.4239 |

**Verdict: null / mixed, not a pass.** CV and holdout log loss both move a hair in arrmode's favour
(0.0038 and 0.0004), but the copy-class Brier score gets slightly worse (+0.0002) and mean P(copy) on
true copy rows drops (0.4351 -> 0.4239) - arrmode does not sharpen lane B's copy-class confidence. No
threshold was set for this test in the brief; on the numbers, arrmode is not worth adding to lane B.

## Files

- `analysis/arrmode.py` - feature build, `data/cache/arrmode.parquet`
- `analysis/arr_feat_test.py` - Test A, `data/analysis_arr_feat_test.csv`
- `analysis/arr_feat_test_b.py` - Test B, `data/analysis_arr_feat_test_b.csv`
