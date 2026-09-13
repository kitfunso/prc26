# Physical-tail mechanism: lane A's largest error pool

Finds the mechanism behind lane A's biggest error pool (`mix.label()` class 3, "physical", no
known stamp copy) and tests one label-free feature for it. `y` = takeoff minus recorded off-block,
`c = y - z_aobt` (`z_aobt` = takeoff minus the flight plan's `AOBT_3`), `r = y - p` where `p` is
the shipped v25 blend (`ra_cb_arr+ra_lgb_arr+nn3_arr`). Study: `analysis/phys_tail.py`
(subcommands `harness census tail mech cluster dup arvt rank all`). Feature test:
`analysis/phys_tail_test.py`.

## 0. Harness

v25's OOF parquet doesn't exist (`blend.py` only writes a holdout file); `_oof_blend()` rebuilds
it with the same NNLS weights on `ra_cb_arr_oof.parquet` + `ra_lgb_arr_oof.parquet` +
`nn3_arr_oof.parquet`.

| | RMSE | want |
|---|---|---|
| holdout laneA | 223.65 | 223.65 |
| OOF laneA (reconstructed blend) | 204.96 | 204.96 |

Exact match. All steps below use the combined OOF+holdout population (2,062,577 lane A rows) -
the study's given facts (8.3% row share / 64.4% SSE share for the phys-tail within physical, see
section 1) only reproduce on that combined population, not on either part alone.

## 1. Census

SSE share by label class (`r = y - p`, the shipped blend's residual):

| part | class | n | row share | SSE share |
|---|---|---|---|---|
| oof | aobt | 342,841 | 0.1989 | 0.0707 |
| oof | eobt | 162,875 | 0.0945 | 0.0730 |
| oof | sched | 28,248 | 0.0164 | 0.0982 |
| oof | physical | 1,189,567 | 0.6902 | 0.7581 |
| holdout | aobt | 65,663 | 0.1937 | 0.0625 |
| holdout | eobt | 30,322 | 0.0894 | 0.0774 |
| holdout | sched | 5,190 | 0.0153 | 0.0991 |
| holdout | physical | 237,871 | 0.7016 | 0.7611 |

Combined: physical is 69.21% of lane A rows (matches `docs/LANEA_CLF.md`'s "69%" almost exactly)
and 75.87% of lane A SSE against `r` (higher than that page's "66%"). Same row share, different
SSE share: `LANEA_CLF.md` measured the older `mix_cb_wx` model. The shipped v25 blend fixes more
of the copy-class error (aobt/eobt/sched) than physical, so physical's error *share* rises even
though its absolute error is somewhat lower too.

Inside physical, by `|c|` band:

| part | band | n | row share | SSE share of physical | SSE share of lane A |
|---|---|---|---|---|---|
| oof | <600 | 1,093,157 | 0.9190 | 0.5836 | 0.4424 |
| oof | 600-1800 | 90,344 | 0.0759 | 0.2802 | 0.2124 |
| oof | 1800-3600 | 5,117 | 0.0043 | 0.1047 | 0.0794 |
| oof | >3600 | 949 | 0.0008 | 0.0315 | 0.0239 |
| holdout | <600 | 215,804 | 0.9072 | 0.5290 | 0.4026 |
| holdout | 600-1800 | 20,220 | 0.0850 | 0.3043 | 0.2316 |
| holdout | 1800-3600 | 1,501 | 0.0063 | 0.1144 | 0.0870 |
| holdout | >3600 | 346 | 0.0015 | 0.0524 | 0.0399 |

**Phys-tail** = physical with `|c| > 600` (bands 600-1800 + 1800-3600 + >3600). Combined
population: 118,477 rows, 8.30% of physical, 5.74% of all lane A rows - but 64.36% of physical
SSE against `c` (the raw z_aobt residual, matches the given fact exactly) and 32.38% of ALL lane
A SSE against `r` (the shipped model's actual residual: it already absorbs some of this, via
d_ae/d_al/occ etc., but a third of its total error still sits in 5.7% of rows).

Top 1% of rows by `r^2`:

| part | physical | sched | eobt | aobt | phys-tail share of top1% |
|---|---|---|---|---|---|
| oof (n=17,236) | 0.774 | 0.117 | 0.087 | 0.023 | 0.628 |
| holdout (n=3,391) | 0.771 | 0.116 | 0.098 | 0.014 | 0.653 |

## 2. Phys-tail profile (vs other class-3 rows)

n phys-tail = 118,477, n other class-3 = 1,308,961.

Geography and carrier concentrate hard. Top airport x sign(c) by row count: phys-tail is led by
LTFM (Istanbul) at `c<0` (28,123 rows, recorded off-block earlier than AOBT_3) and LIRF (Rome) at
`c>0` (12,436). Top 20 airlines by phys-tail SSE: THY (Turkish Airlines) alone is 23.5% of
phys-tail rows and 17.6% of phys-tail SSE, far above #2 AFR (6.6% rows / 8.8% SSE) and #3 BAW
(6.9% rows / 8.4% SSE). No single stand concentrates SSE (top 20 stands each under 0.5%) - this
is an airline/airport effect, not a gate effect.

Hour distribution shifts toward the day versus other class-3 rows: phys-tail peaks 08:00-13:00 UTC
(0.066-0.078/hour) and is thin overnight (00:00-03:00, 0.004-0.010 vs other class-3's 0.002-0.009)
- consistent with daytime hub traffic, not a uniform clock artifact.

Marker shares, phys-tail vs other class-3 (existing features, all roughly flat, none discriminate
alone): `aobt_sec==0` 0.9670 vs 0.9773, `aobt_m5` 0.2016 vs 0.2091, `eobt_m5` 0.8077 vs 0.8623.

Distributions, phys-tail vs other class-3 (existing NUM features `d_ae`/`d_al`, already in the
model): both the **level** and the **spread** are much larger for phys-tail.

| stat | d_ae phys-tail | d_ae other class3 | d_al phys-tail | d_al other class3 |
|---|---|---|---|---|
| mean | 560.55 | 390.49 | 561.47 | 384.11 |
| std | 1241.18 | 621.12 | 1302.63 | 701.43 |
| median | 540 | 300 | 540 | 300 |

`occ`, `t_occ`, `pa_gap`, `pa_missing` are all close to flat between the two groups (e.g. `pa_gap`
median 6,544 vs 5,525, same order of magnitude) - arrival-side congestion is not the driver.

## 3. Which stamp is off

Per `(apt, runway)`, take the median `y` of the *other* (non-tail) class-3 rows as a stable local
norm (118,470 / 118,477 phys-tail rows have a group with n>=20):

| | share |
|---|---|
| `y` within 300s of the norm, `z_aobt` is not (off-block stamp looks wrong) | 0.3413 |
| `z_aobt` within 300s of the norm, `y` is not (genuinely long/short taxi) | 0.3627 |
| both within 300s | 0.0000 (definitional: `\|c\|>600` rules this out) |
| neither within 300s | 0.2960 |

No single story dominates. About a third of phys-tail looks like a stale/wrong AOBT_3 stamp,
about a third looks like real unusual taxi time, and a third is extreme on both clocks at once.

## 4. Clustering

Excess dispersion of the phys-tail rate vs binomial (ratio = `sum((p_g-p_parent)^2) /
sum(p_parent(1-p_parent)/n_g)`, groups n>=20):

| grouping | parent | ratio | n groups |
|---|---|---|---|
| airport-day | airport | **18.40** | 3,649 |
| airport-day-hour | airport-day | 3.03 | 52,158 |

18.4x is far above the 2.6-4.6x range `docs/LANEA_CLF.md` found for the copy-class mechanism on
the older model. The phys-tail rate is set mostly at the airport-day level (a bad AOBT feed on a
given day at a given airport), with a smaller extra hour-to-hour wobble on top.

## 5. Duplicates

Another departure row, same `FLIGHT_mvt` (fallback `CALLSIGN_flt`), same airport, within +-12h:
phys-tail 0.21% vs other class-3 0.25%. No difference. Ruled out.

## 6. ARVT_3 (measurement only, never a feature)

`corr(ARVT_3 - takeoff, r) = -0.0017` (n=2,062,577) - no relationship. Label-leak check: the share
where `ARVT_3 - recorded_offblock == filed duration +-60s` is 0.1117, *lower* than the share where
`ARVT_3 - AOBT_3 == filed duration +-60s` (0.1366) - the opposite of what a leak would show. Clean
null both ways; `ARVT_3` stays measurement-only, never a feature.

## 7. Candidate label-free marks

Coverage = share of phys-tail SSE captured by rows matching the mark; lift = phys-tail rate among
marked rows / base rate (0.0574 over all lane A).

| mark | n marked | coverage of phys-tail SSE | lift |
|---|---|---|---|
| `aobt_sec==0` [existing NUM] | 2,015,132 | 0.9790 | 0.99 |
| `eobt_m5` [existing NUM] | 1,754,772 | 0.7481 | 0.95 |
| `h_dae_std` top decile [NEW] | 206,269 | 0.4200 | 2.38 |
| `h_dal_std` top decile [NEW] | 206,287 | 0.4062 | 2.15 |
| `day_dal_std` top decile [NEW] | 206,353 | 0.3382 | 1.89 |
| `day_dae_std` top decile [NEW] | 206,797 | 0.3370 | 2.08 |
| `apt==LTFM` [existing CAT] | 269,320 | 0.2402 | 2.37 |
| `airline==THY` [existing CAT] | 218,875 | 0.1758 | 2.21 |

`aobt_sec==0`/`eobt_m5` cover most of the SSE only because they cover most of the *rows* (lift
~1, no signal). The new marks are day/hour standard deviation of `d_ae`/`d_al` (per `apt`,
computed only from AOBT/EOBT/LOBT stamps, no `y`), motivated directly by section 4's 18.4x
day-level clustering. Not tested before: `docs/FE.md`'s groups used leave-out target encodings of
the *label*; `docs/ARRMODE.md`'s 16 columns are arrival-side, not AOBT/EOBT dispersion. `h_dae_std`
and `h_dal_std` (hour-level) beat their day-level counterparts on both coverage and lift, and both
beat the existing `apt`/`airline` categoricals on coverage.

## Feature test

`analysis/phys_tail_test.py`: `ra_lgb_arr`, CPU LightGBM, `cfg["num"]` and `cfg["reg"]["num"]`
extended with the new columns, 5 `train.FOLDS` + holdout refit at 1.1x mean fold iters. Harness
check (unmodified config, fold 0 only) reproduced `data/cache/ra_lgb_arr_oof.parquet` exactly
(max abs diff 0.00e+00). Two variants:

- `hour_disp`: `h_dae_std`, `h_dal_std`
- `full_disp`: `day_dae_std`, `h_dae_std`, `day_dal_std`, `h_dal_std`

| fold | months | base RMSE | hour_disp | full_disp |
|---|---|---|---|---|
| 0 | (2,3) | 213.597 | 212.349 | 212.742 |
| 1 | (4,5) | 200.163 | 199.713 | 199.625 |
| 2 | (6,8) | 224.339 | 224.079 | 224.326 |
| 3 | (9,10) | 217.248 | 216.921 | 216.572 |
| 4 | (11,12) | 204.811 | 204.417 | 204.316 |
| holdout | (1,7) | 231.643 | 230.982 | 230.818 |

`train.boot_delta` on holdout lane A (positive = new variant better than base):

| variant | delta | 95% CI | beats base every fold | verdict |
|---|---|---|---|---|
| hour_disp | +0.661 | [+0.212, +1.171] | yes | **PASS** |
| full_disp | +0.826 | [+0.265, +1.409] | yes | **PASS** |

Both pass: better than base on all 5 folds, better on holdout, bootstrap lower bound above zero.
Head to head, `full_disp` has the larger holdout gain and wins on holdout plus 4 of 5 folds
against `hour_disp` (loses narrowly on fold 0 and fold 2) - it is the one to prefer if this ships
into `lanea.py`'s `ra_lgb_arr` config (out of scope for this study; not applied).

## Bottom line

The physical-tail mechanism is a **day-level AOBT anomaly concentrated at specific
airport/airline pairs** (LTFM/THY above all), not gate congestion, not duplicate rows, and not
an ARVT_3 leak. It shows up in both level and spread: on phys-tail rows `d_ae`/`d_al` run 80%
higher at the median and roughly double the standard deviation of the rest of class 3, and the
phys-tail rate itself is 18.4x overdispersed at the airport-day level versus binomial, far above
the 2.6-4.6x found for the copy-class mechanism on the older model. The which-stamp split found
no single smoking gun: about a third of rows look like a stale AOBT_3 stamp, a third look like
genuinely unusual taxi, a third are extreme on both. Day/hour dispersion of `d_ae`/`d_al`,
computed only from visible AOBT/EOBT/LOBT stamps, is the strongest new label-free mark (up to 42%
coverage of phys-tail SSE, 2.4x lift) and passes the feature test on `ra_lgb_arr`: both variants
beat base on every fold and on holdout, bootstrap lower bound above zero; `full_disp` is the
stronger of the two.

## Files

- `analysis/phys_tail.py` - the 7-step study (`harness census tail mech cluster dup arvt rank all`)
- `analysis/phys_tail_test.py` - feature test (`hour_disp`, `full_disp` vs `ra_lgb_arr`)
- `data/analysis_phys_tail_test.log` - feature-test run log
- `data/cache/analysis_phys_tail_hour_disp_oof.parquet`, `..._holdout.parquet`
- `data/cache/analysis_phys_tail_full_disp_oof.parquet`, `..._holdout.parquet`
- `docs/PHYS_TAIL.md` - this file
