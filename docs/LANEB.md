# Lane B mechanism model

`PYTHONPATH=/c/Users/skf_s/prc26 python laneb.py cv mix_cb_wx` then `python laneb.py fit 117 238`. `cv` writes its
tables to `docs/LANEB_RUN.md`; this file is the analysis. Sections 1 to 3 and the verdict describe v25; section 4 adds
the hull that v26 ships; section 5 the classifier weights, tau columns and hull prior that v27 ships; section 6 the
copy history in v28. Sections 7 to 10 record leads that did not help.

**Result: FAIL** vs donor mix_cb_wx (holdout lane B RMSE 1900.6) on the strict pass mark (copy class no worse than
donor's 2707.4), by 0.2%. New 1874.6, donor 1900.6, paired day-block bootstrap delta +26.0 [+5.2, +57.5] (lower
bound above zero: the overall gain is significant). Copy class 2713.0 vs donor's 2707.4 (0.2% worse). This is the
best variant found; see "Donor-blend grid" below for why the two pass criteria cannot both be hit at once here.

## 1. Diagnosis: why the copy class broke

The task's hypothesis was that the day-shift rule (LIRF, z_sched > 57600s, empirical rate 0.714 on 7 train rows)
leaks onto true schedule-copy rows. Re-checked on the un-blended model (near folded into phys, no donor blend):

**By airport (copy rows n=253, shift rows n=6, holdout)**

| apt | mech | n | mean P(copy) | mean donor p_sched | SSE (ours) | SSE (donor) |
|---|---|---|---|---|---|---|
| EDDF | copy | 5 | 0.071 | 0.040 | 1,095,406 | 735,457 |
| EHAM | copy | 21 | 0.310 | 0.330 | 387,664 | 367,234 |
| LEBL | copy | 6 | 0.165 | 0.257 | 834,931 | 550,470 |
| LFPG | copy | 5 | 0.080 | 0.130 | 240,925 | 75,908 |
| LFPG | shift | 1 | 0.013 | 0.011 | 6,955,383,000 | 6,900,003,000 |
| LIRF | copy | 134 | 0.633 | 0.628 | 3,067,021,000 | 1,845,868,000 |
| LIRF | shift | 5 | 0.756 | 0.622 | 460,676,700 | 3,026,229,000 |
| LSZH | copy | 24 | 0.389 | 0.261 | 1,257,201 | 1,163,560 |
| LTFM | copy | 58 | 0.156 | 0.172 | 6,037,665 | 5,671,380 |

LIRF holds 134 of the 253 copy rows and 99% of the copy-class SSE gap. That one LFPG shift row (a genuine day-shift
mechanism row the LIRF-only rule cannot catch, already counted in the "6 shift rows" class, not a new bug) dwarfs
everything else in raw SSE but is a `shift` row, not `copy`.

**Copy rows only, by z_sched bucket**

| bucket | n | mean P(copy) | mean donor p_sched | SSE (ours) | SSE (donor) |
|---|---|---|---|---|---|
| <1h | 125 | 0.2 | 0.2 | 14,118,020 | 10,272,743 |
| 1-2h | 44 | 0.6 | 0.5 | 301,046,100 | 314,270,290 |
| 2-4h | 48 | 0.7 | 0.7 | 402,966,000 | 288,025,117 |
| 4-8h | 25 | 0.6 | 0.6 | 1,191,831,000 | 871,828,118 |
| 8-16h | 10 | 0.8 | 0.8 | 1,096,552,000 | 348,337,843 |
| >16h | 1 | 0.5 | 0.5 | 70,361,790 | 21,697,793 |

Total un-blended copy-row SSE: 3,076,874,690 (RMSE 3487.3 vs donor's 2707.4). Of that, only 70,361,786 (2.3%,
1 row) comes from a shift-rule-hit row. **The task's own hypothesis (shift rule contaminating copy) is not the
main cause**: 97.7% of the SSE gap sits in the 4-8h and 8-16h buckets, where the classifier is only moderately
confident (mean P(copy) 0.6-0.8) and our own `tau` regressor badly misjudges the physical-taxi remainder for the
rows it gets wrong. The donor's paired probability/physical-value estimate is calibrated together and does not
have this failure mode at the same rows.

**Classifier reliability: mean P(copy)**

| | true copy | true phys |
|---|---|---|
| holdout (n=253 / 5075) | 0.441 | 0.026 |
| OOF (n=1146 / 15730) | 0.452 | 0.033 |

The classifier separates phys rows cleanly (P(copy) near 0 both holdout and OOF) but is only moderately confident
on true copy rows (mean 0.44-0.45): a large share of true copy rows sit in the 0.2-0.8 P(copy) range where `tau`'s
error dominates. This, not the shift rule, is the root cause.

## 2. Fix, tried in order

**(a) Fold near into phys.** Ablation (`no_near` in the original run) showed near never earned its keep
(1817.7 vs full 1834.3, i.e. removing it *helped*). Folded permanently: `near` is no longer a modeled class in
`predict()`/`clf_logloss()` (still returned by `classes()` unchanged, for the other agent's import). `near_delta()`
deleted as dead code.

**(b) Conditional shift cell.** Rejected: every LIRF z_sched>57600 train row is already `cross==1` (a no-op
restriction), and the classifier-weighted shift formula (`p_shift_given_phys`) rests on only 4 non-copy train
rows, too few to trust; both made holdout lane B RMSE worse in earlier testing this session. Kept the original
unconditional per-airport shift rule.

**(c) Classifier/probability swap.** Rejected: substituting the donor's raw `p_sched` for our own P(copy) (paired
with our own `tau`) made lane B holdout RMSE worse (1891.3 vs the un-blended baseline), as did a CatBoost copy
classifier trained the same way (1934.3). Pairing a probability estimate with a `tau` it was not calibrated
against breaks more than it fixes; the donor's `p_sched` and physical component are a matched pair, ours are not
interchangeable with either half alone.

**(d) Blend with the donor's prediction.** This is the fix that shipped. `blend_donor()` blends our final
prediction with the donor's *cached final prediction* (available for both holdout and rank, unlike `p_sched`,
which the rank cache lacks) for rows where our own classifier's P(copy) exceeds `DONOR_COPY_THRESH`. Donor-blend
grid, re-run against the real retrained (not fixed-iteration) pipeline:

| DONOR_COPY_W | lane B RMSE | bootstrap delta [lo, hi] | copy RMSE | copy vs donor |
|---|---|---|---|---|
| 0.15 | 1863.3 | +37.3 [+11.2, +80.0] | 2719.3 | 0.44% worse |
| **0.10 (chosen)** | 1874.6 | +26.0 [+5.2, +57.5] | 2713.0 | 0.21% worse |
| 0.07 | 1881.7 | +18.9 [+1.5, +42.7] | 2711.6 | 0.16% worse |
| 0.05 | 1886.6 | +14.0 [-0.7, +33.4] | 2711.7 | 0.16% worse |
| 0 (no blend) | 1793.96 | (not computed; SSE unavailable per-class) | (not blended) | far worse (see diagnosis) |

Copy RMSE flattens near 2711-2712 as the blend weight drops toward zero and does not reach parity with the
donor's 2707.4: rows the classifier never gates in (P(copy) <= 0.12, i.e. true copy rows the classifier missed
entirely) still get our own miscalibrated `tau`, and no amount of *weight* tuning at a fixed threshold reaches
them. At the same time, weight 0.05 already drops the bootstrap lower bound below zero. **No single
(threshold, weight) pair in this family satisfies both pass criteria at once**; this is a genuine, confirmed
tradeoff, not a search that stopped early. `DONOR_COPY_W=0.10` was chosen as the point with a comfortable
bootstrap margin (+5.2 lower bound) and the smallest copy-class miss compatible with that margin.

Interesting side finding: `no_blend` (1794.0 lane B RMSE) beats every blended variant on overall lane B RMSE,
because the blend trades some of that overall gain for copy-class safety. Reported for honesty, not adopted,
since `no_blend` reproduces the diagnosed copy-class failure.

## 3. Final result (DONOR_COPY_THRESH=0.12, DONOR_COPY_W=0.10)

Shift rule: LIRF z_sched > 57600s, n=7 train rows, empirical rate 0.714.

Day-feature multiclass log loss on CV folds: with 0.1231, without 0.1227 (day features do not help, kept anyway
since they were part of the shipped baseline and cost nothing).

### By airport (holdout lane B)

| apt | n | rmse_new | rmse_donor |
|---|---|---|---|
| EDDF | 599 | 386.3 | 366.1 |
| EDDM | 237 | 631.5 | 613.9 |
| EGLL | 466 | 1090.4 | 1062.5 |
| EHAM | 808 | 553.7 | 565.8 |
| LEBL | 496 | 337.3 | 321.9 |
| LEMD | 309 | 317.0 | 337.8 |
| LFPG | 880 | 3451.1 | 3436.9 |
| LIRF | 397 | 4171.0 | 4362.9 |
| LSZH | 524 | 604.5 | 607.8 |
| LTFM | 657 | 591.0 | 582.0 |

### By mechanism class (holdout lane B)

| mech | n | rmse_new | rmse_donor |
|---|---|---|---|
| copy | 253 | 2713.0 | 2707.4 |
| near | 39 | 4649.1 | 4752.6 |
| phys | 5075 | 1137.1 | 1153.0 |
| shift | 6 | 40029.9 | 40674.0 |

### Ablation (holdout lane B RMSE, one piece removed)

| variant | rmse |
|---|---|
| full | 1874.6 |
| donor_only | 1900.6 |
| no_shift | 1904.6 |
| no_blend | 1794.0 |
| no_day | 1873.6 |

The shift rule remains the single biggest lever (removing it costs 30 points). The donor blend costs 80.6 points
of overall lane B RMSE to buy copy-class safety (see grid above).

Overall holdout (344,419 rows, lane A unchanged): donor blend 326.99 -> lane B swapped 324.64.

## Verdict

Not a full pass: copy class is 0.2% worse than the donor (2713.0 vs 2707.4), not "no worse." Everything else
passes cleanly: holdout lane B RMSE 1874.6 < 1900.6, bootstrap lower bound +5.2 > 0. This is the best variant
found after a genuine grid search that shows the two pass criteria trade off against each other in this family
of fixes; closing the remaining 0.2% would need a fix for the classifier's false negatives on true copy rows
(P(copy) <= 0.12 on an actual copy row), not more blend weight. Shipped as the best available variant per
instruction.

## 4. Hull on LIRF rows over 4 h late (v26)

A LIRF lane B row whose takeoff is 4 h or more after its schedule (`z_sched >= 14400`) carries one of two labels,
read on all 127 such rows in 2025 train:

- a schedule copy: y within 60 s of `z_sched`, or in the near band [`z_sched` - 3100, `z_sched` + 700];
- a composed stamp: the real off-block clock time written on some date, so y = tau + 86400 k with tau an
  ordinary taxi.

Which day count the composed rows use: kn = round((`z_sched` - tau) / 86400) is the count nearest the schedule,
kd = local date(`t_off` - tau) - local date(`t_off` - `z_sched`) in Europe/Rome is the calendar count. Counts at
tau = 1100 (session scratchpad `laneb_hull.py`):

| kn | kd | composed, y <= 72000 | composed, y > 72000 | copy, y <= 72000 | copy, y > 72000 |
|---|---|---|---|---|---|
| 0 | 0 | 15 | 0 | 51 | 0 |
| 0 | 1 | 0 | 0 | 33 | 0 |
| 1 | 0 | 0 | 0 | 2 | 0 |
| 1 | 1 | 0 | 11 | 14 | 0 |
| 2 | 1 | 0 | 0 | 0 | 1 |

Every composed row takes k = kn (15 at k = 0, 11 at k = 1), and kd agrees with kn on all 26 of them.

Off LIRF the same filter finds 415 train rows, all composed at k = 0: y runs from -5 to 8,937 s (median 1,026), and
21 are above 3,600 s, 17 of them LTFM. There is no copy mode to bound there, and a cap at 3,600 s would cut 21 real
labels, so the hull is LIRF only.

**Hull.** On those LIRF rows p is clipped to [lo, hi] (`hull()`), with lo = min(`z_sched` - 3100, 86400 k_min) and
hi = max(`z_sched` + 700, 3600 + 86400 k_max), k running over kn and kd at tau in {0, 3600}. Every label the two
mechanisms can produce (copy, near band, composed with tau in [0, 3600] at those day counts) lies inside, and
clipping onto an interval that holds the label never moves a prediction away from it. That rests on the support:
0 of the 127 train rows fall outside their own interval. kd never changes a composed row's k here; it stays in to
keep the interval wide.

**Results.**

| set | rows moved | effect |
|---|---|---|
| OOF, months 2-12, no donor blend | 6 of 17,097, all closer to the label | lane B SSE 37,929M to 37,339M |
| holdout Jan+Jul 2025 | 5 (3 composed, 2 copy), all closer | lane B 1874.6 to 1816.9; all rows with v25's lane A 322.59 to 317.40, +5.19 [+0.01, +12.01] vs v25 |

The `laneb.py cv` rerun with the hull reproduces the holdout figure (1816.93, `docs/LANEB_RUN.md`). With the hull in
place the donor blend costs 33.0 points of lane B RMSE (`no_blend` 1783.9 vs 1816.9), down from 80.6. It stays:
it was chosen for the copy class, and dropping it on this ablation would be a choice made on the holdout.

**Rank 2026.** 43 LIRF lane B rows have `z_sched >= 14400`. `python laneb.py fit 100 449` moves 6 of them and leaves
every other lane B row as v25 had it (the new `data/laneb_rank.parquet` equals hull(v25), max difference 0.000000).

| MVT_ID | z_sched | v25 | v26 |
|---|---|---|---|
| 202680695 | 39,365 | 40,396 | 40,065 |
| 213142469 | 47,705 | 22,435 | 44,605 |
| 213145828 | 36,062 | 50,140 | 36,762 |
| 213142746 | 41,395 | 60,341 | 42,095 |
| 213152353 | 87,484 | 67,814 | 84,384 |
| 213154831 | 39,060 | 40,222 | 39,760 |

Board arithmetic for these six rows alone (N = 344,841, from v25's 288.68): all six copies gives 280.71, all six
composed at k = kn gives 257.93, and all six composed at k = 0 gives 293.16. The last case puts the two kn = 1 rows
outside the support, which no train row does (0 of 11). v26 ships the hull together with the lane A change, and the
board change is not read row by row.

**Non-LIRF rows over 4 h late in 2026.** The rank set has 280 (215 in January). v25 puts 101 of them above 3,600 s,
against 21 of 415 labels in 2025 train. 100 of the 101 are EHAM, which has 148 such rows against 91 in all of 2025
(v25 median 4,701 s, max 6,460; the donor puts 55 above 3,600 s), and 99 of those 100 fall on 2026-01-03 to 05.
METAR shows a snow spell: SN in 31, 28 and 15 reports on those three days, and lane A's median `z_aobt` there is 1,998,
2,352 and 2,110 s against an EHAM 99th percentile of daily medians of 1,202 s. The one 2025 analogue is LTFM in the
February 2025 snow (19, 22 and 23 February): lane A median about 2,100 s, lane B median label about 5,100 s (2.43 times
the lane A median). Out of sample, laneb predicted about 2,300 s on those days (RMSE about 3,085); the full refit trains
on them, and its EHAM predictions near 5,000 s match the analogue (section 5 finds that the analogue does not transfer;
v27 predicts about 3,100 s there). No interval bounds a composed label, so these rows stay as the model has them
(session scratchpad `nonlirf_big.py`, `eham_snow.py`, `eham_snow2.py`).

## 5. Cost-weighted classifier, tau columns and a cell prior on the hull rows (v27)

Three changes, all on lane B; lane A stays as v26 has it. The runs below are fold-honest (five folds over months 2 to
12, the holdout fitted on all ten) and compared with v26's lane B by the paired day-block bootstrap, positive = better.

**Cost-weighted classifier (`_cost_w`).** Calling a copy row physical, or a physical row a copy, costs about
(`z_sched` - tau)^2 of SSE, while log loss charges every row the same. Each lane B row now weighs |`z_sched` - tau0|,
clipped to [300, 14400] s and scaled to mean 1, with tau0 the airport's median physical label in the training rows;
the early-stopping rows take the same weights. `w2` below squares that weight.

**Tau columns.** The physical-taxi regressor gains lane A's 16 `ARR` and 20 `WZ` columns.

Lane B RMSE (session scratchpad `laneb_feat_test.py`, logs `lbt.log`, `lbt2.log`, `lbt3.log`):

| variant | OOF | vs v26 | holdout | vs v26 |
|---|---|---|---|---|
| v26 | 1477.8 | | 1816.9 | |
| tau + ARR | 1485.1 | -7.2 [-14.3, -2.0] | 1813.8 | +3.1 [+0.1, +5.5] |
| tau + WZ | 1472.2 | +5.7 [-3.5, +17.5] | 1813.1 | +3.9 [-0.1, +9.3] |
| tau + ARR + WZ | 1472.0 | +5.8 [-3.9, +18.2] | 1811.6 | +5.4 [+2.6, +9.8] |
| w1 | 1437.9 | +39.9 [+16.1, +70.1] | 1808.3 | +8.6 [-4.9, +26.5] |
| w2 | 1421.5 | +56.3 [+23.3, +104.3] | 1815.0 | +2.0 [-19.2, +23.4] |
| w1 + tau ARR + WZ (v27) | 1431.4 | +46.4 [+21.9, +77.0] | 1803.2 | +13.8 [+0.8, +31.8] |
| w2 + tau ARR + WZ | 1414.4 | +63.4 [+29.2, +110.9] | 1809.5 | +7.4 [-13.2, +29.8] |

Adding columns to the classifier instead (`ARR`, `WZ`, stand, aircraft type, same-day counts of late lane B rows)
lost on the OOF months in every run, with or without the weights. w2 leads w1 on the OOF months and trails it on the
holdout. v27 takes w1; that pick saw the holdout, so the holdout column overstates what the weighting is worth by an
unknown amount. The OOF column does not depend on it.

**Hull cell prior (`hull_prior()`, `_cells()`).** On the 127 hull rows of 2025 train (section 4), w1t's P(copy) does
not sort copies from composed stamps: the copy share is 0.74 on the 38 rows with P(copy) above 0.9, 0.83 on the 29
between 0.7 and 0.9, and 0.40 on the 10 between 0.5 and 0.7 (A0 and A1 cells below; session scratchpad
`hull_air.py`). The mechanism of section 4 sorts them better. Every composed row takes k = kn and has kd = kn, so a
row with kn != kd can only be a copy; the rest split by kn, and at kn = 1 by `z_sched` either side of 57,600 s, the
shift rule's threshold:

| cell | rule (kn, kd at tau = 1100) | train rows | copies |
|---|---|---|---|
| D | kn != kd | 36 | 36 |
| A0 | kn = kd = 0 | 66 | 51 |
| A1lo | kn = kd = 1, `z_sched` <= 57,600 | 13 | 11 |
| A1hi | kn = kd = 1, `z_sched` > 57,600 | 12 | 3 |

Each hull row then takes the expected label of its cell, P(copy) `z_sched` + (1 - P(copy)) (1100 + 86400 kn), with
P(copy) = (copies + 1) / (rows + 2) over the training rows of its cell (the other OOF months in CV, all ten on the
holdout, all of 2025 for the rank set), and `hull()` clips as before. Copies here include the near band.

Against v26 on those 127 rows (SSE in millions of s^2, positive = better; session scratchpad `prior_hull2.py`):

| variant | 12 months (127) | OOF (82) | holdout (45) |
|---|---|---|---|
| w1t, no prior | +2,132 | +2,145 | -13 |
| one share per kn | +3,280 | +2,786 | +494 |
| per kn, kn = 1 split at 57,600 | +3,705 | +2,478 | +1,228 |
| cells D, A0, A1 | +3,893 | +3,574 | +319 |
| cells D, A0, A1lo, A1hi (v27) | +4,952 | +3,794 | +1,158 |

The v27 cells lead on the OOF months alone, so the choice among the four did not need the holdout; over all 12 months
the RMSE on these rows improves by 2,081 s [+546, +3,938]. An airline term was checked and left out: sorted by the
airline's own copy share on LIRF rows 1 to 4 h late, the copy share on A0 and A1 rows runs 0.60, 0.56, 0.55 and 0.81
from the lowest band to the highest (20, 9, 11 and 43 rows).

**Result** (`python laneb.py cv mix_cb_wx`, `docs/LANEB_RUN.md`). Holdout lane B 1741.7 against v26's 1816.9 and the
donor's 1900.6 (+158.9 [+43.7, +347.7] vs the donor); LIRF 3331.1 (v26 3809.8); physical class 1112.4 (donor
1153.0). The copy class is 2793.9 against the donor's 2707.4, so the strict pass mark of section 3 still fails, now
by 3.2%.

| variant | holdout lane B RMSE |
|---|---|
| full | 1741.7 |
| no_prior | 1803.2 |
| no_blend | 1742.4 |
| no_shift | 1741.7 |
| no_day | 1740.5 |
| donor_only | 1900.6 |

The prior is worth 61.5 points. The donor blend now changes little (0.7 points). Removing the shift rule changes
nothing: every row it touches is a hull row, and the prior replaces those predictions whole. It stays in the code
until `laneb.py` is next reworked.

Full file, lane A unchanged (session scratchpad `gate_lb.py`): holdout 308.47 all against v26's 315.17, +6.70
[+0.37, +14.41]; lane B over all 12 months 1486.9 against 1565.6, +78.7 [+17.4, +156.9]. Of the holdout's 1,438M of
lane B SSE gain, 1,171M comes from the prior on 45 hull rows.

**Rank 2026.** The 43 hull rows of the rank set fall 22 in A0, 13 in D, 4 in A1lo and 4 in A1hi, and each now equals
its cell prediction. Against v26 they move by 7,599 s RMS; the largest moves (session scratchpad `rank_cmp.py`):

| MVT_ID | cell | `z_sched` | v26 | v27 |
|---|---|---|---|---|
| 202681830 | A0 | 40,615 | 9,238 | 31,317 |
| 213147307 | A0 | 33,057 | 47,410 | 25,538 |
| 213145579 | A1hi | 111,654 | 106,525 | 94,401 |
| 213154026 | A1hi | 60,656 | 67,737 | 79,830 |
| 213142469 | A1lo | 47,705 | 44,605 | 55,664 |
| 213142746 | A0 | 41,395 | 42,095 | 31,914 |

Each is a two-outcome bet. Row 202681830 gains 4.6 s of board RMSE if it is a copy and loses about 4.3 s if it is
composed at k = 0 with a 1,100 s taxi (1 s = 2 x 282.42 x 344,841 = 195M of SSE at v26's board score). Summed,
(v27 - v26)^2 over the 43 rows is 2,483M, 12.75 s: what the rows are worth in expectation if every cell share is
right, and what they cost if every label sits where v26 put it (session scratchpad `swing.py`).

**EHAM snow rows, re-read.** Section 4 took LTFM's February 2025 snow as the analogue for EHAM's 3 to 5 January 2026
and found v26's predictions near 5,000 s consistent with it. Three checks say it does not transfer:

1. On LTFM's snow days lane A's median `z_aobt` is 2,098 to 2,105 s in every lateness band (`z_sched` 1 to 1.7 h,
   1.7 to 2.8 h, 2.8 to 4.2 h, 4.2 to 6.9 h, over 6.9 h), while lane A's median label is about 3,200 s in the first
   three. A measured pushback does not sit at one value across bands. Lane B's copy share there runs 0.98, 0.77,
   0.17 and 0.00 by band, and its physical labels (median 4,769 to 6,235 s from the second band on) rise with
   lateness. On EHAM's snow days lane A's median `z_aobt` moves with the band (2,806, 2,007, 1,548, 3,341 and
   1,618 s), as measured times do (session scratchpad `snow_band.py`).
2. Over 2025 train, on the 1,642 airport-days with at least 5 lane B physical rows and a lane A median `z_aobt` up to
   2,000 s, the lane B physical median is 0.78 to 1.06 times lane A's median `z_aobt` in every band. The three days
   above 2,000 s are all LTFM in February 2025 and give 0.38, 1.48 and 2.59 (session scratchpad `ab_ratio.py`).
3. Fold-honest, on lane B rows over 1 h late (like for like: no donor blend on OOF months, blend on the holdout
   months), w1t's SSE is 24,873M against v26's 26,958M, and 1,731M against 1,985M on the 6 most disrupted train
   airport-days (lane A late median `z_aobt` over 2,000 s, 221 rows, copy share 0.68). w1t loses in the two bands
   between 1,300 and 2,000 s, 1,892M against 1,781M (session scratchpad `ab_late.py`).

v27 therefore keeps w1t's level on those rows: the 315 EHAM lane B rows of 3 to 5 January 2026 average 3,099 s against
v26's 4,741 s (lane A's late median `z_aobt` on the three days: 2,108, 3,043 and 2,324 s). The risk stays: (v27 -
v26)^2 over those rows sums to 987M, 5.07 s of board RMSE, gained if the labels sit at w1t's level and lost if they
sit at v26's (session scratchpad `swing.py`). One board number cannot split this from the hull rows.

## 6. Copy history by airline and flight number (in v28)

After section 5, the LIRF rows 1 to 4 h late are the largest block of lane B error outside the shift and hull rows:
over the 12 months of 2025, v27 leaves 7,771M of SSE on 1,247 rows, copy share 0.460 (session scratchpad
`mech_hist.py`).

**What decides a copy.** Leave-one-out target encoding on those 1,247 rows (session scratchpad `mech_sep.py`): the
airline sorts copies from physical labels with AUC 0.846, the stand 0.668, the calendar day 0.626 and the month 0.587;
v27's implied copy share reaches 0.875. LTFM's near class goes the other way: the day reaches 0.923 there. By airline
at LIRF, over 2025 rows with the takeoff 30+ min after the schedule (session scratchpad log `lirf_regime.log`):

| airline | lane B rows | lane B copy share | lane A copy share |
|---|---|---|---|
| ISR | 56 | 1.000 | 0.993 |
| LAV | 22 | 1.000 | 1.000 |
| TKJ | 20 | 1.000 | 1.000 |
| TWB | 19 | 1.000 | 1.000 |
| ETH | 21 | 1.000 | 0.221 |
| MSC | 11 | 1.000 | 0.255 |
| WMT | 173 | 0.480 | 0.131 |
| RYR | 234 | 0.329 | 0.046 |
| ITY | 75 | 0.040 | 0.033 |
| BAW | 11 | 0.000 | 0.038 |
| EWG | 14 | 0.000 | 0.006 |
| KLM | 17 | 0.000 | 0.001 |

Across airlines the two lanes' copy shares correlate at 0.754. A copy follows who stamps the off-block (most likely the
ground handler), not the day's traffic. RYR is the one large airline whose lane B share moves with the calendar: 0.60,
0.62 and 0.67 from January to March, 0.18 to 0.26 from April to October except June's 0.49, 0.33 in November and 0.73
in December, on 5 to 70 rows a month, while its lane A share stays between 0.02 and 0.11. The classifier has no
season column.

A season column would not help (session scratchpad `season_d.py`, log `season_d.log`, not run through `laneb.py`).
On all LIRF lane B rows 30 min to 4 h late the copy share is 0.761 and 0.820 in February and March and 0.638 and
0.750 in November and December, but 0.408 in January (49 rows) and 0.263 in July (289 rows). Over the OOF months the
IATA winter season reads 0.733 against summer's 0.460, so a season column fitted there would raise January's
predictions, which v28 already puts 491 s too high on average (July 282 s too high). LIRF lane A copy shares do not
move with the month: schedule 0.070 to 0.089, EOBT 0.108 to 0.138.

**History columns (`copy_hist()`).** Each lane B row gets three columns from lane A rows with the takeoff 30+ min
after the schedule: the copy share of its (airport, airline), shrunk toward the airport share with weight 20; the copy
share of its (airport, flight number), shrunk toward the airline share with weight 5; and log(1 + lane A rows behind
the flight share). A 2025 row sees only lane A months 3+ away from its own, nearer the gap the 2026 rows face; a rank
row sees all of 2025.

Before building them, a two-point check on the same 1,247 rows: p = P `z_sched` + (1 - P) t0, P a logistic stacked on
v27's implied share and fitted leave-one-month-group-out (six groups, the holdout pair one of them), t0 the airport's
median physical label. The two-point form has no tau regressor, so its SSE sits above v27's 7,771M; only the changes
between rows compare (logs `mech_hist.log`, `mech_far.log`):

| stacked on | other month groups: AUC | SSE | months 3+ away: AUC | SSE |
|---|---|---|---|---|
| v27 share | 0.865 | 7,997M | 0.865 | 7,997M |
| + airline share | 0.886 | 7,541M | 0.883 | 7,602M |
| + flight share and count | 0.920 | 6,382M | 0.907 | 6,765M |
| + airline, flight, stand | 0.920 | 6,377M | 0.911 | 6,632M |

Lane A history covers the flight numbers of 87.0% of those rows from the other month groups and 66.5% from months 3+
away. On all lane B rows 30 min to 4 h late (16,289 rows, copy share 0.039) the same stacks run from 16,615M to
15,294M (15,518M from months 3+ away). On the rank set's lane B rows 30 min to 4 h late, coverage from all of 2025
runs from 0.454 (LIRF) and 0.477 (LTFM) to 0.917 (EDDF): flight numbers change between years, so the rank set gets
less from the flight column than 2025 does.

The hull rows do not gain. On the 127 of section 5 the flight share sorts copies within cells (AUC 0.745 on A0's 66
rows, 0.864 on A1lo's 13), but a logistic on cell plus history raises the SSE of the prior form from 7,964M to 8,595M
with the airline share and 8,566M with airline, flight and count (session scratchpad `hull_hist.py`). `hull_prior()`
keeps its cell shares.

**Result** (`python laneb.py cv mix_cb_wx`, `docs/LANEB_RUN.md`; flight history on 66.1% of the OOF months' lane B
rows and 74.8% of the holdout's). Holdout lane B 1736.9 against v27's 1741.7; LIRF 3299.7 (v27 3331.1); physical class
1099.8 (v27 1112.4); copy class 2834.9 (v27 2793.9). Ablation: no_hist 1742.5, no_day 1733.9, no_prior 1798.7,
no_blend 1740.5. The day columns lose on the holdout in v27 and here while the fold log loss favours them (0.1211
against 0.1212); they stay, since dropping them would be a pick made on the holdout.

Full file, lane A unchanged (session scratchpad `gate_lb.py`, log `gate_hist.log`): holdout 308.05 against v27's
308.47, +0.42 [-0.26, +1.21]; lane B over all 12 months 1473.4 against 1486.9, +13.6 [+3.7, +26.5].

Where the 12-month gain lands (session scratchpad `hist_blk.py`, log `hist_blk.log`): lane B SSE falls from 49,681M
to 48,779M, and LIRF rows 1 to 4 h late take 931M of the 902M (14,704M to 13,774M on 1,248 rows). There the copy
class gains 677M and the physical class 401M, while the one day-shift row loses 141M. Every other airport and band
moves by 7M or less, except LIRF under 30 min late (77 rows, 50M worse). The holdout months carry 102M of the LIRF gain
and the OOF months 828M. Without the shift row, LIRF 1 to 4 h falls from 7,770M to 6,700M, close to the two-point
stack's fall from months 3+ away (7,997M to 6,765M). The forms differ, so the match is rough, but the classifier
already takes most of what a dedicated LIRF copy stack would, and that stack is not built.

**Verdict.** The mechanism holds over the 12 months, but the full-file holdout gain does not clear zero, so this lane
B is not uploaded alone. The columns stay in `laneb.py`, and the next candidate that carries them is gated against v27
on the full file. That candidate is v28, which adds the lane A second stage of `docs/RECAL.md`: holdout +1.71
[+0.75, +2.88] against v27. `python laneb.py fit 117 238` wrote its `data/laneb_rank.parquet` (5,290 rows, mean
1675.9).

## 7. The plan from the destination's arrival record (not used)

A lane B departure carries no flight plan, but when it flies to one of the other nine airports, the arrival record
there can carry the plan stamps (session scratchpad `dest_a.py`, log `dest_a.log`; join on flight number, origin and
destination, arrival 0 to 12 h after takeoff). On lane A the join finds the right flight: the arrival record's AOBT_3
equals the departure's own within 60 s on 99.8% of 182,429 rows. Of 4,644 lane B departures to one of the ten
airports, 2,906 match an arrival and 39.2% of those carry AOBT_3: 912 train rows and 228 rank rows. On the train rows
the label lies within 5 min of t_off - AOBT_3 on 63.2% (lane A 75.1%).

It does not help (session scratchpad `dest_b.py`, log `dest_b.log`, against v28's lane B; the OOF months without the
donor blend). The recovered stamp as the prediction, max(t_off - AOBT_3 + 60 s, 0) with the offset the OOF median,
loses on lane B: OOF months 1380.2 to 1414.0, -33.8 [-51.0, -20.1]; holdout 1736.9 to 1784.9, -48.0 [-136.6, +0.7].
A blend weight fitted on the OOF months comes out negative (-0.211) and is flat: OOF +1.1 [-1.5, +4.3], holdout -4.2
[-12.9, +0.2]. The reason is the section 6 mechanism. Of the 37 OOF rows with the stamp that are more than 1 h late,
22 are LIRF, and on 16 of those the label equals the schedule to within 6 s while t_off - AOBT_3 reads 724 to 1,260
s. The departure recorder copied the schedule, the plan kept the real pushback, and the label follows the recorder.

## 8. Near rows and the local lane A taxi level (not used)

After v28, the near rows (`classes()`: off-block recorded between 700 s before and 3,100 s after the schedule, label
over 3,000 s) still under-predict: by 4,459 s on average on the 34 holdout rows off LIRF 1 to 4 h late (session
scratchpad `blk_v28.py`, log `blk_v28.log`). In the OOF months 162 of the 201 near rows off LIRF are LTFM, 108 of
them on 2025-02-19 and 2025-02-23. The physical-taxi regressor, which predicts them since near is folded into
physical, sees the neighbours' taxi excess only through `WZ`, clipped at 1,800 s. The unclipped local lane A level
(`w_za_p90`, the p90 of lane A takeoff minus AOBT within 2 h at the airport) and the lane A near share (`w_est_share`)
reach only the copy classifier. Would they help the regressor?

Against v28's lane B residual off LIRF without the shift rows (session scratchpad `near_d.py`, log `near_d.log`;
16,005 OOF rows, 4,975 holdout):

| | OOF months | holdout |
|---|---|---|
| corr of y - p with `w_za_p90` | 0.133 | -0.005 |
| corr with `w_est_share` | 0.002 | 0.064 |
| corr with `w_tail_share` | -0.030 | 0.056 |
| `w_za_p90` 45 to 60 min: rows, near share, mean y - p | 159, 0.296, +979 s | 118, 0.034, -33 s |

In the OOF months the near rows sit beside a slow lane A; on the holdout they do not (near share 0.004 where the p90
is under 30 min, 0.028 at 30 to 45 min). A band-mean correction fitted on the OOF months loses on the holdout: lane B
1736.9 to 1748.7, -11.8 [-25.0, -5.2]. The OOF pattern rests on the two LTFM days. A near row is a stamp on its own
flight, not congestion that lane A shows, so the day columns stay out of the regressor and no `laneb.py cv` run was
spent on it.

## 9. Previous arrival, local delay and runway twins (not used)

Three more label-free signals for the LIRF lane B rows 30 min or more late, each read against the copy share and
against v28's lane B residual (the OOF months without the donor blend, and the holdout).

**Previous arrival on the same stand** (session scratchpad `pa_d.py`, log `pa_d.log`; 1,399 rows, shift rows left
out). Where that arrival's in-block lies within 60 s of its own schedule (`pa_sc`), the copy share is 0.636 in the OOF
months against 0.508 where the in-block copies neither the schedule nor ARVT_1, and 0.387 against 0.331 on the
holdout. Within an airline it mostly goes away: over the 12 months RYR reads 0.304 against 0.337 and WMT 0.312 against
0.493, while AEZ reads 0.889 against 0.760 and WZZ 0.900 against 0.727 on 18 and 10 `pa_sc` rows. Weighted by the
`pa_sc` rows of the eight largest airlines, the within-airline gap is +0.01 against +0.10 pooled. The column carries the
airline mix: AEZ and WZZ, which copy often, have the highest shares of `pa_sc` rows (0.39 and 0.18, against 0.10 at
RYR and 0.09 at WMT). The stand does the same: stand copy shares correlate 0.590 between
the OOF months and the holdout over 28 stands, while within RYR and WMT they spread about as far as binomial noise on
10 to 27 rows allows. One small pocket: rows with no arrival on their stand in the 24 h before the schedule
(`pa_missing`) copy on 0.811 of 53 OOF rows and 0.923 of 13 holdout rows, and v28 sits 1,139 s and 690 s low on them.
The classifier does not see that column, but removing both biases outright is worth about 75M of SSE (53 x 1,139^2 +
13 x 690^2), so no run was spent on it.

**Local lane A delay** (`dly_d.py`, `dly_d.log`). `w_late` is the share of LIRF lane A takeoffs within 2 h of the row
whose takeoff came 30+ min after the schedule. It follows the season (July 0.703 with copy share 0.330, November 0.252
with 0.653), and within an airline the copy share does not order with it (RYR 0.200, 0.636, 0.278 and 0.333 over the
four bands from 0.1 up, on 5 to 183 rows). Its correlation with v28's residual is -0.021 in the OOF months and 0.030
on the holdout.

**Same-runway twins** (`twin_d.py`, `twin_d.log`). 373 of the 22,470 train lane B rows (1.7%, against 0.9% of lane
A) have another departure off the same runway within 30 s. The labels do not follow each other: the twin's label is
within 60 s of the row's own on 24.1%, duplicate records among them (one EDDM flight number four times at one
takeoff), and takeoff minus the twin's AOBT on 8.0%. At LIRF 30+ min late 2.7% of rows have a twin, with copy share
0.368 against 0.493 without.

None of the three goes into `laneb.py`.

## 10. The inbound parked after the schedule (already in v28)

LIRF lane B rows 30 min or more late, relinked to the last arrival that blocked on the same stand at least 20 min
before takeoff (session scratchpad `pb_d.py` and `pb_s.py`, logs `pb_d.log` and `pb_s.log`; 1,399 rows, shift rows
left out, an arrival found for 96.6%). Where that arrival blocked after the departure's schedule, the aircraft came in
late and the departure copies its schedule far more often: 0.791 against 0.345 in the OOF months (426 and 554 rows)
and 0.599 against 0.179 on the holdout (142 and 229 rows). The gap holds inside every large airline: RYR 0.500
against 0.209, WMT 0.738 against 0.244, ITY 0.625 against 0.056. The arrival's own recorder mode adds nothing (in-block
on its schedule against a physical in-block, within airline -0.071).

It is not new information. `features.occupancy` already gives the copy classifier `occ` (the first arrival after the
anchor parked before takeoff) and `t_occ`, and for lane B the anchor is the schedule, so `occ` is this column with a
different cutoff. v28's implied copy share, (p - t0) / (z_sched - t0) with t0 = 1,027 s the median non-copy label,
tracks it band by band:

| late by | inbound | rows | copy share | v28 implied |
|---|---|---|---|---|
| 1-2 h | after the schedule | 196 | 0.704 | 0.661 |
| 1-2 h | before | 538 | 0.243 | 0.297 |
| 2-4 h | after the schedule | 286 | 0.738 | 0.738 |
| 2-4 h | before | 193 | 0.358 | 0.430 |

A logistic regression stacked on logit(implied share), leave-one-month-group-out over the six groups, moves AUC from
0.878 to 0.888 in the OOF months and not at all on the holdout (0.859 both). The two-point SSE falls 173M in the OOF
months and rises 148M on the holdout; the inbound's lateness in hours on top changes neither. The rows v28 misses most
change sides between the two: in the OOF months it sits 96 s high on the before rows and 10 s high on the after rows,
on the holdout 149 s and 354 s.

Over 4 h late the column runs the wrong way: in cell A0 the copy-or-near share is 0.711 on 45 after rows against
0.857 on 14 before rows, and cell D copies on all 36. Lane A shows nothing either: rows 30 min or more late whose last
same-stand arrival (by `pa_gap`, anchored at AOBT) blocked after the schedule copy the schedule on 0.007 against 0.014,
and v28's lane A residual is 10 s higher on them in both halves.

Not used.

## 11. A lane A physical-taxi prediction as a lane B column (in v29, lost on the board)

Lane B's physical-taxi regressor learns from 12,000 lane B rows. Lane A holds 1.8 million physical rows (`mix.label`
0 or 3, 0 < y < 7,200 s) with the same lane-B-visible columns: airport, runway, stand, type, airline, hour, weekday,
`z_sched`, the day, occupancy, weather, the non-`pa_` arrival columns and `WZ`. `tau_a()` fits LightGBM (`LGB` at
learning rate 0.1, early stopping on a tenth of the days) on those rows and scores the lane B rows, out of fold by
month group with the holdout months excluded for 2025 rows and on all of 2025 for the rank rows (session scratchpad
`lbx.py`, log `lbx.log`). On the 5,075 holdout lane B physical rows tau_a scores 968.5 against v28's 1,099.8; it runs
90 to 280 s high at every airport but LIRF and LTFM (the lane B label is shorter than lane A's for the same cell),
which the tau regressor absorbs as a column.

| variant | OOF folds | holdout lane B | full file vs v28 |
|---|---|---|---|
| baseline (fixed occupancy frame) | 1381.5 | 1737.3 | -0.04 [-0.41, +0.69] |
| tau_a in the tau regressor | 1375.2 | 1732.7 | +0.37 [-0.08, +0.99] |
| tau_a in tau, tau_a and z_sched - tau_a in the classifier | 1374.7 | 1729.5 | +0.65 [-0.03, +1.46] |

The third is v29's lane B (iters (107, 257); lane B over the OOF months +5.6 [+0.2, +13.5]). Holdout SSE gain by
airport: LIRF +79M, EGLL +26M, LSZH +19M, EHAM +13M; the top row alone 63M, which is why the interval touches zero.
With the lane A change of `docs/RECAL.md` the full file reads +0.873 [+0.104, +1.708] against v28, and the board
went from 271.783 to 272.2175. v30 (v29's lane A with v28's lane B) scored 271.9284, so this change cost 54M SSE on
the 2026 rows where the holdout promised 138M. Reverted; the code is commit `88afb7e`. The
rank lane B rows moved 157 s on average (v28 to v29), and the January rank set sits in different cells from the
holdout's (EHAM Jan median `z_sched` 144 min against 84 in training, LSZH 98 against 21, session scratchpad `tz_d.py`),
so a column fitted to 2025's lane B level can move the wrong way there.

## 12. Where the board seconds sit: day shifts and the hull rows (after v30)

**Day shifts outside the hull.** Session scratchpad `shiftx.py`, log `shiftx.log`. 2025 has 34 rows with y over
43,200 s in either lane: 31 LIRF (30 lane B, 1 lane A), 2 LFPG, 1 LSZH. Every LIRF one is a hull row (schedule 4 h or
more late): a schedule copy with `z_sched` above 43,200, or a composed stamp at k = 1. The three others are ordinary
flights (schedule gap 939 to 2,043 s) whose recorded off-block sits 16 to 24 h before takeoff; two are EJU at LFPG
in January 2025. Of the 595 non-LIRF 2025 rows more than 12 h late by schedule, 0 shifted, so the 128 such rank rows
(EDDF 47, EHAM 42, LFPG 19, EDDM 8, LEBL 5, LEMD 4, EGLL 3) get no hedge. Nothing predicts the three ordinary-flight
shifts: 41,939 train rows take off between local midnight and 03:00 and 3 of them shifted, all LIRF hull rows.
Arrivals are not a sensor either (scratchpad `arrshift.py`, log `arrshift.log`): 5 train arrivals have a block time
more than 12 h from landing, none on the 30 airport-days that hold the departure shifts. A shift the model calls
physical costs 86,400^2 = 7.5B of SSE, 37 s of board RMSE at v28's score, and every team pays it alike.

**Hull rows.** Scratchpad `hullvar.py`, log `hullvar.log`. Reading v28's prediction on each of the 43 LIRF hull rows
of 2026 as the expected value of a two-outcome label (copy at `z_sched`, composed at 1,100 + 86,400 kn) gives an
implied copy share per row (0.765, 0.8, 0.286 or 0.974, the section 5 cells) and an expected residual SSE of
P (1 - P) gap^2 per row: 4.01B over the 43 rows, 2.59B on the 32 rows at kn = 0 and 1.42B on the 11 at kn = 1.
Knowing every label would move v28 from 271.78 to 249.5; the leader sits at 245.02. Lane A's whole label-free
ceiling is about 8 s (`docs/STATE_ORACLE.md`), so most of the gap to the leader is these 43 rows, and no 2025 check
can gate a bet on them.
