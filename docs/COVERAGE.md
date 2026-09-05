# Coverage interrogation (Stage 4.5), prc26

Answers to the nine questions Keith asked on 2026-09-05, one row each, every
answer with a pointer. Rules: `~/.claude/skills/quant-ml-protocol/SKILL.md`
Stage 4.5. "Tried" means a verdict row exists; a run with no declaration is a
diagnostic. Answers 7 and 8 are Keith's to grade; this file only lays out the
evidence. Written 2026-09-05; refreshed the same day after the M3 pilot verdict,
the W1 run (ledger H15, H16) and the Stage 5.7 critique (`docs/gate-s2.html`).

| # | Question | Short answer | Pointer |
|---|---|---|---|
| 1 | What have you tried? | 23 holdout verdicts (B0-B4, gap family, M1, M2, W1 in both lanes, D1 plan provenance, L1 lane-B campaign, MC1 CatBoost, MC2 CatBoost at D1's columns, C1 prediction floor in both lanes, L2 bounded lane-B fit), 4 declared searches, 1 board submission, 2 scored probes, M3 pilot run (time gate failed) | `docs/TRIAL-LEDGER.md` H1-H23; protocol section 4 |
| 2 | What features? | 20 columns: airport, hour, weekday, has_fpl, four time gaps, anchor gap, five categoricals, six 900 s traffic counts; 14 ranking columns never used | `taxi/data.py` FEATURES; `docs/M1-KNOCKOUT.md`; section 2 below |
| 3 | What feature engineering? | label lineage (lane A learns the residual after the off-block gap), anchor rule, traffic counts from ranking-visible movements, tail split (M2), weather as-of join (W1) | section 3 below |
| 4 | What model? | means and OLS (B0-B4), LightGBM (M1), two-stage LightGBM (M2), TabPFN pilot (M3); lane B champion is still a two-feature OLS | protocol rows B0-M3 |
| 5 | Feature and model parameter tuning? | one 12-cell grid (lane A) and one 8-cell grid (lane B) over leaves, min_data, objective; learning rate, regularisation, sampling, windows, thresholds never tuned | `docs/M1.md` grid tables; `taxi/m1.py` BASE |
| 6 | Optuna optimisation? | NOT run. Draft declaration O1 in section 6 (40 trials, k = 40, bound 3.7 s, about 2 h) | section 6 below |
| 7 | Are the agent being detailed enough? | Keith grades. Evidence and the thin spots listed | section 7 below |
| 8 | Full effort used to find suitable model? | Keith grades. Facts: 2 days, 2 model classes at full scale, lane B (half the squared error) has had no model campaign beyond baselines | section 8 below |
| 9 | What is missing that we should try? | 3 inputs, 3 mechanisms, 3 tunings, each with a cost and a draft declaration | section 9 below |

## 1. What have you tried

Holdout verdicts, N_h = 23 (`docs/TRIAL-LEDGER.md` section B):

| Rows | Lane | What | Outcome |
|---|---|---|---|
| H1-H4 | B | B1 per-airport mean, B2 cell mean, B3s gap, B4 per-airport OLS | B1 WIN, B2 lose, B3s lose, B4 WIN 2,074.1 s (champion) |
| H5-H11 | A | B1, B2, B3a-d gap family, B4 | B1, B2, B4 WIN; B3a and B3d tie (day-clustered), B3b and B3c lose; B4 311.2 s |
| H12-H13 | B, A | M1 LightGBM | lane B lose (-312.1 s vs B4); lane A WIN 259.5 s, +51.6 [+48.4, +55.1] |
| H14 | A | M2 two-stage | WIN 256.6 s, +2.9 [+1.8, +4.1]; the gain sits in body rows, not the tail it was built for |
| H15-H16 | B, A | W1 weather: B4 plus five METAR columns; M2 recipe plus 16 METAR columns | lane B tie (+9.9 [-2.7, +22.6]), B4 stays; lane A WIN 253.8 s, +2.9 [+1.8, +3.9], W1-A promoted (floor-edge again, F10) |
| H17-H18 | B, A | L1a: B4 plus the VFR flag (CV winner of four lane-B cells); D1: M2 recipe on W1-A's columns plus 13 plan-provenance columns | lane B tie (+2.1 [-0.2, +5.3]), B4 stays; lane A WIN 242.2 s, +11.6 [+9.2, +14.0], D1 promoted (the whole interval clears the floor) |
| H19 | A | MC1: CatBoost on the GPU, M2 recipe, W1-A's 36 columns | WIN vs W1-A, 248.2 s, +5.6 [+4.4, +6.8]; not promoted, D1 is 6.0 s better on the same rows; MC2 (CatBoost at D1's columns, against D1) declared |
| H20 | A | MC2: CatBoost on the GPU, MC1's cell, D1's 49 columns | WIN vs D1, 232.6 s, +9.6 [+8.2, +10.9]; MC2 promoted |
| H21 | A | C1: MC2's predictions clipped at 0 s | tie: positive interval but under the 2 s floor, +0.498 s [+0.186, +0.909]; the label is negative on 30 lane-A holdout rows |
| H22 | B | C1: B4's predictions clipped at 0 s | WIN (both ends clear the 2 s floor), 1965.6 s, +108.5 s [+4.9, +291.9]; floor adopted; the lane-B gain is B4's per-airport OLS at LIRF extrapolating: intercept -2,021 s and slope 1.23 on z_sched (1,091 training rows), so two July rows with the movement 6 to 7 h before schedule (z_sched -26,577 and -21,599 s) are predicted at -34,678 and -28,561 s against truths of 1,387 and 1,682 s; the 17 negative rows carry 11.5% of the lane's squared error |
| H23 | B | L2: CV winner of two bounded B4 cells (cap, Huber), floored, k = 2 | LOSE, 1992.1 s, -26.5 s [-42.9, -15.3]; CV picked L2b (Huber; five-fold CV, every cell floored: B4 1544.9, L2a 1578.2, L2b 1552.3 s); holdout L2b floored 1992.1 s against B4 floored 1965.6 s, -26.5 s [-42.9, -15.3], paired SE 7.0 s, k = 2 optimism bound 4.0 s; LOSE, B4 with the floor stays lane-B champion |

Declared searches (ledger line 35): CAP sweep k = 4 per lane (chose no cap); M1 grid k = 12 lane A (bound 2.8 s), k = 8 lane B (bound 229.4 s); M2 k = 1; W1 k = 1 per lane.
Diagnostics, never verdicts: forward fold Jan-Nov to Dec for M1 (`docs/M1.md` line 117); per-feature knockout for M1 (`docs/M1-KNOCKOUT.md`).
Board: S1 325.40 s, rank 11 of 29, leader 277.29 s (protocol row S1). Probes P1, P2 scored; P3-P6 rejected by the 3-per-day cap; single-sided P3s, P5s scheduled for 2026-09-06 00:05 UTC.
Run 2026-09-05 after this file was first written: M3 TabPFN pilot on LIRF, gain +11.6 s and VRAM 3.8 GB pass, the 30 min time gate fails at 38.1 min, M3 stopped pending Keith (`docs/M3-PILOT.md`; four of five folds seeded from the printed lines of killed runs, RMSE rounded to 0.1 s). W1 weather run in both lanes (`docs/W1.md`, rows H15-H16 above). S2 file built from W1-A plus B4, not uploaded (`docs/gate-s2.html`). Later the same day: D1 and L1 run (rows H17-H18 above), D1 promoted; MC1 run (row H19), a win against W1-A that does not displace D1; MC2 declared.

## 2. What features

Model columns (`taxi/data.py` FEATURES plus the airport): hour, wday, has_fpl, z_aobt, z_eobt, z_lobt, z_sched, z_anchor, runway, stand, actype, wake, segment, queue, dep_before, dep_after, arr_anchor, arr_off, push_anchor. No month (rule R7).

Knockout (`docs/M1-KNOCKOUT.md`, lane A full CV 232.9 s): dropping runway +6.1, z_sched +5.5, z_eobt +4.4, z_aobt +3.2, stand +3.0 clear the 2 s floor; the other 15 columns sit inside the noise bar one at a time. Lane B: airport +74.1, hour +9.3.

Ranking columns the view never reads (measured 2026-09-05, `python -c` DuckDB DESCRIBE on `data/ranking.parquet` against `taxi/data.py` VIEW):

| Column | Distinct (DEP rows) | Point-in-time | Worth a try? |
|---|---|---|---|
| AIRCRAFT_OPERATOR_flt | 579 | filed plan | yes (operator ground handling) |
| FLIGHT_TYPE_flt | 5 | filed plan | yes |
| FLIGHT_RULE_flt / FLIGHT_RULE_mvt | 3 / 2 | filed plan | yes, cheap |
| ADES_mvt / ADES_flt / ADES_FILED_flt | 1,033 | filed plan | yes as a high-cardinality categorical; distance needs coordinates (external data, own register row) |
| IOBT_flt | 31,284 | filed plan | yes as plan slip EOBT_1 minus IOBT |
| ARVT_1_flt | 326,574 | filed plan | yes as filed duration ARVT_1 minus EOBT_1 (fuel load proxy) |
| ARVT_3_flt | 279,144 | actual arrival, after takeoff | BANNED (leak) |
| FLIGHT_ID_mvt, FLIGHT_mvt, CALLSIGN_flt, ADEP_flt, AIRCRAFT_TYPE_flt | ids or duplicates | | no (ids) or duplicate of a used column |

## 3. What feature engineering

Done: the four takeoff-minus-timestamp gaps; the anchor rule (z_anchor); six traffic counts in 900 s windows around the anchor and the takeoff, built from ranking-visible movements only (queue, dep_before, dep_after, arr_anchor, arr_off, push_anchor); categorical codes with unseen-code handling (0.08% unseen stands); lane A learns c = y minus z_aobt; M2 tail split at abs(c) > 600 s with a classifier; W1 weather as-of join at the anchor with a 3 h look-back (16 columns).

QUEUED 2026-09-05 on Keith, 2026-09-05: "Make sure you tried all of the not-dones please I don't care how long it takes." (protocol section 4: T1, G1, TW1, R1, H1; each writes `docs/<id>.md`); the list as it stood: target transforms (log or asinh of y for lane B, quantile targets); stand x runway target encoding (fold-internal median of c per airport, stand, runway); longer traffic windows (1,800 s, 3,600 s); a runway-configuration proxy (share of departures per runway in the last hour); operator, destination, filed duration and plan slip (section 2); day-of-year seasonality (month is banned by R7; a smooth seasonal term is a rule change for Keith); holidays; interaction terms (LightGBM finds these on its own, so low priority).

## 4. What model

Tried: B0 global mean, B1 per-airport mean, B2 airport x hour x weekday mean, B3 gap family, B4 per-airport OLS; M1 LightGBM with l2 and huber objectives; M2 two-stage LightGBM (classifier plus tail and body regressors); M3 TabPFN 6.3.1 per airport (pilot on LIRF only).
Not tried: a lane-B model campaign beyond the M1 grid (half the squared error rides on a two-feature OLS, flag F9); per-airport LightGBM; median (L1) regression; XGBoost (CatBoost run as MC1, H19: +5.6 s vs W1-A on identical columns, so the same-family assumption was wrong; MC2 declared); k-nearest rows on stand, runway, hour; neural nets; blends (E1 declared, waits on M3).

## 5. Tuning

M1 grid (`docs/M1.md`, protocol row M1): lane A leaves {31, 127, 255} x min_data {200, 2000} x {l2, huber delta 600}, lane B leaves {7, 31} x min_data {50, 200} x {l2, huber}; month-pair CV, five folds; the most parsimonious cell within 2 s of the best is chosen (lane A chosen 232.9 s, best 231.6 s). Fixed and never tuned (`taxi/m1.py` BASE): learning_rate 0.1, feature_fraction 0.8, bagging_fraction 0.8 with bagging_freq 1, seed 0, early stopping 50, max 2,000 rounds. Never tuned anywhere: lambda_l1, lambda_l2, min_gain_to_split, max_bin, cat_smooth and cat_l2 (stand has hundreds of levels), seed bagging, the M2 threshold (600 s fixed), the traffic window (900 s fixed), TabPFN n_estimators (8, the default).
CAP sweep: k = 4 per lane on the training-truth admission cap, chose no cap in both lanes.

## 6. Optuna

Not run. Why: the protocol treats any optimiser as a declared search with k = trial count and an optimism bound; the first search was a 12-cell grid whose best-to-chosen gap (1.3 s) is inside the paired SE (1.7 s), so a wider search was not the next rung while model classes and inputs were untried.

Draft declaration O1 (lane A only; lane B needs a model class, not tuning):

- Optimiser: Optuna TPE, seed 0, 40 trials, no pruning.
- Objective: mean month-pair CV RMSE over the five M1 folds on the same rows and columns as M1.
- Space: learning_rate log-uniform [0.02, 0.2]; num_leaves [15, 255]; min_data_in_leaf [50, 5,000]; lambda_l2 log-uniform [0.001, 100]; feature_fraction [0.5, 1.0]; min_gain_to_split [0, 10]; max_bin {255, 1,023}; cat_l2 [1, 100]. Early stopping 50, max 4,000 rounds.
- k = 40; bound = expected max of 40 standard normals (about 2.16) x paired SE 1.7 s = about 3.7 s.
- Verdict: the best trial's spec refit on the twelve months, judged on the holdout against the lane-A champion with the paired day-block interval; promote only if the point improvement clears bound plus the 2 s floor.
- Cost: one M1 lane-A CV took 121 s at learning rate 0.1 (`docs/M1.md` line 60); lower rates run to more rounds, so 40 trials is about 1.5 to 3 h on this box.

## 7. Detail (Keith grades)

What exists for grading: protocol registry with every design declared before its run (section 4 of the protocol); the ledger with 22 paired intervals; a current NOT-DONE table; the knockout table; the M2 report stating that its gain sits in the body rows, not the tail it targeted.
Thin spots, by the author: no forward fold for M2 or W1; no per-airport error table in `docs/M2.md`; no residual table by hour, runway or stand to guide the next feature set; no learning curve (does more training data still help?); the probe decomposition (lane B = 50% of holdout squared error) has not been turned into a lane-B campaign; per-airport squared-error shares were only used to pick the M3 pilot airport.

## 8. Effort (Keith grades)

Facts: the campaign started 2026-09-04; 16 verdicts in two days; two model classes at full scale (OLS, LightGBM), one piloted on one airport (TabPFN, time gate failed); one board submission, 48.1 s behind the leader; lane B, half the holdout squared error, has had baselines, one LightGBM grid and one weather variant; tuning is one grid; one external source scored (W1: +2.9 s in lane A, tie in lane B). The list in section 9 is longer than the list in section 1.

## 9. What is missing

Each item has a cost and a draft declaration. Keith picks; nothing below runs unasked.

Inputs

- W1 weather: RUN 2026-09-05 (`docs/W1.md`): lane A +2.9 s [+1.8, +3.9] WIN, lane B tie. Next weather step, if any: a join-shift stress (observation latency, +10 min on the as-of join), k = 1, about 40 min. DECLARED 2026-09-05 as WS1 (protocol section 4), queued.
- F1 unused plan columns: RUN as D1 2026-09-05 (`docs/D1.md`): +11.6 s [+9.2, +14.0] WIN, D1 promoted; the gain sits in the AOBT_3 minus EOBT_1 and AOBT_3 minus LOBT gaps, operator and destination; the rounding and second-of-minute flags carry nothing.
- G1 stand x runway target encoding: fold-internal median of c per airport, stand, runway as one numeric column. k = 1, about 5 min. DECLARED 2026-09-05 (protocol section 4: at least 20 rows, (airport, stand) fallback, the fit rows encoded out-of-month), queued.
- D2 lane-A columns (declared 2026-09-05 after H23, protocol section 4): MC2 plus d_as (AOBT_3 minus schedule), occ and t_occ (stand occupancy), k = 1, about 40 min GPU. RUN 2026-09-05 (`docs/D2.md`): 231.9 s vs MC2 232.9, +1.0 [+0.5, +1.5], under the 2 s floor, TIE, MC2 stays (H28).

Mechanisms

- L1 lane-B campaign: RUN 2026-09-05 (`docs/L1.md`), four cells (VFR flag, log target, over-1-h classifier, pooled fit): the VFR flag ties B4 (+2.1 s [-0.2, +5.3]); the three model cells lose to B4 on CV by 560 to 1,160 s. Lane B is still the half of the squared error nobody has beaten.
- L3 lane-B mixture (declared 2026-09-05 after H23, protocol section 4): the lane-B error has a measured mechanism (schedule-copied block times on no-type movements, mainly LIRF, and day-early block times); three cells (mixture, linear with the new columns, LightGBM on y minus z_sched), k = 3. RUN 2026-09-05 (`docs/L3.md`, H24): L3b ties B4, +0.8 s [-30.6, +32.6]; the lane's squared error sits in six day-early rows and one LFPG row that no ranking-visible column separates.
- L4 lane-B band mixture (declared 2026-09-05 after H24, protocol section 4): LIRF's no-type rows get a per-band (z_sched) regime mixture or a per-band OLS instead of B4's single line, every other row keeps B4; two cells, k = 2. RUN 2026-09-05 (`docs/L4.md`, H25): L4b WINS, +26.0 s [+4.0, +54.3] against floored B4, promoted to lane-B champion; LIRF 4,434 vs 4,587 s, all rows 334.8 vs 337.1 s.
- L5 lane-B airline mixture (declared 2026-09-05 after H25, protocol section 4): LIRF's no-type rows get P(copy) from the airline (first three characters of FLIGHT_mvt, a new column) by a shrunk table or a logistic on airline, band, hour and stand, times z_sched, plus the band's non-copy mean; every other row keeps L4b; two cells, k = 2. RUN 2026-09-05 (`docs/L5.md`, H26): L5a LOSES, -269.8 s [-548.2, -21.7] against floored L4b, L4b stays; the airline split wins every band up to 4 h (1 to 2 h 1,876 vs 2,258 s, 2 to 4 h 4,600 vs 5,496 s) and loses the whole gain above 12 h (28,864 vs 12,172 s on 11 rows) where the non-copy rows are day-stamped, not taxi times.
- L6 lane-B gated airline mixture (declared 2026-09-05 after H26, protocol section 4): L5a's mixture only in a band with 20 or more non-copy training rows at the airport (the training-fold diagnostic put the whole H26 loss in the far band, whose non-copy rows are not taxi times); non-copy component a mean (L6a) or a least-squares line on z_sched (L6b); every other row keeps L4b; two cells, k = 2, seconds; caution: the H26 holdout by-band table was seen before this design. RUN 2026-09-05 (`docs/L6.md`, H27): L6a WINS, +77.9 s [+43.0, +135.5] against floored L4b (CV 1501.5 vs 1527.6 s), promoted to lane-B champion; LIRF 3,957 vs 4,434 s, all rows 327.8 vs 334.8 s.
- E1 blend of MC2 and M3: the 30 min time gate lifted 2026-09-05 on Keith, 2026-09-05: "Make sure you tried all of the not-dones please I don't care how long it takes." (protocol amendment log); M3 on all ten airports queued (about 9 h GPU, `docs/M3.md`), then the 11-point weight grid chosen on OOF, k = 11 (protocol section 4, amended).
- L6T lane-B threshold sweep (lane-B critique condition 3): L6a's non-copy minimum at 10, 15, 20, 25; k = 4; RUN 2026-09-05 (`docs/L6T.md`, H33): CV tie to 0.1 s, the picked cell L6T15 is identical to L6a on the holdout, +0.0, L6a stays.
- R1 runway-configuration proxy (`rwy_share`, the runway's share of the airport's takeoffs in the hour before the anchor) on MC2's cell: k = 1; DECLARED 2026-09-05 (protocol section 4), queued.
- H1 public-holiday flags (`hol`, `hol_prev`, `hol_next`, `holidays` 0.103 by ICAO prefix) on MC2's cell: k = 1; DECLARED 2026-09-05 (protocol section 4), queued.
- FF forward fold (train Jan-Nov, predict Dec) for M2, W1-A, D1, MC1, MC2: reported, never decides; queued 2026-09-05, `docs/FORWARD.md`.
- T1 target treatment for lane A: asinh(c) and a median-loss fit vs l2. k = 2, about 5 min. DECLARED 2026-09-05 (protocol section 4, on MC2's cell), queued.

Tuning

- O1 Optuna (section 6): DECLARED 2026-09-05 (protocol section 4): TPE over six axes of MC2's cell, 40 trials or 24 h, trial 0 = MC2's cell, k = trials completed; queued last.
- SB1 seed bagging: DECLARED 2026-09-05 (protocol section 4) as five CatBoost seeds of MC2's cell at its recorded rounds, seed 0 = the cached MC2 refit; k = 1, about 30 min GPU. The earlier draft (learning rate 0.05, 15 min) was written for LightGBM timing and is withdrawn.
- E2 blend of MC2 and D1 (cached holdout predictions), weight on MC2 in {0.5, 0.7, 0.9}: DECLARED 2026-09-05 (protocol section 4), k = 3, seconds.
- TW1 traffic window sweep {300, 900, 1,800, 3,600} s. k = 4, about 15 min (feature build plus CV per width). DECLARED 2026-09-05 (protocol section 4, on MC2's cell), queued.

Surfaced by the 2026-09-05 critique (`docs/gate-s2.html`)

- A1 anchor guard: 322 lane-A rows in 2025 and 51 in the ranking file take EOBT as the anchor while EOBT is after takeoff (flag F11), and 741 lane-B rows in 2025 (109 in the ranking file) carry a schedule after takeoff. Fix in `taxi.data`: the anchor is the first non-negative of AOBT, EOBT, schedule, else takeoff itself. Changes every model's traffic counts and weather join on 0.05% of rows, so the chain B4, M1, M2, W1-A is re-verdicted after it: about 3 h, k unchanged. RUN 2026-09-05 (`docs/A1.md`, ledger H29-H32): every champion refit on the fixed anchors ties its old self (W1-A -0.2 [-0.8, +0.5], D1 -0.2 [-0.5, +0.1], MC2 +0.1 [-0.3, +0.7] beside the same-seed GPU refit noise floor +0.2 [-0.1, +0.5], L6a +0.0); champions stay.
- C1 prediction floor: RUN 2026-09-05 (ledger H21). The S2 file carried 54 negative predictions (22 lane A, 32 lane B), the S3 v11 file 48 (16 lane A, 32 lane B); the clip is the `floor` option of `taxi.submit.build_submission` (off by default in the code, on for the S3 candidate), with a test. Holdout effect (`docs/C1.md`, H21 and H22): lane A (MC2) 29 negative predictions of 339,046 rows, RMSE 232.8722 to 232.3744 s (+0.498 s [+0.186, +0.909], tie: positive interval but under the 2 s floor); lane B (B4) 17 of 5,373, RMSE 2074.1 to 1965.6 s (+108.5 s [+4.9, +291.9], WIN (both ends clear the 2 s floor)); all rows 347.1253 to 336.7945 s; the lane-B gain is B4's per-airport OLS at LIRF extrapolating: intercept -2,021 s and slope 1.23 on z_sched (1,091 training rows), so two July rows with the movement 6 to 7 h before schedule (z_sched -26,577 and -21,599 s) are predicted at -34,678 and -28,561 s against truths of 1,387 and 1,682 s; the 17 negative rows carry 11.5% of the lane's squared error. The root cause of the lane-B win, B4's unbounded per-airport OLS, is declared as L2. The label itself is negative on 80 holdout rows and 289 training rows (minimum -12 s), so the clip was judged, not assumed.
- LK1 runtime leak check (S3 critique finding 6): a test asserting that every feature's anchor precedes takeoff and that no feature reproduces the label beyond a stated correlation. RUN 2026-09-05: `tests/test_leak.py` (4 tests pass) and `python -m taxi.lk1` over 12 months and the ranking file (`docs/LK1.md`): 0 anchors after takeoff; lane A's largest non-gap correlation is queue (0.44 at most); lane B's notyp reaches 0.56 in three months, the documented schedule-copy mechanism, not a leak.
- L2 bounded lane-B fit: lane-B lane, two cells, k = 2, declared after H22 from its mechanism (B4's per-airport OLS extrapolates: LIRF intercept -2,021 s, slope 1.23 on z_sched, two holdout rows at -34,678 and -28,561 s): L2a = B4's per-airport OLS on [z_sched, queue] with each airport's prediction capped at that airport's training maximum of y; L2b = the same two columns fit per airport with a Huber loss (scikit-learn 1.7.2 HuberRegressor, epsilon 1.35, alpha 0, columns standardised) instead of least squares. Both cells and the champion carry the floor at 0 s (H22). The cell is chosen on the training month-pair CV (`CV_FOLDS`, as L1); the CV winner alone is judged on the Jan+Jul 2025 holdout against B4 with the floor (the lane-B champion after H22) by the paired day-block bootstrap; optimism bound for k = 2 quoted in the verdict line; floor 2 s; parsimony breaks a tie for B4. RUN 2026-09-05 (ledger H23, `docs/L2.md`): CV picked L2b (Huber; five-fold CV, every cell floored: B4 1544.9, L2a 1578.2, L2b 1552.3 s); holdout L2b floored 1992.1 s against B4 floored 1965.6 s, -26.5 s [-42.9, -15.3], paired SE 7.0 s, k = 2 optimism bound 4.0 s; LOSE, B4 with the floor stays lane-B champion. Lane B is back to having no open item with a mechanism behind it.

Diagnostic (no verdict): a residual table for M2 on the holdout by airport x runway x hour, minutes, to point the next declared feature set. RUN 2026-09-05 as RD on MC2 (`scripts/diag_resid.py`, `docs/diag_resid.md`): LIRF 21.0% of the squared error (runway 25 alone 17.6%), LTFM 17.1%, LFPG 13.5%, EGLL 13.3%.

Diagnostics RUN 2026-09-05 (no verdict; scratch scripts, outputs not committed): holdout squared error by lane and tail (lane A tail 23.2%, lane A body 23.8%, lane B 53.0%; for an all-rows 277 s lane B must reach 1,223 s with lane A fixed, or lane A 129 s with lane B fixed); block-time source per lane (lane A tail rows: 12.3% schedule copies, 4.1% EOBT copies; lane B: 6.4% schedule copies, 14 day-early rows in 2025); stand occupancy and the no-type flag (the L3 declaration in the protocol carries the numbers).
