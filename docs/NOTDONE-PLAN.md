# Not-done queue (2026-09-05)

Keith, 2026-09-05: "Make sure you tried all of the not-dones please I don't care how long it takes." This file is the plan for that
instruction. Every item below is declared in `docs/EXPERIMENT-PROTOCOL.md` section 4 before it runs; results land in the ledger as H rows.
Nothing is uploaded; the S3 decision on v14 stays with Keith.

Champions at the start: lane A MC2 (CatBoost GPU, D1's 49 columns, cached holdout 232.9 s), lane B L6a. N_h = 28.

## Order and design

Data layer first, so every later verdict runs on the fixed data.

| # | id | design | k | comparator | cost | ship path if it wins |
|---|---|---|---|---|---|---|
| 1 | A1 | anchor guard in `taxi.data`: the anchor is the first non-negative of AOBT, EOBT, schedule, else the takeoff itself (z = 0); changes z_anchor, the six traffic counts, occ, t_occ and the weather join on about 0.05% of rows. Re-verdict: every cached lane-A champion (W1-A, D1, MC2) and lane-B L6a refit on the fixed data; each new-vs-old paired delta reported as one H row per champion (a data fix, no promotion). Old caches kept as `.pre-a1` copies. | 1 per champion | the same champion on the old data | about 30 min (MC2 refit 8 min GPU) | the v15 file is built from the fixed data; lane A needs a fresh 12-month MC2 refit (v12's predictions were made on the old anchors) |
| 2 | LK1 | leak test `tests/test_leak.py`: after A1 every z_anchor >= 0 on Feb; no non-gap feature's correlation with y exceeds 0.5 (the four gap columns and their plan differences are declared exceptions, they are the label's own scale); the ranking view never selects the label columns (already `test_masked_view`) | none | none | 30 min | none |
| 3 | E2 | fixed-weight blend of cached MC2 and D1 holdout predictions, w on MC2 in {0.5, 0.7, 0.9} | 3 | MC2 cached | seconds | lane A = w x v12 MC2 (by id) + (1 - w) x D1 12-month refit |
| 4 | L6T | L6a's non-copy threshold sweep {10, 15, 20, 25} on lane-B CV (critique condition 3); the CV winner, if not 20, meets the holdout against L6a | 4 | L6a floored | seconds | lane B refit with the winning threshold |
| 5 | SB1 | five-seed bag of MC2's cell at its recorded rounds (seed 0 = the cached refit) | 1 | MC2 cached | 30 min GPU | lane A = mean of v12 MC2 and four 12-month seed refits |
| 6 | WS1 | weather join-shift stress: the as-of join reads the latest observation at or before anchor minus 600 s; MC2's cell at its recorded rounds on the shifted columns | 1 | MC2 cached | 10 min GPU | none (a stress; a loss says the join is fragile, a tie says it is not) |
| 7 | T1 | lane-A target treatment on MC2's cell, two cells chosen on CV: T1a regressors fit asinh(c) and predict sinh; T1b regressors fit c with the MAE loss (median); the classifier unchanged | 2 | MC2 cached | 2 x 40 min GPU | as MC2, from the same fit |
| 8 | G1 | fold-internal target encoding: median of c per (airport, stand, runway) over the fit rows, NaN where unseen, as one numeric column on MC2's cell; recomputed inside every fold and for the final fit | 1 | MC2 cached | 40 min GPU | as MC2 |
| 9 | TW1 | traffic window sweep: the six counts rebuilt at 300, 1,800 and 3,600 s (900 = MC2); each width CV'd on MC2's cell; the CV winner, if not 900, meets the holdout | 4 | MC2 cached | 3 x 40 min GPU | as MC2 on the rebuilt columns |
| 10 | R1 | runway-configuration proxy: share of the airport's takeoffs on this departure's runway in the hour before the anchor (NaN when none), one column on MC2's cell | 1 | MC2 cached | 40 min GPU | as MC2 |
| 11 | H1 | public-holiday flag for the airport's country (ICAO prefix: ED DE, EG GB, EH NL, LE ES, LF FR, LI IT, LS CH, LT TR; `holidays` 0.103) plus day-before and day-after flags, three columns on MC2's cell | 1 | MC2 cached | 40 min GPU | as MC2 |
| 12 | FF | forward fold (train Jan-Nov 2025, predict Dec 2025) for M2, W1-A, D1, MC1, MC2 at their recorded rounds; reported, never decides | none | none | 30 min | none |
| 13 | RD | residual table for MC2 on the holdout by airport x runway x hour, `docs/diag_resid.md`; diagnostic | none | none | minutes | none |
| 14 | M3 | TabPFN per airport on all ten airports as the pilot (50,000-row seeded subsample per fit, M1's columns, folds cached per airport), plus one holdout fit per airport (fit on the ten training months' subsample, predict Jan+Jul) cached in `data/cache/m3_holdout.parquet`. The 30 min time gate is lifted by Keith's instruction above (amendment log). | none on its own | M1 per fold (as the pilot, reported) | about 8 h GPU | none on its own |
| 15 | E1 | blend of cached MC2 and M3 holdout predictions, w on MC2 over an 11-point grid {0.0, 0.1, ..., 1.0}; the best on the holdout is the challenger | 11 | MC2 cached | seconds after M3 | lane A = w x v12 MC2 (by id) + (1 - w) x M3 12-month refits per airport |
| 16 | O1 | Optuna TPE seed 0 on MC2's cell: learning rate log [0.02, 0.2], depth [4, 10], l2 log [0.1, 30], border count {128, 254}, random strength [0, 5], bagging temperature [0, 2]; objective = five-fold month-pair CV RMSE on MC2's columns and recipe; 40 trials or 24 h of wall clock, whichever first; the best trial's cell refit at its CV rounds meets the holdout | trials completed (40 at most) | MC2 cached | 10 to 24 h GPU | as MC2, from the same fit |

Not in the queue and why: day-of-year seasonality (rule R7 bans month; a smooth seasonal term is a rule change only Keith can make);
interaction terms (the boosters find them); the E1 draft's "W1-A and M3" partner is replaced by the current champion MC2.

## Queues

One detached sequential runner (`scripts/run_queue.py`, launched with PowerShell Start-Process) runs the GPU work one job at a time, in
this order: the MC2 and D1 OOF caches, E2, SB1, WS1, T1, G1, TW1, R1, H1, FF, M3, E1, O1. Each job logs to `data/queue/<job>.log` and
leaves `<job>.done` or `<job>.failed`; a failure is recorded and the queue moves on; a relaunch skips finished jobs. CPU items (A1,
LK1, L6T, RD) ran or run in the foreground first; LK1's full-year pass runs detached beside the queue (CPU only).

## Reviewer findings applied before launch

- E1 and E2 pick the weight on the month-pair OOF predictions (`taxi.champ.champion_oof`), not on the holdout; the holdout is touched once.
- A1's MC2 row is reported beside the same-seed GPU refit noise floor (+0.24 s [-0.05, +0.54]).
- Ship paths refit on the 12 months after A1 via `taxi.s2`; no v12 predictions are copied.
- T1, TW1 and L6T: the CV winner alone meets the holdout; TW1 skips the holdout when no width beats MC2's CV.
- O1's k = trials completed is a floor on the optimism (TPE draws are not independent); the study resumes from `data/o1.db`.
- M3 is compared with MC2's OOF and holdout per airport, reported only; E1 carries the verdict.
- G1: at least 20 rows per (airport, stand, runway), (airport, stand) fallback, then NaN; the fit rows are encoded out-of-month.
- LK1 is a test (`tests/test_leak.py`) plus the full-year script (`taxi.lk1`).

## Short searches

Rule B of protocol section 3: a declared search under its declared k is a NOT-DONE row here, with the reason, or
`tests/test_search_completion.py` goes red. Counts stay OUT of this table on purpose, so the contract does not churn
once per trial; `python -m taxi.searches` prints declared against achieved from the study databases themselves.

| search | why it stopped short | what is being done |
|---|---|---|
| O1 | Killed by hand 2026-09-07 when O2 was declared to supersede it, at 8 of 40 trials. Its 24 h budget was feasible against 22 h of measured need, so nothing forced the kill. **The restart this row claimed for 2026-09-08 never ran.** `data/o1.db` was found on 2026-09-10 holding the optuna schema and zero studies, and `data/o1.log` holding zero bytes since 2026-09-08 22:18, so the eight trials are gone and no replacement trial was ever attempted. The quoted best of 212.68 belonged to the killed study and is no longer reproducible from anything in this repo. | Re-queued 2026-09-10 behind the LIVE O2 process through `scripts/gpu_queue.sh`, which tops O2 up to its 40 first. The card order was CHANGED later the same day, and O1 now runs FOURTH, after DQ1 and NN3REF, because O1's own protocol row says its verdict is against MC2 on D1's columns and MC2 is two champions behind, so it "cannot promote anything", while DQ1 can. k is untouched, the study still runs to 40, only its place in the queue moved. The card is why it waits rather than starts: O2 holds 15.6 GB of a 16 GB card. This row is deleted when the study reaches 40, and it now says what a reader can check, `python -m taxi.searches`, rather than a number no file still holds. |
| O2 | Its declared "40 trials or 10 h" cannot both hold, and the rate that showed it was itself wrong twice over. 40 min a trial ended the first run near trial 15 at 4 completed trials; the real rate, read off the study on 2026-09-10 by `taxi.searches.measured_s`, is 73 min a trial, because 14 of 35 trials never completed and a mean over completions cannot see the clock those burn. | `TIMEOUT` raised 10 h to 40 h on 2026-09-08, then 40 h to 56 h on 2026-09-10 so 40 trials at the measured rate fit inside one launch. k stays 40 and so does the k = 40 bound; only wall clock moved. 21 complete as of 2026-09-10, best trial 12 at 213.47 s against trial 0 at 214.33, so 0.85 s of CV against its own base and below the 1.0 s bound. |
