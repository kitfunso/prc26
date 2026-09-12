# PRC 2026 taxi-out, campaign 2 (from first principles)

Deadline 11 Oct 2026 23:59:59 CET. Board 2026-09-10: ours 294.99 (rank 30/95), leader ~245.
Board = RMSE over Jan+Jul 2026 departures at 10 airports; best file per team counts; 3 scored files a day.
Campaign 1 lives on branch `v1` (100 modules, 64 holdout verdicts, exhausted at 2 s gains).

## Reframe
Label = takeoff time (given) minus off-block time (blanked). The unknown is ONE timestamp.
Composite squared error splits ~51/49 between lane A (339,551 rows with a flight plan, ~206 s) and
lane B (5,290 rows with no plan, ~1,618 s). Reaching 245 needs lane B ~1,000 s or lane A ~142 s.

## Steps
0. Archive v1, orphan master, keep data + keys + licence.           [done]
1. Data pass: per-airport mechanism of the recorded off-block (AOBT copy / schedule copy / physics / day shift).
2. Constraints in the test file itself: stamps, stand bounds from arrivals, runway sequence, turnaround.
3. Noise floor from the simplest mixture; decides whether lane A or lane B gets the time.
4. External data: no ADS-B (licence, and not decided by Keith). Web check of what other teams say they use.
5. Model dev: baselines, leakage+drift audit, CV by month with Jan/Jul holdout, family comparison,
   Optuna tuning on CV only, ablation, error analysis, seed bag + blend on OOF, full-2025 refit, file checks.
6. Board: LEDGER.md row per upload (expected vs actual). Upload only expected gains > ~3 s, CI excluding zero,
   after `board.py check`. Day budget: 3 scored files; never more than 2 on a day without a fresh holdout win.
7. Ship: README reproduces the best file, requirements pinned, repo public GPLv3 before the deadline.

## Rules folded in from the plan review
- As-of rule: features may use any movement in the same file, before or after the takeoff (the rules allow it;
  dep_after and arr_off look forward). State it in the README.
- Predictions are floored at 0 (no upper clip). Month and year are never features; hour/wday are.
- Optuna: price 2-3 trials in wall time before setting the budget; SQLite study; CV only, holdout untouched.
- Every result line: per lane and per month, plus a paired day-block bootstrap against the previous best.
- Lane B is a mechanism mixture (schedule copy, day shift, physics); the RMSE-optimal answer is its conditional mean.

## Files
data.py (load + features, cached), explore.py (steps 1-3 numbers), train.py (CV, tune, refit),
submit.py (build + check + upload), board.py (leaderboard), LEDGER.md.
