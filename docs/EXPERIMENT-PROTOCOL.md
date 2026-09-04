# Experiment Protocol: prc26 taxi-out regression

Instantiated 2026-09-04 from `~/.claude/skills/quant-ml-protocol/templates/EXPERIMENT-PROTOCOL.md`.
Every number below was measured on this box on 2026-09-04; the command that regenerates it sits beside it.
Rules and data layout: `NOTES.md`. Site: https://prc-data-challenge-2026.netlify.app/ (deadline 11 Oct 2026 23:59:59 CET).

## 0. Sample-size math (before anything else)

| Quantity | Value | Source / regenerate with |
|---|---|---|
| Training anchors (DEP rows with target, 2025-01..2025-12) | 2,085,047 | `python -c "import json;print(json.load(open('data/view.json'))['n_dep_train'])"` |
| Ranking DEP rows (to predict) | 344,841 | same file, key `n_dep_rank` |
| Scored rows on the board (usedPairs) | 215,876 | `python lb.py`; identical on all 50 page-1 entries |
| Holdout = Jan + Jul 2025 DEP rows | 344,419 (153,706 + 190,713) | `view.json` key `months` |
| Holdout rows with AOBT_3 | 339,050 | `python scripts/noise_bar.py` |
| Independence unit | day x airport block (flights in one block share the queue) | design choice, attackable |
| n_independent, holdout | 620 (62 days x 10 airports) | `python scripts/noise_bar.py` |
| n_independent, training | 3,650 (365 x 10) | arithmetic on the row above |
| Feature cap (n_indep / 8, holdout) | 77 | arithmetic |

Noise yardstick, measured with a day-block bootstrap (300 reps, each resampled to the scored size of 215,876 rows)
on the raw AOBT_3 residual (takeoff minus AOBT_3 minus truth) over the holdout:

| Statistic | Value |
|---|---|
| RMSE of raw AOBT_3 residual, holdout | 428.2 s |
| SE of the RMSE LEVEL, day-block bootstrap | 10.2 s (95% band 406.5 to 446.1) |
| SE of the RMSE level, iid-row subsample | 2.2 s |
| SE of the level, Gaussian formula rmse/sqrt(2n) | 0.65 s (wrong here by 3x to 15x: heavy tails plus day clustering) |
| Paired SE of delta-RMSE between two predictors on the same rows | 1.03 s (for a delta of 19.4 s) |
| Tail: rows with abs(residual) > 600 s | 7.6% of rows, 72.4% of squared error |

Regenerate: `python scripts/noise_bar.py` (DuckDB over `data/training_*.parquet`, numpy, seed 0).

Consequences written into the decision rule:
- **Detectable-effect floor = 2 s paired delta-RMSE** on the holdout (2 x paired SE). A delta under 2 s is a tie.
- **Levels do not transfer.** A different set of days moves the RMSE level by about 10 s (1 SE). A local holdout level is never compared with a board level; only within-set deltas are.
- The earlier session note "SE about 0.4 s, gaps under 1 s are noise" used the Gaussian formula and is withdrawn.

## 1. Instrument and target

- Instrument: N/A. No trade, no costs. The money metric is the leaderboard RMSE; a submission is free and unlimited (the site states no cap; checked 2026-09-04, `NOTES.md`).
- Target: `TAXITIME_SEC_mvt` in raw seconds for `PHASE_mvt = 'DEP'`. The scorer is RMSE on raw seconds, so the primary fit minimises MSE on raw seconds. Log-target or Huber fits are labelled diagnostics and cannot flip a verdict.
- Decomposition: the part already paid is takeoff minus AOBT_3 (`MVT_TIME_UTC_mvt - AOBT_3_flt`), present in the ranking file for 339,551 of 344,841 DEP rows. The model works on the residual around it.
- Naive-baseline check: predicting takeoff minus AOBT_3 with no model scores 428.2 s on the holdout; the board leader is 249.46 s (`python lb.py`, 2026-09-04). The leak alone does not solve the target, so the target stands. The per-airport-debiased residual scores 408.8 s.
- Hidden simple strategy: the per-airport mean and per-airport x hour means are baselines B1 and B2 below; every model is reported next to them.

## 2. Data register

| Source | Files | Vintaged? | Revision risk | Staleness cap | Point-in-time rule |
|---|---|---|---|---|---|
| Organiser bucket `prc-2026-datasets` (MinIO, `s3.opensky-network.org`) | 12 x `training_YYYY-MM-01_..parquet`, `ranking.parquet`, `submitting.parquet`; sha256 list below | fixed snapshot downloaded 2026-09-04 | UNKNOWN whether the organisers re-issue files; re-run `sha256sum data/*.parquet` before every submission build and refuse to build on a changed hash | N/A | see below |
| External data (weather, METAR, other OpenSky, prior years) | none | N/A | N/A | N/A | NOT admitted. The site says nothing either way (WebFetch of `data.html` and `index.html`, 2026-09-04, both silent). Admitting any needs an amendment with its own row. |

Point-in-time rule for this competition (the scorer is offline on a fixed file):
1. Every column of `ranking.parquet` is admissible as a feature for any row, including the times of other movements after the own AOBT (takeoffs and landings before, around and after it). The site sets no restriction.
2. In ranking, every DEP row has `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` blanked. So the feature builder runs on a **masked view of training** that blanks those two columns on every DEP row. Arrivals keep their block times in both files.
3. A test asserts the feature matrix for a training month is identical whether built from the masked training file or from a ranking-shaped copy of it. The build fails if not.
4. `AOBT_3_flt` is minute-granular; it is missing on 5,290 ranking DEP rows (344,841 - 339,551). Every model declares its fallback for those rows.

sha256 (2026-09-04, `sha256sum data/*.parquet`):

```
9db75e3a56fa09b87aab72e191a53da3fe763d9887ad2b23aea9cde6a6325f86  ranking.parquet
0d383408ed5ac27bfcfd12d6b961cee77c8a6364b10dad9fe3ec81115c79652b  submitting.parquet
a4dc8f152c082c650cd7eb745fe9c641089f34f74df127e6944a188990a0ed94  training_2025-01-01_2025-02-01.parquet
b6f9ff2cdb1cfe07c5442f4e1f11bbcdbeafe52852dcb0b8175f62ab84c6a066  training_2025-02-01_2025-03-01.parquet
58fde7e5f8b539546af63b41d7f6ec0f2884fdc3d3dd9b6bfca7c59a420dcdd3  training_2025-03-01_2025-04-01.parquet
9ffe0cf5741c4863799c41be81475dfaecedabe0ba687b8f8e4eacba992f9390  training_2025-04-01_2025-05-01.parquet
5481c0642a8fbb4dc395f93bc713c57f4cc16854fb361d404967077e1fce2ff9  training_2025-05-01_2025-06-01.parquet
c3ef5080f2cc636c3416baf1555f7d0ee3519f9046e8043c1c0c8f017a244025  training_2025-06-01_2025-07-01.parquet
34d1c5bba7c8596a1c20c09edd3cb2f63277b5a684fa801911660fe0eaf63c9d  training_2025-07-01_2025-08-01.parquet
6c8d2aa7a1b6c9dff9a0305a37eb02de1e87a68d07557264f41b77738272c540  training_2025-08-01_2025-09-01.parquet
955308a4000947e32a00586d3ee13928949d2231b67f84f6dfb0631893f3754a  training_2025-09-01_2025-10-01.parquet
4a31a35601c1bdcd74b8cd6379231ba516a48324e4c2478bf1c9360cd23e687f  training_2025-10-01_2025-11-01.parquet
ac36b17db7a537e3baeb682fa4763dacaa1fcc5d9bfa6bf32b3ee5d9f91300b0  training_2025-11-01_2025-12-01.parquet
2d357f2f403a86778a69e93f72d8754effe77f1ac0942399b3d1b7728241f724  training_2025-12-01_2026-01-01.parquet
```

## 3. Decision rule (declared before the first run)

Primary metrics, both required:
1. **Holdout RMSE** on Jan + Jul 2025 DEP rows (all 344,419, fallback rules included), model fit on the other ten months of 2025. The verdict statistic is the **paired delta-RMSE vs the current champion**, day-block bootstrap, 300 reps, seed 0. A challenger wins only if the 95% interval of the delta excludes zero AND the point delta is at least 2 s.
2. **Board RMSE** on the 215,876 scored rows. Every submission is a ledger trial (section 8). For learning what works, a board delta under 2 s is a tie; for the ranking itself the board is exact and needs no interval.

Selection honesty: the holdout selects the champion; the board judges it. The haircut for each submission is (holdout delta) minus (board delta), recorded in the ledger. Two promoted champions in a row with a negative board delta trip REVIEW (section 8).

Tie-break: parsimony. Fewer features, simpler model class, fewer stages. A tie promotes the simpler model.

Nulls: any lane that searches (feature sets, hyperparameters, tail thresholds, ensemble weights) re-runs its whole recipe on 5 target-permuted copies of the training months (permutation within day x airport block). The real gain must beat the best permuted gain by 2 s; otherwise the gain is a flag, not a result. Nulls are vetoes for promotion, flags for reporting.

Every verdict sentence quotes the effect next to the 2 s floor ("delta -6.3 s, floor 2 s").

A number inside the floor never justifies another run of the same kind. The next run must change a declared axis.

Stage-6 gate (Keith): (a) the first leaderboard submission S1; (b) every new model class (LightGBM, TabPFN-3, ensemble) before its first submission; (c) any amendment to this file. Between gates, submissions of promoted champions run in waves of at most 3, each logged.

Stage-6 surface: honest table, paired intervals for both metrics, this rule restated with "attack the rule, not just the numbers", all caution flags, the ledger line (N submissions so far).

## 4. Declared experiments registry

Common design unless stated: train = 2025 months 02-06 and 08-12; holdout = 2025-01 + 2025-07; features built on the masked view (section 2); fallback where AOBT_3 is missing = B1 prediction. Hash = commit of this file at declaration (see `git log -- docs/EXPERIMENT-PROTOCOL.md`).

| Id | Declared | Design | Null design | Status | Verdict |
|---|---|---|---|---|---|
| B0 | 2026-09-04 | global mean of the target | none (no search) | DECLARED | |
| B1 | 2026-09-04 | per-airport mean | none | DECLARED | |
| B2 | 2026-09-04 | per-airport x UTC-hour x weekday mean | none | DECLARED | |
| B3 | 2026-09-04 | takeoff minus AOBT_3, fallback B1 | none | DECLARED | |
| B4 | 2026-09-04 | B3 plus per-airport OLS on [residual, queue-ahead count] (queue = same-airport takeoffs between own AOBT_3 and own takeoff) | none | DECLARED | |
| M1 | 2026-09-04 | LightGBM, MSE objective, at most 77 features (airport, hour, weekday, month, runway, stand, AOBT_3 residual, EOBT/LOBT deltas, queue and arrival counts in windows around AOBT_3 and takeoff); grid of at most 12 cells chosen by month-block CV inside the 10 training months; holdout touched once per frozen fit | 5 permutation replays of the full grid | DECLARED | |
| M2 | 2026-09-04 | two-stage tail model: classifier on abs(AOBT_3 residual) > 600 s from ranking-visible traffic, then separate regressors; compared with M1 on the same rows | 5 permutation replays | DECLARED | |
| M3 | 2026-09-04 | TabPFN-3 regressor per airport (GPU, tabpfn 6.3.1), same feature set as M1, training subsample per airport within the 1,000,000-row limit | 5 permutation replays if any per-airport selection is searched | DECLARED | |
| E1 | 2026-09-04 | weighted blend of M1 and M3 (and M2 if promoted), weights chosen by month-block CV on training months | 5 permutation replays over the weight search | DECLARED | |
| S1 | 2026-09-04 | first board submission = the champion after B0-B4 and M1; ranking file built from all 12 months of 2025 with the frozen spec | N/A (a trial, not a search) | DECLARED, gated on Keith | |

## 5. NOT-DONE table

| Declared item | Why not run | Effect on the verdict |
|---|---|---|
| all of section 4 | protocol instantiated 2026-09-04; no run yet | no verdict exists yet |

## 6. Accrual list (thin windows)

N/A. The smallest airport (LSZH, 134,907 training departures) has about 22,000 holdout rows; no per-airport lane falls under the 300-anchor floor.

## 7. Caution flags (ride every report)

- F1 **Scored subset unexplained.** The board scores 215,876 of 344,841 ranking DEP rows. Checked 2026-09-04: no column non-null count equals it, no simple predicate equals it, no airport subset sums to it, the site states no exclusion. UNKNOWN which rows count.
- F2 **2026 tail transfer unknown.** In every 2025 month the AOBT_3-residual tail (5.9% to 8.2% of rows) carries 59% to 77% of the squared error. Whether Jan/Jul 2026 look the same is unknown until a submission lands.
- F3 **Truth outliers in training.** LIRF: 1,967 taxi-outs over 1 h, max 131,167 s; LSZH: 313 non-positive; LFPG: 65 non-positive (`view.json` key `airports`). If rows like these sit in the scored set they dominate the RMSE. Training truth is not cleaned without a declared diagnostic.
- F4 **AOBT_3 missing** on 5,290 ranking DEP rows; every model declares its fallback.
- F5 **Level transfer.** One SE of the RMSE level across day blocks is 10.2 s; local levels are never compared with board levels.
- F6 **Board volume.** The leader file is `_v344`; the board ranks each team by its best submission, so volume is rewarded. Our ledger counts N so the winner-take-best effect on our own history stays visible.

## 8. Prospective ledger contract

- Ledger: `docs/TRIAL-LEDGER.md`, append-only, one row per board submission. Columns: id, date, file name, spec tag, git hash, holdout RMSE, holdout delta vs previous champion, board RMSE, board delta, haircut, usedPairs. Old-spec rows stay, labelled.
- Tripwires: REVIEW when two promoted champions in a row score a negative board delta (the holdout has stopped mirroring the board; stop and re-examine the split). KILL: N/A, no money at risk. Operational kill: a submission that fails the site validation (missing or extra MVT_ID) is logged as VOID and the builder is fixed before the next one.
- Drift monitor: N/A.

## Amendment log

- v1 2026-09-04: instantiated. Baselines B0-B4 and lanes M1-M3, E1, S1 declared; noise bar measured and the 2 s floor set.
