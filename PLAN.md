# PRC 2026 taxi-out — current campaign plan

Updated 2 October 2026. Active branch: `campaign2`.
Latest recorded champion: **v53, 270.43 s**, `data/merry-octopus_v53.parquet`.
The official deadline is **11 October 2026, 23:59:59 CET**.

The next objective is a serious first-place challenge through better recording-mechanism predictions
and transferable Lane A gains. Matching the September 23 leader's 235.3074 s would require roughly
**24.3% less SSE**; refresh the live leaderboard before execution. That score is a historical
benchmark, not a verified current first-place target.

## Execute this plan

The detailed queue, hypotheses, acceptance tests, budgets and dates are in
[docs/CONTENDER-PLAN.md](docs/CONTENDER-PLAN.md). Experiment declarations and submission authority
are in [docs/EXPERIMENT-PROTOCOL.md](docs/EXPERIMENT-PROTOCOL.md), amendment v9.

| Order | Work | Required result |
| --- | --- | --- |
| CP0 | Freeze the exact champion; repair fold-aware history, nested feature/meta fitting and scored-fold early stopping | Passing hidden-label mutation and actual-file gates; clean frozen baseline |
| CP1 | Calibrated hierarchical copy/physical/date-shift mechanisms, especially catastrophic Lane B rows | Supported whole-file SSE reduction and uncertainty/calibration checks |
| CP2 | Within-day disruption and recorder-state interactions in Lane A | Gains across outer blocks and episodes; no clear hot-slice harm |
| CP3 | True all-data NN refit; bounded Huber-versus-MSE comparison | Complete fit-day coverage and measured ensemble contribution |
| CP4 | Nested blend/recalibration using surviving members | One robust frozen recipe |
| CP5 | Full-data build, exact candidate validation and reproduction package | Signed-off submission and verified final release |

CP0 comes first: the review confirmed that `laneb.copy_hist()` can use held-out January labels
for July history and vice versa. Development tree folds also select their early-stopping point,
and global OOF caches are not sufficient isolation for later pipeline stages. The final NN
reserves the last 10% of fit days without an all-data gradient refit. These findings do not establish
a measured ranking gain from their repair.

## Dates and compute

- 2–3 October: validation, champion lineage and runtime pricing.
- 3–5 October: recording-mechanism CPU screen.
- 5–7 October: disruption screen, NN correction and at most two expensive survivors.
- 8 October: freeze the recipe.
- 9 October: full-data fit and assembled-file verification.
- 10 October: substantive signed-off upload(s) and release rehearsal.
- 11 October: final verification before the official deadline.

Limit initial model recipes to eight including the control; avoid another broad 200-trial search
or blend grid. Price runtime first. Use one GPU job at a time and respect the standing instruction
to wait for unrelated `fifty` jobs. No GPU is exposed to the reviewing workspace.

## Operating rules

- Target the recorded label, including artifacts. Do not rewrite training targets into physically
  plausible taxi times. Preserve the champion's zero prediction floor and no upper clip as control.
- Features may use permitted movements before or after takeoff in the same supplied file.
  Evaluation must hide departure targets/block times and match ranking-file context boundaries.
- Every label-derived feature, model, stopping choice and meta-stage needs an explicit fit scope.
- Keep the current hour/weekday feature policy for the control. Calendar variants, if ever proposed,
  require a distinct declaration; the plan does not silently add month/year features.
- Jan/July 2025 is a repeatedly studied historical stress set, not an untouched model-choice holdout.
- Compare whole-file pooled SSE/RMSE, with per-month/lane results and paired calendar-day/episode checks.
- The existing submission floor is 1.5 s all-row gain, target about 3 s, paired interval excluding zero,
  plus explicit Keith sign-off for each upload. These are budget rules, not guaranteed transfer laws.
- Five files per day is the published scorer cap, not a target. Avoid scorer probes.
- Inspect the actual candidate artifact, exact IDs and unchanged lanes; preserve v53 as fallback.
- Record every run and upload. Final code and documented external sources must satisfy eligibility.

[README.md](README.md) contains historical reproduction instructions.
[LEDGER.md](LEDGER.md) is the measured score record.
[docs/HANDOFF.md](docs/HANDOFF.md) has the latest execution pointer and local-machine constraints.
[docs/CONTENDER-PLAN.md](docs/CONTENDER-PLAN.md) is the current next-step plan.
