# First-place campaign plan — 2 October 2026

**7 October execution checkpoint:** complete January/July and February/March clean
references pass their construction/replay/file audits, but winter promotion fails.
Four separately declared original-member CP4 follow-ups are measured and rejected.
The exact scored v53 fallback copy passes actual file/template/hash checks. No
stronger ranking candidate or scored improvement is established. Preserve v53
through the existing recipe freeze; final release/write-up readiness continues.
Current evidence is in HANDOFF.md and STATUS-2026-10-03.html. The initial-screen
status below is historical; it does not describe an active fitting process.

Status: **initial screens measured; full-data MSE lead rejected**. The 2026-10-03 continuation
implemented cold monthly inputs, nested CPU neighbour donors, training-only
stopping and full supplied-row neural refits. CP1/CP2 initial screens and a
sampled CP3/CP4 comparison are measured; none passes the full promotion gate.
The first full-capacity block compares both original neural losses on every
permitted fit day. MSE reverses against Huber with both paired intervals below
zero, so the declared stop rule rejects it before completing the second block.
Complete clean-v53 and remaining confirmation folds remain unfinished.
No new recipe qualifies for upload. The scoped campaign driver supplies the
current evidence; legacy global PW/tune/recal/blend paths are not a clean-v53
reconstruction and must not be used as promotion evidence.
Declarations: [CONTENDER-RUN-2026-10-03.md](CONTENDER-RUN-2026-10-03.md).
See [STATUS-2026-10-03.html](STATUS-2026-10-03.html) for checks, measurements and limits.
Implementation baseline: `campaign2` at `a3503dcd148c21a511a134754503621254b60a8b`.
Keith requested this plan to seriously challenge first place and authorized committing and pushing it.

This is the current execution plan. [EXPERIMENT-PROTOCOL.md](EXPERIMENT-PROTOCOL.md), amendment v9,
declares the experiment families and preserves submission sign-off. [LEDGER.md](../LEDGER.md)
remains the record of measured board results. Historical results are not rewritten.

## 1. Objective and size of the challenge

Predict the **recorded** `TAXITIME_SEC_mvt`, including timestamp-copy and date artifacts.
Minimize pooled squared error; report RMSE in raw seconds. Do not clean extreme or negative training
labels into a more physically plausible target. Keep the champion's prediction floor at zero and
no upper clip as the control policy; any change to that policy needs its own declared comparison.

Latest recorded champion: `data/merry-octopus_v53.parquet`, **270.43 s**, 344,841 ranking IDs.
Its Lane A members and recorded weights are:

- `ra_cb_wz_t7`: 0.442565.
- `ra_lgb_pw_t7`: 0.168605.
- `nn3_pw_t7`: 0.388830.
- Residual recalibration after the blend; Lane B supplied by the established `laneb_rank` artifact.

Preserve the actual champion file and its hash. The displayed six-decimal weights are a description,
not a substitute for the original build inputs, overrides and model artifacts.

The September 23 ledger records the leader at **235.3074 s**. This is a historical benchmark,
not a verified current leader. At kickoff, refresh with the existing read-only `python board.py lb`
on a connected machine and record its timestamp. Do not assume that 235.3074 will still win.

| Benchmark RMSE | Gain from 270.43 | Required SSE reduction | Required SSE gain |
| --- | ---: | ---: | ---: |
| 250 s | 20.43 s | 14.54% | 3.666 billion seconds² |
| 245 s | 25.43 s | 17.92% | 4.520 billion seconds² |
| 240 s | 30.43 s | 21.24% | 5.356 billion seconds² |
| 235.3074 s, historical leader | 35.12 s | 24.29% | 6.125 billion seconds² |

Calculation: `N * (270.43**2 - target_rmse**2)`, with `N = 344841`.

The v34-era holdout census attributes about 50.5% of SSE to 1.56% of rows in Lane B.
Its 14 labels above 12 hours carry 35.0% of holdout SSE. These are historical diagnostics,
not observations of hidden 2026 errors, and must be recomputed after validation repairs.

A previous two-outcome model estimates 4.01 billion seconds² of uncertainty on 43 LIRF rows.
Even entirely removing that amount from v53 would imply about 248.0 s. That estimate is not a
measured removable gain or a universal ceiling. First place therefore needs predictable improvement
on catastrophic recording outcomes **and** additional gains elsewhere.

The aim is a mechanism discovery large enough to matter, followed by reliable fitting and assembly.
More regressor trials or a new grid of nearly identical blends has a weak prior for closing this gap.

## 2. Work queue and ownership

The implementing developer owns code, fit-scope checks, run manifests and reproducible result files.
Keith owns the GPU access decision and the existing per-submission sign-off. Reserve one GPU job at
a time. Respect the standing instruction in [HANDOFF.md](HANDOFF.md): wait for unrelated `fifty`
jobs; do not pause or terminate them.

| ID | Priority | Work | Depends on | Output and acceptance |
| --- | --- | --- | --- | --- |
| CP0 | P0 | Correct validation isolation, champion preservation and actual-file gates | None | Passing label-mutation and assembly tests; a clean frozen baseline |
| CP1 | P1 | Calibrated hierarchical recording-mechanism model, especially catastrophic Lane B rows | CP0 | Whole-file SSE gain, calibrated outcome probabilities and sparse-cell stress report |
| CP2 | P1 | Conditional disruption and recording-state features in Lane A | CP0 | Gain across outer blocks and independent episodes; no clear hot-slice harm |
| CP3 | P1 | True all-data NN refit, then loss comparison | CP0 | Best-epoch tracking, complete training-day coverage and clean ensemble comparison |
| CP4 | P2 | Nested meta-fitting and a small robust ensemble | CP1–CP3 survivors | One frozen recipe; no global OOF or blend-weight evaluation leakage |
| CP5 | P0 before release | Full-data build, file validation, submission and reproduction package | CP4 or validated fallback | Exact candidate hash, passing gate, signed-off upload and verified release |

CP0 is a correctness requirement. It is not itself a claim of lower ranking error.
Keep two named controls: the immutable uploaded **v53 artifact**, and **clean-v53**, the same
modelling recipe evaluated under the corrected protocol. New experiments compare with clean-v53;
the original artifact remains available as the valid board fallback.

## 3. CP0 — establish a trustworthy comparison

Target files: `laneb.py`, `train.py`, `lanea.py`, `tune.py`, `features.py`,
`recal.py`, `blend.py`, `gate.py`; new focused fit-scope tests and an evaluation driver.

### CP0.1 Freeze data and artifact lineage

- Record raw file hashes, schema, departure/arrival counts, target missingness, negative/extreme
  target counts, duplicate IDs and join multiplicity.
- Require a finite valid AOBT anchor for AOBT-residual modelling; use an explicit fallback for rows
  with another plan stamp but no usable AOBT. Diagnose whether any such rows exist before changing lanes.
- Record champion output hash, template ID hash, member/checkpoint hashes, exact weights and CLI
  overrides, feature/cache versions, Lane B source and the originating commit.
- Keep research runs under separate run directories. Existing global caches and `recal_rank.parquet`
  must not be concurrently overwritten by competing recipes.
- Apply the checks in draft [PR #1](https://github.com/kitfunso/prc26/pull/1), or an equivalent reviewed
  implementation, before submission. That PR has not been merged by this plan.

### CP0.2 Make every label-derived stage fit-scope aware

Confirmed finding: `laneb.copy_hist()` computes copy labels over all twelve 2025 months;
a three-month gap allows January and July held-out labels into each other's history.
A five-row synthetic test changed `h_al` and `h_flt` from 0.5 to 0.0 by changing only
hidden January/July Lane A labels. The real-data score effect has not been measured.

Introduce explicit permitted-fit IDs for history fit/transform. Exclude the entire outer evaluation
block from source labels across both lanes, and cross-fit aggregates for training rows.

Rebuild neighbour-prediction features within each outer fit scope. A row's donor prediction being OOF
does not prove that the donor excluded a later learner's outer evaluation labels.
Apply the same discipline to gates, blending and residual recalibration.
`recal.cv()` currently uses Jan/July labels when generating non-Jan/July recalibration OOF outputs.

### CP0.3 Separate stopping from scoring

Tree development routines currently pass scored-fold labels into early stopping.
Choose iterations on a training-only inner split; refit the outer training set at those iterations;
predict the outer block once. Fit gate and expert stopping budgets inside that scope too.

Audit CatBoost's zero-indexed best-iteration convention and configured caps. Store explicit iteration
counts instead of silently treating best-iteration indices as counts. Do not turn this audit into an
unbounded iteration-cap search.

### CP0.4 Reconstruct ranking-like evaluation inputs

For each outer month block, blank departure target and recorded block time before feature generation.
Retain allowed arrival fields and all movement context provided in that pseudo-ranking file.
Match month-file boundaries and context truncation to the ranking files rather than using
unavailable adjacent-month context at the edges.

This is retrospective event regression: legitimate later movements within the supplied file remain
usable. Do not impose an online forecast restriction or drop allowed future arrival context.

### CP0 acceptance checks

| Check | Required behavior |
| --- | --- |
| Hidden-label mutation | Mutating every excluded departure label/block time cannot alter fitted training features or evaluation predictions |
| Donor provenance | Every outer feature/model/meta-stage lists its permitted label IDs; no excluded IDs occur |
| Early stopping | Outer labels are absent from all fit, stopping and selection calls |
| NN budget bookkeeping | Stored best epoch matches restored weights; stopping epoch is separately recorded |
| Cold feature build | Independent cold builds agree under the documented tolerance; warm-cache equality alone is insufficient |
| Actual-file assembly | Intended unchanged lanes match the control exactly by ID and prediction; the actual candidate file is inspected |
| Submission schema | Exact template ID set, unique IDs, numeric finite nonnegative predictions and no missing/extra rows |

Use the existing data-dependent test suite plus these focused regressions when implementing CP0.
A completed source audit is not a completed real-data test run. Record missing data/hardware explicitly.

## 4. Evaluation contract for every new recipe

**Primary endpoint:** paired whole-file SSE gain over clean-v53 on identically constructed outer
evaluation rows. Report pooled RMSE; do not optimize an unweighted average of fold RMSEs.

Use training-only inner selection and outer month blocks. Start with cheap CPU screens on two
predeclared development blocks, including contrasting regimes. Confirm survivors across the remaining
development blocks and Jan/July 2025 with the entire pipeline rebuilt inside each outer fit scope.
If compute forces fewer confirmation blocks, label the reduced coverage; it cannot support a full-CV claim.

Existing splits have been examined repeatedly. No retrospective split becomes untouched because the code
is repaired. Preregister future recipe variants and selection decisions before their next run.
There is no new 2026 label test available locally.

For every result, save:

- Pooled SSE/RMSE and paired gain; January and July separately; Lane A and Lane B separately.
- Airport, signed residual mechanism, schedule-gap and predeclared hot-regime slices.
- At least 2,000 paired calendar-day bootstrap resamples; month-stratified 3–7-day moving-block
  sensitivity where serial disruption dependence matters. These do not correct adaptive selection.
- Gain concentration in the top 1, 10 and 50 contributing rows; leave-largest-day and
  leave-disruption-episode-out checks. Distinguish the physical/recorded-label mechanism of each gain.
- Conditional probability reliability, Brier/log loss where applicable, and final SSE.
- Coverage/support of ranking feature regions, runtime, peak memory and the actual selection count.

Evaluate complete files, including unchanged lanes. Keep target-derived diagnostic strata out of
features and promotion rules that require knowing hidden ranking labels.

Record an experiment before running it: ID, hypothesis, control, variants, outer/inner scopes,
fixed parameters or bounded search space, seed, metric, stress slices, budget and stop rule.
Then record the result and verdict. The family declarations below are not permission to quietly
add more variants after seeing a promising slice.

## 5. CP1 — catastrophic recording mechanisms

**Hypothesis:** some high-cost ambiguity has predictable recording-process structure beyond current
cell priors. Better calibrated conditional probabilities reduce SSE more than another generic regressor.

Correct the existing history in CP0 first. CP1 is a new mechanism/calibration comparison against that
corrected baseline, not a relabelled claim of the leakage repair's performance.

Predeclare at most three initial recipes:

1. Hierarchically shrunk airport/airline/flight-number copy probabilities, with support counts and
   uncertainty, using only permitted source labels.
2. A small mechanism classifier using stamp agreement, rounding, valid context and flight metadata,
   followed by training-only probability calibration.
3. An explicit outcome mixture separating physical, timestamp-copy and signed date-shift outcomes
   where training evidence supports those classes.

For mechanism `k`, estimate `P(k | X)` and `E[y | X, k]`; predict their probability-weighted mean.
Do not force uncertain rows into a hard physical-versus-copy decision. Sparse cells need shrinkage
and conservative fallback to the control.

For a two-outcome model with separation `D`, incorrect probability `q` rather than true `p`
adds approximately `D² * (p-q)²` of expected squared error when outcome means are correct.
Measure gap-sensitive calibration and whole-file SSE. Classification AUC alone is insufficient.

Stress tests: leave-airline/flight group where feasible; independent shift episodes; gap bands;
out-of-hull targets; posterior sensitivity in very small cells; support overlap with ranking rows.
Cross-record recovery must demonstrate a new signal beyond the already tested arrival joins.

**Stop:** no whole-file gain on the predeclared screen, a gain explained only by unsupported cells,
or reversal on confirmation. Do not reopen the failed B1 flight-string heuristic.
A ranking-only mechanism with no labelled analogue remains an uncertainty estimate, not a validated win.

**Contender planning checkpoint:** assess whether the supported mechanism improvement plausibly addresses
a substantial fraction of the 6.125-billion-SSE historical gap. Do not manufacture a point forecast
of ranking gain from 2025 data or from the 43-row model. A small robust gain can still be useful,
but cannot by itself justify a first-place claim.

## 6. CP2 — Lane A conditional disruption and recorder state

**Hypothesis:** within-day interactions remain predictable even when airport-day mean residuals are
near zero. Existing aggregate nulls do not rule out correcting individual flights in opposite directions.

Predeclare at most two feature-family recipes:

1. Reporting-normalized weather severity/duration plus observation support, interacted with movement
   pressure, stand occupancy and stamp disagreement.
2. Recording-state/context-support features with separate positive and negative residual mechanisms,
   using nested neighbour predictions and arrival recorder proxies.

Start on a small LightGBM control. Promote a feature family to CatBoost/NN confirmation only after
a whole-file CPU gain. Report hot-regime effects and independent disruption-episode sensitivity.

Recompute adversarial validation using matched January/July samples and day-grouped classification.
If weighting is tested, replace one of the two recipe slots with a predeclared clipped-density-weight
variant and report effective sample size. Stable `P(y | X)` is an assumption, not a consequence of
adversarial AUC. Severe ranking-only weather lacks direct labelled validation.

**Stop:** repeat of the existing raw-snow-count or mean-ATFM experiment without a new conditional
hypothesis; benefit limited to a selected severity band; clear harm on hot/disruption rows.
A simple season/DST indicator is not a new major lever: related tests already have null outcomes.

## 7. CP3 — full-data NN and RMSE-aligned loss

**Hypothesis:** restoring the final training days and estimating the conditional mean more accurately
can improve the neural member or its ensemble contribution.

Current `nn.fit()` calls a routine which always reserves the last 10% of fit days. No subsequent
gradient fit uses those days. The returned epoch is the stopping epoch, not the best-weight epoch.

Implementation:

1. Track best and stopping epochs separately; determine budgets inside permitted training data.
2. For outer predictions, refit preprocessing and weights on every outer training day at the selected budget.
3. For final ranking prediction, refit the encoder, target scale and network on all eligible 2025 days.
4. Compare at most two initial recipes: corrected Huber control and squared-error training.
   Keep architecture/features fixed so the loss effect is interpretable.
5. Run two or three seeds only for a survivor, with a documented ensemble benefit over its single seed.

Do not assume Huber is wrong: it may complement tree errors despite differing from the scored loss.
Signed experts are CP2's mechanism hypothesis, not an undeclared extra NN architecture search.

**Stop:** no clean ensemble contribution, unstable seed/episode effect, or insufficient GPU time
to complete confirmation and the final fit by the recipe-freeze date.

## 8. CP4 — final meta-stage

Build meta-training and validation inputs inside each outer scope. Fit weights and residual
recalibration on training-only meta-data, then evaluate once.

Compare at most three inexpensive meta-recipes: the clean champion method, simplex-constrained
nonnegative weights, and weights shrunk toward an equal/established blend. NNLS followed by
normalization is not generally the simplex least-squares optimum.

Calibrate gate probabilities where CP1/CP2 has shown a specific failure. Gate/second-stage tuning
is bounded to the surviving mechanism; it is not another broad 200-trial search.

Choose the smallest ensemble with robust additional whole-file gain. Keep Lane B exactly unchanged
unless CP1 has separately passed. Freeze parameters, feature groups, iteration/epoch budgets,
seed count, weights, fallback policy and artifact lineage before the final all-data build.

## 9. Compute, dates and decisions

Dates below are London local dates. The official site states the deadline as
**11 October 2026, 23:59:59 CET**; retain that wording rather than silently changing the time zone.
Aim to finish uploads on October 10. Do not depend on a last-minute interpretation of CET/CEST.

No GPU is connected to the reviewing workspace. GPU execution requires the existing local machine
or another explicitly configured runner with the data. A plan item is not a started training job.

| Date | Work | Decision/output |
| --- | --- | --- |
| 2–3 Oct | CP0, runtime pricing, immutable champion and clean CPU control | Validation/file gates pass before modelling conclusions |
| 3–5 Oct | CP1 CPU screen; corrected SSE census | One mechanism survivor or a recorded null |
| 5–7 Oct | CP2 screen and CP3; confirm at most two substantive survivors | Outer predictions, stress results and budget decision |
| 8 Oct | CP4 and final comparison | Freeze one recipe and a validated fallback |
| 9 Oct | Full-data fit and actual ranking-file assembly | Complete manifest, ID/lane parity and reproduced output |
| 10 Oct | Substantive signed-off upload(s); release rehearsal | Confirm score and code readiness with a day of buffer |
| 11 Oct | Final verification before the published deadline | Valid best artifact and eligible code/documentation |

Budget:

- Maximum eight initial model recipes including the clean control: CP1 three, CP2 two, CP3 two,
  control one. Count every attempted variant, including failed/pruned runs.
- Price the first fit of each new family in wall time and memory. Use reduced-capacity CPU screens
  for hypothesis testing, not as substitutes for the final ensemble comparison.
- At most two substantive survivors receive expensive full confirmation.
- Initial GPU envelope: 48 available GPU-hours through recipe freeze, shared by confirmation
  and fitting needs. Reprice after the first full fold; reduce the recipe count if necessary.
  The envelope is a budget, not a throughput estimate or a reservation.
- One GPU process at a time; wait for other projects. Isolated CPU work may continue while waiting.
- No new broad 200-trial searches, transformer rebuild, massive ADS-B ingestion or near-duplicate blend grid.
- By October 5, record a candid contender assessment: which supported mechanisms address the large
  gap, which gains are small, and what is still unjudgeable. Do not equate correctness fixes with a score jump.
- By October 8, freeze the recipe even if the first-place hypothesis is unresolved. Preserve the best
  valid artifact if no candidate passes.

## 10. Promotion and release gate

Retain the existing protocol's **1.5 s all-rows gain floor**, **approximately 3 s target**,
paired interval excluding zero and explicit submission sign-off. These are campaign budget rules,
not universal laws that guarantee transfer to 2026. A smaller gain is recorded as a useful result
but does not automatically receive a slot; any exception is declared and signed off.

Also require:

- Complete fit-scope/mutation checks and sufficient outer confirmation.
- No statistically clear hot/disruption-slice harm; ambiguous negative effects are reviewed explicitly.
- A mechanism explanation and error-concentration report. An unsupported rare-cell bet fails validation.
- Actual assembled-file comparison against the champion; exact equality for intended unchanged lanes.
- `python board.py check <candidate>` on the final file, its hash and a successful upload/readback check.
- Candidate, expected effect, uncertainty, limitations and exact file presented to Keith for the
  existing per-upload sign-off. This plan does not grant automatic scorer access.
- No row-level scorer probes or hidden-label reverse engineering. Use whole-model submissions.

Prefer one primary candidate and, if independently justified, one distinct fallback/survivor.
The published scorer cap is five files per day; it is not a target number of uploads.

Record the result in `LEDGER.md` with the artifact hash. Release commands, dependency versions,
external-data sources/licences and the fit/predict instructions must reproduce the final best file.
Document GPU numerical nondeterminism and tolerances; retain the actual scored artifact.

The repository records Windows task `prc26-public-20261011` for 21:30 UTC on October 11.
Verify that task, its paths, credentials and a read-only scheduler rehearsal on the local machine.
Check the publication outcome and default branch after it runs. This plan does not alter that task,
repository visibility or the already approved publication timing.

## 11. Run record and immediate next task

Each run needs a machine-readable record with:

`run_id, experiment_id, commit, raw_hashes, outer_scope, inner_scope, donor_scopes,
feature_hash, parameters, selected_iterations_or_best_epoch, seeds, model_hashes,
blend_weights, lane_sources, output_hash, runtime, peak_memory, metrics, verdict`.

Store large data, models and prediction files outside git as already established; commit code,
small manifests without credentials, declarations and result summaries. Mark missing evidence
as missing rather than filling in assumed values.

**Immediate implementation task:** implement CP0.2/CP0.3 fit-scope isolation and its synthetic
hidden-label mutation regression; then freeze the actual v53 manifest and produce one clean CPU
baseline with the same ranking-like input construction. Keep modelling changes separate so the
evaluation repair can be reviewed and its real-data effect measured.

## Sources and limits

- [Official competition and deadline](https://prc-data-challenge-2026.netlify.app/).
- [Data specification](https://prc-data-challenge-2026.netlify.app/data.html),
  [ranking rules](https://prc-data-challenge-2026.netlify.app/ranking.html),
  [eligibility](https://prc-data-challenge-2026.netlify.app/eligibility.html).
- [Nested validation guidance](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
  and [Cawley & Talbot, JMLR 2010](https://jmlr.org/papers/v11/cawley10a.html).
- Repository evidence: [LEDGER.md](../LEDGER.md), [LANEB.md](LANEB.md),
  [STATE_ORACLE.md](STATE_ORACLE.md), [AUDIT.md](AUDIT.md), and
  [EXPERIMENT-PROTOCOL.md](EXPERIMENT-PROTOCOL.md).

The strategic review inspected 47 files and executed one synthetic history diagnostic.
No new competition-data training, live leaderboard verification or GPU execution was performed.
The identified code paths are evidence; the expected usefulness of CP1–CP4 remains a hypothesis.
