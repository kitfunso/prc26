# Contender run declaration, 3 October 2026

This declaration precedes the screen. The earlier scope correction is diagnostic,
not a new mechanism recipe. The frozen champion remains the fallback.

Run: `data/runs/contender-20261003`. Sources: monthly training parquet files read
individually, with every departure target and block time blanked before feature
construction. Labels are joined by unique ID afterwards. Arrival context stays
visible. The run verifies an independent cold rebuild and real hidden-field
mutation on January. Global model/PW caches are not inputs to the screen.

Two development blocks: February/March and September/October. Each fit excludes
both its scored months and January/July. All evaluation rows are scored together.
Outer membership follows the supplied file: the audit found two Lane B
movements at boundaries whose UTC clock month differs from their file month.
This is not evidence of a faulty timestamp. The earlier chat wording that
called them misplaced clocks was too strong.
The initial September/October control is rebuilt with that entire-file scope;
its first clock-scope attempt is retained as a failed construction, not counted
as confirmation. February/March's ID union is identical under both definitions.
Lane A and NN fits exclude Lane B labels, so their predictions can be joined
onto the corrected complete-file control without retraining those models.
The Lane A CPU control is residual LightGBM, 350 fixed trees, 63 leaves, minimum
100 rows, learning rate .08, seed 0, NUM/DAY/OCC/WX/ARR/WZ. This reduced-capacity
control is not clean-v53. Lane B uses the corrected current mechanism model,
inner training-day stopping and fixed-budget full outer refit, then its existing
cell prior/hull. No cached donor is used. Recipe count starts at four: this
control and the following three alternatives.

CP1 hierarchy changes only LIRF gaps at least four hours. The permitted Lane B
copy/near labels estimate cell, cell/airline and cell/flight probabilities with
fixed pseudo-counts 2, 20 and 10. Outcomes remain the existing schedule and
date-adjusted physical values. Flight support is reported; no flight-string
hard rule is introduced.

CP1 calibration uses a small binary LightGBM without label-derived histories:
120 trees, 15 leaves, minimum 60 rows, learning rate .05, L2 10. Three month-group
cross-fits inside permitted training produce calibration data. A logistic
calibrator weights squared outcome separation, clipped at 60,000 seconds,
with C=10. In the same high-gap region, the probability averages this estimate
and the existing cell prior equally. Reliability and final SSE are recorded.

CP1 outcomes fits the same bounded gate for schedule/near copy, physical,
positive day shift and negative day shift. The shifted classes require 20
permitted airport examples; unsupported probabilities return to the other
classes. A 300-tree physical expert estimates the conditional outcome. Sparse
LIRF cells retain the control. The candidate floor stays zero; this recipe
explicitly tests replacing the old hull policy for supported outcomes.

Primary: paired whole-file SSE and pooled RMSE. Report both blocks, Lane B,
airports, gap bands, hot days, 2,000 calendar-day resamples, month-stratified
five-day blocks, top contributing rows and leave-best-day/episode sensitivity.
No variant receives a full confirmation unless both blocks improve and the
gain is supported outside one day. Confirm survivors on remaining development
blocks and Jan/July before comparing against the frozen champion and presenting
an upload. Existing splits are reused; none is described as untouched.

Budget: one initial screen per declared recipe, CPU only, stop on null/reversal,
no post-result parameter search. The campaign's eight-recipe cap still applies.
CP2/CP3 declarations will precede their runs if these mechanisms do not earn
confirmation. No scorer access is part of this declaration.

## CP2 declaration before running

The two CP2 recipe slots are now fixed. Use the identical two outer blocks,
fit IDs, complete-file Lane B control and 350-tree Lane A capacity above.
CP2 weather replaces raw reporting counts with elapsed exposure: integrate
snow, freezing and low visibility over six hours, snow over one day, and record
observed coverage. A report is carried at most one hour across missing periods.
Add explicit severity interactions with movement pressure, stamp disagreement
and occupancy. No selected severity band controls promotion.

CP2 recorder adds arrival-copy/support, rounding, signed stamp disagreement,
neighbour density and occupancy interactions. It explicitly models negative
residuals below -600 seconds, body and positive residuals above 600 seconds:
a fixed 120-tree/31-leaf gate, three fixed 350-tree experts with the control
parameters. This is one declared recipe, not an extra architecture search.
Both use seed 0 and no outer-label stopping. Check hot weather subsets and
each month, keep the paired resampling and concentration checks. Confirm only
if both blocks improve without clear hot-slice harm. Six recipes are declared
so far, including the control. CPU only; no refits with edited parameters.

## CP3 declaration before running

Fix the final two initial recipes: corrected Huber and MSE, both using CELL_T7,
the same numeric/categorical/PW feature list, seed 0, at most 30 selection
epochs. The initial runtime screen fits every fifth calendar day measured from
January 1, keeps complete movement context, and evaluates every departure in
the two declared development blocks. This is a sampled capacity screen;
the CP3 full-data claim requires later fitting on every permitted day.

PW uses a declared small 80-tree residual LightGBM donor with the CPU control
parameters. Three month cross-fits build training neighbour predictions;
evaluation donors use the complete permitted outer labels. A second donor
build excludes all inner stopping days before NN epoch selection. Encoders
and target scales are fit inside each training scope. Query labels are blanked
before donor inference. The restored best epoch sets a fixed-budget NN refit
on every sampled outer fit row. Store selection/refit row/day counts, separate
best/stopping epochs, model checkpoints and donor ID hashes.

This donor is a cheap screen substitute for the champion's three-member donor
blend, so this does not reconstruct clean-v53. Compare MSE against corrected
Huber and both against the frozen CPU control; no architecture or loss search
follows these results. Whole-file paired SSE, hot slices, episode/concentration
checks remain required. Eight initial recipes are now declared, the full cap.
Only a consistent survivor can earn unsampled confirmation and an expensive
comparison against the champion. One GPU job at a time; price the first block.

## CP4 bounded screen declaration

If MSE beats corrected Huber on both neural screen blocks, test its ensemble
contribution even if its standalone RMSE trails the CPU tree. This does not
promote MSE to an expensive fit yet. Exactly three cheap meta recipes are
declared: fixed NN weight .388830 (the champion's recorded contribution),
the two-member simplex least-squares weight, and its equal average with .388830.
The fixed weight is a reference method, not a reconstruction of the champion.

Estimate the learned weight only on the NN's inner stopping dates. Reproduce
the selected NN weights by fixed-budget fitting on the inner rows, rebuild PW
with those permitted labels, and train a 350-tree control excluding every
stopping date. Fit neither weight nor model with outer labels. Predict the
complete outer files using the saved full sampled NN refit and full CPU
control. Save the calibration dates/rows, donor scopes and frozen weights.
Require positive whole-file gain in both blocks, no clear hot harm and
leave-episode support before granting unsampled confirmation. The campaign's
1.5-second floor still applies; uncertain/sub-floor results are recorded and
do not earn an upload. No extra blend grid or tuning follows.

## Full-capacity CP3 check, declared before its run

Keith requested continued work after the initial screens. The sampled MSE
loss advantage is positive in both blocks, but the sampled ensemble falls
below the promotion floor and shows winter weather harm. No model is promoted.
The cheap donor/sample combination is not enough evidence to close the intended
full-training NN comparison, so perform that missing capacity check now.

Run `python analysis/neural_screen.py full`. Use both original loss recipes,
all permitted outer Lane A days, CELL_T7's original 150-epoch selection cap,
its original patience/scheduler/architecture, seed 0, identical feature lists
and the same declared 80-tree nested donor. Choose the budget inside training
and refit every permitted outer row. Save per-stage runtime, true peak allocated
GPU memory, nested PW inputs, checkpoints and interim results after each loss.
The full model still uses a proxy donor and is not clean-v53.

This is a capacity/correctness comparison within the existing two NN slots,
not another loss or architecture recipe. Price the first block before adding
more confirmation folds. Budget at most four GPU-hours for these two blocks;
stop on failed scope/fit coverage, clear full-capacity reversal, or the budget.
No upload, new seeds or extra recipe is authorized by a sampled gain. The full
capacity result must earn any later confirmation/assembly under the same gates.

If the full MSE member beats full Huber on both blocks, repeat the same three
declared CP4 methods at full capacity with `python analysis/meta_screen.py full`.
Reuse the recorded inner PW inputs and donor scopes exactly. Fit the selected
MSE epoch budget on every inner training row; the inner stopping dates alone
fit the weight. The CPU control excludes all stopping dates during weight fit.
Use saved full-outer members for complete-file evaluation, require exact ID
coverage and unchanged Lane B, and retain the whole-file/hot/episode gates.
Budget this conditional check within the four GPU-hours already declared.
This repeats the existing three methods, with no weight grid or new recipe.

## Ranking-support audit declaration

While the full NN runs, audit cold ranking inputs against January/July 2025
and all 2025 inputs. Record category coverage, numeric ranges/missingness,
weather severity/support and permitted raw/weather hashes. Recompute year
separation using the fixed 300-tree/63-leaf classifier from audit.py, seed 0,
on CAT/NUM/DAY/OCC/WX/ARR/WZ, with matched airport/month samples capped at
3,000 rows per year and stratum. Five outer groups keep each five-day calendar
block together; no movement from a scored day block enters its fit.
This is a shift diagnostic, not a taxi prediction recipe or a density-weight
variant. It uses no ranking targets and does not select model parameters.
