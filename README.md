# PRC Data Challenge 2026: taxi-out time, team merry-octopus


Current recorded champion: **v71, 241.0532 s**: frozen v53 (270.43 s) corrected by the ADS-B pushback stage of
10 October ([section below](#ads-b-pushback-stage-v64-on-docsadsb_stagemd), `LEDGER.md`). The campaign plan is
[PLAN.md](PLAN.md) and [docs/CONTENDER-PLAN.md](docs/CONTENDER-PLAN.md).
The October 2 review identified validation isolation and final-NN-fit issues; prospective work
is declared in [protocol amendment v9](docs/EXPERIMENT-PROTOCOL.md#amendment-v9--2026-10-02-first-place-campaign-and-validation-corrections).
The reproduction history below describes existing recipes, not a newly measured contender.

The 8 October recipe freeze retains exact scored v53. Both complete clean reference
blocks passed construction audits, but winter promotion failed. The four fixed CP4
alternatives and the separately declared Lane B residual correction were rejected.
No improved recipe qualified; no new fitting or retuning follows the freeze. On 10 October a new source of
information, each departure's own pushback read from open ADS-B traces, was added as a separate stage on top of the
frozen v53 file; v53 itself is unchanged.

On 8 October the frozen file was designated as `merry-octopus_final.parquet`, as
required by the [official final-submission rule](https://prc-data-challenge-2026.netlify.app/ranking.html#final-submission).
The final population adds February and June 2026 (670,790 rows in `final_submitting.parquet`),
so on 10 October the final file was rebuilt and replaced. Branch `final-febjun` holds the v53 code
plus `scripts/final_chain.sh` (the v53 recipe refit on `final_ranking.parquet`, stage by stage),
`scripts/final_assemble.py` (January and July keep scored v53, moved by each row's takeoff re-jitter
between the two ranking files; February and June take the refit; template checks) and
`scripts/final_upload.py`. That v53 final (SHA-256 `1c3830a45b5ab1cb...`) was then replaced by the same file
corrected with the ADS-B stage (`adsb_stage.py apply final`); the stored file's SHA-256 and the scorer result are in
`LEDGER.md`. The 9 October [incident](docs/FINAL-SCORER-INCIDENT-2026-10-09.html) was a scorer
outage plus the missing months, both closed. The [official deadline](https://prc-data-challenge-2026.netlify.app/)
for the final file and the repository is 14 October 2026 at 10:00 UTC.
See [release readiness](docs/RELEASE-READINESS-2026-10-08.html) for stage checks and gaps.

The checked campaign source has been pushed to the private `campaign2` release
branch under Keith's 8 October continuation approval. Public code eligibility
remains pending until scheduled publication is verified. Historical native-model lineage is partial:
the preserved prediction files reproduce the exact scored artifact, while the
recipe commands below do not establish bit-identical historical model refits.

Predict `TAXITIME_SEC_mvt` (takeoff minus off-block, seconds) for departures at ten European airports.
Scored by RMSE on January and July 2026. Licence: GPLv3 (`LICENSE`).

## How the label was made, and why the model looks like it does

The takeoff time is given in the test file. The off-block time is blanked. So the unknown is one timestamp,
and in the training data that timestamp is one of four things:

| recorded off-block | share of 2025 departures | prediction if we knew the class |
|---|---|---|
| the AOBT from the flight plan (within 60 s) | ~19% | takeoff minus AOBT |
| the EOBT from the flight plan | ~9% | takeoff minus EOBT |
| the scheduled time (heavy at LIRF, up to hours before takeoff) | ~1.5% | takeoff minus schedule |
| a physical time (real pushback), occasionally shifted by a day | ~70% | needs a regressor |

The model is that mixture: a multiclass classifier gives P(class | features), the three stamp classes use their
known timestamp, and a gradient-boosted regressor fitted only on physical rows gives the rest.
`yhat = sum_k P(k) * z_k + P(other) * mu(x)`. RMSE is minimised by the conditional mean, and the mixture is the
conditional mean written out. See `docs/EXPLORE.md`, `docs/AUDIT.md` and `docs/RESULTS.md` for the numbers.

## Features (`features.py`, `weather.py`)

From the challenge files: gaps from takeoff to each stamp, differences between stamps, stamp rounding, calendar
(hour, weekday; month is never a feature), categorical ids (airport, runway, stand, type, operator, destination,
airline prefix), counts of movements in the same file around the estimated off-block and the takeoff, airport-day
aggregates (`DAY`), and stand occupancy (`OCC`): whether the next arrival parked on the departure's stand before
its takeoff, and how long before. Occupancy carries most of the feature gain (`docs/RESULTS.md`, ablation lines).

Weather (`WX`): IEM ASOS METAR archive, one CSV per airport under `data/weather/<pull date>/`, fetched with
`python weather.py fetch 2026-09-05` (stations EDDF EDDM EGLL EHAM LEBL LEMD LFPG LIRF LSZH LTFM, UTC,
2024-12-31 to the pull date; the shipped pull ends 2026-08-01, and later rows touch no anchor). Joined as-of the estimated off-block: wind, gust, visibility, temperature, ceiling,
present-weather codes, a 3 h thunderstorm count and the age of the observation. Precipitation (`p01i`) is 0 on
every European row, so it is dropped. The base model reads no ADS-B; from v64 a separate ADS-B stage corrects it
(next section).

## ADS-B pushback stage (v64 on, `docs/ADSB_STAGE.md`)

Data declared: open ADS-B traces from adsb.lol (`adsblol/globe_history_2025` and `_2026` on GitHub, Open Database
License 1.0), used under the organisers' 18 September 2026 ruling that open trajectory data may be used to extract
off-block times when declared. No OpenSky state vectors are used. Each departure is matched to its own ground track,
which gives candidate pushback clocks; a LightGBM stage fitted on the 2025 residual of the base model moves each
matched prediction, with LIRF lane B left uncorrected (its labels often copy the schedule, which no pushback clock
sees). The stage trains on every 2025 month (363 days; 2025-12-31 was never published, 2025-10-15's download was
corrupt). It also sees each stand's and operator's rate of labels that copy a plan stamp (`cr`, leave-one-month-out on
training rows). Holdout Jan+Jul 2025, each month corrected by a stage trained on the other 2025 months: 304.99 to
290.13 all rows, +14.86 s [+11.90, +18.85]. Board: v71 241.0532 against v53 270.43. The final file is the same stage
on the final population. Rebuild:

```
python adsb_fetch.py                 # download and cut the days, about 2 to 3.5 GB each, deleted after the cut
python adsb_events.py train          # 2025 matches; then: adsb_events.py rank data/ranking.parquet
python adsb_stage.py gate skipB=LIRF cr
python adsb_stage.py apply rank data/merry-octopus_v53.parquet data/merry-octopus_v71.parquet skipB=LIRF cr
python adsb_events.py final          # the 120 final days, then:
python adsb_stage.py apply final data/merry-octopus_final_v53full.parquet data/merry-octopus_final_v71.parquet skipB=LIRF cr
```

Source access: [IEM ASOS/METAR downloads](https://mesonet.agron.iastate.edu/request/download.phtml).
The [IEM usage policy](https://mesonet.agron.iastate.edu/disclaimer.php) places its
materials in the public domain and permits lawful reuse. Credit: Iowa Environmental
Mesonet, Iowa State University. Challenge inputs are obtained through the
[challenge data instructions](https://prc-data-challenge-2026.netlify.app/data.html).

As-of rule: a feature may use any movement in the same file, before or after the takeoff (the rules allow it;
`dep_after` and `arr_off` look forward).

## Lane B (no flight plan)

About half of the rank-set squared error sits on the ~5k departures without a plan, mostly LIRF, where the recorded
off-block is often the schedule. A second, small LightGBM classifier fitted on lane B only gives P(schedule copy)
and overrides the stamp probabilities on those rows (`bclf` in `mix.py`). It helps the LightGBM mixture
(+3.65 s on holdout, CI excludes zero) and not the CatBoost one, so it enters only through the blend.

## Validation (`train.py`, `mix.py`)

Historical evaluation uses January and July 2025 plus five month-pair folds over the other ten months.
Jan/July has been read repeatedly and is not an untouched model-choice test. The October 2 audit found
held-out-label dependencies in copy history and multi-stage caches, plus scored-fold early stopping
in development CV. [CP0](docs/CONTENDER-PLAN.md#3-cp0--establish-a-trustworthy-comparison)
defines the corrections. The complete January/July and February/March references are
audited; winter promotion fails. September/October full reference remains deferred.
This is reduced coverage using adaptively reused development data, rather than fresh
confirmation. Every result reports per lane and month.
Predictions are floored at 0. Board uploads are logged in `LEDGER.md` with expected and actual scores.

## Historical recipe commands (v28 base; frozen v53 is the file of record)

The lane A blend is 0.5223 * CatBoost residual lane-A model with arrival recorder-mode and window-state features (`ra_cb_wz`)
+ 0.1564 * the LightGBM version (`ra_lgb_wz`) + 0.3213 * the MLP version (`nn3_wz`). Lane A is v25's design
(residual to AOBT, tail-gated tree cells or an MLP on the residual, plus the 16 `ARR` arrival recorder-mode
features, `docs/ARR_SHIP.md`) with 20 window-state columns (`WZ` in `features.py`): label-free statistics of the
neighbouring departures' stamps within +-30 min, +-1 h and +-3 h at the airport and +-1 h within the airline
(`docs/WINDOW_STATE.md`). New in v28, a LightGBM second stage on the blend's residual, from the blend, the airport
and the blend's distance to each plan stamp, lets rows whose recorded off-block copies a stamp move toward it
(`recal.py`, `docs/RECAL.md`). Lane B (no plan) takes `laneb.py`'s LIRF-only mechanism model as a row-level override
(`combine.py --over`). LIRF lane B rows over 4 h late are clipped onto the interval of labels the two recorder
mechanisms can produce (a schedule copy, or the real clock time written on a nearby date), and clipping onto an
interval that holds the label cannot move a prediction away from it (v26, `docs/LANEB.md` section 4). From v27,
all on lane B: the copy classifier weighs each row by what a wrong call costs in squared error, the physical-taxi
regressor gains the `ARR` and `WZ` columns, and before the clip each late LIRF row takes the expected label of its
cell, P(copy) times the schedule gap plus (1 - P(copy)) times the composed stamp, with P(copy) the cell's copy share
in training (`docs/LANEB.md` section 5). New in v28, the copy classifier also sees how often the row's airline and
flight number copy the schedule in lane A (`docs/LANEB.md` section 6). Weights are non-negative least squares on the
CV out-of-fold predictions (`blend.py`), never on the holdout. Holdout Jan+Jul 2025: 306.76 all, 218.60 lane A,
1736.9 lane B; paired day-block bootstrap against v27 +1.71 [+0.75, +2.88] all, second stage alone +1.29 [+0.76,
+2.03].
v28 on the board: 271.783, rank 11 of 132 (`LEDGER.md`).

v29 added a schedule-copy head to the second stage (`copyhead.py`: P(the recorded off-block copies the schedule) from
the flight's lane A copy history, the stamps and identity columns, as a `recal.py` column) and a lane A physical-taxi
prediction to the lane B tau regressor and copy classifier (`python laneb.py taua`). Holdout 305.89, +0.87 [+0.10,
+1.71] against v28; board 272.2175, 0.43 s worse. v30 (v29's lane A with v28's lane B) scored 271.9284, so both
changes lost on the 2026 rows: the copy head 27M SSE, the lane B column 54M. v28 stays the file of record and the
code at head is v28's again; the v29 code is commit `88afb7e`.

v31 (2026-09-17, the file of record, board 271.68) is v28 with one change: the LightGBM lane A member also sees
five windows of the neighbours' predicted residual, the v28 blend minus the AOBT gap, leave-one-out over the
airport-day (`features.pw_cols`, `docs/WINDOW_STATE.md`). NNLS on OOF reweights the blend to 0.501509 `ra_cb_wz`
+ 0.187349 `ra_lgb_pw` + 0.311142 `nn3_wz`; the second stage is refit on it; lane B rows are v28's. Holdout 306.52
all, 218.26 lane A, +0.240 [+0.108, +0.402] against v28. The extra commands, after the v28 chain:

```
python lanea.py cv ra_lgb_pw mix_cb_wx     # builds data/cache/pw.parquet from the v28 caches on first use
python blend.py ra_cb_wz ra_lgb_pw nn3_wz --base=blend_ra_cb_wz+ra_lgb_wz+nn3_wz
python lanea.py fit ra_lgb_pw mix_cb_wx 237 293 470
python recal.py cv && python recal.py fit
python combine.py data/merry-octopus_v31.parquet data/recal_rank.parquet:1 --over=data/laneb_rank.parquet
python board.py check data/merry-octopus_v31.parquet
```

v32 (2026-09-17, the file of record, board 271.5521) is v31 with one change: the MLP lane A member
also sees the five `PW` neighbour-predicted-residual windows in place of the plain residual
(`nn3_pw` in `nn.py`, `docs/WINDOW_STATE.md`), trained on CPU (`NN_DEV=cpu`, venv `.venv-cpu`,
CPU-only torch 2.14.0) because the GPU driver was wedged by another job. NNLS on OOF reweights the
blend to 0.490727 `ra_cb_wz` + 0.175394 `ra_lgb_pw` + 0.333879 `nn3_pw`; the second stage is refit
on it; lane B rows are v28's. Holdout 305.77 all, 217.18 lane A, +0.754 [+0.392, +1.314] against
v31 (paired day-block bootstrap); `nn3_pw` alone vs `nn3_wz` +6.900 [+2.555, +13.026] lane A
holdout. Board 271.5521, rank 20 of 152: NEW BEST by 0.128 s, but only 24.0M of the 159M SSE the
holdout promised carried, 0.15 against v31's 0.4, so each `pw` member carries less of its holdout
gain to the board. The extra commands, after the v31 chain (the CPU venv is only needed while the
GPU driver is wedged; `python nn.py cv nn3_pw mix_cb_wx` is the normal form):

```
NN_DEV=cpu CUDA_VISIBLE_DEVICES="" .venv-cpu/Scripts/python.exe nn.py cv nn3_pw mix_cb_wx
python blend.py ra_cb_wz ra_lgb_pw nn3_pw --base=blend_ra_cb_wz+ra_lgb_pw+nn3_wz
NN_DEV=cpu CUDA_VISIBLE_DEVICES="" .venv-cpu/Scripts/python.exe nn.py fit nn3_pw mix_cb_wx
python recal.py cv && python recal.py fit
python combine.py data/merry-octopus_v32.parquet data/recal_rank.parquet:1 --over=data/laneb_rank.parquet
python board.py check data/merry-octopus_v32.parquet
```

v34 (2026-09-23, the file of record, board 270.5655) is v32 with the LightGBM and MLP lane A members swapped for
their R7-tuned re-fits, `ra_lgb_pw_t7` and `nn3_pw_t7` (Optuna TPE, 200 trials each, configs read back from
`data/optuna.db`, `docs/EXPERIMENT-PROTOCOL.md` section 4). NNLS on OOF reweights the blend to 0.345484 `ra_cb_wz`
+ 0.265653 `ra_lgb_pw_t7` + 0.388863 `nn3_pw_t7`; the second stage is refit on it; lane B rows are v28's. Holdout
305.24 all, 216.42 lane A, +0.526 [-0.123, +1.043] against v32, so the declared slot rule held it; Keith asked for
the upload anyway. Board 270.5655, rank 28 of 178: NEW BEST by 0.99 s, 1.66 times the SSE the holdout promised.
The extra commands, after the v32 chain:

```
python lanea.py cv ra_lgb_pw_t7 mix_cb_wx
NN_DEV=cpu .venv-cpu/Scripts/python.exe nn.py cv nn3_pw_t7 mix_cb_wx
python blend.py ra_cb_wz ra_lgb_pw_t7 nn3_pw_t7 --base=blend_ra_cb_wz+ra_lgb_pw+nn3_pw
python lanea.py fit ra_lgb_pw_t7 mix_cb_wx 237 6938 8665
NN_DEV=cpu .venv-cpu/Scripts/python.exe nn.py fit nn3_pw_t7 mix_cb_wx
python recal.py cv && python recal.py fit
python gate.py recal_holdout.parquet recal_holdout_v32.parquet
python combine.py data/merry-octopus_v34.parquet data/recal_rank.parquet:1 --over=data/laneb_rank.parquet
python board.py check data/merry-octopus_v34.parquet
```

v53 (2026-09-27, the file of record, board 270.43) is v34 with the CatBoost member swapped for its R7-tuned re-fit,
`ra_cb_wz_t7`. It is the `CLN` file of `grid.py`, which builds every 1-, 2- and 3-member mix of the three tuned
members (NNLS and equal weights) and runs each through the second stage. NNLS on OOF weights it 0.442565
`ra_cb_wz_t7` + 0.168605 `ra_lgb_pw_t7` + 0.388830 `nn3_pw_t7`; lane B rows are v28's. Holdout 304.99 all, 216.06
lane A, against v34's 305.24 and 216.42. Board 270.43, rank 35 of 195: NEW BEST by 0.136 s, about half the 0.254 s
the holdout promised. Every other grid file lost to v34 (`LEDGER.md`, v53 to v62 rows). The extra commands, after
the v34 chain:

```
cp data/cache/recal_holdout.parquet data/cache/recal_holdout_v34.parquet   # the gate's base arm
python lanea.py cv ra_cb_wz_t7 mix_cb_wx     # about 12 h on the GPU
python lanea.py fit ra_cb_wz_t7 mix_cb_wx 2503 4354 6594
python grid.py CLN                           # NNLS, recal.py cv and fit, gate vs v34 -> data/grid/CLN.parquet
cp data/grid/CLN.parquet data/merry-octopus_v53.parquet
python board.py check data/merry-octopus_v53.parquet
```

```
pip install -r requirements.txt            # Python 3.12, CUDA GPU for CatBoost
# put training_*.parquet, ranking.parquet, submitting.parquet in data/
python weather.py fetch 2026-09-05         # data/weather/2026-09-05/<ICAO>.csv
python data.py && python features.py       # caches under data/cache/, incl. ARR (arrival recorder-mode) and WZ (window-state) features
python regime.py                           # disruption-day map -> docs/REGIME.md + data/cache/regime.parquet; blend.py and recal.py slice every result by it
python -m pytest tests/ -q                 # label timing, scope, features, coverage and submission guards
python mix.py cv mix_cb_wx                 # lane-B donor: CV + holdout line in docs/RESULTS.md, iteration counts at the end
python mix.py fit mix_cb_wx 2309 3779      # -> data/mix_cb_wx_rank.parquet
python lanea.py cv ra_cb mix_cb_wx         # residual lane A, lane B copied from the donor
python lanea.py cv ra_cb_s1 mix_cb_wx
python lanea.py cv ra_cb_s2 mix_cb_wx
python lanea.py cv ra_lgb mix_cb_wx
python nn.py cv nn3 mix_cb_wx              # MLP on the residual, GPU, early stopping on the last 10% of fit days
python blend.py ra_cb ra_cb_s1 ra_cb_s2 ra_lgb nn3 --base=mix_cb_wx    # v24 blend; its holdout cache is --base below
python lanea.py cv ra_cb_arr mix_cb_wx     # same configs plus ARR (16 arrival recorder-mode features)
python lanea.py cv ra_lgb_arr mix_cb_wx
python nn.py cv nn3_arr mix_cb_wx
python blend.py ra_cb_arr ra_lgb_arr nn3_arr --base=blend_ra_cb+ra_cb_s1+ra_cb_s2+ra_lgb+nn3    # v25 blend
python lanea.py cv ra_cb_wz mix_cb_wx      # v25 configs plus WZ (20 window-state features)
python lanea.py cv ra_lgb_wz mix_cb_wx
python nn.py cv nn3_wz mix_cb_wx
python blend.py ra_cb_wz ra_lgb_wz nn3_wz --base=blend_ra_cb_arr+ra_lgb_arr+nn3_arr
python laneb.py cv mix_cb_wx               # LIRF-only lane B mechanism model: CV + holdout, tables in docs/LANEB_RUN.md
python laneb.py fit 117 238                # full 2025 refit at the CV iteration counts -> data/laneb_rank.parquet
python lanea.py fit ra_lgb_wz mix_cb_wx 229 286 494    # full 2025 refit at the CV iteration counts
python nn.py fit nn3_wz mix_cb_wx
python lanea.py fit ra_cb_wz mix_cb_wx 2373 1991 4509
python recal.py cv                         # lane A second stage on the blend: holdout check, docs/RECAL.md
python recal.py fit                        # -> data/recal_rank.parquet (its lane B rows are replaced by --over)
python combine.py data/merry-octopus_v28.parquet data/recal_rank.parquet:1 --over=data/laneb_rank.parquet
python board.py check data/merry-octopus_v28.parquet
```

`laneb.py` is a LIRF-only mechanism model for lane B (schedule copy vs. physical, day-shifted; `docs/LANEB.md`).
Its own strict pass mark (copy class no worse than the donor) fails, by 4.7% at v28, but its holdout RMSE is better
overall with a bootstrap CI clear of zero, so v25 to v28 use its predictions as a row-level override on lane B
(`combine.py --over`) rather than through the blend weights.

CatBoost on GPU is not bit-reproducible run to run; expect about 1 s of holdout noise. `tune.py` runs Optuna over
the mixture regressor on CV folds only (`python tune.py N_TRIALS 2-3,6-8`, study in `data/optuna.db`).
`window.py` prices a rolling training window against everything-before on a later month pair
(`python window.py ra_lgb_wz 11,12 26 99`); it found expanding wins, `docs/EXPERIMENT-PROTOCOL.md` R4.
Every declared experiment, its null bar and its verdict live in `docs/EXPERIMENT-PROTOCOL.md`; coverage against
the 44-step time-series roadmap is `docs/ROADMAP-COVERAGE.md`.
