# PRC Data Challenge 2026: taxi-out time, team merry-octopus

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
every European row, so it is dropped. No ADS-B.

As-of rule: a feature may use any movement in the same file, before or after the takeoff (the rules allow it;
`dep_after` and `arr_off` look forward).

## Lane B (no flight plan)

About half of the rank-set squared error sits on the ~5k departures without a plan, mostly LIRF, where the recorded
off-block is often the schedule. A second, small LightGBM classifier fitted on lane B only gives P(schedule copy)
and overrides the stamp probabilities on those rows (`bclf` in `mix.py`). It helps the LightGBM mixture
(+3.65 s on holdout, CI excludes zero) and not the CatBoost one, so it enters only through the blend.

## Validation (`train.py`, `mix.py`)

Holdout = January and July 2025, never used for model choice. CV = five folds over the other ten months,
by month pair. Every result line reports per lane (A = has a flight plan, B = no plan) and per month.
Predictions are floored at 0. Board uploads are logged in `LEDGER.md` with the expected and actual score.

## Reproduce the current build (v25)

v25 = 0.5824 * CatBoost residual lane-A model with arrival recorder-mode features (`ra_cb_arr`) + 0.1399 * the
LightGBM version (`ra_lgb_arr`) + 0.2777 * the MLP version (`nn3_arr`). Lane A is the same residual-to-AOBT
design as v24 (tail-gated tree cells, or an MLP fit on the residual directly), now with 16 more features from
the arrival recorder-mode analysis (`ARR` in `features.py`; each `_arr` config passed a paired day-block
bootstrap gate against its non-ARR parent, `docs/ARR_SHIP.md`). Lane B (no plan) takes `laneb.py`'s LIRF-only
mechanism model as a row-level override (`combine.py --over`) instead of the donor mixture. Weights are
non-negative least squares on the CV out-of-fold predictions (`blend.py`), never on the holdout. Holdout
Jan+Jul 2025: 322.59 all, 223.65 lane A, 1874.62 lane B (`docs/RESULTS.md`); paired day-block bootstrap
against the shipped v24 blend +4.40 [+2.18, +6.84] all, +3.00 [+2.38, +3.64] lane A, +25.98 [+5.24, +57.48]
lane B (`docs/ARR_SHIP.md`). Uploaded 2026-09-13: board 288.6796, down from v24's 290.7995 (`LEDGER.md`).

```
pip install -r requirements.txt            # Python 3.12, CUDA GPU for CatBoost
# put training_*.parquet, ranking.parquet, submitting.parquet in data/
python weather.py fetch 2026-09-05         # data/weather/2026-09-05/<ICAO>.csv
python data.py && python features.py       # caches under data/cache/, incl. ARR (arrival recorder-mode features)
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
python blend.py ra_cb_arr ra_lgb_arr nn3_arr --base=blend_ra_cb+ra_cb_s1+ra_cb_s2+ra_lgb+nn3
python laneb.py fit 100 449                # LIRF-only lane B mechanism model -> data/laneb_rank.parquet
python lanea.py fit ra_lgb_arr mix_cb_wx 254 249 414   # full 2025 refit at the CV iteration counts
python nn.py fit nn3_arr mix_cb_wx
python lanea.py fit ra_cb_arr mix_cb_wx 2356 2743 5048
python combine.py data/merry-octopus_v25.parquet data/ra_cb_arr_rank.parquet:0.5824 data/ra_lgb_arr_rank.parquet:0.1399 data/nn3_arr_rank.parquet:0.2777 --over=data/laneb_rank.parquet
python board.py check data/merry-octopus_v25.parquet
```

`laneb.py` is a LIRF-only mechanism model for lane B (schedule copy vs. physical, day-shifted; `docs/LANEB.md`).
Its own strict pass mark (copy class no worse than the donor) narrowly fails, but its holdout RMSE is better
overall with a bootstrap CI clear of zero, so v25 uses its predictions as a row-level override on lane B
(`combine.py --over`) rather than through the blend weights.

CatBoost on GPU is not bit-reproducible run to run; expect about 1 s of holdout noise. `tune.py` runs Optuna over
the mixture regressor on CV folds only (`python tune.py N_TRIALS 2-3,6-8`, study in `data/optuna.db`).
