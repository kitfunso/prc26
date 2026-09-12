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

## Features (`features.py`)

All from the challenge files only (no ADS-B, no weather in the shipped file). Per departure: gaps from takeoff to
each stamp, differences between stamps, stamp rounding, calendar (hour, weekday; month is never a feature),
categorical ids (airport, runway, stand, type, operator, destination, airline prefix), and counts of movements
in the same file around the estimated off-block and the takeoff.

As-of rule: a feature may use any movement in the same file, before or after the takeoff (the rules allow it;
`dep_after` and `arr_off` look forward). Airport-day aggregates (`DAY`) also come from the same file.

## Validation (`train.py`, `mix.py`)

Holdout = January and July 2025, never used for model choice. CV = five folds over the other ten months,
by month pair. Every result line reports per lane (A = has a flight plan, B = no plan) and per month.
Predictions are floored at 0. Board uploads are logged in `LEDGER.md` with the expected and actual score.

## Reproduce the submitted file

```
pip install -r requirements.txt            # Python 3.12, CUDA GPU for CatBoost
# put training_*.parquet, ranking.parquet, submitting.parquet in data/
python data.py && python features.py       # caches under data/cache/
python mix.py cv mix_cb                    # CV + holdout line in docs/RESULTS.md
python mix.py fit mix_cb IT_CLF IT_REG     # iteration counts from that line -> data/mix_cb_rank.parquet
python board.py check data/merry-octopus_vN.parquet
```
