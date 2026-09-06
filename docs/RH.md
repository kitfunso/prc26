# RH: a recent history of the realised clock, in both directions

Regenerate with `python -m taxi.rh`. Diagnostic: no cell declared, no champion moves, no verdict decided.
MC2 holdout RMSE 232.7269 s on 339,046 lane-A rows, against the 2.0 s improvement floor.

## 1. Why the question is open

The lane-A target is `c = AOBT_3 - BLOCK_TIME`, a disagreement between the flight-plan clock and the movement
clock. A clock offset should be persistent and airport-specific, so the natural missing column is what that
offset recently was. `rec_taxi` looks like it, but averages the PLANNED taxi `z_aobt` of the last 10 takeoffs;
no feature in the build is a realised outcome of a past flight. The construction below is point-in-time by the
same argument `rec_taxi` already relies on: `searchsorted(t, anchor, 'left')` admits only events already complete.

## 2. Availability decides the departure direction before signal does

| ranking.parquet | rows | labelled | with a block time |
|---|---|---|---|
| departures | 344,841 | 0 | 0 |
| arrivals | 344,693 | 344,693 | 344,693 |

The ranking set carries no departure label at all, so a recent realised DEPARTURE offset cannot be computed
there: the preceding departures are exactly the rows we are asked to predict. It is measured below anyway,
because a large holdout gain would have been worth finding a proxy for, and because the measurement is what
makes the closure a fact rather than an assumption.

The same table opens the other direction. Every one of the 344,693 arrivals in the ranking set carries
its realised taxi-in and its block time. That is a labelled measurement of the same surface, at the same
airports, in the same window, and no feature reads it: arrivals enter the build only as event times, through
`LANDINGS` for the counts and `ONBLOCK` for stand occupancy.

## 3. Both directions, against MC2's cached holdout residuals

Realised departure offset (not available on the ranking set):

| channel | populated | corr with y | corr with the MC2 residual | linear ceiling (s) | out-of-fold gain (s) |
|---|---|---|---|---|---|
| last 10 departures, airport | 100.0% | +0.1723 | -0.0776 | 0.701 | +0.504 |
| last 30, airport | 100.0% | +0.1600 | -0.0593 | 0.409 | +0.297 |
| last 100, airport | 100.0% | +0.1386 | -0.0345 | 0.138 | +0.120 |
| last 10, airport and airline | 97.8% | +0.1824 | -0.0490 | 0.274 | +0.171 |
| last 10, airport and runway | 99.9% | +0.1843 | -0.0722 | 0.607 | +0.359 |

Realised arrival taxi-in (available on the ranking set):

| channel | populated | corr with y | corr with the MC2 residual | linear ceiling (s) | out-of-fold gain (s) |
|---|---|---|---|---|---|
| last 10 arrivals | 100.0% | +0.1243 | -0.0240 | 0.067 | -0.171 |
| last 30 arrivals | 100.0% | +0.1322 | -0.0237 | 0.065 | -0.100 |
| last 60 arrivals | 100.0% | +0.1347 | -0.0221 | 0.057 | -0.166 |
| arrivals on stand in the previous 30 min | 97.2% | +0.1252 | -0.0246 | 0.069 | -0.150 |
| arrivals on stand in the previous 60 min | 99.1% | +0.1320 | -0.0241 | 0.067 | -0.124 |

## 4. Answer: the family is closed, for two different reasons

The departure direction has the stronger raw signal of the two and still does not reach the floor. Its best
out-of-fold correction is +0.504 s, about a quarter of the 2.0 s floor, and its linear ceiling is the
same size: a residual correlation of -0.078 caps an additive correction at 0.70 s by arithmetic. So the channel
would have been parked under the floor even if the ranking set had carried the labels, and the availability
fact makes it moot rather than being the only thing against it.

The arrival direction is available and is the one that matters, and it is flat: it correlates with the label
but not with the residual, every out-of-fold correction is negative, the best of them -0.100 s, and its linear
ceiling is 0.069 s, so the closure does not rest on the sign of a bin correction. MC2's 49
columns already span it. `queue`, `occ`, `dep_before`, `arr_anchor`, the hour and the airport describe the same
congestion the arrival taxi-in measures, and they describe it without the noise of averaging ten arrivals.

Consequence for the campaign: do not re-ask for a recent-realised-history feature in either direction, and do
not treat the labelled 2026 arrivals as an unused channel. They are unused as a LABEL and spanned as a SIGNAL.

