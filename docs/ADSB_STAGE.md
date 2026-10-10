# ADS-B pushback stage (10 October 2026)

## Data source, declared

Open ADS-B traces from the adsb.lol `globe_history_2025` and `globe_history_2026` archives
(https://github.com/adsblol, one release per day, Open Database License 1.0). The organisers ruled on Discord on
18 September 2026 that open trajectory data may be used to extract off-block times when it is declared; this file is
that declaration. No OpenSky state-vector data is used (ruled inadmissible on 3 September 2026).

Days pulled: every day of January, February, June and July 2026 (the final file's months) and every day of 2025 that
the archive publishes (June 2025 has 20 days). The 2025 days are the stage's training set, since every 2025 departure
has a label and an out-of-sample v53 prediction. `python adsb_fetch.py` streams each day's split
tar, keeps the points within 8 km of one of the ten airports and under 3,000 ft above it, writes
`data/adsb/pts/YYYY-MM-DD.parquet` (about 17 MB a day) and deletes the raw archive (about 3.2 GB a day).

## Match and clocks

`python adsb_events.py SRC [RANKING]` matches each departure to the aircraft whose ground-to-air transition at the same
airport lies within 120 s of `MVT_TIME_UTC_mvt`. The trace callsign equal to `CALLSIGN_flt` breaks ties, then the
aircraft type, then the time difference; each track serves one departure. The ground points before that take-off
(back to the aircraft's landing or a 30 min gap) give candidate off-block clocks, each in seconds before take-off:
first point seen, first point 5, 10, 20 and 30 m from it, the end of the last stop of 120 s or more held 400 m or
more from the take-off roll, and the leave time of the longest stop. Quality columns ride beside them (dwell before
the first move, largest gap, point count, stop count and length, speed at the first point, callsign and type
agreement, candidate count).

Why several clocks: the label is the airport's recorded off-block, and how much of the pushback the open feeders see
differs by airport. At EDDM and EHAM the transponder is usually on at the stand and the 10 to 20 m leave time sits
within a minute of the label for about half the rows; at EDDF, LEBL, EGLL and LIRF it mostly comes on after pushback,
so the clock is a lower bound 300 to 900 s short of the label.

## Stage

`python adsb_stage.py` fits LightGBM on the residual `y - p` of matched departures, `p` being v53's out-of-sample
prediction (the Jan+Jul 2025 holdout, out-of-fold for other months), with the clocks, each clock minus `p`, the quality
columns and the airport. `gate` judges it on the Jan+Jul 2025 holdout, each month corrected by a stage trained on the
other months only; `apply` trains on every labelled day and corrects a submission file, then projects onto the flight
plan window (`|BLOCK - LOBT| <= 3,606 s` on every 2025 departure that has a LOBT) and floors at zero.

## Results

Filled in from the runs below.
