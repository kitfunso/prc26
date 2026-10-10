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

Gate, 10 October 2026, `python adsb_stage.py gate skip=LFPG,LIRF`, trained on the January and July 2025 ADS-B days
only (each month corrected by a stage fitted on the other): 168,725 of 344,419 holdout departures matched (49 %).

| | v53 | v53 + stage |
|---|---|---|
| holdout, all rows | 304.99 | 293.64 |
| lane A | 216.06 | 200.24 |
| lane B | 1736.89 | 1731.20 |

Paired day-block bootstrap against v53, all rows: +11.35 s [+9.08, +14.56]. Matched lane A rows by airport, RMSE
before and after: EDDF 169.6 to 118.7, EDDM 162.3 to 89.8, EGLL 221.7 to 184.2, EHAM 160.3 to 112.0, LEBL 212.7 to
185.1, LEMD 196.3 to 152.8, LSZH 159.1 to 104.8. LIRF lane A got worse with the stage (419.2 to 439.0): many LIRF
labels copy the schedule, which no pushback clock sees, so LIRF stays uncorrected. LFPG has almost no coverage and
stays uncorrected too.

Coverage is higher in 2026 than in the training year: 63.5 % of January 2026 departures match, with EDDM at 97 %,
LEMD at 96 % and EGLL at 78 % (2025: 62 %, 14 % and 34 %).

Transfer to airports the training year barely covers: without the airport column the stage gains +10.93 s
[+8.73, +14.00]. Fitted with each airport left out of its own training (`gate noapt loao`), it still gains +9.15 s
[+7.27, +11.73], and every corrected airport improves (LEMD 196.3 to 150.2, EGLL 221.7 to 192.2, EDDF 169.6 to
131.0). The stage carries across airports, so the higher 2026 coverage at EDDM, LEMD and EGLL is safe to correct.
