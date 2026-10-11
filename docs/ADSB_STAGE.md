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

More training months help. With February 2025 added to January and July, the gate gives +12.33 s [+9.87, +15.78]
(holdout 292.65, lane A 198.86; EDDM matched rows 89.8 to 79.0, EGLL 184.2 to 174.0), so the stage trains on every
2025 day the archive publishes.

Neighbours (`adsb_nb.py`): a second layer that reads the first layer's corrections of matched departures at the same
airport within 30 and 60 min, nested-cross-fitted on the same three months, adds nothing: +12.30 s against +12.33 s
for the first layer alone. It is not used.

Third match pass: a trace carries its callsign forward from where it is first broadcast, often after take-off, so the
ground points before pushback still hold the inbound flight's callsign and the callsign pass misses them. Pass 3 finds
the aircraft by the departure's callsign on its climb-out inside the airport cut, then reads that aircraft's ground
track. It lifts the 2026-01-15 match share from 63.6 % to 67.8 % and the gate from +12.44 s to +13.10 s
[+10.53, +16.64] (holdout 291.89, lane A 197.74), most at EGLL and EDDF.

Round of 10 October, Jan+Feb+Jun+Jul 2025 training days (364,558 matched rows): June adds nothing over three months
(+12.95 s against +13.10 s). A stage of its own for LIRF, trained on the June and July LIRF rows, still loses (LIRF
lane A 421.4 to 427.5, lane B worse), so LIRF stays uncorrected. With pass 3 matching more LFPG departures,
correcting LFPG now helps (+13.02 s; LFPG matched lane A 267.2 to 252.8), so only LIRF is skipped from here.

Coverage in the final file's new months: 2026-02-10 matches 61.6 % of departures, close to January, but EDDF falls
to 11 to 39 % across February (2026-02-03 has 4,170 EDDF ground points against 42,768 on 2026-01-28), as a receiver
near Frankfurt appears to drop out. Those EDDF rows keep v53.

Checks of 10 October, afternoon:

- LIRF lane B only (`skipB=LIRF`): the pooled stage now improves LIRF lane A on the holdout (428.8 to 418.5, all from
  July, since January 2025 has no LIRF matches), and LIRF lane B, whose labels run to 88,000 s, stays uncorrected.
  The gate rises from +13.02 to +13.18 s, but paired against `skip=LIRF` it is +0.17 s [-0.19, +0.72], with 15 of 31
  days gaining. Not adopted; it is retested on the late-2025 training months. Skipping every lane B row (`laneA`)
  scores +12.64 s, so the stage helps lane B elsewhere.
- OPDI (EUROCONTROL's open flight events, from OpenSky) as a second pushback source: its ground events cover the same
  airports as adsb.lol and miss the same ones. For 10 to 19 January 2026 it has 1,517 runway entries at LFPG, 781 at
  LIRF and 25 at LTFM, and its EDDF stand exits fall from about 500 a day to 1 or 2 from 5 February, the same day as
  adsb.lol's. It adds nothing, so it is not used.
- The 2026 ranking set matches 63.3 % of departures. LIRF has almost no ground points in July 2026 (0 to 44 on 7 of 8
  sampled days, against 9,000 to 16,000 in January) and LEMD about half, so those gaps are receivers, not matching.
- The largest remaining holdout errors are labels, not clocks: two LFPG lane B rows (84,240 s and 58,206 s, no plan
  and no feature near either value) hold 32 % of the holdout squared error. Among matched lane A rows outside LIRF
  missed by more than 600 s, 109 have an ADS-B clock within 120 s of the label, 385 a plan stamp within 60 s, and 621
  neither. LEBL plan-copy rates do not depend on the airline (12 to 18 % EOBT copies for every large airline).

Rounds of 10 and 11 October, late-2025 training months. Two rules were set before the runs: more training days are
adopted unless the paired holdout says they are clearly worse, and `skipB=LIRF` needs a paired interval against
`skip=LIRF` that excludes zero.

- Nov+Dec 2025 added (596,342 matched rows, v69 recipe): `skip=LIRF` +13.56 s [+10.91, +17.21] against the base;
  `skipB=LIRF` against `skip=LIRF` +0.32 s [-0.01, +0.86]. Not adopted. v69 was built and not uploaded.
- Every 2025 month (1,219,616 matched departures over 363 days; 2025-12-31 has no release, 2025-10-15 failed to
  decompress): against v69, `skip=LIRF` +0.25 s [+0.14, +0.38]. `skipB=LIRF` against `skip=LIRF` +0.53 s
  [+0.18, +1.08], all from July (January 2025 has no LIRF matches), so LIRF lane A is now corrected. Gate against
  the base +14.33 s [+11.44, +18.18]; holdout 290.65 all, lane A 195.87, lane B 1730.42. This is v70.
- v70 on the board: 241.4228 against v68's 243.0747. It corrects 8,415 LIRF lane A rows, which v68 left alone, and
  no LIRF lane B row.
- `rec` (the LOBT and IOBT stamps and their distance to the clocks, the stamp seconds flags, and the base's day and
  arrival recording-mode columns, 0 to 2.3 % missing on both 2025 and 2026 rows): +0.01 s [-0.12, +0.13] against
  v70's stage. Null; the base already carries what those columns know.
- Correcting LIRF lane B as well (no skip flag): -1.07 s [-2.14, -0.07] against v70 (LIRF lane B 3572.8 to 3708.5),
  so LIRF lane B stays uncorrected.
- Error left after v70 on the holdout: lane B rows hold 55 % of the squared error (5,373 rows, top 50 rows of all
  holdout rows 47 %), unmatched lane A 27 % (LTFM 8.1 %, LFPG 7.1 %), matched lane A 18 % (LIRF 8.2 %, LEBL 3.0 %).
  adsb.lol has 9 points near LTFM on 2026-01-15 against 344,918 near EHAM and 44,880 near LFPG, so LTFM and most of
  LFPG are out of reach of this source.
- `cr`, copy rates: a label "copies" when it sits within 60 s of the EOBT, schedule or LOBT gap while the pushback
  clock is more than 180 s away (4.4 % of matched lane A rows at EDDM to 27.3 % at LIRF). The rate per stand is
  stable across months (train-month rate against holdout-month rate, correlation 0.64 to 0.90 at 8 of 9 airports,
  0.46 at EGLL; EHAM stands run 0.3 % to 11 %), and so is the rate per operator (0.93 at LIRF). Two columns, the
  smoothed rate per (airport, stand) and per (airport, operator) with 50 rows of the airport rate as the prior
  (fixed before the run); each training row's rate leaves its own month out, so it sees its key the way an unseen
  2026 month does. Against v70's stage: +0.53 s [+0.40, +0.70], January +0.37, July +0.69, gains at 8 airports.
  Gate against the base +14.86 s [+11.90, +18.85]; holdout 290.13, lane A 195.12. Keys with 50 or more training rows
  cover 97.0 % (stand) and 97.8 % (operator) of matched 2026 rows, against 99.1 % and 98.6 % in the holdout. This is
  v71, board 241.0532 against v70's 241.4228.
- `cr3`, the copy rate split by stamp (EOBT, schedule, LOBT) on top of `cr`: +0.04 s [-0.04, +0.11] against v71.
  Null.
- `cb`, clock bias per stand: the smoothed mean of the label minus the `first` and `lv10` clocks (clipped to
  ±900 s) over the stand's rows that do not copy a stamp, with the same 50-row airport prior and month left out as
  `cr`. It tells the stage how long a stand's pushback runs before the feeders see it. On top of `cr`, against v71:
  +0.21 s [+0.12, +0.31], January +0.14, July +0.28, gains at 8 of 9 matched airports (LFPG -0.4M SSE). Gate against
  the base +15.07 s [+12.05, +19.09]; holdout 289.92, lane A 194.84. Stand keys with 50 or more clean training rows
  cover 96.8 % of matched 2026 rows, against 98.9 % in the holdout. This is v72, board 240.8664 against v71's
  241.0532.
- `cbo`, the same clock bias per operator on top of v72: +0.02 s [-0.05, +0.10] against v72. Null; the stand
  carries what the operator knows.
- `crh`, the copy rate per (airport, hour) on top of v72 (hourly rates are stable across months, correlation 0.64
  to 0.87 at 7 of 8 airports, but span less than the stand rates, 24 to 35 % at LIRF): +0.00 s [-0.07, +0.08]
  against v72. Null. With `cbo` that is two null rounds in a row, so the per-key statistics are closed. The neighbour layer already read matched neighbours for unmatched departures and
  was null, so that route stays closed.
- Error left after v72 on the holdout: unmatched lane B at LFPG holds 36 % of the squared error (871 rows) and LIRF
  lane B 15 %; unmatched lane A at LTFM 8.2 % and LFPG 7.1 %, where adsb.lol has no traces. Matched lane A rows missed
  by more than 600 s hold 9.9 % (1,938 rows; 892 at LIRF, 530 at LEBL): 5.4 % on labels within 60 s of a plan stamp,
  4.1 % on labels near neither a stamp nor a clock, 0.4 % near a clock.
