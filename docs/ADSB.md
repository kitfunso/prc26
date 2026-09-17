# Open ADS-B ground traces (pilot, 2026-09-17): null, closed

Question: do open ADS-B ground traces show the off-block time the label hides?
Source: adsb.lol `globe_history_2025`, ODbL, release `v2025.07.15-planes-readsb-prod-0` (3.5 GB, one day).
Eligible under the rules ("openly accessible/usable and documented"). Script and tables: `data/adsb_pilot/pilot.py`,
`data/adsb_pilot/REPORT.md` (gitignored; rerun `python data/adsb_pilot/pilot.py`). One day = 2 min download, 4 min parse.

Method: departures on 2025-07-15 matched to the aircraft whose ground-to-airborne transition at ADEP is within 120 s of
`MVT_TIME_UTC_mvt`; clean rows = lane A matches whose trace callsign equals `CALLSIGN_flt`. `taxi_move` = takeoff minus the
first ground point at 2 kt or more after the latest 20 min gap and after any landing. Model `p` = `recal_holdout` (v32 lane A).

| | value |
|---|---|
| departures that day | 6,120 |
| matched within 120 s | 49 to 63% at nine airports; LTFM 0% (no feeders) |
| callsign agrees | 1,517 of 2,796 lane A matches (54%) |
| matches with ground points over 60 s before takeoff | EHAM 99.7%, EDDF 95%, LSZH 93%, LIRF 92%, LEBL 92%, LEMD 75%, EGLL 53%, LFPG 20%, EDDM 16% |
| clean rows, five airports with 90%+ coverage (n = 1,241) | `taxi_move` RMSE 691 s, model 265 s |
| label minus `taxi_move` | median +308 s, 5th/95th -29 / +1,009 s; 36.7% within 120 s |
| out of sample, odd/even id split | OLS(p, taxi_move, taxi_seen, airport) 262 to 266 s vs p alone 265 to 266 s; LightGBM 269 to 283 s (worse) |
| corr(label - p, taxi_move - p) | 0.070 |
| stamp-copy rows vs the rest | `taxi_move` RMSE 621 s (n = 337) vs 755 s (n = 1,075) |

EHAM is the one airport where the trace reads the label: median absolute error 20 s, 74.5% within 120 s (n = 322), but a
tail takes its RMSE to 398 s against the model's 130 s. Sized generously (clean matches are 46% of EHAM rows, 75% of them
in band, model error 130 s to 60 s there) it is about 200M of the board's 25.4B SSE, 0.4 s, for a 220 GB pull of the two
ranking months plus training months. Not taken.

Verdict: transponders come on after pushback at most stands, half the matches cannot be verified, and the estimate's error
is uncorrelated with the model's residual. Do not retry with this source.
