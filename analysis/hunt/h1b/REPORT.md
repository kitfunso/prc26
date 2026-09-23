# H1b: file-structure leak hunt

Protocol v6 section 4. A fresh Sonnet agent, 2026-09-23. It returned its report as text (a sub-agent policy blocked
the file write); the session filed it here, condensed. Every script runs from the repo root as
`PYTHONPATH=. python analysis/hunt/h1b/<script>.py`; `common.py` holds the dev/holdout split (the ten dev months,
asserted).

**Verdict: no exploitable leak.** Holdout reads spent on a candidate: 0 of 3. No override file written.

Disclosure from the agent: before `common.py` fixed the split, `s01_explore.py` and `s02_explore2.py` read January
2025 (a holdout month) for schema probes only (phase counts, airport roles, id order). No SSE or candidate number came
from that read.

## The seven checks

| # | check | result | script |
|---|---|---|---|
| 1 | row order | Null. Position within (airport, day) tracks MVT, BLOCK and SCHED equally (0.1308, 0.1307, 0.1304 over 1,740,628 dev rows). `submitting.parquet` keeps `ranking.parquet`'s DEP order. | `s1_row_order.py` |
| 2 | ids | Real structure, not a gain. `FLIGHT_ID_mvt` orders time within (airport, day) at 0.999 and is unused as a feature. v34's out-of-fold residual correlates 0.100 with it overall and 0.577 on the worst 1,961 rows (38.9% of dev SSE). Both attempts to use it lost on held-out months: a neighbour mean (RMSE 463 against v34's 241) and a binned bias fix (7.4M to 12.9M SSE worse). | `s2_ids.py`, `s3_flightid_residual.py`, `s4_bigresid_composition.py`, `s5_fid_resid_correction.py` |
| 3 | links between rows | Null. No BLOCK or TAXITIME value on any of the 344,841 rank rows; no duplicate rows; no MVT_ID collisions. 50,923 rank flights have a DEP row and an ARR row at the destination; the ARR row always comes after takeoff (median 85 min), so it cannot leak the off-block time. | `s6_crosslinks.py`, `s6b_croslink_followup.py`, `s6c_id_overlap_check.py` |
| 4 | time-stamp rounding | Null. SCHED is on the whole minute on every dev row; MVT's seconds differ weakly by stamp class, and a bias fix from them lost 8.23M SSE on held-out months. | `s7_granularity.py` |
| 5 | time zones and clock changes | Null. Labels near a whole hour: 0.071% of rows, against 3.33% by chance. "Off-block = stamp plus or minus 3,600 s" hits 0.34% of rows, and less on the two 2025 clock-change dates. | `s8_timezones.py` |
| 6 | missing-value patterns | Null. Eight flight-plan columns go missing together (99.02% present in train, 98.47% in rank): the lane A / lane B split we already use. | `s9_nulls.py` |
| 7 | parquet metadata | Null. One row group per file, standard schema metadata only, no sort key. | `s00_schema.py` |

## Ranked by SSE at stake

1. The LIRF no-plan extreme rows: 20 of 1,740,628 dev rows (labels over 12 h) hold 18.15% of dev SSE; all lack a
   plan, 95% are LIRF, 60% are schedule copies. v34 gets the order of magnitude but not the precision. In the rank
   set 383 rows are LIRF without a plan and 112 clear the dev-derived gap threshold. This is the known hull problem,
   owned by lane B, so the agent left it alone (`s10_extreme_rows.py`, `s11_rank_sizing.py`).
2. `FLIGHT_ID_mvt`'s time order: real, but both uses lost.
3. Everything else: zero or noise.

## Not done

- `classify_offblock`'s DAY_SHIFT detector catches 1 dev row; the direct checks above cover the cases that matter.
- Turnaround links were not rebuilt: `features.py` already carries them (`ad_arr_*`, `w_arr_*`, `al_arr_*`, `pa_*`).
- `STAND_mvt`, `RUNWAY_mvt` and `AIRCRAFT_TYPE_mvt` were not re-checked by stamp class; they are 99.9 to 100% present.

## Session verdict

Accepted as a null. No candidate reached the pass bar (279.1M less SSE), so N is unchanged.
