# H1a: where the off-block stamp comes from on the giant rows

Protocol v6 section 4. A fresh Opus agent, 2026-09-23. It returned its report as text (a sub-agent policy blocked
the file write); the session filed it here as written, with the session verdict at the end. Every script runs from
the repo root as `PYTHONPATH=. python analysis/hunt/h1a/<script>.py`.

**Outcome.** No candidate passes the bar. I spent 2 of the 3 holdout reads.

- **Read #1** backs v34 on rank row ITY340 (213147416). v34 predicts 82,117 s there. If that row were a normal taxi,
  v34 would lose 6.56B on it alone.
- **Read #2** backs bet B1: 7 of 7 holdout rows are copies. It saves 237.3M on the holdout, with `boot_delta`
  +1.131 s [+0.078, +2.485]. That is below 279.1M, and the rows are unjudgeable hull rows, so B1 stays a bet.
- **What the numbers use.** v34's dev OOF (1,740,628 rows, SSE 101.016B), the holdout files, and label-free counts on
  2026 rank rows. Nothing uses board scores.

## How to regenerate

Run each script from the repo root as `PYTHONPATH=. python analysis/hunt/h1a/<script>.py > out_<script>.txt`.

| Script | What it prints | Needs |
|---|---|---|
| `origin.py` | where BLOCK comes from, dev rows with y > 1 h | |
| `q4_decomp.py` | v34 error split into wrong class and wrong size; LIRF copy rates by zs | |
| `stamps.py` | whether big off-blocks match another row's stamp, against a null | |
| `daterule.py`, `dayshift.py`, `utcrule.py` | the day-shift rules at LIRF lane B | |
| `copymodel.py` | label-free copy detector (OOF); writes `copyprob_lirf.parquet` | |
| `evalmix.py` | mixture, stack and combined rules against v34 | copymodel |
| `hull.py` | hull cells, strategy H, rank risk | copymodel |
| `bet_hull.py` | bet B1; writes both `bet_hull_*_override.parquet` files | copymodel |
| `arrcong.py` | congestion features, level-2 GBM on v34's residual; writes `arrcong_feat.parquet` | |
| `lanea_day.py` | lane A rows whose schedule or flight plan sits more than 12 h before takeoff | |
| `tables.py` | tables A to F cited below | copymodel, arrcong |
| `read_holdout.py B1\|L1` | the counted reads; appends to `holdout_reads.log` and refuses a repeat | bet_hull |

## Findings, ranked by SSE at stake

| # | Mechanism | Rows (dev / holdout / rank) | Dev ΔSSE vs v34 OOF | Holdout ΔSSE, `boot_delta` | Verdict |
|---|---|---|---|---|---|
| 1 | L1: in lane A, the whole flight plan is a day old. LOBT, IOBT, EOBT_1, AOBT_3 and SCHED all sit more than 12 h before takeoff. On the one 2025 row, BLOCK = SCHED. | 0 / 1 / 1 | no dev rows | Read #1. Day-sized p = MVT − AOBT_3: **+67.7M**, +0.322 s [0.000, +1.185]. Normal-taxi p: **−7,309.7M**, −32.98 s [−106.97, 0.000]. | Keep v34's 82,117 on ITY340. Never hedge it down. |
| 2 | Rows with y > 12 h: 12 LIRF lane B schedule copies, 7 LIRF lane B physical off-blocks stamped on the schedule's UTC date, and 1 LSZH row with BLOCK = SCHED − 86,402 s. | 20 dev rows, SSE 18.332B. The flag "LIRF lane B and zs > 12 h" touches 18 / 11 / 11. | best rule (strategy H): **−1.044B** | not read | No candidate. |
| 3 | B1: LIRF lane B rows with 4 h ≤ zs ≤ 12 h and a flight number that is not a plain AAA123 are schedule copies. | 29 / 7 / 12 | **+0.432B** OOF (0.467B → 0.035B); +0.447B with in-sample q | Read #2: **+237.3M**, +1.131 s [+0.078, +2.485]; 7 of 7 copies | Bet. Below the bar, and the rows are unjudgeable. |
| 4 | LIRF schedule copies with y of 1-12 h (wrong class). | LIRF rows with zs > 30 min. Lane A: 48,128 / 11,234 / 11,167. Lane B: 1,028 / 383 / 355. | Lane A stack **+66M** (every fold positive). Lane B stack −36M. Mixtures lose 1.2B (lane B) and 8.3B (lane A). | Not read. Expected holdout gain is only about 15M. | No candidate. |
| 5 | Long physical taxis (wrong size). They are not another row's stamp, and visible congestion does not explain them. | 1,805 dev rows | Level-2 GBM: **−2.199B** with congestion features, −3.151B without | not read | No candidate. |
| 6 | LIRF lane B rows whose takeoff crosses UTC midnight relative to the schedule, with zs < 12 h. | 29 / 7 / 15. The 29 dev rows are 28 copies and 1 shift. | Not run as a rule. The one shift (ITY332) costs 7.07B. | not read | Rejected: a coin flip on one row. |
| 7 | The other hull cells: 4-12 h with a plain number, and zs > 12 h. | 35 / 27 / 20 and 18 / 11 / 11 | cell hedge **−0.447B** and **−1.044B** | not read | Keep v34. |

Where each finding's numbers come from:
- **Finding 1:** `holdout_reads.log` lines 1-5 and `out_lanea_day.txt`.
- **Finding 2:** `out_q4_decomp.txt` lines 26-32 and `out_hull.txt` line 18.
- **Finding 3:** `out_hull.txt` line 19, `out_bet_hull.txt` and `holdout_reads.log` lines 6-14.
- **Finding 4:** `out_evalmix.txt` and `out_copymodel.txt`.
- **Finding 5:** `out_stamps.txt`, `out_arrcong.txt` and `out_tables.txt` part B.
- **Finding 6:** `out_tables.txt` parts E and F.
- **Finding 7:** `out_hull.txt` lines 20-22.

**Why single rows matter.**
- **Cost of one miss.** A day-sized miss costs about 86,400² ≈ 7.46B of SSE. Our board SSE is 25.24B, and third place
  is 5.58B below us.
- **Two dev rows hold 14.6B** of the 101.0B dev SSE (`out_tables.txt` part E). Nothing visible on either row flags it:
  - LSZH NJE389D: zs 939, y 87,341, p 590, SSE 7.53B.
  - LIRF ITY332: zs 11,694, y 87,168, p 3,058, SSE 7.07B.
- **Hull exposure.** v34's dev hull SSE is 6.670B over 82 rows, about 81M per row. If 2026 behaves like 2025, the 43
  rank hull rows carry about 3.5B. That figure is arithmetic, not a measurement.

## The four questions

**Q1. What is BLOCK on each y > 12 h row?** It is never another row's stamp.
- **12 rows** copy their own schedule.
- **7 rows** are physical off-blocks stamped on the schedule's UTC date, so y = 86,400·k + taxi.
  - The rule "k = UTC date of the actual off-block − UTC date of the schedule" matches 502 of 504 LIRF lane B
    non-copy rows. It fails on WMT7TL and ITYTY630.
  - Lane A never shifts: 0 of 109,024 LIRF lane A non-copy rows. That includes 51 rows whose off-block falls on a
    later UTC date than the schedule.
- **1 row** (LSZH) has BLOCK = its own schedule − 86,402 s.
- **Stamp test** on the 1,805 physical rows with y > 1 h, hits against the null: same flight 5 vs 0, same FLIGHT_ID
  0 vs 0, same stand 48 vs 98, same prefix 1,091 vs 1,089. So there is no signal.
- Sources: `out_origin.txt`, `out_utcrule.txt`, `out_daterule.txt` and `out_stamps.txt`.

**Q2. Is there a label-free signal for y > 12 h?** Yes for 18 of the 20 rows, and v34 already uses it.
- **The flag.** "LIRF lane B and zs > 12 h" fires on 18 dev rows, and all 18 have y > 12 h. v34's implied copy
  probability on them is 1.0.
- **What is left** is copy (y = zs) against shift (y = 86,400 + taxi). My detector's AUC on that split is only 0.639.
- **Outside LIRF lane B,** zs > 12 h fires on 497 dev rows, and all are normal taxis. v34 predicts normal taxis there:
  mean p/zs is 0.017 on dev, 0.028 on holdout and 0.027 on rank (`out_tables.txt` part A).
- **The other 2 rows:**
  - ITY332 crosses UTC midnight, but only 1 of 29 such dev rows shifted.
  - LSZH NJE389D has no flag at all.

**Q3. The same for y of 1-12 h (3,125 dev rows, 15.749B).**
- **Wrong class: LIRF copies,** 738 lane A rows (3.254B) and 535 lane B rows (3.256B).
  - Lane B detector AUC is 0.887 and 0.849. v34's implied AUC is 0.896 and 0.893, so v34 already has this.
  - Lane A detector AUC is 0.834 and 0.886, above v34's 0.778 and 0.844. But its best use, a stack, only saves 66M
    on dev.
- **Wrong size: long physical taxis.**
  - The biggest groups: LTFM (1.580B lane A, 1.802B lane B), LIRF lane B (1.921B), LFPG (0.905B lane A, 0.432B
    lane B).
  - None of it is explained by congestion that is visible on rank days. I tried arrival taxi-in, neighbours' schedule
    delay and runway counts over the 30, 60 and 120 minutes before takeoff.
  - v34's residual is flat across deciles of those features. The largest top-decile bias is 45 s at LTFM, worth
    45-48M in-sample.

**Q4. Wrong class or wrong size?**

| Band | v34 SSE | Wrong class | Wrong size |
|---|---|---|---|
| y > 12 h | 18.332B | 18.331B (100%) | 0.001B |
| y 1-12 h | 15.737B | 6.733B (43%) | 9.004B (57%) |

In the 1-12 h band, 6.51B of the 6.73B class error is LIRF copies.

## The bet: B1 (unjudgeable hull rows)

- **Mechanism.** LIRF lane B rows with 4 h ≤ zs ≤ 12 h and a garbled flight number record their own schedule as
  BLOCK. Garbled means a doubled prefix (ITYTY672, WZZZZ401), no number, or another odd shape.
  - With a plain number, the same band is only 24 copies out of 35 rows.
  - Below 4 h, garbled rows are only 60-80% copies, so the rule stops at 4 h.
- **2025 evidence.** 29 of 29 dev rows and 7 of 7 holdout rows are copies.
- **Prediction.** p = q·zs + (1 − q)·1,026, with q = 30/31 = 0.968 taken from dev rows only.
- **Rank rows it touches (12):** 202681830, 202683690, 202683962, 202687662, 202689463, 202689507, 213141089,
  213142471, 213151252, 213153920, 213147307, 213155642.
- **Risk both ways on rank:**
  - If all 12 are copies, it saves 0.196B, about 1.0 s of board RMSE.
  - If all 12 are physical, it costs 1.221B (0.102B per physical row on average).
  - It pays only if at most 1 of the 12 is physical.
  - Under a Beta(1, 37) posterior, P(at most 1 physical) ≈ 0.94 and the expected gain is about +0.16B. That is
    arithmetic, not a measurement.
- **Files** (written by `bet_hull.py` from dev rows only):
  - `bet_hull_rank_override.parquet` (MVT_ID_mvt, TAXITIME_SEC_mvt; 12 rows).
  - `bet_hull_holdout_override.parquet` (id, p; 7 rows).

**ITY340: keep v34, no second bet.**
- **The row.** SCHED is 94,560 s before takeoff. LOBT and IOBT are 86,820 s before it, EOBT_1 86,280 s and AOBT_3
  86,340 s.
- **2025 evidence:**
  - No dev row has a day-old flight plan.
  - When AOBT is only 2-12 h early, BLOCK never equals it (0 of 15 rows).
  - The one holdout match (AEZ146R) was a schedule copy, y = 87,002. Hedging it toward a normal taxi would have cost
    7.31B.
- **Raising ITY340 to p = zs** changes SSE by:
  - +154.8M if the label is a SCHED copy.
  - −37.8M to −51.2M if it matches a flight-plan stamp.
  - −2,170.8M if it is a normal taxi.
- I do not propose that move, because it rests on one matching row.

## NOT DONE

1. Holdout read #3 is unused.
2. I did not pool dev and holdout (36 of 36) for B1's rank q. That would give q = 0.974, a change of about 0.6% of zs
   per row.
3. The two 7B dev rows have no flag beyond "1 in 29 midnight-crossers shift". I did not search weather, NOTAMs or the
   inbound leg's delay.
4. I have no mechanism for WMT7TL (0.72B) or WMT8PV (0.50B). In both, BLOCK is 7-9 h before takeoff, and neither is a
   copy or a shift.
5. I did not trace why garbled numbers are always copies at 4-12 h but only 60-80% copies below 4 h.
6. I did not build a lane A copy model stronger than the linear stack. The stack only reached +66M on dev.
7. I tried no disruption features beyond arrival taxi-in, neighbour delay and runway counts, and I did not tune the
   level-2 GBM. The decile table puts its ceiling under 0.1B.
8. LFPG lane B (35 rows, 0.432B) only got the general tests.
9. The hull cells "4-12 h plain" (20 rank rows) and "zs > 12 h" (11 rank rows) have no rule better than v34, and were
   not read on the holdout.

## Holdout reads: 2 of 3

- **#1, L1** (1 row): day-sized is right.
- **#2, B1** (7 rows): 7 of 7 copies, +237.3M.

Both are logged in `holdout_reads.log`, and the script refuses to read the same candidate twice.

## Session verdict: B1 taken as the one declared bet (v41); no candidate passes; N rises by 2

- **Where the risk sits.** On rank, 7 of the 12 rows already sit near 97.5% of the gap in v34, and the bet moves them
  under 1%. The other 5 (RYRR8830, AEZZ2023, SVII3006, RYRR388G and one row with no number, 213153920) sit near 77%;
  the bet pays on them if their copy rate is above 0.87, the midpoint of 0.77 and 0.968.
- **2025 evidence by shape** (`b1_shapes.py`): in the 4-12 h band, doubled one-letter prefixes are 12 of 12 copies,
  doubled two-letter prefixes 13 of 13, no number 3 of 3, other 1 of 1. Garbled rows in the bands around it copy less:
  73% at 0.5-2 h, 76% at 2-4 h, 62% at 12-24 h. The clean check is the holdout's 7 of 7, read once on a cell defined
  on dev.
- **Drift check, null** (`b1_arrivals.py`): LIRF arrivals never copy the schedule once they land more than 30 min
  late, in 2025 or 2026, garbled or plain, so arrivals cannot test the departure mechanism. Near the schedule, the
  garbled share and copy rates are alike in both years.
- **The call.** The best file counts, so a losing bet costs a slot, not the score. Expected board gain +0.85 s under
  Beta(1, 37); +1.05 s if all 12 are copies, -6.47 s if all are physical. Taken: `merry-octopus_v41.parquet` = v34 plus
  the 12 rows (`combine.py --over`), sha256 prefix 5c9c06d5c9e14c63, uploads at 00:15 UTC on 2026-09-24 (`LEDGER.md`
  v41 row). Declared before the upload: B1 rides every later file whatever v41 scores, and it gets no second try.
- **Not taken.** L1 (raise ITY340 to p = zs): one matching row, and the agent does not propose it. Pooled q = 0.974
  (NOT DONE 2): worth about 0.05 s, and it would tune the bet on the holdout read that judged it.
- **Holdout reads.** H1a spent 2 (L1, B1); H1b and H1c spent none. Both reads go in `docs/RESULTS.md`.
