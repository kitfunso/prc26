# PRC Data Challenge 2026 — state

Site: https://prc-data-challenge-2026.netlify.app/

## Task
Predict taxi-out time (`TAXITIME_SEC_mvt`) for departures at 10 European airports.
Scored on RMSE over Jan + Jul 2026 movements. Deadline **11 Oct 2026 23:59:59 CET**.
Prize: 5000 EUR split across the top 3. Winning entries must be GPLv3 on GitHub.

## Data (needs MinIO keys, see Blocker)
- Bucket `competition-data` on `s3.opensky-network.org` (path-style, https).
- 12 monthly `training_2025-MM-01_2025-MM-01.parquet` files, 4,167,797 movements.
- `ranking.parquet` — Jan + Jul 2026. For `PHASE_mvt='DEP'`, `BLOCK_TIME_UTC_mvt`
  and `TAXITIME_SEC_mvt` are blanked. Everything else is present.
- `submitting.parquet` — template, two columns: `MVT_ID_mvt`, `TAXITIME_SEC_mvt`.
- Submit as `<team-name>_v<n>.parquet` into the team's own bucket. Row set must
  match the ranking departures exactly (no missing, no extra, ids must match).

## The one thing to check first
Target is `MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt`. Takeoff time is given; only the
airport's off-block is blanked. NM's own actual off-block, `AOBT_3_flt`, is a
near-duplicate of it and is **not listed as blanked**. If it survived, the target
is arithmetic, not a model. `probe.py` tests exactly this on training data.

Counter-evidence: the top of the leaderboard is 249s RMSE after 542 submissions.
A live `AOBT_3_flt` would put someone near zero. So expect it to be blanked too,
or to disagree with the airport clock by minutes. Check it, then move on.

## Live leaderboard (pulled 2026-09-04, `python lb.py`)
Per-team best; the API returns every submission, not team bests.

| # | team | RMSE | subs |
|---|------|------|------|
| 1 | enthusiastic-daisy | 249.46 | 542 |
| 2 | quick-boat | 250.52 | 18 |
| 3 | jovial-uniform | 253.18 | 8 |
| 4 | merry-rose | 258.44 | 8 |
| 5 | intelligent-ladder | 259.83 | 4 |
| 10 | hardworking-kiwi | 275.50 | 10 |

37 teams have scored, of 208 registered. Top-3 money needs ~253s or better.

API (leaderboard is broken on the site, Observable migration):
`https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard`
Swagger: `https://datacomp.opensky-network.org/api/swagger-ui/`

## BLOCKER — waiting on bucket keys
Keith submitted the team-creation request on 2026-09-04. Not yet visible in the
teams API: still 208 teams, newest registration 19:15Z, checked at 20:05Z. The
list moves in ~15-minute batches, so a fresh request lagging is expected.

Remaining gates, in order:
1. PRC approves the request by hand, sends a verification email.
2. Reply to it; they then send the MinIO bucket keys.
3. `export PRC_KEY=... PRC_SECRET=...` then `python probe.py`.

Team name is assigned by them (an adjective-animal pair) and arrives in that
email. Record it here when it lands; it is also the submission filename prefix.
