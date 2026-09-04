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

## Team
`merry-octopus`, accepted 2026-09-04. Submission bucket `prc-2026-merry-octopus`.
Submissions go in as `merry-octopus_v<N>.parquet`.

## BLOCKER: access keys not generated yet
Unlike 2024, the acceptance email carries no key pair. It grants the OSN account
access; the S3 key and secret are minted by hand in the MinIO console:

1. https://s3-console.opensky-network.org/access-keys
2. Login: "Other Authentication Methods" then "Login with SSO", then OpenSky credentials.
3. Left nav "Access Keys", button "Create access key". Copy both values once, the
   secret is shown a single time.
4. Put them in `.env` as `PRC_KEY` / `PRC_SECRET`, then `python probe.py`.

Endpoint for clients is `s3.opensky-network.org`, path-style, TLS. The console
host (`s3-console.`) is a different name from the API host; do not mix them.

Data bucket name is not stated on the 2026 site. It was `competition-data` in
2024 and `probe.py` assumes the same. First thing to confirm once keys exist:

    SELECT * FROM glob('s3://competition-data/*');
