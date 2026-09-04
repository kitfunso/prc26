# PRC Data Challenge 2026 — state

Site: https://prc-data-challenge-2026.netlify.app/

## Task
Predict taxi-out time (`TAXITIME_SEC_mvt`) for departures at 10 European airports.
Scored on RMSE over Jan + Jul 2026 movements. Deadline **11 Oct 2026 23:59:59 CET**.
Prize: 5000 EUR split across the top 3. Winning entries must be GPLv3 on GitHub.

## Site rules (quoted 2026-09-04)
- index.html: "The applied ranking will use Root Mean Square Error (RMSE) for submitted predictions for the movements in January and July 2026."
- ranking.html: "Teams will be ranked based on the best Root Mean Square Error (RMSE) achieved across their submissions." No separate final set and no submission cap are mentioned.
- eligibility.html: additional datasets must be "openly available under an open source license" and documented; source code goes public on GitHub under GPLv3.

## Data
- Bucket `prc-2026-datasets` on `s3.opensky-network.org` (path-style, https). Keys live in `.env`
  as `PRC_KEY` / `PRC_SECRET` (gitignored; never commit or log them). All 14 files are under `data/`,
  hashes in `docs/EXPERIMENT-PROTOCOL.md` section 2.
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

Checked 2026-09-04: `AOBT_3_flt` survives in ranking, and takeoff minus AOBT_3 scores
428.2 s RMSE on the Jan+Jul 2025 holdout rows that have it. It is a feature, not a
shortcut. The 5,290 ranking departures without it have no flight-plan record at all
(protocol section 1, lane B).

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

## Access keys
The acceptance email carries no key pair; the S3 key and secret are minted by hand in
the MinIO console (https://s3-console.opensky-network.org/access-keys, SSO login, "Create
access key", the secret is shown once). The console host (`s3-console.`) is a different
name from the API host `s3.opensky-network.org`; do not mix them.
