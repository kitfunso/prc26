# prc26

Entry for the [PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/):
predict taxi-out time (`TAXITIME_SEC_mvt`) for departures at 10 major European
airports, scored on RMSE over a fixed ranking set of 344,841 rows from January and
July 2026.

Team `merry-octopus`. Deadline 11 Oct 2026. Licensed GPLv3, as the challenge
requires of prize-eligible entries.

## What the model is

Two lanes, split on whether a departure has a matched flight plan
(`AOBT_3_flt` present). They share no model and no verdict.

| lane | rule | ranking rows | model |
|---|---|---|---|
| A | flight plan present | 339,551 | `SB3`, a five-seed CatBoost bag on the D2 column set, predictions floored at 0 s |
| B | no flight plan | 5,290 | `L8c`, a per-airport gated mixture over the schedule-copy artifact |

Lane A predicts the residual `c = y - z_aobt` (taxi time minus the takeoff-minus-AOBT
gap) and adds `z_aobt` back, so the model never sees the raw label. Lane B has no
such anchor: it is a per-airport least-squares fit on `z_sched` and a queue count,
with a CatBoost classifier for the rows whose recorded block time is a copy of the
scheduled time. Where that classifier fires, the prediction is
`P(copy) * z_sched + (1 - P(copy)) * the band's non-copy mean`, which is the
squared-error optimal point prediction under that mixture.

The current champion pair is `taxi/champ.py:CURRENT`. Every promotion that put a name
there has a report in `docs/` and a row in `docs/TRIAL-LEDGER.md`.

## Reproducing a submission

Python 3.12. A CUDA GPU is needed for the lane-A fit (RTX 5080 here, about 40 min).

    pip install -r requirements.txt

    cp .env.example .env          # MinIO keys from the OpenSky console
    set -a; . ./.env; set +a

Put the challenge files in `data/`: the twelve
`training_2025-MM-01_2025-MM-01.parquet` months (4,167,797 movements),
`ranking.parquet` and `submitting.parquet`. `data/` is not committed. `probe.py`
lists the buckets and dumps the schema if you need to find them.

Fetch the weather, once:

    python -m taxi.weather fetch     # writes data/weather/<date>/ with a sha256 manifest

Build the file:

    python -m taxi.s2 data/merry-octopus_v16.parquet SB3 0

The three arguments are the output path, the lane-A champion name and the floor in
seconds. Lane B always comes from `taxi.champ.CURRENT`. The command prints, for each
lane, the training and ranking row counts and the prediction quantiles, then the row
count and sha256 of the file it wrote. Upload is a separate step and is never
automatic:

    python -m taxi.upload data/merry-octopus_v16.parquet
    python -m taxi.upload status      # the scorer writes <key>_result.json beside the file

    pytest                            # 39 tests, about 20 s, no GPU

## Reproducing a verdict

Nothing is promoted on a leaderboard score. Every model is judged on a development
holdout of January and July 2025, fit on the other ten months, so the holdout has the
same month shape as the ranking set. Verdicts are a paired day-block bootstrap,
2,000 replications, seed 0, against the standing champion, and a challenger wins only
if the 95% interval lies above zero and the point improvement is at least 2 s.

    python -m taxi.lanes W1          # one declared lane, writes its own docs/*.md
    python -m taxi.sb3               # the lane-A champion's own run
    python -m taxi.l8                # the lane-B champion's own run

Each writes its report to `docs/` and prints the verdict line it recorded.

## Where the method is written down

- `docs/EXPERIMENT-PROTOCOL.md` is the contract. Section 4 declares every search
  before it runs; section 3 holds the decision rule; the flag list carries what each
  finding closed. An undeclared run is a diagnostic and can never decide a verdict.
- `docs/TRIAL-LEDGER.md` is every paired verdict in order, with its interval.
- `docs/COVERAGE.md` answers what was tried, what was not, and why.
- `NOTES.md` holds the challenge rules, the data layout and the live leaderboard.
- `docs/gate-s*.html` are the promotion gates: what was known at each decision point.

## Data and attribution

Challenge data from the PRC Data Challenge 2026 organisers, not redistributed here.

Weather features (lane W1) come from the ASOS/METAR archive of the Iowa Environmental
Mesonet, Iowa State University (public domain; attribution appreciated per
https://mesonet.agron.iastate.edu/disclaimer.php), pulled 2026-09-05 with
`python -m taxi.weather fetch` into `data/weather/2026-09-05/`; the per-file sha256 in
its `manifest.json` pins the pull.

Public holiday flags come from the `holidays` package.
