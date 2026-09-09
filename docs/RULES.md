# Challenge rules, quoted (read 2026-09-07)

The repo held no copy of the published rules until now, so a claim about them could not be checked against anything. That is
how the sentence "the rules ban none of it" survived in three files while the ranking page said the opposite. Quotes below are
verbatim from the pages named. Re-read them before any claim about what is allowed; they can change.

The old paths under `ansperformance.eu/study/data-challenge/` now 301 to Netlify, and `ranking.html` under the old path
returns 404, so a stale bookmark reads as "page gone" rather than as a changed rule.

## Ranking and submission (https://prc-data-challenge-2026.netlify.app/ranking.html)

> We will monitor submissions for attempts to learn from or exploit the ranking process. We consider such practices unfair and
> inconsistent with the purposes and goals of the Data Challenge.

Teams are ranked on their best submission by RMSE. The cap is three submissions per day and 1 GB per team bucket. A file must
fill `TAXITIME_SEC_mvt` against the template's `MVT_ID_mvt`, named `<team-name>_v<incremental integer>.parquet`; mismatched
identifiers, missing rows or extra rows are rejected.

No private, final or hidden evaluation set is described anywhere on the page. The board score is the score.

## Prize eligibility (https://prc-data-challenge-2026.netlify.app/eligibility.html)

> All produced **source code** are made openly available on GitHub under the **GNU GPLv3 license**.

> Sufficient **documentation** is provided **to understand and reproduce** the results.

> The solution is **_original_**: Teams must use their own original solutions.

Also required: additional datasets openly available under an open source licence and documented, and no relation to sanctioned
non-EUROCONTROL states.

## What this closes

The board-probe route (`docs/PB.md`, flag F38, gate question 10) reads information out of the scorer. The ranking page names
that practice unfair. It is therefore not an open question about arithmetic, and the campaign does not pursue it. The banked
F38 correction rests on probes P1, P3s and P5s that were already submitted, so it inherits the same objection.
