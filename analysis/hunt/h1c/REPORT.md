# H1c: the public record (web only)

Protocol v6 section 4. A fresh Sonnet agent, web only, 2026-09-23. It returned its report as text; the session filed
it here, condensed, with the verdict below. Discord was not accessed: every Discord quote is second-hand, from team
kind-mango's public notes.

## Findings, as ranked by the agent

1. Organiser: BLOCK==SCHED fallback rows are in the ranking set: "the ranking comes from whatever we got from the
   relevant airports for Jan/Jul 2026" (kind-mango `RECAP.md`, entry 2026-09-09).
2. Organiser: "The decision on how to deal with strange/noisy/messy data is a full part of the challenge and a
   modelling decision" (same entry).
3. Organiser: 630 "neither" rows where BLOCK_TIME precedes AOBT by about 50 min at EGLL, LFPG, LIRF and EHAM, with no
   known recurring cause (same entry).
4. kind-mango's ITY340 rule: LIRF, any flight record, `sd > 70,000`, predict `(5/6) * (86,400 + 1,150) + (1/6) * sd`.
   It fires on one rank row (`sd` 94,560) and is worth about 18 s in expectation (`README.md:77`, `README.md:214`,
   `RECAP.md:257`).
5. One 24-h row served at a normal taxi costs about +34 s on this board (`docs/MODEL_ANALYSIS.md` section 0.3).
6. kind-mango v48 capped five non-LIRF extreme predictions (EDDM 21,465 to 7,499): -2.33 s live. v51 retrained the
   base on served rows only (LIRF and y > 80,000 excluded): -5.99 s live (`RECAP.md`).
7. `AOBT_3_flt` is on 98.9% of rank rows. kind-mango leaves it out on ethics grounds, citing a "brief section 7.2"
   the agent could not find on the public site. The organiser, in the same notes: "Any provided value is fair to use
   as a proxy (or as itself) if it helps" (`RECAP.md:357`).
8. Drift: seven same-day LIRF columns in kind-mango's own feature set go from full coverage in January 2026 to none in
   July 2026; the null-flight-record share roughly doubles in 2026; EOBT_1 differs from the schedule on 35.7% of LFPG
   rows in 2025 and 61.6% in July 2026 (`docs/MODEL_ANALYSIS.md` D1, D3, D4).
9. Rules: outside data is allowed if declared (METAR, OPDI, adsb.lol with attribution); Trino-derived features
   disqualify; reusing code needs the authors' rights and significant changes.
10. The 2024 and 2025 PRC challenges left nothing public on extreme-label handling.

Sources: https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango (`README.md`, `RECAP.md`,
`docs/MODEL_ANALYSIS.md`); https://prc-data-challenge-2026.netlify.app/ (data, eligibility);
https://github.com/Phoenix-Ops-LTD/prc2026-taxiout; https://github.com/satam2/knowledgeable-helicopter.

## Session verdict: no candidate; every lever H1c found is already in v34

| lever | check | result |
|---|---|---|
| ITY340 day shift | `PYTHONPATH=. python analysis/hunt/h1c/big_sd.py`: rows with `z_sched` > 70,000 | The one lane A LIRF rank row (id 213147416, `z_sched` 94,560) has `z_aobt` 86,340, so the residual-to-AOBT anchor already carries the day: v34 predicts 82,117. The two lane B LIRF rank rows are hull rows, predicted 87,495 and 94,401. The six 2025 LIRF rows of this kind are all train months: labels 87,002 to 131,167, out-of-fold predictions 78,773 to 132,872. |
| cap wild predictions | `PYTHONPATH=. python analysis/hunt/h1c/wild_preds.py`: rank against holdout prediction tails per airport | The largest non-LIRF rank prediction is 12,088 (LFPG), under LFPG's holdout maximum of 13,987. 62 rank rows sit above their airport's holdout maximum, mildly: an EHAM January 2026 cluster with `z_aobt` near 5,400 s, and an LTFM January 2026 cluster. Under 1 s in total at most. No cap. |
| base trained without giant labels | `lanea.py:1`, `nn.py:106` | The lane A trees fit rows with \|y - z_aobt\| > 600 in a separate tail model, and the MLP trains on a Huber loss, so giant labels never reach the body model. |
| `AOBT_3_flt` ethics | `RECAP.md:357` (organiser, second-hand) | Our use stands: a provided field used as a feature, with no scorer reads. |

Leads passed on, not judged here: the 630 "BLOCK about 50 min before AOBT" rows (H1a's ground) and the 2026 drift in
stamp shares (H1b's ground).

No candidate was scored on the holdout, so N is unchanged. `wild_preds.py` reads the holdout's label tails per airport
as a description only.
