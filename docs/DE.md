# DE: the block time recorded one day early (diagnostic, never a verdict)

Regenerate with `python -m taxi.de`. Lane-B champion L8c, floored, holdout Jan + Jul 2025. Nothing here decides a verdict and no cell is declared from it: every answer it prices either loses or lands under the 2 s floor.

## 1. The artifact, exactly

A lane-B row whose taxi time sits within an hour of a full day is not a long taxi. Its recorded block time carries the right time of day and the wrong date. Subtract the day and what is left is an ordinary taxi time (the lane's own median is 900 s):

| set | y minus 86,400, one row each |
|---|---|
| lane B train | 768, 847, 941, 1,080, 1,143, 1,198, 1,205, 1,992 |
| lane B holdout | -2,160, 770, 777, 786, 961, 1,732 |

That is the whole mechanism. It is distinct from the schedule-copy artifact the L4 to L8 chain models, where the block time field was filled with the scheduled time and y equals z_sched. Both produce a taxi time of many hours; only the copy family is reachable from the features, because a copied row's z_sched IS its answer.

## 2. How much error it carries

| set | rows | normal (under 1 h) | stale, 1 to 12 h | over 12 h | of those, day-shifted |
|---|---|---|---|---|---|
| lane B train | 17,097 | 16,216 | 861 | 20 | 8 |
| lane B holdout | 5,373 | 5,167 | 193 | 13 | 6 |

Where the champion's holdout squared error sits, by the same bands:

| band | rows | share of lane-B squared error | mean prediction | mean truth |
|---|---|---|---|---|
| normal | 5,117 | 20.6% | 1,127 s | 1,011 s |
| stale, 1 to 12 h | 193 | 12.1% | 6,577 s | 9,188 s |
| over 12 h | 13 | 67.2% | 53,295 s | 67,841 s |

### The same error, banded on what the model can actually see

Bands on y are a post-mortem. A band on the model's own prediction is a lane: a per-band constant needs no new feature and no new fit, so if one of these bands held a systematic offset it would be free to collect.

| band, on the prediction | rows | share of lane-B squared error | mean signed error | bias share of the band |
|---|---|---|---|---|
| 0 to 900 s | 1,385 | 3.3% | -63 s | 1.0% |
| 900 to 1,800 s | 3,621 | 65.6% | -54 s | 0.1% |
| 1,800 to 3,600 s | 156 | 4.7% | +372 s | 2.6% |
| 3,600 to 7,200 s | 123 | 6.3% | +1,262 s | 17.8% |
| 7,200 to 43,200 s | 77 | 10.9% | +635 s | 1.6% |
| over 43,200 s | 11 | 9.3% | -4,442 s | 13.3% |

Two bands are not flat. Both are tiny and neither is a lane, and the way to see that is to grant the whole table its own constants: subtract each band's OWN holdout mean error, fitted on the rows it is scored on. That is an oracle, not a model, so it can only be an upper bound. It takes lane B from 1,808.2 s to 1,783.2 s, +24.9 [-11.5, +67.1] against a promotion floor of 30 s. A real per-band constant, fitted on the training months, would have to beat that ceiling from below.

And the concentration, by dropping the worst rows one at a time. Nothing is predicted here; the rows are removed from the score to show what carries it.

| lane-B holdout | rows | RMSE | share of the lane's squared error dropped | of the dropped rows, over 12 h |
|---|---|---|---|---|
| all rows | 5,373 | 1,808.2 | | |
| worst 1 dropped | 5,372 | 1,408.5 | 39.3% | 1 of 1 |
| worst 5 dropped | 5,368 | 1,034.0 | 67.3% | 4 of 5 |
| worst 10 dropped | 5,363 | 979.7 | 70.7% | 6 of 10 |
| worst 50 dropped | 5,323 | 770.4 | 82.0% | 6 of 50 |

The lane is not a population, it is a handful of rows. One row of 5,373 carries 39.3% of the squared error (LFPG, prediction 1,115 s against a truth of 84,240 s), the worst five carry 67.3%, and 6 of the worst ten are the over-12-hour rows section 2 tabulates one by one. So the concentration is not a new structure to search: it IS the artifact family, which section 4 prices and closes. Below it, the remaining 5,360 rows score 1,037.2 s and the band table above says their error is spread, not offset.


Of 13 holdout rows over 12 h, 6 are schedule copies and 6 are day-shifted. Every one of them, with the champion on it:

| airport | day | truth | z_sched | prediction | family |
|---|---|---|---|---|---|
| LIRF | 2025-07-21T00:00:00.000000 | 88,132 | 84,236 | 87,484 | day-shifted |
| LIRF | 2025-07-03T00:00:00.000000 | 87,361 | 73,260 | 75,763 | day-shifted |
| LIRF | 2025-07-30T00:00:00.000000 | 87,186 | 58,563 | 60,069 | day-shifted |
| LIRF | 2025-01-25T00:00:00.000000 | 87,177 | 60,118 | 61,729 | day-shifted |
| LIRF | 2025-07-14T00:00:00.000000 | 87,170 | 93,535 | 97,415 | day-shifted |
| LFPG | 2025-01-19T00:00:00.000000 | 84,240 | 1,740 | 1,115 | day-shifted |
| LIRF | 2025-07-14T00:00:00.000000 | 64,437 | 64,438 | 66,342 | schedule copy |
| LFPG | 2025-01-23T00:00:00.000000 | 58,206 | 2,043 | 1,096 | neither |
| LIRF | 2025-01-25T00:00:00.000000 | 56,633 | 56,636 | 58,011 | schedule copy |
| LIRF | 2025-07-07T00:00:00.000000 | 46,369 | 46,374 | 47,052 | schedule copy |
| LIRF | 2025-07-14T00:00:00.000000 | 45,791 | 45,785 | 46,423 | schedule copy |
| LIRF | 2025-07-31T00:00:00.000000 | 45,368 | 45,364 | 45,974 | schedule copy |
| LIRF | 2025-07-30T00:00:00.000000 | 43,863 | 43,861 | 44,369 | schedule copy |

The two LFPG rows F17 names as holding 28.4% of the whole composite's squared error are in that table, and they are not the same thing as each other: one is day-shifted and one belongs to neither family.

## 3. Where it lives, and why the 2026 arrivals cannot help

The ranking file blanks the departure taxi time and keeps the arrival one, so an arrival is the only place a 2026 artifact rate could be read. It is not there.

| set | rows | over an hour | day-shifted |
|---|---|---|---|
| 2025 arrivals, all 12 months | 2,082,749 | 440 | 2 |
| 2026 arrivals, Jan + Jul (ranking) | 344,693 | 145 | 0 |
| 2025 lane-B departures, all 12 months | 22,470 | | 14 |

Two day-shifted arrivals in 2.08 M rows of 2025 and none at all in 2026, against fourteen in 22,470 lane-B departures. The artifact belongs to off-block recording on movements with no flight plan, so the arrivals carry no estimate of its 2026 rate.

## 4. The ceiling

Predicting 87,300 s on exactly the 6 day-shifted holdout rows and changing nothing else takes lane B from 1,808.2 s to 1,297.4 s, **+510.7 s** [+16.2, +1,244.9]. That is about 45 s of composite, larger than the whole distance from our S1 board score to the leader. Six rows of 344,419. It is a ceiling and not a lane: the label is not visible in 2026.

The per-row table says the ceiling is narrower still, and says why. On five of the six the SCHEDULE was shifted with the block time, so z_sched already carries the day and the champion lands within a few thousand seconds unaided. On one the schedule stayed normal and only the block time moved. Splitting the oracle along that line:

| oracle applied to | rows | lane B RMSE | paired against the champion [95%] |
|---|---|---|---|
| the schedule moved too (z_sched over 40,000 s) | 5 | 1,722.7 | +85.5 [+6.0, +224.7] |
| only the block time moved | 1 | 1,409.0 | +399.2 [+0.0, +1121.9] |

Most of the ceiling is one row, but not all of it, and the remainder is the part that matters. The single LFPG row is unreachable: its schedule is an ordinary 1,740 s and nothing marks it. The five LIRF rows are worth +85.5 s between them and they are NOT unreachable, because the thing that identifies them, a schedule already past 40,000 s, is a ranking-visible column. The champion under-predicts three of them badly (75,763 against 87,361, 60,069 against 87,186, 61,729 against 87,177) because it follows z_sched instead of the day. That looked like a lane. It is not, and the population says why:

| lane-B rows with z_sched over 40,000 s | rows | schedule copies | day-shifted | ordinary flights |
|---|---|---|---|---|
| train | 61 | 18 | 6 | 38 |
| holdout | 25 | 8 | 5 | 12 |

A schedule already many hours out is the only thing these rows share, and it is mostly a stale schedule on an ordinary flight. Of the 25 such holdout rows, 12 have perfectly normal taxi times (681, 865, 888, 1,093, 1,167, 1,183 s and so on). Any rule keyed on that signature pays about 86,000 s of error on each of those to win about 25,000 s on each of five. The champion already gets them right, which is what its classifier is for. No cell is declared: the lane was killed by its own population before it cost a verdict.

## 5. Every blind answer, priced

### 5a. A uniform shift at the training rate

The RMSE-optimal response to an artifact of magnitude A on a row with probability p is to add p x A. The training rate is 8 of 17,097 rows, 0.0468%, so the shift is 40.4 s on every lane-B prediction.

| shift | lane B RMSE | paired against the champion [95%] |
|---|---|---|
| 40.4 s | 1,808.3 | -0.2 [-2.0, +1.1] |
| 80.9 s | 1,809.4 | -1.2 [-5.4, +1.5] |
| 100.0 s | 1,810.2 | -2.1 [-7.5, +1.4] |

The best of these lands under the 2 s floor, and the arithmetic says why it cannot do better. The gain from an optimal shift is n p squared A squared, so at a rate near one in two thousand the whole lane is worth a couple of seconds, and every row that is not an artifact pays the shift squared.

### 5b. Splitting the champion's own high predictions between the two artifact values

The one structured answer available. Where the champion already predicts over 40,000 s the copy mixture has fired, so the row is one artifact or the other; predict q x z_sched + (1 - q) x (86,400 + a normal taxi), with q the copy share of the fit months' big rows. Five month-pair folds:

| fold | eval rows | predictions over 40,000 s | fit copies | fit day-shifts | q | champion | mixed | gain |
|---|---|---|---|---|---|---|---|---|
| (2, 3) | 2,192 | 2 | 12 | 7 | 0.632 | 1626.0 | 1606.9 | +19.1 |
| (4, 5) | 2,690 | 3 | 12 | 7 | 0.632 | 1919.5 | 1992.3 | -72.9 |
| (6, 8) | 4,832 | 9 | 7 | 6 | 0.538 | 1600.1 | 1772.7 | -172.6 |
| (9, 10) | 4,378 | 6 | 10 | 7 | 0.588 | 1055.8 | 1098.1 | -42.3 |
| (11, 12) | 3,005 | 4 | 11 | 5 | 0.688 | 1111.3 | 992.0 | +119.3 |
| pooled | 17,097 | | | | | 1464.5 | 1524.9 | -60.4 |

It loses on CV and it loses on the holdout: at q = 0.62 the same recipe scores 1,873.1 s against 1,808.2, -64.9 [-173.1, +10.0]. The reason is in section 1: a day-shifted row need not have a strange schedule at all. Of the 14 holdout rows the champion pushes over 40,000 s, 5 are day-shifted, so the split pays out on the wrong rows: 9 of them are not day-shifted at all and take the pull toward 86,400 for nothing, while the 1 day-shifted row the champion leaves low is never reached, its z_sched being an ordinary 1,740 s.

## 6. What this closes

The largest single gain in the campaign is real, measured, and out of reach by every route tested. Knowing which six rows carry it is worth more than the distance from our board score to the leader. Nothing reaches them: the one row that holds most of the gain has an ordinary schedule and no mark on it; the five that are marked share their mark with twice as many ordinary flights and with the schedule copies, so a rule keyed on it loses more than it wins; the 2026 arrivals carry no day-shifted rows at all and cannot price the rate; a uniform shift at the measured rate lands under the 2 s floor by the n p squared A squared arithmetic; and the one structured split loses three of five folds, pools negative and loses on the holdout. Section 2 closes the last blind route with it: banded on the model's OWN prediction, which is the only banding a model can act on, the lane holds no offset worth collecting, and an oracle that fits a constant per band on the very rows it is scored on lands under the lane-B promotion floor with an interval through zero. Flags F21 and F43 carry this so no later report re-opens it as an opportunity, and so that a board gap of a few tens of seconds is not read as proof that a better recipe exists.
