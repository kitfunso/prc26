# TAIL: what carries the holdout composite

Diagnostic, never a verdict (protocol section 2). Regenerate with `python -m taxi.tail`. Current champions, both lanes floored at 0 s:
lane A MC2 from the cached holdout refit, lane B L8c. 344,419 development-holdout rows (Jan and Jul 2025), composite 322.5796 s.

## Levels and their day-block bootstrap spread

`taxi.verdict.level_ci`, 2,000 reps, seed 0, whole days resampled. These are LEVELS, not paired deltas.

| model | RMSE | bootstrap SD | 95% range | rows |
|---|---|---|---|---|
| lane A MC2 | 232.15 | 4.47 | 223.46 to 240.81 | 339,046 |
| lane B L8c | 1808.17 | 400.57 | 1086.10 to 2621.83 | 5,373 |
| composite | 322.58 | 33.04 | 266.16 to 389.96 | 344,419 |

Read this correctly. The 33 s is how far the composite moves when the test set is redrawn, not doubt about a score on a fixed set.
The board scores one fixed set of rows for every team, so a difference between two teams on it is a real difference on those rows.
What the spread does say is where that difference comes from: a handful of rows, not the other 344,000.

## Squared error by error size

| absolute error | rows | share of squared error |
|---|---|---|
| under 1,000 s | 341,843 | 37.2% |
| 1,000 to 3,000 s | 2,365 | 15.8% |
| 3,000 to 10,000 s | 202 | 12.7% |
| 10,000 to 30,000 s | 7 | 6.0% |
| over 30,000 s | 2 | 28.4% |

Nine rows of 344,419 carry 34.4% of the squared error, and two of them carry 28.4%.

## The nine rows

| lane | airport | month | truth | prediction | z_sched | no type |
|---|---|---|---|---|---|---|
| B | LFPG | 1 | 84,240 | 1,115 | 1,740 | no |
| B | LFPG | 1 | 58,206 | 1,096 | 2,043 | no |
| B | LIRF | 7 | 87,186 | 60,069 | 58,563 | yes |
| B | LIRF | 1 | 87,177 | 61,729 | 60,118 | yes |
| B | LIRF | 7 | 1,377 | 17,926 | 20,878 | yes |
| B | LIRF | 7 | 959 | 12,793 | 14,339 | yes |
| B | LIRF | 7 | 87,361 | 75,763 | 73,260 | yes |
| B | EHAM | 1 | 11,032 | 412 | -364 | no |
| B | LIRF | 7 | 87,170 | 97,415 | 93,535 | yes |

Every one is lane B, and they split into two kinds. The LIRF rows carry a day-scale schedule and the copy branch already predicts
them within tens of per cent, in both directions: five are day-scale truths predicted day-scale, two are normal truths the copy
branch over-predicted. The two LFPG rows are the ones nothing sees. Their schedule is an ordinary 29 and 34 minutes, they carry an
aircraft type, and the truth is a day. No ranking-visible column separates them from the 878 other LFPG lane-B holdout rows.

## The far band beats every constant mixture, on 22 rows

22 holdout lane-B rows have a schedule over 12 h. Their training copy share is 0.226 on 53 rows. The current per-row P(copy) scores
8,652 s there; the best constant mixture weight on the same rows, chosen with hindsight on the holdout itself, scores 35,355 s.
The gated mixture is four times better than the best single weight an oracle could pick. That says the constant-weight family is
badly specified, not that the band holds no gain: 22 rows with a training copy share estimated on 53 (binomial SE 0.057) cannot
settle the question. Untested at this n.

## Oracle bound on the extreme rows

Perfect prediction on the k worst lane-B rows, everything else unchanged:

| k | lane B | composite |
|---|---|---|
| 0 | 1808.2 | 322.5796 |
| 1 | 1408.4 | 289.8196 |
| 2 | 1173.2 | 272.9938 |
| 5 | 1033.5 | 264.0362 |
| 9 | 988.2 | 261.3154 |
| 20 | 897.0 | 256.1349 |

## What this closes, and what it does not

Two rows perfectly predicted is worth 49.6 s of composite. The distance from S1's board score to the leader's is 48.1 s
(325.3982 against 277.29). Those two numbers are the same size, and the 49.6 s is bought by nothing a model can learn.

A rate-based shift is NOT closed here. An earlier draft of this section argued it was, citing `docs/APT.md`. That citation was
wrong and is withdrawn: APT measures per-key recalibration of MC2 on 1,723,531 **lane-A** out-of-fold rows, and the two rows in
question are **lane-B** rows that MC2 never predicts (`taxi/champ.py` routes lane B to B4, L4, L6 and L8 only). The argument also
fails on its own terms, because the copy branch's fit rows do not contain a day-scale LFPG truth at all (F3: the training months
hold no LFPG lane-B row above 9,182 s), so no leaf can carry a term learned from them.

What can honestly be said is weaker. An unconditional rate term for a 2-in-880 day-scale event at LFPG is worth roughly
2 x 71,000 / 880 = 161 s of uplift on every LFPG lane-B row, which would wreck the 878 ordinary ones; nothing measured here tells
us whether a conditional version exists, because no ranking-visible column separates the two rows (LK1). Treat this as untested at
this power, not as closed.

What is left for the model is the 65.7% of squared error under 3,000 s, spread over 344,208 rows. That is lane A's broad middle,
which is what the queued lanes work on.
