# F18 gate: the promotion floor, the one decision the campaign cannot take for itself

Protocol Stage 6. Flag F18 (`docs/EXPERIMENT-PROTOCOL.md:228`) says the flat `FLOOR = 2.0` in `taxi/verdict.py:8` means
opposite things in the two lanes. SW1 was declared as the instrument that should decide it, and SW1 has now run. This page is
the surface, not a verdict: nothing here promotes anything.

## The question

`SB3 + NN2` beat the champion by +1.7 s [+1.1, +2.3] at a paired SE of 0.3 s and was refused, because 1.7 is under 2
(`docs/NN2.md:27`). Should the floor stand as written?

## The two numbers

| model | holdout RMSE | ranking-table RMSE |
|---|---|---|
| SB3 + L8c, the champions as they stand | 230.10 | 286.12 |
| SB3 + NN2 at its out-of-fold weight 0.4160 | 228.41 | 285.05 |

Regenerate the right-hand column:
`python -c "import numpy as np; f=lambda a: 0.894619*np.sqrt((339551*a**2+5290*1808.1729**2)/344841); print(f(230.0951), f(228.41))"`

**Neither ranking-table number has been scored.** Both are projections from ONE calibration point, S1's holdout 365.24 against its board 325.3982 (`docs/BL.md:5`), carried at the equal ratio 0.8946. `docs/BL.md:37` prices the same file across the plausible family at **281.4 to 288.0**, which spans rank 6 to rank 8 on the 2026-09-07 04:17 UTC snapshot. The absolute position is therefore not known to better than three ranks without an upload.

**What survives that band is the difference.** 286.12 minus 285.05 is 1.07 s, and it is the same ratio applied to two holdouts, so the calibration error cancels out of it: promoting NN2 is worth about one board second wherever the band actually lands. At the equal-ratio point alone both rows sit above merry-rose at 284.7897 (`docs/CEIL.md:29`) and are rank 8, but that is one point of the band, not the answer. The floor question is a question about the method; the rank cannot be settled by any model run, only by a submission.

## What the evidence says

- **Romano-Wolf step-down, 5% family-wise error rate** (`docs/SW1.md`): of the six lane-A challengers whose per-row errors are
  cached, `SB3+NN2` is the only one that survives, at p = 0.0000 and studentised t = +5.26. Every other lane reads p = 1.0000.
  A lane that survives step-down is not a multiple-comparisons artifact, whatever the floor says about it.
- **The floor is not one floor** (`docs/LEV.md:43`): in composite seconds it asks 1.41 s of lane A and 0.176 s of lane B,
  an eight-fold difference from the same constant. On the ranking table itself the two constants are 0.8723 and 0.057521 s
  of board per lane second, so the same 2 s rule buys 1.74 s of board in lane A and 0.12 s in lane B. **Those two rates
  replaced 0.633 and 0.0776 on 2026-09-09 and the replacement widened the gap rather than closing it.** The old pair was
  differentiated out of a slope fitted through a SINGLE scored anchor; `taxi.ceil.per_lane` now measures lane A off the
  controlled v15/v16 pair, which moves lane A alone, and fits the slope through four scored anchors. Lane A's rate rose by
  38% and lane B's fell by 26%, so a lane-A second is now worth 15 times a lane-B second on the board, not 8.
- **The lanes are gated by opposite halves of the rule** (`docs/MDE.md:32`): lane A's median detectable effect is 0.4 s, so the
  floor binds and the interval is slack; lane B's is 32.0 s, so the interval binds and the floor is slack.
- **Cost so far**: F18's own row counts four lane-A challengers with intervals strictly above zero killed by the floor alone,
  about 2.7 s of lane A discarded, before NN2's 1.7 s.

## The decision rule, restated

A challenger is promoted only if the 95% paired day-block interval lies above zero AND the point improvement is at least
`FLOOR` seconds, after an optimism bound of `expected_max(k)` standard errors (`taxi/w1.py:64`). **This verdict is only as good
as this rule. Attack the rule, not just the numbers.**

## The three answers open to you

1. **Leave the floor at 2 s.** SB3 stays. NN2 stays a measured, unpromoted lane. Costs nothing on the board today, because the
   rank does not move either way, and keeps one constant governing 60 ledger rows.
2. **Replace the flat floor with the step-down.** Promote what survives at 5% FWER. That promotes `SB3+NN2` today, and it makes
   the family the judge, which is the instrument Stage 2 actually asks for.
3. **Keep the floor but express it per lane.** 2 s of *board*, not 2 s of lane, which inverts `docs/BL.md:93` to 3.2 s on
   lane A and 25.8 s on lane B. That is stricter than today on lane A, so it promotes nothing now and closes lane B for good.

Changing the rule is a protocol rule change and needs an explicit yes, per protocol section 0. No option above has been taken.

## Why this is not urgent for the score

The lane that moves the rank is NN3, not the floor. At NN2's measured error agreement of 0.9403 a partner at 233.0 s blends to
227.91 holdout and 284.74 at the equal-ratio point, which passes merry-rose there and carries the same three-rank band as every other projection on this page (`docs/BF.md`, re-derived from `taxi.bf.best_blend`). That is
1.2 s of net away and it is running.
