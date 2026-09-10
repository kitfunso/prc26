# MS1: where in the year each lane's holdout gain sits, and how much of it reached the board

Regenerate with `python -m taxi.ms1`. Declared in `docs/EXPERIMENT-PROTOCOL.md` section 4 before it ran. k = 0 and N_h does not move: every lane below has already met the holdout under its own ledger row, and this page decomposes numbers that are already read. It promotes nothing and cannot decide anything. Rows 339,046, months [1, 7].

## The 4 pairs whose board carry is MEASURED

The gain column is holdout lane-A RMSE, base minus challenger. The carry is that pair's board delta divided by its holdout delta, from two files whose lane B is byte-identical. The month columns give the RMSE gain inside the month and, in brackets, that month's share of the pair's TOTAL squared-error reduction, which is the part that actually adds up across months.

| pair | holdout gain | board carry | month 1 | month 7 |
|---|---|---|---|---|
| v15/v16, MC2 to SB3 | +2.0551 | 0.8723 | +1.811 (35%) | +2.234 (65%) |
| v17/v18, SB3+RC1 to SB3+NN3 | +2.1328 | 0.5895 | +1.780 (33%) | +2.385 (67%) |
| v17/v19, SB3+RC1 to SB3+NN2 | +1.7437 | 0.4279 | +0.458 (11%) | +2.620 (89%) |
| v17/v20, SB3+RC1 to SB3+LT2 | +0.9949 | 0.9582 | +1.061 (43%) | +0.959 (57%) |

## Verdict on the hypothesis

**Not falsified. The ordering is perfect and the magnitudes are not.** Sorted by January's share of the squared-error reduction, the 4 pairs run 11% carrying 0.43, 33% carrying 0.59, 35% carrying 0.87, 43% carrying 0.96. Rank correlation is +1.00, and under the exact permutation null over all 4 factorial orderings that is p = 0.042. Every lane that put more of its gain in January kept more of it on the board, with no exception, and the falsifier written into the declaration did NOT fire.

Say the other half as loudly. The relationship is an ORDER, not a size: **v15/v16, MC2 to SB3 and v17/v18, SB3+RC1 to SB3+NN3 split their gain almost identically, 35% against 33% in January, and carry at 0.87 against 0.59**, a factor of 1.48 on a 2-point difference in share. So month share is not itself the mechanism; it is tracking something the mechanism also tracks. And 4 points, against a hypothesis written after the inversion was seen, is one lane's worth of evidence: p = 0.042 is the chance of this ordering alone, not of the hypothesis being right.

What it is worth today is a tie-break, not a model. When two candidate files are close on the holdout and only one slot is free, this page says prefer the one whose gain is spread across the holdout year rather than concentrated in July, and it says so with four measured points behind it. What would actually test it is the next controlled pair: predict its carry from its January share BEFORE the upload, and write the prediction into the protocol the way every other lane's is.

## What is missing from this page

Every controlled pair the campaign has scored is read above: 4 of 4.

