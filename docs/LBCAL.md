# LBCAL: is the lane-B prediction calibrated in magnitude? (diagnostic, never a verdict)

Regenerate with `python -m taxi.lbcal`. No cell declared, no champion moves. Champion L8c, floored, out of fold over the five month-pair folds on the 17,097 lane-B training rows.

## 1. Why it was asked, and the misreading that asked it

`docs/DE.md` section 2 reports a band of 193 holdout rows whose truth sits between 1 and 12 hours, carrying 12.1% of
lane-B squared error, with a mean prediction of 6,577 s against a mean truth of 9,188 s. Read as a 2,611 s bias on rows
the model has already flagged, that is a magnitude error rather than an identification error, and magnitude errors are
usually free to fix.

The reading is wrong, and the error is worth naming because it is easy to repeat. That band is defined by the TRUTH.
Conditioning on the truth makes every honest predictor look biased toward the mean, because the rows selected into a
high-truth band are exactly the rows where the noise ran high. The quantity that matters is the other conditional: given
what the model PREDICTS, which is the only thing visible in 2026, what is the mean truth?

## 2. The prediction-conditioned answer

| predicted band (s) | rows | mean prediction | mean truth | bias | share of squared error |
|---|---|---|---|---|---|
| 0 to 600 | 2,399 | 467.3 | 464.4 | -2.9 | 21.5% |
| 600 to 900 | 2,883 | 723.7 | 769.4 | +45.7 | 2.2% |
| 900 to 1,200 | 8,133 | 1,062.2 | 1,141.1 | +78.9 | 13.5% |
| 1,200 to 1,800 | 2,590 | 1,386.5 | 1,345.2 | -41.3 | 6.9% |
| 1,800 to 3,000 | 341 | 2,279.3 | 1,719.9 | -559.4 | 3.4% |
| 3,000 to 6,000 | 351 | 4,415.2 | 4,069.8 | -345.4 | 9.9% |
| 6,000 to 12,000 | 322 | 8,066.8 | 8,267.6 | +200.8 | 26.3% |
| 12,000 to 25,000 | 42 | 15,659.6 | 14,791.7 | -867.9 | 5.9% |
| 25,000 to 45,000 | 18 | 36,968.4 | 36,451.0 | -517.4 | 0.1% |
| over 45,000 | 18 | 62,012.9 | 68,245.0 | +6,232.1 | 10.4% |

The biases alternate in sign and none of the populated bins carries one worth having. The 2,611 s gap of section 1 does
not appear anywhere.

## 3. The ceiling, and why it closes the family

Subtracting every bin's own bias, fitted in sample on the same rows it is then scored on, is an oracle recalibration and
an upper bound on any honest version. It removes 958,267,625 of 36,670,065,441 squared seconds, 2.6%, and takes
out-of-fold lane-B RMSE from **1,464.5 s to 1,445.3 s: +19.3 s**.

At the lane-B carry of `docs/BL.md` section 5 that is about **1.5 s of board**, for a correction
fitted and judged on the same rows. An honestly cross-fitted version is worth less. The floor is 2 s of lane B and this
is a ceiling, so the size is not the objection; the objection is that the ceiling is an oracle and that
10.4% of the squared error, the largest single contributor in the table, sits in the 18 rows above
45,000 s, which are the day-shift artifact that flag F21 already prices at zero through every route tested.

## 4. What this closes

The lane-B prediction is calibrated in magnitude. Do not re-open a recalibration, a per-bin correction, an isotonic fit
or a variance model on top of it, and do not read a truth-conditioned band table as evidence of bias. `docs/LB2.md`
section 3 closed recalibration of the copy PROBABILITY; this closes recalibration of the predicted VALUE, which is the
other half and was open.
