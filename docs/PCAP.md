# PCAP: an upper cap on the prediction, out of fold (diagnostic, never a verdict)

Regenerate with `python -m taxi.pcap`. No cell is declared and nothing here can decide a verdict. C1 measured the FLOOR at 0 s and promoted it; nothing had ever measured the cap, and `docs/LEV.md` section 3 shows the composite is carried by a handful of extreme rows, so a one-parameter clip is exactly the shape of knob that would pay there if one paid. Lane A is MC2's cached out-of-fold prediction on the ten training months, floored; lane B is the champion L8c refit per fold by `taxi.lbcal.oof`. Choosing on out-of-fold rows and not on the holdout is the point: a cap read off the verdict set would be free by construction.

## Lane A (1,723,531 rows)

Prediction max 14,965 s, p99.9 3,013; truth max 21,308 s, p99.9 3,842.

| cap (s) | rows capped | OOF RMSE (s) | gain (s) |
|---|---|---|---|
| none | 0 | 212.65 | +0.00 |
| 3,000 | 1,758 | 218.19 | -5.54 |
| 5,000 | 202 | 213.94 | -1.29 |
| 8,000 | 37 | 212.83 | -0.18 |
| 15,000 | 0 | 212.65 | +0.00 |
| 30,000 | 0 | 212.65 | +0.00 |

## Lane B (17,097 rows)

Prediction max 137,707 s, p99.9 45,558; truth max 131,167 s, p99.9 46,541.

| cap (s) | rows capped | OOF RMSE (s) | gain (s) |
|---|---|---|---|
| none | 0 | 1464.52 | +0.00 |
| 2,000 | 998 | 3105.72 | -1641.20 |
| 3,000 | 751 | 3016.59 | -1552.07 |
| 5,000 | 504 | 2868.15 | -1403.62 |
| 8,000 | 223 | 2706.21 | -1241.69 |
| 12,000 | 78 | 2546.65 | -1082.13 |
| 20,000 | 42 | 2286.94 | -822.42 |
| 30,000 | 34 | 2021.39 | -556.87 |
| 45,000 | 18 | 1746.01 | -281.49 |

## Reading

The cap is harmful at every level tried, on both lanes, and monotonically: the tighter the cap the worse the lane. It is not a knob with a bad setting, it is a knob with no live side. The reason is in the two summary lines above. Lane B's truth reaches 131,167 s and its p99.9 is 46,541 s against a predicted p99.9 of 45,558, so the model's extreme predictions are landing on rows whose answers really are that large; clipping them turns a row that was roughly right into a row that is wrong by tens of thousands of seconds. Lane A never predicts above 15,000 s at all, so a cap there is either inert or destructive with nothing in between.

What this closes: the clip family, both directions. The floor is measured and kept (C1), the cap is measured and dead, and neither needs asking again. What it does not license: any claim about the extreme rows themselves, which `docs/DE.md` and flag F21 price at zero by their own separate argument.
