# CB1: the lane-B chain judged end to end (holdout Jan+Jul 2025, floored)

Regenerate with `python -m taxi.cb1`. Train months (2, 3, 4, 5, 6, 8, 9, 10, 11, 12), 5373 lane-B holdout rows, every prediction floored at 0 s (H22). Paired day-block bootstrap, 2,000 reps, seed 0.

## 1. The whole chain in one verdict (H45)

Each promotion priced its own cell count; nothing priced the chain. This is the contrast the chain claims, judged once.

| champion | challenger | RMSE champion | RMSE challenger | improvement [95%] |
|---|---|---|---|---|
| B4 floored | L8c floored | 1965.6 | 1808.2 | +157.5 [+88.4, +268.5] |

Priced once for the whole lane at k = 25 declared cells: expected max 1.97 x SE gives a bound of 90.3 s against an observed +157.5 s. Verdict: clears the chain bound.

## 2. Each step, for the record

| step | RMSE champion | RMSE challenger | improvement [95%] |
|---|---|---|---|
| B4 to L4b | 1965.6 | 1939.7 | +26.0 [+4.0, +54.3] |
| L4b to L6a | 1939.7 | 1861.8 | +77.9 [+43.0, +135.5] |
| L6a to L8c | 1861.8 | 1808.2 | +53.6 [+29.9, +95.0] |

## 3. Does the L8c step survive the extreme rows?

Rows removed in descending order of the champion's absolute error, the construction the control table below reproduces from `docs/LB-CONC.md`. If the lead collapses as the biggest truths leave, the step is a level artifact of a handful of rows.

| rows dropped | n | L6a | L8c | improvement [95%] |
|---|---|---|---|---|
| 0 | 5373 | 1861.8 | 1808.2 | +53.6 [+29.9, +95.0] |
| 1 | 5372 | 1476.7 | 1408.5 | +68.2 [+40.7, +105.3] |
| 2 | 5371 | 1254.5 | 1173.4 | +81.0 [+53.5, +111.4] |
| 5 | 5368 | 1125.1 | 1034.0 | +91.2 [+62.2, +123.1] |
| 20 | 5353 | 1010.0 | 903.9 | +106.0 [+75.7, +134.4] |

The same table for the earlier step, as a control (LB-CONC reports the first two rows):

| rows dropped | n | B4 | L6a | improvement [95%] |
|---|---|---|---|---|
| 0 | 5373 | 1965.6 | 1861.8 | +103.9 [+57.0, +178.1] |
| 1 | 5372 | 1605.7 | 1476.7 | +129.0 [+71.7, +191.4] |
| 2 | 5371 | 1404.1 | 1254.5 | +149.6 [+89.1, +206.1] |
| 5 | 5368 | 1298.8 | 1125.1 | +173.7 [+118.3, +225.1] |
| 20 | 5353 | 1160.1 | 1030.6 | +129.5 [+86.4, +168.3] |
