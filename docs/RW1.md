# RW1: training window and recency weighting, Dec-2025 forward fold

Regenerate with `python -m taxi.rw1`. MC2's cell and its 49 columns throughout; train on 2025 months 1-11, judge 164028 December lane-A rows. `MC2-fixed` refits at MC2's recorded rounds (1804, 1532, 3726) and reproduces `docs/FORWARD.md`; every other arm early-stops on the last 10% of fit days and is compared to `MC2-es`, which early-stops the same way on the same rows. k = 7 declared arms (protocol section 4, RW1).

| arm | months | half-life | fit rows | trees | Dec RMSE (s) | vs MC2-es (s) | fit s |
|---|---|---|---|---|---|---|---|
| MC2-fixed | 1-11 | - | 1898549 | fixed | 204.3 |  | 416 |
| MC2-es | 1-11 | - | 1898549 | (2260, 2520, 4714) | 205.6 |  | 551 |
| W3 | 9-11 | - | 526222 | (1243, 731, 1630) | 213.8 | -8.3 | 183 |
| W6 | 6-11 | - | 1082489 | (792, 346, 1876) | 216.3 | -10.7 | 162 |
| W9 | 3-11 | - | 1603656 | (1481, 1277, 2812) | 211.0 | -5.4 | 335 |
| W11 | 1-11 | - | 1898549 | (2127, 1721, 4325) | 205.9 | -0.3 | 507 |
| H3 | 1-11 | 3 mo | 1898549 | (1696, 1206, 4275) | 207.3 | -1.7 | 448 |
| H6 | 1-11 | 6 mo | 1898549 | (2289, 1388, 4845) | 206.4 | -0.8 | 547 |
| H12 | 1-11 | 12 mo | 1898549 | (2538, 1683, 4397) | 206.2 | -0.6 | 533 |

This fold cannot promote a champion. The holdout is Jan + Jul 2025, interior to the training year, so a recency window has no meaning against it, while the submission trains on all of 2025 and predicts 2026. Section 3 declares the forward fold reported, never deciding, so a winning arm is a gate question about how `taxi/s2.py` builds the submission, not a promotion.

Hazards recorded at declaration: December is one month and holds Christmas, so this is one seasonal draw; and `docs/HB.md` arm D measured seasonal weighting at -0.9444 s, a loss, though it weighted by calendar similarity rather than by time distance.

