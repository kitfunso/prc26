# LK1: anchor and label-leak check over 2025 and the ranking file

Regenerate with `python -m taxi.lk1`. Anchor rule after A1: the first non-negative of the AOBT, EOBT and schedule gaps, else the takeoff itself. The correlation columns are the largest |corr(feature, y)| among the features outside the declared gap family (['t_occ', 'z_anchor', 'z_aobt', 'z_eobt', 'z_lobt', 'z_sched'], limit 0.5); `tests/test_leak.py` asserts the same on February.

| month | rows | anchor = AOBT | EOBT | schedule | takeoff itself | anchors after takeoff | max non-gap corr, lane A | lane B |
|---|---|---|---|---|---|---|---|---|
| 01 | 153,706 | 152,209 | 31 | 1,404 | 62 | 0 | queue 0.431 | notyp 0.286 |
| 02 | 143,731 | 142,592 | 21 | 1,056 | 62 | 0 | queue 0.435 | queue 0.449 |
| 03 | 164,450 | 163,315 | 36 | 1,022 | 77 | 0 | queue 0.406 | notyp 0.528 |
| 04 | 175,288 | 174,155 | 34 | 1,030 | 69 | 0 | queue 0.380 | notyp 0.522 |
| 05 | 185,202 | 183,518 | 54 | 1,578 | 52 | 0 | queue 0.363 | notyp 0.388 |
| 06 | 183,142 | 180,544 | 71 | 2,444 | 83 | 0 | queue 0.328 | notyp 0.410 |
| 07 | 190,713 | 186,597 | 112 | 3,922 | 82 | 0 | queue 0.335 | notyp 0.364 |
| 08 | 191,181 | 188,778 | 52 | 2,292 | 59 | 0 | queue 0.325 | notyp 0.483 |
| 09 | 183,951 | 181,328 | 115 | 2,451 | 57 | 0 | queue 0.344 | notyp 0.404 |
| 10 | 185,674 | 183,666 | 71 | 1,887 | 50 | 0 | queue 0.369 | notyp 0.563 |
| 11 | 162,332 | 160,921 | 38 | 1,331 | 42 | 0 | queue 0.428 | notyp 0.413 |
| 12 | 165,677 | 163,942 | 55 | 1,616 | 64 | 0 | queue 0.388 | notyp 0.473 |
| ranking | 344,841 | 339,382 | 118 | 5,229 | 112 | 0 | no label | no label |

Largest non-gap correlation over the year: 0.563 (limit 0.5). Anchors after takeoff over the year and the ranking file: 0.

## Reading

No anchor lands after takeoff in any month or in the ranking file (F11 closed by A1). Lane A's largest non-gap correlation is `queue` (0.33 to 0.44), under the 0.5 limit every month. Lane B's largest is `notyp` (the movement has no aircraft type) at 0.52 to 0.56 in March, April and October, over the limit; it is not a leak: `notyp` is read from the same ranking-visible column in both files, and its correlation with y is the documented lane-B mechanism (schedule-copied block times on no-type movements, mainly LIRF, protocol L3 to L6), which the lane-B champion L6a is built on. The February test (`tests/test_leak.py`) passes at 0.449 (`queue`). The limit stays at 0.5 and the three months are reported, not suppressed.
