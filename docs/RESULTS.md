
## Baselines on the Jan+Jul 2025 holdout (fit on the other 10 months)

- constant mean: {'all': 686.62, 'laneA': 476.59, 'laneB': 3986.01, 'm1': 604.89, 'm7': 746.01}
- per airport x lane mean: {'all': 637.07, 'laneA': 440.66, 'laneB': 3709.95, 'm1': 570.69, 'm7': 685.92}
- takeoff - anchor + airport x lane offset: {'all': 698.91, 'laneA': 448.71, 'laneB': 4313.57, 'm1': 668.16, 'm7': 722.74}
- raw takeoff - AOBT_3 on lane A only: 428.17
| lgb0 | {"all": 283.16, "laneA": 217.62, "laneB": 1840.94, "m1": NaN, "m7": NaN} | {"all": 391.22, "laneA": 273.34, "laneB": 2257.56, "m1": 365.84, "m7": 410.54} | 923 | 502s |
