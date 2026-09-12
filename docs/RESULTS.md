
## Baselines on the Jan+Jul 2025 holdout (fit on the other 10 months)

- constant mean: {'all': 686.62, 'laneA': 476.59, 'laneB': 3986.01, 'm1': 604.89, 'm7': 746.01}
- per airport x lane mean: {'all': 637.07, 'laneA': 440.66, 'laneB': 3709.95, 'm1': 570.69, 'm7': 685.92}
- takeoff - anchor + airport x lane offset: {'all': 698.91, 'laneA': 448.71, 'laneB': 4313.57, 'm1': 668.16, 'm7': 722.74}
- raw takeoff - AOBT_3 on lane A only: 428.17
| lgb0 | {"all": 283.16, "laneA": 217.62, "laneB": 1840.94, "m1": NaN, "m7": NaN} | {"all": 391.22, "laneA": 273.34, "laneB": 2257.56, "m1": 365.84, "m7": 410.54} | 923 | 502s |
| mix_lgb | {"all": 273.06, "laneA": 217.79, "laneB": 1676.13} | {"all": 363.01, "laneA": 259.41, "laneB": 2049.53, "m1": 360.34, "m7": 365.14} | (146, 1682) | 1024s |
| mix_lgb_wx | {"all": 266.76, "laneA": 215.62, "laneB": 1599.29} | {"all": 355.48, "laneA": 250.49, "laneB": 2034.88, "m1": 353.79, "m7": 356.83} | (147, 1380) | 704s |
| blend:mix_lgb_wx+mix_lgb w=[0.861, 0.139] | {"all": 266.25, "laneA": 215.34, "laneB": 1594.4} | {"all": 355.09, "laneA": 251.09, "laneB": 2025.92, "m1": 353.75, "m7": 356.16} | - | -  vs mix_lgb: +7.91 [+0.72, +18.04] |
