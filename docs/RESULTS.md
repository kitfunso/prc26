
## Baselines on the Jan+Jul 2025 holdout (fit on the other 10 months)

- constant mean: {'all': 686.62, 'laneA': 476.59, 'laneB': 3986.01, 'm1': 604.89, 'm7': 746.01}
- per airport x lane mean: {'all': 637.07, 'laneA': 440.66, 'laneB': 3709.95, 'm1': 570.69, 'm7': 685.92}
- takeoff - anchor + airport x lane offset: {'all': 698.91, 'laneA': 448.71, 'laneB': 4313.57, 'm1': 668.16, 'm7': 722.74}
- raw takeoff - AOBT_3 on lane A only: 428.17
| lgb0 | {"all": 283.16, "laneA": 217.62, "laneB": 1840.94, "m1": NaN, "m7": NaN} | {"all": 391.22, "laneA": 273.34, "laneB": 2257.56, "m1": 365.84, "m7": 410.54} | 923 | 502s |
| mix_lgb | {"all": 273.06, "laneA": 217.79, "laneB": 1676.13} | {"all": 363.01, "laneA": 259.41, "laneB": 2049.53, "m1": 360.34, "m7": 365.14} | (146, 1682) | 1024s |
| mix_lgb_wx | {"all": 266.76, "laneA": 215.62, "laneB": 1599.29} | {"all": 355.48, "laneA": 250.49, "laneB": 2034.88, "m1": 353.79, "m7": 356.83} | (147, 1380) | 704s |
| blend:mix_lgb_wx+mix_lgb w=[0.861, 0.139] | {"all": 266.25, "laneA": 215.34, "laneB": 1594.4} | {"all": 355.09, "laneA": 251.09, "laneB": 2025.92, "m1": 353.75, "m7": 356.16} | - | -  vs mix_lgb: +7.91 [+0.72, +18.04] |
| mix_lgb_occ | {"all": 268.18, "laneA": 216.76, "laneB": 1607.97} | {"all": 353.36, "laneA": 250.7, "laneB": 2009.4, "m1": 355.44, "m7": 351.67} | (141, 1274) | 1140s |
| blend:mix_lgb_wx+mix_lgb_occ w=[0.684, 0.316] | {"all": 266.01, "laneA": 215.17, "laneB": 1592.83} | {"all": 353.89, "laneA": 249.66, "laneB": 2023.61, "m1": 353.45, "m7": 354.25} | - | -  vs mix_lgb_wx: +1.59 [+0.77, +2.56] |
| mix_lgb_wx_s1 | {"all": 266.03, "laneA": 215.74, "laneB": 1585.35} | {"all": 356.8, "laneA": 254.33, "laneB": 2019.69, "m1": 352.91, "m7": 359.91} | (152, 1521) | 1390s |
| blend:mix_lgb_wx+mix_lgb_wx_s1+mix_lgb_occ w=[0.323, 0.541, 0.135] | {"all": 264.94, "laneA": 214.84, "laneB": 1579.03} | {"all": 354.78, "laneA": 251.53, "laneB": 2018.91, "m1": 352.56, "m7": 356.56} | - | -  vs mix_lgb_wx: +0.70 [-2.04, +2.75] |
| mix_cb_wx | {"all": 257.53, "laneA": 213.13, "laneB": 1474.08} | {"all": 342.8, "laneA": 249.26, "laneB": 1900.6, "m1": 345.52, "m7": 340.59} | (2309, 3779) | 1919s |
| blend:mix_cb_wx+mix_lgb_wx+mix_lgb_wx_s1 w=[0.705, 0.033, 0.262] | {"all": 255.77, "laneA": 211.33, "laneB": 1469.12} | {"all": 341.11, "laneA": 247.85, "laneB": 1892.68, "m1": 345.04, "m7": 337.91} | - | -  vs mix_cb_wx: +1.69 [-2.17, +6.43] |
