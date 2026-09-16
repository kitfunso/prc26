# Window state: the recorder's state around each departure, from stamps only

About 31% of lane A labels are recorder copies of a known stamp (`mix.label`), and the tail
rows cluster in time. `docs/STATE_ORACLE.md` puts a ceiling on knowing that state: the
realised tail share and physical-residual level of each +-1h window (its A4 oracle) is
worth **+7.98 s [+5.99, +10.25]** on `ra_lgb_arr`'s lane A holdout. Day grain is weak
(+1.12) and knowing which stamp got copied is small (at most +2.28). This file is the
label-free estimator of that window state.

## Estimator

For each departure, over the other departures in the same (`split`, `apt`, `day`) whose
takeoff is within +-`win`, leave-one-out, using stamps and never `y`:

- `dz = clip(z_aobt - median z_aobt over (apt, runway, hour), +-1800)`, with the median taken
  over train and rank together. This is how far each neighbour's AOBT sits from the usual
  AOBT-to-takeoff gap. Stats: mean, sd, share above 600, share below -600.
- `d_ae = AOBT - EOBT`, clipped at +-1800: mean, sd, share with `|d_ae| < 60`.
- `d_al = AOBT - LOBT`, clipped at +-1800: mean, sd.
- `n`, the number of neighbours that have both stamps.

Scales: `wz` is +-1h on the same airport-day, `wa` is +-1h among the same airline,
`wz30` is +-30 min and `wz3h` is +-3h. Code: `features.window_state` and `features.winstate`.
The 20 shipped columns are `features.WZ`.

## How much of the oracle state the columns see

Linear R^2 against the oracle's leave-one-out window columns, on lane A train rows:

| columns | `win_tail_shr` | `win_physc_mean` |
|---|---|---|
| `wz`, 8 | 0.308 | 0.536 |
| `wz` + `wz2`, 20 | 0.379 | 0.555 (on the 1.50M of 2.06M rows where all 20 are finite) |

## Feature test

`PYTHONPATH=C:/Users/skf_s/prc26 python analysis/window_state_test.py wz wz_disp wz2`
appends the columns to the `ra_lgb_arr` config and runs `train.FOLDS`, then the holdout
refit, as `lanea.cv` does. The harness reproduces the cached `ra_lgb_arr` OOF exactly
(max abs diff 0). Deltas are `train.boot_delta` on the lane A holdout (339,046 rows).
Positive means the variant is better.

| variant | cols | folds (2,3)/(4,5)/(6,8)/(9,10)/(11,12) | holdout lane A | vs `ra_lgb_arr` |
|---|---|---|---|---|
| `ra_lgb_arr` | 0 | 213.60 / 200.16 / 224.34 / 217.25 / 204.81 | 231.643 | - |
| `wz` | 8 | 211.49 / 197.11 / 222.28 / 214.07 / 202.42 | 228.807 | +2.836 [+1.883, +3.878] |
| `wz_disp` | 12 | 211.18 / 197.13 / 222.15 / 214.02 / 202.29 | 228.829 | +2.814 [+2.016, +3.722] |
| `wz2` | 20 | 210.18 / 195.74 / 221.15 / 213.20 / 201.45 | **227.220** | **+4.424 [+3.205, +5.671]** |

- `wz2` against `wz`: +1.588 [+1.135, +1.976], better in all 5 folds. The airline window
  and the other widths add real information, so all 20 columns ship.
- `wz_disp` shows the day and hour dispersion columns (`docs/PHYS_TAIL.md`) add nothing
  once `wz` is present.
- `wz2` recovers 4.42 of the 7.98 s ceiling, about 55%.

## Arrival taxi-in level: tested, null

Hypothesis: an offset `d` on the block-time recorder adds `d` to arrivals' taxi-in and takes `d`
off departures' taxi-out, and arrivals are visible in both splits. `analysis/arr_level_test.py`
builds 9 columns from physical arrivals (not stamp copies): the deviation of taxi-in from its
(apt, runway, hour) median, over +-30 min, +-1h and +-3h of the departure's estimated off-block,
and over +-3h within the same airline.

- The sign refutes the mechanism. The 1h mean deviation correlates +0.169 with the oracle's
  physical-residual mean; a shared offset predicts a negative sign. The columns read congestion.
- Linear R^2 on the oracle columns rises a little on the same rows: `win_physc_mean` 0.561 to
  0.589, `win_tail_shr` 0.384 to 0.393.
- `wz2` plus the 9 columns: holdout lane A 227.249 vs 227.220, delta -0.029 [-0.356, +0.341],
  better in 2 of 5 folds. Not shipped.

## Production move

`features.build` appends the 20 `WZ` columns after the CAT cast (`features.winstate`), so the
refreshed `data/cache/feat.parquet` is (2,429,888, 105). Checked against a backup of the old
85-column cache: every old column is unchanged, and the 20 new columns equal a recompute on the
old frame (max abs diff 0, identical NaN pattern). The `lanea.py` configs `ra_cb_wz` and
`ra_lgb_wz` and the `nn.py` config `nn3_wz` are the v25 designs with `WZ` appended.
`python lanea.py cv ra_lgb_wz mix_cb_wx` reproduces the analysis run fold for fold (210.18 /
195.74 / 221.15 / 213.20 / 201.45) and on holdout (lane A 227.22 at refit iterations
(229, 286, 494)).

## Shift between the holdout months and the rank months

Lane A, standardised mean difference of each column in rank 2026 against the holdout month
of 2025. Script: the session scratchpad `wz_shift.py`.

- January: every column within +-0.25.
- July: the `dz` columns within 0.06; the `d_ae` columns -0.37 to -0.59; the `d_al` columns
  -0.33 to -0.39; `wz_dae0` +0.32.
- By airport, months pooled: LFPG `wz_dae` -1.05, EHAM -0.31; EDDF `wz_dz` +0.77, LSZH +0.29.

Against the spread of the twelve 2025 monthly means (lane A, scratchpad `month_range.py`):

- The `d_ae` and `d_al` means of both rank months sit inside it. July 2026 `wz_dae` is 357.4
  against 492.8 in July 2025, between the May (329.2) and October (377.3) 2025 levels.
- January 2026 `wz_dz` is 73.5, above the 2025 range of 23.5 to 49.3.
- Per airport, LFPG July 2026 `wz_dae` is 196.8 against a 2025 range of 331.9 to 668.0.
  LIRF January sits just below its 2025 minimum on `wz_dae` (219.6 vs 227.5) and `wz_dal`
  (162.1 vs 170.6).

The shipped model already uses row-level `d_ae`, `d_al` and `day_dae_med`, which carry the
same shift. This is the risk to check first if the board gain falls short of the holdout gain.

After v29 and v30 (session scratchpad `shift.py`, log `shift2.log`): a LightGBM told to separate
the holdout month's lane A rows from the same rank month's, validated on a fifth of the days held
out from both years, reaches AUC 0.73 for January and 0.92 for July, on the airport-day columns
(`n_day`, `day_dae_med`, `day_za_med`, `day_zs_med`, the `ad_arr_*` arrival shares). A random row
split gives AUC 1.00 because those columns name the day. Row-level identity coverage is not the
shift: 99.0 to 99.7% of rank rows use a stand, operator, destination and airline seen in the
holdout month. The two v29 columns that lost on the board rested on lane A copy history by flight
number (rank LIRF coverage 0.454 against 0.973, `docs/LANEB.md` section 6) and on lane B cell
levels, neither of which a 2025-only check can see; a column built from identity history needs its
rank coverage checked before it ships.

## Neighbours' predicted residual (after v30, not shipped yet)

Two more window families on `ra_lgb_wz`, same harness as the feature test above (session scratchpad `nw.py`, log
`nw.log`; `predict` per fold, refit at mean iterations x 1.1, `boot_delta` on the lane A holdout):

- `wr`, 6 columns: the `wz` statistics keyed by runway as well as airport and day.
- `pw`, 5 columns: leave-one-out count, mean and share above 600 s of the neighbours' predicted residual, the
  `ra_lgb_wz` prediction minus `z_aobt` clipped at +-1,800 s (OOF for train months, the refit for the holdout), over
  +-1 h on the airport-day, plus the +-1 h runway mean and the +-30 min mean. Rank rows get it from the rank
  predictions, so rank coverage equals holdout coverage by construction.

| variant | folds (2,3)/(4,5)/(6,8)/(9,10)/(11,12) | holdout lane A | vs base |
|---|---|---|---|
| base | 209.97 / 195.68 / 221.24 / 213.33 / 201.47 | 227.311 | - |
| `wr` | 210.29 / 195.50 / 220.72 / 212.49 / 201.13 | 227.113 | +0.198 [-0.186, +0.618], 4 of 5 folds |
| `pw` | 209.46 / 195.02 / 220.73 / 212.18 / 201.35 | 225.938 | **+1.373 [+0.706, +2.043], 5 of 5** |
| both | 209.72 / 195.02 / 220.76 / 212.57 / 201.31 | 226.038 | +1.273 [+0.618, +1.944], 5 of 5 |

The runway key adds nothing. The neighbours' predictions do: a second pass in which each row sees the hour's
predicted tail share and level, which is the state the oracle prices. The second stage cannot take it instead
(scratchpad `recalpw.py`, log `recalpw.log`): `recal.py`'s columns plus the same windows built from the v28 blend
give holdout lane A 218.878 against 218.597, -0.281 [-0.653, +0.052], 3 of 5 folds. The gain needs the base models
to see the columns, which means a second round of `ra_cb_wz`, `ra_lgb_wz` and `nn3_wz` with `pw` from the v28 blend.

## v26 ship

`python lanea.py cv ra_cb_wz mix_cb_wx` (GPU): folds 204.66 / 191.67 / 216.59 / 207.78 / 197.82,
holdout lane A 223.71 at refit iterations (2373, 1991, 4509). Each `WZ` config against its v25
parent on holdout lane A (`docs/RESULTS.md`):

| config | holdout lane A | v25 parent | parent's holdout lane A |
|---|---|---|---|
| `ra_cb_wz` | 223.71 | `ra_cb_arr` | 226.85 |
| `ra_lgb_wz` | 227.22 | `ra_lgb_arr` | 231.64 |
| `nn3_wz` | 233.77 | `nn3_arr` | 235.05 |

`python blend.py ra_cb_wz ra_lgb_wz nn3_wz --base=blend_ra_cb_arr+ra_lgb_arr+nn3_arr`: NNLS weights
on OOF 0.522 / 0.156 / 0.321 (v25: 0.583 / 0.140 / 0.278), holdout lane A 220.43 against 223.65,
+2.17 [+1.39, +3.11].

v26 = this blend, with lane B from `laneb.py` plus the hull (`docs/LANEB.md` section 4). Gate
against v25 on all 344,419 holdout rows (session scratchpad `gate_v26.py`, paired day-block
bootstrap, positive = better than v25):

| file | all | lane A | lane B | vs v25 |
|---|---|---|---|---|
| v25 | 322.59 | 223.65 | 1874.6 | - |
| lane A blend only | 320.40 | 220.43 | 1874.6 | +2.19 [+1.40, +3.14] |
| lane B hull only | 317.40 | 223.65 | 1816.9 | +5.19 [+0.01, +12.01] |
| v26 | 315.17 | 220.43 | 1816.9 | +7.42 [+2.12, +14.51] |

Uploaded 2026-09-13 as `merry-octopus_v26.parquet` (sha256 prefix fe320454aa7839c8): board 282.4241 against v25's
288.6796, 6.26 s against 7.42 on the holdout. Lane A alone projected 286.23 at v25's 1:1 SSE carry, and the board
went further. The hull ships in the same file, so the split between lane A and the six clipped rows is not read
(reading rank rows off board differences is a probe, and probes are Keith's call).
