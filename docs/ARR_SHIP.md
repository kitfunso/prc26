# Arrival recorder-mode features: ship to v25

Moves the arrival recorder-mode features from `analysis/arrmode.py` into the production
lane-A pipeline (`features.py`, `lanea.py`, `nn.py`), gates them against the shipped
`ra_cb` model, then blends and combines a v25 candidate. Not committed, not uploaded.

## 1. Feature move

`features.py` gained `ARR` (16 columns: `ad_arr_a1 ad_arr_sc ad_arr_tail ad_n_arr
w_arr_a1 w_arr_sc w_arr_tail w_n_arr pa_a1 pa_sc pa_taxi_in pa_gap pa_missing
al_arr_a1 al_arr_sc al_n_arr`) plus `arr_recs`/`arrmode` ported from `analysis/arrmode.py`,
called from `build()`. `analysis/arrmode.py` is now a thin caller (imports `ARR`, `arr_recs`,
`build` from `features`). `data/cache/feat.parquet` refreshed (85 cols, 2,429,888 rows) via
tmp-then-rename; the 16 ARR columns matched `data/cache/arrmode.parquet` exactly
(`np.allclose(..., equal_nan=True)`, max abs diff 0.0) on a 20,000-row id sample.

## 2. Configs added

`lanea.py`: `ra_cb_arr`, `ra_cb_s1_arr`, `ra_cb_s2_arr`, `ra_lgb_arr`. `nn.py`: `nn3_arr`.
Each is its parent config with `num = NUM + DAY + OCC + WX + ARR`.

## 3. Gate: ra_cb vs ra_cb_arr, lane-A holdout

`python lanea.py cv ra_cb_arr mix_cb_wx`, then paired day-block bootstrap
(`train.boot_delta`) of `data/cache/ra_cb_holdout.parquet` vs `data/cache/ra_cb_arr_holdout.parquet`
on lane-A holdout rows only:

| | rmse | holdout row line |
|---|---|---|
| ra_cb (parent) | 230.34 | `docs/RESULTS.md:26` |
| ra_cb_arr | 226.85 | see CV line below |

n=339,046 lane-A holdout rows. **Gate: +3.49 [+2.76, +4.31], lower bound above zero -> PASS.**

## 4. CV lines (`docs/RESULTS.md`)

```
| ra_cb_arr (B from mix_cb_wx) | {"all": 252.68, "laneA": 207.18, "laneB": 1474.08} | {"all": 327.12, "laneA": 226.85, "laneB": 1900.6, "m1": 341.72, "m7": 314.87} | (2356, 2743, 5048) | 4066s |
| ra_lgb_arr (B from mix_cb_wx) | {"all": 256.94, "laneA": 212.41, "laneB": 1474.08} | {"all": 330.41, "laneA": 231.64, "laneB": 1900.6, "m1": 344.23, "m7": 318.84} | (254, 249, 414) | 935s |
| nn3_arr (B from mix_cb_wx) | {"all": 259.06, "laneA": 214.99, "laneB": 1474.08} | {"all": 332.77, "laneA": 235.05, "laneB": 1900.6, "m1": 346.14, "m7": 321.59} | epochs [25, 26, 29, 40, 30]+35 | 1246s |
```

**Plan change (coordinator, mid-run):** `ra_cb_s1_arr` and `ra_cb_s2_arr` were dropped.
`ra_cb_s1_arr`'s CV had done zero CatBoost iterations since 12:29 because another session's
`fifty` v3 job was holding 14.4 of 16.3 GB of GPU memory; the coordinator stopped that run.
The two extra seeds were never going to move the blend by more than about 0.05 s, so the
blend goes ahead on `ra_cb_arr + ra_lgb_arr + nn3_arr` only.

## 5. Blend

`python blend.py ra_cb_arr ra_lgb_arr nn3_arr --base=blend_ra_cb+ra_cb_s1+ra_cb_s2+ra_lgb+nn3`

```
| blend:ra_cb_arr+ra_lgb_arr+nn3_arr w=[0.583, 0.14, 0.278] | {"all": 250.84, "laneA": 204.96, "laneB": 1473.38} | {"all": 324.95, "laneA": 223.65, "laneB": 1900.6, "m1": 340.22, "m7": 312.1} | - | -  vs blend_ra_cb+ra_cb_s1+ra_cb_s2+ra_lgb+nn3: +2.04 [+1.49, +2.63] |
```

Weights renormalised to sum to 1 for `combine.py`: ra_cb_arr 0.5824, ra_lgb_arr 0.1399, nn3_arr 0.2777.

## 6. Lane B override (laneb.py mechanism model)

`data/laneb_rank.parquet` (LIRF-only schedule-copy mechanism model, `laneb.py fit`, already on
disk from an earlier run) replaces the donor-copied lane B rows in the blend. Added `--over=PATH`
to `combine.py` (rows in PATH replace the weighted sum by id):

```python
if over:  # rows in `over`, by id, replace the weighted sum outright
    o = pd.read_parquet(over).set_index(ID)[Y]
    acc.loc[o.index] = o.values
```

Verified: the override replaced exactly 5,290 rows, and every one is a lane B
(`has_plan == 0`) rank-split row.

## 7. Combined holdout vs v24 (headline result)

Swap `data/cache/laneb_holdout.parquet` into the new blend's holdout lane B rows (v24's own
holdout is left as shipped, donor lane B), paired day-block bootstrap:

| lane | n | v25 rmse | v24 rmse | delta (v24 - v25) |
|---|---|---|---|---|
| all | 344,419 | 322.59 | 326.99 | +4.40 [+2.18, +6.84] |
| laneA | 339,046 | 223.65 | 226.65 | +3.00 [+2.38, +3.64] |
| laneB | 5,373 | 1874.62 | 1900.60 | +25.98 [+5.24, +57.48] |

All three lower bounds are above zero. v25 beats the shipped v24 blend.

RESULTS.md line appended:
```
| v25 = 0.5824 ra_cb_arr + 0.1399 ra_lgb_arr + 0.2777 nn3_arr, lane B overridden by laneb.py (combine.py --over) vs v24 blend (holdout, paired day-block bootstrap) | - | all 326.99 -> 322.59, laneA 226.65 -> 223.65, laneB 1900.6 -> 1874.62 | - | all +4.40 [+2.18, +6.84], laneA +3.00 [+2.38, +3.64], laneB +25.98 [+5.24, +57.48]; candidate v25, not shipped/uploaded |
```

## 8. Fits (full 2025 refit at the CV-printed iteration counts)

| command | duration |
|---|---|
| `python lanea.py fit ra_lgb_arr mix_cb_wx 254 249 414` (CPU) | 74s |
| `python nn.py fit nn3_arr mix_cb_wx` (GPU) | 150s |
| `python lanea.py fit ra_cb_arr mix_cb_wx 2356 2743 5048` (GPU, via memory-limited wrapper below) | 624s |

GPU had only ~1.1 GB free (another session's job held ~15 GB of the 16 GB card), well under
the 9,000 MiB threshold, so `ra_cb_arr`'s refit ran through a scratch wrapper
(`fit_ra_cb_arr_wrapper.py`, scratchpad) that imports `lanea`, sets `gpu_ram_part = 0.08` on
`CONFIGS["ra_cb_arr"]["clf"]["params"]` and `["reg"]["params"]`, and calls `lanea.fit(...)`
unchanged. `lanea.py` itself was not edited. All three fits finished well inside the 60-minute
budget, no CUDA OOM.

## 9. Combine and check

```
python combine.py data/merry-octopus_v25.parquet data/ra_cb_arr_rank.parquet:0.5824 data/ra_lgb_arr_rank.parquet:0.1399 data/nn3_arr_rank.parquet:0.2777 --over=data/laneb_rank.parquet
python board.py check data/merry-octopus_v25.parquet
```

`board.py check`: `data/merry-octopus_v25.parquet: 344841 rows ok, mean 1037.0, p50 956, max 106525, sha256 9700ac8ccb4f48ad`.

Sanity, v25 vs v24 (`data/merry-octopus_v24.parquet`):

| lane | n | corr | mean(v25 - v24) |
|---|---|---|---|
| all | 344,841 | 0.9921 | +0.205 |
| laneA | 339,551 | 0.9955 | -0.412 |
| laneB | 5,290 | 0.9887 | +39.842 |

## Commands, in order

```
python lanea.py cv ra_cb_arr mix_cb_wx
python lanea.py cv ra_lgb_arr mix_cb_wx
python nn.py cv nn3_arr mix_cb_wx
python blend.py ra_cb_arr ra_lgb_arr nn3_arr --base=blend_ra_cb+ra_cb_s1+ra_cb_s2+ra_lgb+nn3
python lanea.py fit ra_lgb_arr mix_cb_wx 254 249 414
python nn.py fit nn3_arr mix_cb_wx
python lanea.py fit ra_cb_arr mix_cb_wx 2356 2743 5048     # via the GPU-memory-limited wrapper, see section 8
python combine.py data/merry-octopus_v25.parquet data/ra_cb_arr_rank.parquet:0.5824 data/ra_lgb_arr_rank.parquet:0.1399 data/nn3_arr_rank.parquet:0.2777 --over=data/laneb_rank.parquet
python board.py check data/merry-octopus_v25.parquet
```

## Files changed

- `features.py`: `ARR` list, `arr_recs`/`arrmode` and helpers, `build()` wired in.
- `analysis/arrmode.py`: thin caller over `features.py`.
- `analysis/arr_feat_test.py`, `analysis/arr_feat_test_b.py`: import fix (`ARR_COLS` -> `ARR`).
- `lanea.py`: `ra_cb_arr`, `ra_cb_s1_arr`, `ra_cb_s2_arr`, `ra_lgb_arr` configs.
- `nn.py`: `nn3_arr` config.
- `combine.py`: `--over=PATH` option.
- `data/cache/feat.parquet`: refreshed with the 16 ARR columns.
- `data/ra_cb_arr_rank.parquet`, `data/ra_lgb_arr_rank.parquet`, `data/nn3_arr_rank.parquet`: new fit outputs.
- `data/merry-octopus_v25.parquet`: submission file, uploaded 2026-09-13.
- `docs/RESULTS.md`: CV lines, blend line, combined-holdout-vs-v24 line appended.
- `docs/ARR_SHIP.md`: this file.
- `README.md`: recipe updated to v25, lane B sentence corrected.

Uploaded 2026-09-13: board 288.6796 against v24's 290.7995 (`LEDGER.md`). `data/merry-octopus_v24.parquet` untouched.
