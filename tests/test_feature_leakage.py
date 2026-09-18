"""Feature leakage audit (lane R5). Forward-looking features are allowed on purpose (PLAN.md lines 25-26:
dep_after, arr_off look forward in the same file) - the leak that matters here is a feature computed from the
BLOCK_TIME_UTC_mvt that the scorer blanks on rank rows. Sources: features.py build()/CAT/NUM/DAY/OCC/WX/ARR/WZ/SNOW/PW."""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

import features
from features import ARR, CAT, DAY, NUM, OCC, SNOW, WX, WZ

NUMERIC_FEATURE_COLS = NUM + DAY + OCC + WX + ARR + WZ + SNOW  # PW excluded: needs with_pw(), not in build()'s output
ALL_FEATURE_COLS = CAT + NUMERIC_FEATURE_COLS
KNOWN_TRAIN_ONLY = {"y"}  # dep["TAXITIME_SEC_mvt"], build() line 288; blanked on rank by the scorer, not a bug


def test_no_feature_is_near_deterministic_function_of_y(feat):
    """abs(corr(col, y)) >= 0.999 on train would mean the model can read the label off a feature almost exactly."""
    tr = feat[feat["split"] == "train"]
    y = tr["y"].astype(float)
    rows = []
    for c in NUMERIC_FEATURE_COLS:
        v = tr[c].astype(float)
        m = y.notna() & v.notna()
        if m.sum() < 2 or v[m].std(ddof=0) == 0:
            continue
        rows.append((c, float(np.corrcoef(y[m], v[m])[0, 1])))
    corr = pd.DataFrame(rows, columns=["col", "corr"])
    corr["abscorr"] = corr["corr"].abs()
    corr = corr.sort_values("abscorr", ascending=False)
    worst = corr[corr["abscorr"] >= 0.999]
    assert worst.empty, (
        f"near-deterministic leak, |corr(col, y)| >= 0.999:\n{worst.to_string()}\n"
        f"top 5 correlations overall:\n{corr.head(5).to_string()}"
    )


def test_feature_density_matches_between_train_and_rank(feat):
    """No feature should be dense on train and entirely null on rank: that means it is usable while training but
    never at inference, which silently shifts what the model learns to rely on."""
    tr, rk = feat[feat["split"] == "train"], feat[feat["split"] == "rank"]
    rows = [(c, tr[c].notna().mean(), rk[c].notna().mean()) for c in ALL_FEATURE_COLS + ["y"]]
    dens = pd.DataFrame(rows, columns=["col", "train_dense", "rank_dense"])
    flagged = dens[(dens["train_dense"] > 0.5) & (dens["rank_dense"] == 0.0) & ~dens["col"].isin(KNOWN_TRAIN_ONLY)]
    assert flagged.empty, (
        f"feature(s) dense on train, all-null on rank (exceptions allowed: {sorted(KNOWN_TRAIN_ONLY)}):\n"
        f"{flagged.to_string()}"
    )


def test_z_aobt_path_does_not_reference_block_time():
    """features.py build() derives z_aobt (and the other z_* stamps) from the flight-plan columns only; read the
    exact source block that assigns them and confirm BLOCK_TIME_UTC_mvt never appears in it."""
    src = inspect.getsource(features.build)
    block = src[src.index("for k, c in"):src.index('f["z_anchor"]')]
    assert "AOBT_3_flt" in block, "expected the z_* loop to build z_aobt from AOBT_3_flt; build() source shape changed"
    assert "BLOCK_TIME_UTC_mvt" not in block, "z_aobt must never reference BLOCK_TIME_UTC_mvt: it is blanked on rank rows"


def test_z_aobt_is_populated_on_rank_despite_blanked_block_time(feat, dep):
    """z_aobt cannot equal the recorded off-block on rank rows because that stamp does not exist there (blanked);
    it must be a reconstruction from AOBT_3_flt. Proof: it stays populated where BLOCK_TIME_UTC_mvt is 100% null."""
    rk_dep = dep[dep["split"] == "rank"]
    assert int(rk_dep["BLOCK_TIME_UTC_mvt"].notna().sum()) == 0, "premise failed: BLOCK_TIME_UTC_mvt is not blanked on rank rows"
    rk_feat = feat[feat["split"] == "rank"]
    nonnull = int(rk_feat["z_aobt"].notna().sum())
    assert nonnull > 0, "z_aobt is all-null on rank rows: consistent with (wrongly) depending on blanked BLOCK_TIME_UTC_mvt"
