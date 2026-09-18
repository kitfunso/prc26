"""Label timing audit (lane R5). y = TAXITIME_SEC_mvt must equal takeoff (MVT_TIME_UTC_mvt) minus the recorded
off-block (BLOCK_TIME_UTC_mvt), only on train rows; the scorer blanks BLOCK_TIME_UTC_mvt on the 344,841 rank rows.
Sources: PLAN.md lines 7-10, docs/LANEB.md, data.py (secs, load_raw, departures), features.py build() line 288."""
from __future__ import annotations

import pandas as pd

from data import secs

TOL = 1.0  # seconds; audit tolerance requested. Measured max deviation on all 2,085,047 train rows is 0.0s.
N_SAMPLE = 5000
NEG_TRAIN_COUNT = 369  # sourced 2026-09-17: (departures(load_raw())["split"]=="train" & TAXITIME_SEC_mvt<0).sum()


def _train(dep: pd.DataFrame) -> pd.DataFrame:
    return dep[dep["split"] == "train"]


def _rank(dep: pd.DataFrame) -> pd.DataFrame:
    return dep[dep["split"] == "rank"]


def _recon_diff(rows: pd.DataFrame) -> pd.Series:
    recon = secs(rows["MVT_TIME_UTC_mvt"], rows["BLOCK_TIME_UTC_mvt"])
    return (recon - rows["TAXITIME_SEC_mvt"].astype(float)).abs()


def test_label_reconstructs_on_a_sample(dep):
    """y must equal takeoff minus the recorded off-block, to within TOL, on a reproducible sample of train rows."""
    sample = _train(dep).sample(n=N_SAMPLE, random_state=0)
    diff = _recon_diff(sample)
    bad = sample.loc[diff > TOL, ["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]]
    assert diff.max() <= TOL, (
        f"{len(bad)} of {len(sample)} sampled train rows: y != MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt "
        f"beyond {TOL}s (max diff {diff.max()}s)\n{bad.head(10)}"
    )


def test_label_never_built_from_a_different_stamp(dep):
    """Full-population reconstruction: a row built from AOBT/EOBT/schedule instead of BLOCK_TIME_UTC_mvt would
    show up here as a deviation beyond TOL, even if it happened to pass the sample check above."""
    tr = _train(dep)
    diff = _recon_diff(tr)
    bad = tr.loc[diff > TOL, ["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]]
    assert len(bad) == 0, (
        f"{len(bad)} of {len(tr)} train rows have y built from a stamp other than BLOCK_TIME_UTC_mvt:\n{bad.head(10)}"
    )


def test_label_nonnegative_or_exact_negative_count(dep):
    """y should be non-negative on train rows; count of exceptions is pinned so any change trips this test."""
    y = _train(dep)["TAXITIME_SEC_mvt"].astype(float)
    neg = int((y < 0).sum())
    assert neg == NEG_TRAIN_COUNT, (
        f"train negative-y count changed: expected {NEG_TRAIN_COUNT}, found {neg} (min y = {y.min()})"
    )


def test_label_absent_on_every_rank_row(dep):
    """The scorer blanks BLOCK_TIME_UTC_mvt on rank rows; y (TAXITIME_SEC_mvt) must be null there too."""
    rk = _rank(dep)
    leaked = rk[rk["TAXITIME_SEC_mvt"].notna()]
    assert len(leaked) == 0, (
        f"SERIOUS: {len(leaked)} of {len(rk)} rank rows carry a non-null y (TAXITIME_SEC_mvt), the held-out "
        f"label. Example MVT_ID_mvt: {leaked['MVT_ID_mvt'].head(5).tolist()}"
    )
