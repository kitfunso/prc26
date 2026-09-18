"""Reproducibility audit (lane R5): build() is idempotent, its on-disk cache matches it, and every model config
carries a fixed seed. `pytest -m "not slow"` skips the two frame-sized checks; the seed audit always runs fast."""
from __future__ import annotations

import pandas as pd
import pytest

import lanea
import train as train_mod
from data import CACHE
from features import build


@pytest.mark.slow
def test_build_is_idempotent(feat):
    """Calling build() again in the same process must return an identical frame: same shape, column order, values."""
    again = build()
    assert list(again.columns) == list(feat.columns), "column order changed between two build() calls in one process"
    assert again.shape == feat.shape, f"shape changed: {feat.shape} then {again.shape}"
    pd.testing.assert_frame_equal(feat.reset_index(drop=True), again.reset_index(drop=True))


@pytest.mark.slow
def test_cache_matches_build_output(feat):
    """data/cache/feat.parquet is what build() reads on a warm cache; its row count and columns must match."""
    cached = pd.read_parquet(CACHE / "feat.parquet")
    assert cached.shape[0] == feat.shape[0], f"cache has {cached.shape[0]} rows, build() returned {feat.shape[0]}"
    assert set(cached.columns) == set(feat.columns), (
        f"cache/build() column sets differ: cache-only {set(cached.columns) - set(feat.columns)}, "
        f"build()-only {set(feat.columns) - set(cached.columns)}"
    )


def _missing_seed(params: dict) -> bool:
    return "random_seed" not in params and "random_state" not in params


def test_every_model_config_has_a_fixed_seed():
    """Every params dict fed to a CatBoost/LightGBM/XGBoost/Ridge model in lanea.CONFIGS or train.CONFIGS should
    carry random_seed or random_state, so a rerun of the same config reproduces the same fit."""
    missing = []
    for name, cfg in lanea.CONFIGS.items():
        for sub in ("clf", "reg"):
            if _missing_seed(cfg[sub]["params"]):
                missing.append(f"lanea.CONFIGS[{name!r}][{sub!r}]['params']")
    for name, cfg in train_mod.CONFIGS.items():
        if _missing_seed(cfg["params"]):
            missing.append(f"train.CONFIGS[{name!r}]['params']")
    assert not missing, "config(s) with no random_seed/random_state key, rerun is not pinned deterministic:\n" + "\n".join(missing)
