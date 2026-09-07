"""Every lane-A champion must predict a frame with no label: the ranking file carries none, and F36 shipped twice."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import taxi.champ as champ
import taxi.mc1 as mc1
import taxi.sb3 as sb3

TAG = "pytest"


def _frame(n: int = 8) -> pd.DataFrame:
    cols = dict.fromkeys(["id", *champ.COLS_D2, *sb3.EXTRA])
    return pd.DataFrame({c: np.arange(n, dtype=float) for c in cols})


@pytest.fixture(autouse=True)
def _stub(monkeypatch):
    """Stub every fit so the paths run in milliseconds; only the frame handling is under test."""
    rounds, model = (1, 1, 1), object()
    pred = lambda m, te, *a, **k: (np.zeros(len(te)), None)  # noqa: E731
    for mod in (champ, sb3):
        monkeypatch.setattr(mod, "final_rounds", lambda *a, **k: rounds)
    monkeypatch.setattr(champ, "chosen_cells", lambda: {"A": ({}, None)})
    monkeypatch.setattr(champ, "stages", lambda *a, **k: model)
    monkeypatch.setattr(champ, "predict_m2", pred)
    for mod in (mc1, sb3):
        monkeypatch.setattr(mod, "stages_cb", lambda *a, **k: model)
        monkeypatch.setattr(mod, "predict_cb", pred)
    monkeypatch.setattr(sb3, "SEEDS", (0,))
    path = f"data/cache/sb3_seed0_{TAG}.parquet"
    if os.path.exists(path):
        os.remove(path)
    yield
    if os.path.exists(path):
        os.remove(path)


@pytest.mark.parametrize("name", [*champ.CHAMPIONS_A, *champ.COMPOSED_A])
def test_champion_predicts_a_frame_that_has_no_label(name):
    tr, te = _frame(), _frame()
    assert "y" not in te.columns
    assert len(champ.refit_predict(name, tr, te, TAG)) == len(te)


def test_a_composed_champion_cache_key_moves_when_its_module_changes(monkeypatch):
    """H53 edited taxi/sb3.py and the champion cache kept serving the withdrawn label-gated predictions: the frames cannot see a model change."""
    tr, te = _frame(), _frame()
    base = champ._fingerprint(tr, te)
    assert champ._cache_fp("MC2", tr, te) == base % (1 << 62)  # a registry champion is its cell and its recorded rounds, not a module
    first = champ._cache_fp("SB3", tr, te)
    assert first != base % (1 << 62)
    monkeypatch.setattr(champ, "_code_fp", lambda name: 7 if name in champ.COMPOSED_A else 0)
    assert champ._cache_fp("SB3", tr, te) != first
