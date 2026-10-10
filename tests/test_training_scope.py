from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

import lanea
import nn as nnmod
import train


def frame() -> pd.DataFrame:
    return pd.DataFrame({"id": range(24), "day": np.repeat(["2025-02-01", "2025-02-02", "2025-02-03"], 8),
                         "y": np.arange(24, dtype=float) + 900., "z_aobt": 900., "x": np.arange(24, dtype=float), "apt": "TEST"})


@pytest.mark.parametrize("module", (train, lanea))
def test_tree_stopping_never_receives_outer_ids(module: object, monkeypatch: pytest.MonkeyPatch) -> None:
    tr, te, calls = frame(), frame().assign(id=lambda d: d.id + 100), []

    def predict(cfg: dict, a: pd.DataFrame, b: pd.DataFrame, va: pd.DataFrame | None = None,
                iters: object = None) -> tuple:
        calls.append((tuple(a.id), tuple(b.id), None if va is None else tuple(va.id), iters))
        return np.zeros(len(b)), 4 if module is train else (4, 5, 6)

    monkeypatch.setattr(module, "fit_predict" if module is train else "predict", predict)
    module.outer_predict({}, tr, te)
    assert set(calls[0][2]).isdisjoint(te.id)
    assert calls[1][2] is None
    assert calls[1][0] == tuple(tr.id)
    assert calls[1][3] == (4 if module is train else (4, 5, 6))


def config() -> dict:
    return {"cat": ["apt"], "num": ["x"], "cell": {**nnmod.CELL, "hidden": (4,), "batch": 8,
                                                   "max_epochs": 5, "patience": 2, "bn": False}}


def test_nn_returns_best_epoch_separately_from_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    scores = iter((3.0, 1.0, 2.0, 4.0))
    monkeypatch.setattr(nnmod, "DEV", "cpu")
    monkeypatch.setattr(nnmod, "rmse", lambda *args: next(scores))
    model, _, _, _, best = nnmod.train(config(), frame())
    assert best == model.run_info["best_epoch"] == 2
    assert model.run_info["stopping_epoch"] == 4


def test_nn_fixed_budget_trains_every_day(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nnmod, "DEV", "cpu")
    model, enc, _, _, epochs = nnmod.train(config(), frame(), epochs=2)
    assert epochs == 2
    assert model.run_info == {"best_epoch": 2, "stopping_epoch": 2, "fit_rows": 24, "fit_days": 3, "mode": "all_rows"}
    assert enc.mu["x"] == frame().x.mean()


def test_nn_predict_refits_all_rows_after_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    original = nnmod.train

    def fit(cfg: dict, rows: pd.DataFrame, epochs: int | None = None) -> tuple:
        calls.append(epochs)
        return original(cfg, rows, epochs=epochs)

    monkeypatch.setattr(nnmod, "DEV", "cpu")
    monkeypatch.setattr(nnmod, "train", fit)
    _, epochs = nnmod.predict(config(), frame(), frame())
    assert calls == [None, epochs]


def test_training_batches_keep_lone_final_row() -> None:
    batches = list(nnmod._batches(17, 8, torch.Generator().manual_seed(0)))
    assert [len(b) for b in batches] == [8, 9]
    assert sorted(torch.cat(batches).tolist()) == list(range(17))
