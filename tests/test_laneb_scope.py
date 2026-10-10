from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import laneb


@pytest.fixture
def history_frame(monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    f = pd.DataFrame({
        "id": range(1, 10), "apt": "LIRF", "airline": "ABC",
        "split": ["train"] * 8 + ["rank"],
        "month": [1, 4, 7, 10, 1, 4, 7, 10, 1],
        "day": ["2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01",
                "2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01", "2026-01-01"],
        "has_plan": [1] * 4 + [0] * 5,
        "z_sched": 4000.0, "y": [4000., 1000., 4000., 1000., 1000., 1000., 1000., 1000., np.nan],
    }, index=np.arange(10, 19))
    raw = pd.DataFrame({"MVT_ID_mvt": f["id"].values, "FLIGHT_mvt": "ABC123", "PHASE_mvt": "DEP"})
    monkeypatch.setattr(laneb.pd, "read_parquet", lambda *args, **kwargs: raw.copy())
    return f


def test_hidden_labels_cannot_change_history(history_frame: pd.DataFrame) -> None:
    permitted = pd.Index([2, 4, 6, 8])
    before = laneb.copy_hist(history_frame, permitted)
    changed = history_frame.copy()
    changed.loc[~changed["id"].isin(permitted), "y"] = -86400.0
    pd.testing.assert_frame_equal(before, laneb.copy_hist(changed, permitted))
    np.testing.assert_array_equal(before.loc[[14, 16], laneb.HIST[:2]], 0.0)


def test_permitted_labels_still_affect_history(history_frame: pd.DataFrame) -> None:
    permitted = pd.Index([2, 4])
    before = laneb.copy_hist(history_frame, permitted)
    changed = history_frame.copy()
    changed.loc[changed["id"].isin(permitted), "y"] = 4000.0
    after = laneb.copy_hist(changed, permitted)
    assert (after.loc[14, laneb.HIST[:2]] > before.loc[14, laneb.HIST[:2]]).all()


def test_training_history_excludes_nearby_months(history_frame: pd.DataFrame) -> None:
    before = laneb.copy_hist(history_frame, pd.Index([1, 2, 3, 4]))
    changed = history_frame.copy()
    changed.loc[10, "y"] = -1000.0
    after = laneb.copy_hist(changed, pd.Index([1, 2, 3, 4]))
    pd.testing.assert_series_equal(before.loc[14], after.loc[14])
    assert before.loc[18, "h_al"] != after.loc[18, "h_al"]


def test_empty_history_uses_label_free_prior(history_frame: pd.DataFrame) -> None:
    history = laneb.copy_hist(history_frame, pd.Index([], dtype=int))
    np.testing.assert_array_equal(history.loc[14], [0.5, 0.5, 0.0])


@pytest.mark.parametrize("fit_ids", ([1, 1], [999], [9]))
def test_invalid_fit_scope_is_rejected(history_frame: pd.DataFrame, fit_ids: list[int]) -> None:
    with pytest.raises(ValueError, match="unique training departure IDs"):
        laneb.copy_hist(history_frame, pd.Index(fit_ids))


def test_outer_labels_never_choose_iterations(history_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def predict(tr: pd.DataFrame, te: pd.DataFrame, va: pd.DataFrame | None = None,
                iters: tuple[int, int] | None = None) -> tuple:
        calls.append((tuple(tr["id"]), tuple(te["id"]), None if va is None else tuple(va["id"]), iters))
        p = te["h_al"].to_numpy() + tr["y"].mean()
        return p, (5, 7), None, 0.0, 0

    monkeypatch.setattr(laneb, "predict", predict)
    fit_mask = (history_frame["split"] == "train") & ~history_frame["month"].isin((1, 7))
    _, _, before, _ = laneb.outer_predict(history_frame, fit_mask, pd.Index([5, 7]))
    assert calls == [((6,), (8,), (8,), None), ((6, 8), (5, 7), None, (5, 7))]
    changed = history_frame.copy()
    changed.loc[~fit_mask, "y"] = -99999.0
    _, _, after, _ = laneb.outer_predict(changed, fit_mask, pd.Index([5, 7]))
    np.testing.assert_array_equal(before, after)


def test_overlapping_outer_scope_is_rejected(history_frame: pd.DataFrame) -> None:
    fit_mask = history_frame["split"] == "train"
    with pytest.raises(ValueError, match="overlap"):
        laneb.outer_predict(history_frame, fit_mask, pd.Index([5]))
