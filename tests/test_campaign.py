"""Regression checks for masked inputs, ID joins and sparse recording estimates."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import campaign
import mechanisms


def test_calendar_day_is_not_split_by_supplied_file_month() -> None:
    rows = pd.DataFrame({"day": pd.to_datetime(["2025-02-28", "2025-02-28", "2025-03-01"]),
                         "month": [2, 3, 3], "y": [10., 20., 30.]})
    base, candidate = np.zeros(3), np.array([1., 2., 3.])
    observed = campaign.paired_metrics(rows, base, candidate)
    regrouped = campaign.paired_metrics(rows.assign(month=3), base, candidate)
    assert observed["day_ci"] == regrouped["day_ci"]
    assert observed["leave_best_day_sse_gain"] == pytest.approx(95.)
    assert observed["leave_best_5day_episode_sse_gain"] == pytest.approx(0.)
    assert observed["by_month"]["2"]["n"] == 1
    assert observed["by_month"]["3"]["n"] == 2


def test_departure_fields_hidden_but_arrival_fields_retained(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = pd.DataFrame({"PHASE_mvt": ["DEP", "ARR"], "TAXITIME_SEC_mvt": [100., 200.],
                        "BLOCK_TIME_UTC_mvt": pd.to_datetime(["2025-01-01", "2025-01-02"], utc=True),
                        "MVT_TIME_UTC_mvt": pd.to_datetime(["2025-01-03", "2025-01-04"], utc=True),
                        **{c: [pd.NaT, pd.NaT] for c in campaign.PLAN_STAMPS}})
    monkeypatch.setattr(campaign, "build_frame", lambda hidden: hidden)
    masked = campaign.masked_frame(raw)
    assert pd.isna(masked.loc[0, "TAXITIME_SEC_mvt"])
    assert pd.isna(masked.loc[0, "BLOCK_TIME_UTC_mvt"])
    assert masked.loc[1, "TAXITIME_SEC_mvt"] == 200
    assert masked.loc[1, "BLOCK_TIME_UTC_mvt"] == raw.loc[1, "BLOCK_TIME_UTC_mvt"]
    assert raw.loc[0, "TAXITIME_SEC_mvt"] == 100


def test_prediction_windows_reject_missing_duplicate_and_nonfinite_ids() -> None:
    rows = pd.DataFrame({"id": [1, 2]})
    for values in (pd.Series([1.], index=[1]), pd.Series([1., 2.], index=[1, 1]), pd.Series([1., np.nan], index=[1, 2])):
        with pytest.raises(ValueError):
            campaign.prediction_windows(rows, values)


def test_sparse_hierarchy_shrinks_to_available_parents(monkeypatch: pytest.MonkeyPatch) -> None:
    def keys(rows: pd.DataFrame) -> pd.DataFrame:
        return rows[["cell", "airline", "flight"]].copy()
    monkeypatch.setattr(mechanisms, "keyed", keys)
    monkeypatch.setattr(mechanisms, "copy_labels", lambda rows: rows["label"].values)
    train = pd.DataFrame({"cell": ["a", "a", "b"], "airline": ["a|X", "a|X", "b|Y"],
                          "flight": ["a|1", "a|1", "b|2"], "label": [1., 1., 0.]})
    query = pd.DataFrame({"cell": ["a", "a", "c"], "airline": ["a|X", "a|Z", "c|Z"],
                          "flight": ["a|1", "a|new", "c|new"]})
    p, n = mechanisms.hierarchy(train, query)
    assert np.all((p > 0) & (p < 1))
    assert p[0] > p[1] > p[2]
    np.testing.assert_array_equal(n, [2, 0, 0])
