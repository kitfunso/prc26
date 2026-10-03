"""Donor inference must exclude query labels at every feature-training level."""
from __future__ import annotations

import numpy as np
import pandas as pd

from neighbours import scoped_predictions


def test_nested_donor_sources_and_hidden_label_mutation() -> None:
    f = pd.DataFrame({"id": np.arange(10), "month": [2, 2, 4, 4, 6, 6, 9, 9, 10, 10],
                      "has_plan": 1, "y": np.arange(10, dtype=float)})
    permitted = pd.Index([0, 1, 2, 3, 4, 5])
    calls = []
    def predict(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
        assert not tr.id.isin(te.id).any()
        assert tr.id.isin(permitted).all()
        assert te.y.isna().all()
        calls.append((set(tr.id), set(te.id)))
        return np.full(len(te), tr.y.mean())
    original, scopes = scoped_predictions(f, permitted, predict)
    f.loc[~f.id.isin(permitted), "y"] = 999999.
    changed, _ = scoped_predictions(f, permitted, predict)
    pd.testing.assert_series_equal(original, changed)
    assert len(scopes) == 4
    assert calls[3][0] == set(permitted)
