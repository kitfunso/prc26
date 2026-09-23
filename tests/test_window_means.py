"""Window means are NaN, never +-inf, on a row with no neighbour (v35: loo_mean gave +-inf on 2,066 pw.parquet cells,
and the MLP encoder then zeroed pw_m, pr_m and pw30_m on every row). Source: features.per, loo_mean."""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import loo_mean, per


def test_per_is_nan_at_zero_count():
    out = per(np.array([1e-14, -1e-14, 6.0]), np.array([0.0, 0.0, 3.0]))
    assert np.isnan(out[:2]).all() and out[2] == 2.0


def test_loo_mean_lone_row_is_nan_not_inf():
    f = pd.DataFrame({"g": [0, 0, 0], "t_off": [0.0, 10.0, 5000.0]})
    v = np.array([600.0 / 7, 1234.567 / 3, 0.123456789])
    cum = np.cumsum(v)
    assert cum[2] - cum[1] - v[2] != 0, "fixture must leave float noise on the lone row, the case that gave +-inf"
    out = loo_mean(f, ("g",), 100.0, v, np.ones(3, bool), "pw")
    assert out["pw_n"].tolist() == [1.0, 1.0, 0.0]
    assert np.isnan(out["pw_m"].iloc[2]) and np.isnan(out["pw_hi"].iloc[2])
    assert np.isclose(out["pw_m"].iloc[0], v[1]) and np.isclose(out["pw_m"].iloc[1], v[0])
