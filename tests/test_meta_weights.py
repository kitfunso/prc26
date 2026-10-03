"""The declared two-member simplex fit minimizes training-only squared error."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.meta_screen import meta_inputs, simplex_weight


def test_known_blend_and_simplex_boundaries() -> None:
    a, b = np.array([0., 1., 2.]), np.array([2., 4., 0.])
    assert np.isclose(simplex_weight(.75 * a + .25 * b, a, b), .25)
    assert simplex_weight(2 * b - a, a, b) == 1
    assert simplex_weight(2 * a - b, a, b) == 0
    assert simplex_weight(a, a, a) == 0


def test_full_meta_reuses_exact_inner_pw_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = pd.DataFrame({"id": range(5), "has_plan": 1,
                      "day": ["2025-12-20", "2025-12-21", "2025-12-22", "2025-02-03", "2025-01-01"],
                      "file": ["training_2025_12.parquet"] * 3 + ["training_2025_02.parquet", "training_2025_01.parquet"]})
    path = tmp_path / "pw-inner-2-3.parquet"
    pw = pd.DataFrame({"id": [3, 2, 1, 0], "pw": [30., 20., 10., 0.]})
    pw.to_parquet(path)
    block = {"inner_stop_days": ["2025-12-22"], "inner_donor_scopes": [{"fit_rows": 2}]}
    monkeypatch.setattr("analysis.meta_screen.with_scoped_pw", lambda *a: pytest.fail("cached full PW must be reused"))
    fit, inner, stop, inside, scopes = meta_inputs(f, (2, 3), True, tmp_path, block)
    assert set(f.loc[fit, "id"]) == {0, 1, 2}
    assert set(inner.id) == {0, 1} and set(stop.id) == {2}
    assert inside.set_index("id").pw.to_dict() == {0: 0., 1: 10., 2: 20., 3: 30.}
    assert scopes == block["inner_donor_scopes"]
    for broken in (pw.assign(id=[3, 2, 1, 1]), pw.assign(id=[4, 2, 1, 0])):
        broken.to_parquet(path)
        with pytest.raises(ValueError, match="PW IDs"):
            meta_inputs(f, (2, 3), True, tmp_path, block)
    pw.to_parquet(path)
    with pytest.raises(ValueError, match="stopping dates"):
        meta_inputs(f, (2, 3), True, tmp_path, {**block, "inner_stop_days": ["2025-01-01"]})
