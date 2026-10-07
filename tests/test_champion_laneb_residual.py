import numpy as np
import pandas as pd
import pytest

from analysis.champion_laneb_residual import COLS, correction, features, scope
from analysis.champion_laneb_residual_audit import checked_table
from champion_pipeline import MEMBERS


def rows() -> pd.DataFrame:
    frame = pd.DataFrame({c: [0., np.nan] for c in COLS if c not in ("apt", "airline")})
    return frame.assign(id=[1, 2], y=[-60., 90000.], has_plan=0, apt=["AAA", "BBB"], airline=["A", "B"])


def test_query_targets_and_keys_never_enter_features() -> None:
    frame = rows().assign(day="2025-01-01", file="hidden", month=1)
    base = np.array([100., 200.])
    expected = features(frame, base)
    observed = features(frame.assign(y=1e9, id=[999, 1000], day="2040-01-01", file="changed", month=12), base)
    pd.testing.assert_frame_equal(expected, observed)
    assert not {"id", "y", "day", "file", "month"}.intersection(expected.columns)
    assert np.isnan(expected.iloc[1].z_sched)


def test_rejects_mixed_lanes_duplicate_scopes_and_overlap() -> None:
    frame = rows()
    with pytest.raises(ValueError, match="Lane B"):
        features(frame.assign(has_plan=[1, 0]), np.array([100., 200.]))
    with pytest.raises(ValueError, match="overlap"):
        scope(frame, frame.iloc[:1])
    with pytest.raises(ValueError, match="unique"):
        scope(frame.assign(id=[1, 1]), frame.assign(id=[3, 4]))
    with pytest.raises(ValueError, match="nonfinite"):
        scope(frame.assign(y=np.nan), frame.assign(id=[3, 4]))


def test_correction_keeps_raw_signed_targets_out_and_only_floors_prediction() -> None:
    class FixedModel:
        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            return np.array([-1000., 200000.])

    frame = rows()
    output = correction(frame, np.array([100., 200.]), FixedModel())
    np.testing.assert_array_equal(output, [0., 100200.])
    np.testing.assert_array_equal(output, correction(frame.assign(y=1e9), np.array([100., 200.]), FixedModel()))


def test_audit_enforces_exact_candidate_schema_and_every_retained_column() -> None:
    cold = ["id", "y"]
    original = pd.DataFrame({c: [1., 2.] for c in [*cold, *MEMBERS, "control", "clean_champion"]})
    reference = original.assign(p=[3., 4.], d_z_sched=[5., 6.])
    candidate = original.assign(**{"CP1-B-residual": [10., 20.]})
    checked_table(candidate, reference, cold)
    with pytest.raises(ValueError, match="schema"):
        checked_table(candidate.drop(columns="y"), reference, cold)
    with pytest.raises(ValueError, match="schema"):
        checked_table(candidate.assign(unexpected=1), reference, cold)
    with pytest.raises(AssertionError):
        checked_table(candidate.assign(y=-100.), reference, cold)
