import numpy as np
import pandas as pd
import pytest

from analysis.champion_meta_weather import applied, hot, simplex
from analysis.champion_meta_context import context_frame
from champion_pipeline import MEMBERS, recal_frame
import recal


def test_simplex_finds_known_mean_optimum_and_checks_kkt() -> None:
    x = np.array([[0., 10., 20.], [20., 10., 0.], [5., 0., 30.]])
    truth = x @ np.array([.2, .3, .5])
    weights, proof = simplex(x, truth)
    np.testing.assert_allclose(weights, [.2, .3, .5], atol=1e-5)
    assert proof["kkt_residual"] <= proof["kkt_tolerance"]


def test_simplex_boundary_and_degenerate_members() -> None:
    x = np.array([[0., 1., 2.], [2., 3., 4.], [4., 5., 6.]])
    weights, _ = simplex(x, x[:, 0] - 1)
    np.testing.assert_allclose(weights, [1., 0., 0.], atol=1e-8)
    same = np.repeat(x[:, :1], 3, axis=1)
    weights, _ = simplex(same, x[:, 0])
    np.testing.assert_array_equal(weights, [1., 0., 0.])
    with pytest.raises(ValueError, match="nonfinite"):
        simplex(x, np.array([1., np.nan, 3.]))


def test_rounding_keeps_simplex_and_nonnegative_boundary() -> None:
    for weights in ([.3333334, .3333334, .3333332], [0., .1234567, .8765433]):
        rounded = applied(np.array(weights))
        assert rounded.sum() == pytest.approx(1., abs=1e-12)
        assert (rounded >= 0).all()


def test_weather_missing_values_keep_existing_predicate() -> None:
    rows = pd.DataFrame({"w_ts": [np.nan, 0., 1.], "w_sn": [np.nan, 0., 0.],
                         "w_fz": [np.nan, 0., 0.], "w_vsby": [np.nan, .5, 10.]})
    np.testing.assert_array_equal(hot(rows), [False, True, True])


def test_context_preserves_original_blend_and_stamp_distances() -> None:
    rows = pd.DataFrame({"apt": ["AAA", "BBB"], **{z: [10., np.nan] for z in recal.Z},
                         **{m: [100., -10.] for m in MEMBERS}})
    weights = np.array([.357257, .218947, .423796])
    expected = recal_frame(rows, rows[list(MEMBERS)].values @ weights)
    observed = context_frame(rows, weights)
    pd.testing.assert_frame_equal(observed[recal.COLS], expected[recal.COLS])
    assert observed.c_disagreement.tolist() == [0., -10.]
