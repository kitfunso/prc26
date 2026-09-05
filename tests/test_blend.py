import numpy as np

from taxi.e2 import blend, pick


def test_pick_returns_the_weight_with_the_lowest_rmse():
    rng = np.random.default_rng(0)
    y = rng.normal(size=1000)
    a, b = y + rng.normal(scale=1.0, size=1000), y + rng.normal(scale=1.0, size=1000)
    w, table = pick(y, a, b, weights=(0.0, 0.5, 1.0))
    assert w == 0.5  # two equally noisy independent components: the even blend halves the error variance
    assert [t[0] for t in table] == [0.0, 0.5, 1.0]
    assert all(t[1] > 0 for t in table)


def test_blend_endpoints_return_the_components():
    a, b = np.arange(5.0), np.arange(5.0) * 2
    assert np.array_equal(blend(a, b, 1.0), a)
    assert np.array_equal(blend(a, b, 0.0), b)
