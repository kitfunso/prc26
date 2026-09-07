"""PB claims a probe's gain is arithmetic, not an estimate: the SSE a measured group sum removes must be exact, overlaps and all."""
import numpy as np

from taxi.pb import removed


def test_the_removed_sse_matches_the_correction_it_predicts_on_overlapping_groups():
    rng = np.random.default_rng(0)
    n = 500
    w = np.column_stack([np.arange(n) < 300, np.arange(n) >= 300, rng.random(n) < 0.1]).astype(float)  # lane B straddles both months
    err = rng.normal(scale=50.0, size=n) + w @ np.array([12.0, -7.0, 250.0])
    c = w.T @ err
    d = w @ np.linalg.solve(w.T @ w, c)
    assert np.isclose(removed(w, c), float((err ** 2).sum() - ((err - d) ** 2).sum()))


def test_a_measured_group_sum_can_never_cost_sse():
    rng = np.random.default_rng(1)
    w = (np.arange(200) % 4 == 0).astype(float)[:, None]  # one group, no structure in it at all
    err = rng.normal(size=200)
    assert removed(w, w.T @ err) >= 0
