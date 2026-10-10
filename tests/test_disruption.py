"""Exposure integration respects irregular reporting and missing intervals."""
from __future__ import annotations

import numpy as np

from disruption import _integral


def test_irregular_reports_integrate_time_and_cap_missing_gaps() -> None:
    t = np.array([0., 60., 120., 7200.])
    state = np.array([1., 1., 1., 0.])
    query = np.array([-1., 60., 120., 3720., 7200., 7300.])
    np.testing.assert_allclose(_integral(t, state, query), [0, 60, 120, 3720, 3720, 3720])
