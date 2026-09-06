import numpy as np

from taxi.lbc import branch_err, known_err


def test_branch_and_known_agree_only_where_the_copy_is_exact():
    y = np.array([100.0, 200.0, 300.0])
    z = np.array([100.0, 260.0, 300.0])  # row 1 is a near copy, not an exact one
    e0 = np.array([50.0, 50.0, 50.0])
    m = np.ones(3, bool)
    b, k = branch_err(e0, y, z, m), known_err(e0, m)
    assert np.array_equal(b[[0, 2]], k[[0, 2]])
    assert b[1] == 60.0 and k[1] == 0.0  # the branch keeps the residual the oracle would erase
    assert np.sum(b ** 2) > np.sum(k ** 2)


def test_untouched_rows_keep_the_champion_error():
    e0 = np.array([7.0, -3.0])
    m = np.array([True, False])
    assert branch_err(e0, np.zeros(2), np.zeros(2), m)[1] == -3.0
    assert known_err(e0, m)[1] == -3.0
