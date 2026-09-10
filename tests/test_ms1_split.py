"""MS1's month shares are read on squared error because that is the part that adds up across months. This pins that."""
import numpy as np

from taxi.ms1 import split


def _case():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    base = np.array([13.0, 24.0, 33.0, 41.0])
    chal = np.array([11.0, 21.0, 32.0, 44.0])
    return base, chal, y, np.array([1, 1, 7, 7])


def test_the_month_shares_sum_to_one():
    assert np.isclose(sum(p[5] for p in split(*_case())), 1.0)


def test_a_month_that_lost_ground_carries_a_negative_share():
    parts = split(*_case())
    assert parts[0][5] > 1.0 and parts[1][5] < 0.0, "July gets worse here, so its share must pull the total down"


def test_the_rows_and_rmses_are_read_inside_the_month():
    base, chal, y, month = _case()
    jan = split(base, chal, y, month)[0]
    assert jan[0] == 1 and jan[1] == 2
    assert np.isclose(jan[2], np.sqrt(np.mean(np.square(base[:2] - y[:2]))))
    assert np.isclose(jan[4], jan[2] - jan[3])
