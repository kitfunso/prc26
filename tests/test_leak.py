"""LK1: the anchor never lands after takeoff, and no feature outside the declared gap family reproduces the label (S3 critique finding 6)."""
import os

import numpy as np
import pandas as pd
import pytest

from taxi.data import DATA, anchor, load
from taxi.lk1 import LIMIT, correlations


def test_anchor_takes_the_first_non_negative_gap_else_the_takeoff():
    df = pd.DataFrame({"has_fpl": [True, True, True, True, False, False],
                       "z_aobt": [100.0, -5.0, -5.0, np.nan, np.nan, np.nan],
                       "z_eobt": [200.0, 300.0, -1.0, np.nan, np.nan, np.nan],
                       "z_sched": [400.0, 500.0, 600.0, -9.0, 700.0, -3.0]})
    assert anchor(df).tolist() == [100.0, 300.0, 600.0, 0.0, 700.0, 0.0]


@pytest.fixture(scope="module")
def feb():
    if not os.path.isdir(DATA):
        pytest.skip("data/ not present")
    return load((2,))


def test_every_anchor_precedes_takeoff(feb):
    assert int((feb.z_anchor < 0).sum()) == 0


@pytest.mark.parametrize("lane", ["A", "B"])
def test_no_feature_outside_the_gap_family_reproduces_the_label(feb, lane):
    over = {c: round(r, 3) for c, r in correlations(feb[feb.has_fpl == (lane == "A")]).items() if r > LIMIT}
    assert not over, over
