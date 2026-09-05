"""The band cells reproduce a pure-copy band as z_sched and a pure-normal band as its mean; untouched rows keep B4."""
import numpy as np
import pandas as pd

from taxi.baselines import b4
from taxi.l4 import CELLS, band, per_band


def _frame(apt, notyp, z, y):
    return pd.DataFrame({"apt": apt, "notyp": np.int8(notyp), "z_sched": np.asarray(z, float), "queue": 3.0, "y": np.asarray(y, float)})


def test_band_edges():
    assert band([-5, 0, 1, 3600, 3601, 43200, 43201]).tolist() == [0, 0, 1, 1, 2, 5, 6]


def test_cells_recover_pure_bands():
    rng = np.random.default_rng(0)
    z1, z3 = rng.uniform(1, 3600, 150), rng.uniform(7201, 14400, 150)
    tr = pd.concat([_frame("LIRF", 1, np.r_[z1, z3], np.r_[z1, np.full(150, 1000.0)]), _frame("EDDF", 0, rng.uniform(0, 3000, 200), 900.0)], ignore_index=True)
    te = pd.concat([_frame("LIRF", 1, [1800, 10000], [0, 0]), _frame("EDDF", 0, [1500], [0])], ignore_index=True)
    ref = b4(tr, te, "B")
    for fit in CELLS.values():
        p = per_band(tr, te, fit)
        assert abs(p[0] - 1800) < 1e-6 and abs(p[1] - 1000) < 1e-6
        assert p[2] == ref[2]
