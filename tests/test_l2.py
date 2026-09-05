"""L2 cells: the cap never exceeds the airport's training maximum, and the Huber fit reproduces a clean linear relation."""
import numpy as np
import pandas as pd

from taxi.l2 import cap, floored, huber


def _frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.uniform(-3000, 6000, n)
    q = rng.integers(0, 20, n)
    return pd.DataFrame({"apt": np.where(np.arange(n) % 2 == 0, "AAAA", "BBBB"), "z_sched": z, "queue": q, "y": 300 + 0.5 * z + 20 * q})


def test_cap_stays_under_each_airport_training_max():
    tr, te = _frame(400, 0), _frame(50, 1)
    te.loc[0, "z_sched"] = 1e6  # an extrapolating row
    p = cap(tr, te)
    top = te.apt.map(tr.groupby("apt").y.max()).values
    assert (p <= top + 1e-6).all()
    assert (floored(p) >= 0).all()


def test_huber_recovers_a_clean_line():
    tr, te = _frame(400, 2), _frame(50, 3)
    assert np.abs(huber(tr, te) - te.y.values).max() < 1.0
