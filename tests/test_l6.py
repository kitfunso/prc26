"""A band with enough non-copy rows follows the airline; a band without them (and every untouched row) keeps L4b."""
import numpy as np
import pandas as pd

from taxi.l5 import champion
from taxi.l6 import CELLS, per_row


def _frame(apt, notyp, airline, z, y):
    z = np.asarray(z, float)
    return pd.DataFrame({"apt": apt, "notyp": np.int8(notyp), "airline": airline, "z_sched": z, "queue": 3.0, "hour": 10, "stand": "501",
                         "y": np.asarray(y, float)})


def test_gate_and_cells():
    rng = np.random.default_rng(0)
    za, zb, zf = rng.uniform(3601, 7200, 150), rng.uniform(3601, 7200, 150), rng.uniform(43201, 60000, 30)
    tr = pd.concat([_frame("LIRF", 1, "AAA", za, za), _frame("LIRF", 1, "BBB", zb, 1000.0), _frame("LIRF", 1, "AAA", zf, zf),
                    _frame("EDDF", 0, "CCC", rng.uniform(0, 3000, 200), 900.0)], ignore_index=True)
    te = pd.concat([_frame("LIRF", 1, "AAA", [5000], [0]), _frame("LIRF", 1, "BBB", [5000], [0]), _frame("LIRF", 1, "AAA", [50000], [0]),
                    _frame("EDDF", 0, "CCC", [1500], [0])], ignore_index=True)
    ref = champion(tr, te)
    for fit in CELLS.values():
        p = per_row(tr, te, fit)
        assert p[0] > 4500 and p[1] < 1500
        assert p[2] == ref[2] and p[3] == ref[3]
