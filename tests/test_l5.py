"""An airline of pure copies predicts near z_sched, an airline of normals near their mean; untouched rows keep L4b."""
import numpy as np
import pandas as pd

from taxi.l5 import CELLS, champion, per_row


def _frame(apt, notyp, airline, z, y):
    z = np.asarray(z, float)
    return pd.DataFrame({"apt": apt, "notyp": np.int8(notyp), "airline": airline, "z_sched": z, "queue": 3.0, "hour": 10, "stand": "501",
                         "y": np.asarray(y, float)})


def test_cells_follow_the_airline():
    rng = np.random.default_rng(0)
    za, zb = rng.uniform(3601, 7200, 150), rng.uniform(3601, 7200, 150)
    tr = pd.concat([_frame("LIRF", 1, "AAA", za, za), _frame("LIRF", 1, "BBB", zb, 1000.0), _frame("EDDF", 0, "CCC", rng.uniform(0, 3000, 200), 900.0)],
                   ignore_index=True)
    te = pd.concat([_frame("LIRF", 1, "AAA", [5000], [0]), _frame("LIRF", 1, "BBB", [5000], [0]), _frame("LIRF", 1, "ZZZ", [5000], [0]),
                    _frame("EDDF", 0, "CCC", [1500], [0])], ignore_index=True)
    ref = champion(tr, te)
    for prob in CELLS.values():
        p = per_row(tr, te, prob)
        assert p[0] > 4500 and p[1] < 1500 and 1500 < p[2] < 4500
        assert p[3] == ref[3]
