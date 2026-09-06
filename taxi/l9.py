"""L9: the copy band that defines the lane-B regime. |y - z_sched| <= 600 s has been the definition since L3 and was never swept, and L3's own
report says the band admits punctual short taxis (24.6% of training rows land in it), so the L8c classifier learns a blurred class. The band moves
the classifier's label, the non-copy rows each band's fit uses, and the gate count together; everything else is L8c."""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from taxi.l3 import COPY
from taxi.l5 import Prob, judge
from taxi.l6 import CELLS as L6_CELLS, per_row
from taxi.l8 import gbm

BANDS = (300.0, 900.0, 1200.0)
CELLS: dict[str, float] = {f"L9-{int(v)}": v for v in BANDS}


def _pred(tr: pd.DataFrame, te: pd.DataFrame, copy_s: float) -> np.ndarray:
    return per_row(tr, te, L6_CELLS["L6a"], prob=partial(gbm, copy_s=copy_s), copy_s=copy_s)


def _champion(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    return _pred(tr, te, COPY)


def main() -> None:
    judge(CELLS, _pred, "L9: lane-B copy band sweep",
          f"Regenerate with `python -m taxi.l9`. L8c's gated mixture throughout; only the copy band changes, at {BANDS} s against the champion's "
          f"{COPY:.0f} s. The band sets which training rows are copies, so it moves the classifier's label, the non-copy rows each band's mean is "
          f"taken over, and the count that opens the gate, together. The CV winner alone meets the holdout.",
          "docs/L9.md", champ=("L8c", _champion), k=4)


if __name__ == "__main__":
    main()
