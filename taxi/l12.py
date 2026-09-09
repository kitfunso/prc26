"""L12: the lane-B copy probability with the airport's own recording-outage intensity as an input.

A block time copied from the schedule is a feed failure, and the failures burst: on 22,470 lane-B departures of 2025 the copy rate is
0.226, but 0.498 on the row after a copy and 0.147 otherwise. The copy itself is invisible in 2026, so L12 reads the burst off two
markers that are not: the share of the airport's recent departures with no aircraft type and with no flight plan (`taxi.data._outage`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from taxi.l3 import COPY
from taxi.l5 import Prob, is_copy, judge
from taxi.l6 import CELLS as L6_CELLS, per_row
from taxi.l8 import CB_KW, GBM_COLS, _frame, gbm

OUT_ALL = ("ntp6", "ntp24", "lb6", "dt_ntp")
OUT_NTP = ("ntp6", "ntp24")
K = 2


def _wide(d: pd.DataFrame, extra: tuple[str, ...]) -> pd.DataFrame:
    f = _frame(d)
    for c in extra:
        f[c] = d[c].values.astype(float)
    return f


def _fit(t: pd.DataFrame, te: pd.DataFrame, extra: tuple[str, ...], copy_s: float = COPY) -> np.ndarray:
    """L8c's classifier at L8c's settings, on L8c's columns plus the outage columns; only the input set changes."""
    c = is_copy(t, copy_s)
    if len(np.unique(c)) < 2:
        return np.full(len(te), float(c.mean()))
    cols = [*GBM_COLS, *extra, "airline"]
    m = CatBoostClassifier(**CB_KW).fit(Pool(_wide(t, extra)[cols], c.astype(int), cat_features=["airline"]))
    return m.predict_proba(Pool(_wide(te, extra)[cols], cat_features=["airline"]))[:, 1]


CELLS: dict[str, Prob] = {"L12a": lambda t, te: _fit(t, te, OUT_ALL), "L12b": lambda t, te: _fit(t, te, OUT_NTP)}


def _pred(tr: pd.DataFrame, te: pd.DataFrame, prob: Prob) -> np.ndarray:
    return per_row(tr, te, L6_CELLS["L6a"], prob=prob)


def _champion(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    return per_row(tr, te, L6_CELLS["L6a"], prob=gbm)


def main() -> None:
    judge(CELLS, _pred, "L12: lane-B copy probability with recording-outage intensity",
          f"Regenerate with `python -m taxi.l12`. L6a's gated mixture and L8c's classifier throughout, unchanged in depth, rounds, learning "
          f"rate and l2: the ONLY change is the input set. L12a adds all four outage columns {OUT_ALL}; L12b adds the two no-type shares "
          f"{OUT_NTP} alone. Each is read at the anchor over the airport's earlier departures, so none of them sees this row or any later one.",
          "docs/L12.md", champ=("L8c", _champion), k=K)


if __name__ == "__main__":
    main()
