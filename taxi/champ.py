"""Lane-A champion holdout predictions, fit once per champion and cached, so every lane-A challenger is judged against the same errors."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from taxi.data import HOLDOUT_MONTHS, PLAN, TRAIN_MONTHS, load
from taxi.m1 import chosen_cells, final_rounds
from taxi.m2 import predict_m2, stages
from taxi.w1 import COLS_A
from taxi.weather import join, load_metar

COLS_D1 = [*COLS_A, *PLAN]
# Lane-A champions by name: columns, and the report whose recorded rounds the refit uses.
CHAMPIONS_A = {"W1-A": (COLS_A, "W1", "docs/W1.md"), "D1": (COLS_D1, "D1", "docs/D1.md")}


def cache_path(name: str) -> str:
    return f"data/cache/{name.lower().replace('-', '')}_holdout.parquet"


def lane_a() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lane-A training and holdout frames with weather joined, in the row order every lane-A verdict shares."""
    metar = load_metar()
    tr, te = (join(load(m), metar) for m in (TRAIN_MONTHS, HOLDOUT_MONTHS))
    tra, tea = (d[d.has_fpl].reset_index(drop=True) for d in (tr, te))
    return tra, tea


def _fingerprint(tra: pd.DataFrame, tea: pd.DataFrame) -> float:
    # Cheap stamp of both frames: a feature-view change (A1) or a different holdout invalidates the cache.
    return float(len(tra) + tra.z_anchor.sum() + tra.queue.sum() + tea.id.sum() + tea.z_anchor.sum())


def champion_holdout(name: str, tra: pd.DataFrame, tea: pd.DataFrame) -> np.ndarray:
    """The named champion refit on the lane-A training rows at its recorded rounds, predicted on the holdout rows."""
    cols, tag, doc = CHAMPIONS_A[name]
    path, fp = cache_path(name), _fingerprint(tra, tea)
    if os.path.exists(path):
        c = pd.read_parquet(path)
        if len(c) == len(tea) and c.fp.iloc[0] == fp and np.array_equal(c.id.values, tea.id.values):
            return c.pred.values
    cell, _ = chosen_cells()["A"]
    pred = predict_m2(stages(tra, cell, rounds=final_rounds(tag, doc), cols=cols), tea)[0]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"id": tea.id.values, "pred": pred, "fp": fp}).to_parquet(path, index=False)
    return pred


if __name__ == "__main__":
    import sys
    tra, tea = lane_a()
    for name in sys.argv[1:] or ["W1-A"]:
        print(name, len(champion_holdout(name, tra, tea)), "holdout predictions cached at", cache_path(name))
