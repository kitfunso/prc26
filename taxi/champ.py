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
# Lane-A champions by name: columns, the report whose recorded rounds the refit uses, and the model family (lgb or cb).
CHAMPIONS_A = {"W1-A": (COLS_A, "W1", "docs/W1.md", "lgb"), "D1": (COLS_D1, "D1", "docs/D1.md", "lgb"),
               "MC2": (COLS_D1, "MC2", "docs/MC2.md", "cb")}
# The registry's current champion per lane; verdict wording promotes only against these names.
CURRENT = {"A": "MC2", "B": "B4"}


def cache_path(name: str) -> str:
    return f"data/cache/{name.lower().replace('-', '')}_holdout.parquet"


def lane_a() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lane-A training and holdout frames with weather joined, in the row order every lane-A verdict shares."""
    metar = load_metar()
    tr, te = (join(load(m), metar) for m in (TRAIN_MONTHS, HOLDOUT_MONTHS))
    tra, tea = (d[d.has_fpl].reset_index(drop=True) for d in (tr, te))
    return tra, tea


def _fingerprint(tra: pd.DataFrame, tea: pd.DataFrame) -> int:
    # Row-wise hash of every column any champion reads, so a changed feature value invalidates the cache (critique S3, finding 1).
    cols = ["id", "y", *dict.fromkeys(c for cs, *_ in CHAMPIONS_A.values() for c in cs)]
    return int(sum(int(pd.util.hash_pandas_object(d[cols], index=False).sum()) for d in (tra, tea)) % (1 << 62))


def refit_predict(name: str, tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    """The named champion refit on tr at its recorded rounds, predicted on te; the one lane-A fit path for verdicts and files."""
    cols, tag, doc, kind = CHAMPIONS_A[name]
    rounds = final_rounds(tag, doc)
    if kind == "cb":
        from taxi.mc1 import predict_cb, stages_cb  # local: taxi.mc1 imports this module
        return predict_cb(stages_cb(tr, iters=rounds, cols=cols), te, cols)[0]
    cell, _ = chosen_cells()["A"]
    return predict_m2(stages(tr, cell, rounds=rounds, cols=cols), te)[0]


def champion_holdout(name: str, tra: pd.DataFrame, tea: pd.DataFrame) -> np.ndarray:
    """The named champion's holdout predictions, fit once and cached."""
    path, fp = cache_path(name), _fingerprint(tra, tea)
    if os.path.exists(path):
        c = pd.read_parquet(path)
        if len(c) == len(tea) and c.fp.iloc[0] == fp and np.array_equal(c.id.values, tea.id.values):
            return c.pred.values
    pred = refit_predict(name, tra, tea)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"id": tea.id.values, "pred": pred, "fp": fp}).to_parquet(path, index=False)
    return pred


if __name__ == "__main__":
    import sys
    tra, tea = lane_a()
    for name in sys.argv[1:] or ["W1-A"]:
        print(name, len(champion_holdout(name, tra, tea)), "holdout predictions cached at", cache_path(name))
