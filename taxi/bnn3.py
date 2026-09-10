"""SB3+NN3 as a composed champion: SB3's five-seed bag blended with NN3's searched net at the weight fitted out of fold in `docs/NN3.md`.

The weight and the net arm are read from `taxi.bfile`, which is also what built the ranking file, so the champion and the submitted
file cannot drift apart. The bag is cached per row set under `taxi.sb3._fp`, exactly as `sb3.seed_pred` caches its own seeds, because
torch on the GPU does not reproduce a fit and a re-run would move the champion under every verdict already read against it.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from taxi.bfile import PARTNERS
from taxi.l2 import floored
from taxi.sb3 import _fp
from taxi.sb3 import predict as sb3_predict

DEPS = ("taxi.bfile", "taxi.sb3")  # read by `champ._code_fp`: the weight and the arm live in bfile, not here
W, ARM = PARTNERS["NN3"][0], PARTNERS["NN3"][1]


def bag(tr: pd.DataFrame, te: pd.DataFrame, tag: str = "holdout") -> np.ndarray:
    """NN3's five-seed bag on any row set, cached; tag keeps the holdout and ranking caches apart the way `sb3.seed_pred` does."""
    path, fp = f"data/cache/nn3bag_{tag}.parquet", _fp(tr, te)
    if os.path.exists(path):
        c = pd.read_parquet(path)
        if len(c) == len(te) and c.fp.iloc[0] == fp and np.array_equal(c.id.values, te.id.values):
            return c.pred.values
    pred = ARM(tr.reset_index(drop=True), te.reset_index(drop=True))
    pd.DataFrame({"id": te.id.values, "pred": pred, "fp": fp}).to_parquet(path, index=False)
    return pred


def predict(tr: pd.DataFrame, te: pd.DataFrame, tag: str = "holdout") -> np.ndarray:
    """The blend as `taxi.sw1` scored it and `taxi.bfile` shipped it: each part floored, blended at W, floored again."""
    return floored(W * bag(tr, te, tag) + (1 - W) * sb3_predict(tr, te, tag))
