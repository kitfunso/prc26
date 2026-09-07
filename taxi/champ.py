"""Champion registry: lane-A holdout predictions fit once per champion and cached, so every lane-A challenger is judged against the same errors;
lane-B champions by name behind one predictor path."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from taxi.data import HOLDOUT_MONTHS, PLAN, TRAIN_MONTHS, WINDOW, load
from taxi.m1 import chosen_cells, final_rounds
from taxi.m2 import predict_m2, stages
from taxi.w1 import COLS_A
from taxi.weather import join, load_metar

COLS_D1 = [*COLS_A, *PLAN]
COLS_D2 = [*COLS_D1, "d_as", "occ", "t_occ"]
# Lane-A champions by name: columns, the report whose recorded rounds the refit uses, and the model family (lgb or cb).
CHAMPIONS_A = {"W1-A": (COLS_A, "W1", "docs/W1.md", "lgb"), "D1": (COLS_D1, "D1", "docs/D1.md", "lgb"),
               "MC2": (COLS_D1, "MC2", "docs/MC2.md", "cb")}
# Composed lane-A champions: built from parts each fixed by an earlier run, so they carry a module instead of a cell (SB3, H52).
# Their bases stay out of CHAMPIONS_A on purpose, because that registry feeds _fingerprint and would invalidate every cache below.
COMPOSED_A = {"SB3": "taxi.sb3"}
# The registry's current champion per lane; verdict wording promotes only against these names.
# SB3 keeps lane A at H53 on a corrected number: its ST1a gate read each row's own label and is withdrawn, taking the verdict from
# +2.38 to +2.0551 [+1.71, +2.37]. Section 3 asks for an interval above zero and a point improvement of at least 2 s, so it still passes,
# by 0.055 s. Thin, and flagged as such on the gate surface.
CURRENT = {"A": "SB3", "B": "L8c"}


def predict_b(name: str, tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    """The named lane-B champion fit on tr and predicted on te, unfloored (the floor is the file's C1 option, H22)."""
    from taxi.baselines import b4
    if name == "B4":
        return b4(tr, te, "B")
    from taxi import l4, l6, l8  # local: each reaches taxi.w1, which reads CURRENT above
    if name in l4.CELLS:
        return l4.per_band(tr, te, l4.CELLS[name])
    if name in l8.CELLS:  # an L8 cell names the copy probability, not the non-copy fit: L6a's mixture carries it
        return l6.per_row(tr, te, l6.CELLS["L6a"], prob=l8.CELLS[name])
    return l6.per_row(tr, te, l6.CELLS[name])


def cache_path(name: str) -> str:
    return f"data/cache/{name.lower().replace('-', '')}_holdout.parquet"


def lane_a(window: float = WINDOW, lag: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lane-A training and holdout frames with weather joined, in the row order every lane-A verdict shares (window: TW1; lag: WS1)."""
    metar = load_metar()
    tr, te = (join(load(m, window), metar, lag) for m in (TRAIN_MONTHS, HOLDOUT_MONTHS))
    tra, tea = (d[d.has_fpl].reset_index(drop=True) for d in (tr, te))
    return tra, tea


def _fingerprint(tra: pd.DataFrame, tea: pd.DataFrame) -> int:
    # Row-wise hash of every column any champion reads, so a changed feature value invalidates the cache (critique S3, finding 1).
    cols = ["id", "y", *dict.fromkeys(c for cs, *_ in CHAMPIONS_A.values() for c in cs)]
    return int(sum(int(pd.util.hash_pandas_object(d[cols], index=False).sum()) for d in (tra, tea)) % (1 << 62))


def refit_predict(name: str, tr: pd.DataFrame, te: pd.DataFrame, rows: str = "ranking") -> np.ndarray:
    """The named champion refit on tr at its recorded rounds, predicted on te; the one lane-A fit path for verdicts and files.

    rows names the row set for a composed champion's own caches; month cannot separate them, since the ranking file is Jan and Jul too.
    """
    if name in COMPOSED_A:
        from importlib import import_module  # local: a composed champion imports this module
        return import_module(COMPOSED_A[name]).predict(tr, te, rows)
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
    pred = refit_predict(name, tra, tea, "holdout")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"id": tea.id.values, "pred": pred, "fp": fp}).to_parquet(path, index=False)
    return pred


def champion_oof(name: str, tra: pd.DataFrame) -> np.ndarray:
    """The named champion's out-of-fold month-pair CV predictions on the training months, run once and cached (E1, E2 pick their weight here)."""
    path, fp = cache_path(name).replace("_holdout", "_oof"), _fingerprint(tra, tra.iloc[:0])
    if os.path.exists(path):
        c = pd.read_parquet(path)
        if len(c) == len(tra) and c.fp.iloc[0] == fp and np.array_equal(c.id.values, tra.id.values):
            return c.pred.values
    cols, tag, doc, kind = CHAMPIONS_A[name]
    if kind == "cb":
        from taxi.mc1 import cv_a
        pred = cv_a(tra, cols, name).oof
    else:
        from taxi.w1 import cv_a
        pred = cv_a(tra, chosen_cells()["A"][0], cols=cols, name=tag)[2]
    pd.DataFrame({"id": tra.id.values, "pred": pred, "fp": fp}).to_parquet(path, index=False)
    return pred


if __name__ == "__main__":
    import sys
    tra, tea = lane_a()
    for name in [a for a in sys.argv[1:] if a != "oof"] or ["W1-A"]:
        if "oof" in sys.argv:
            print(name, len(champion_oof(name, tra)), "OOF predictions cached at", cache_path(name).replace("_holdout", "_oof"))
        else:
            print(name, len(champion_holdout(name, tra, tea)), "holdout predictions cached at", cache_path(name))
