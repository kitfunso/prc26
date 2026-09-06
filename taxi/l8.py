"""L8: learned copy probability at the lane-B gate. An oracle P(copy) is worth +182.9 s on the lane-B holdout against +45.0 s for an oracle
non-copy fit, and three measured separators never reach the probability: stand occupancy (0.409 free against 0.725 occupied), its age
(0.409 / 0.537 / 0.777 / 0.759 by hour) and hour of day (0.446 at 08-11 against 0.886 at 00-03) on LIRF's 1,090 no-type training rows."""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression

from taxi.l3 import COPY
from taxi.l4 import band
from taxi.l5 import NB, PSEUDO, Prob, is_copy, judge, shrunk
from taxi.l6 import CELLS as L6_CELLS, per_row
from taxi.l7 import EPS, WINTER, _season

CB_KW = dict(iterations=300, depth=3, learning_rate=0.05, l2_leaf_reg=10.0, random_seed=0, verbose=False, allow_writing_files=False)
GBM_COLS = ["z_sched", "occ", "t_occ_h", "hour", "winter", "band"]


def _odds(v: np.ndarray | float) -> np.ndarray:
    p = np.clip(v, EPS, 1 - EPS)
    return p / (1 - p)


def _design(d: pd.DataFrame) -> np.ndarray:
    """Band one-hot, the stand-occupancy flag and its age in hours, hour of day as a circle, and the winter flag."""
    zb, h = band(d.z_sched.values), d.hour.values.astype(float)
    return np.column_stack([*(np.asarray(zb == b, float) for b in range(NB)), d.occ.values.astype(float),
                            d.t_occ.fillna(0.0).values / 3600.0, np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), _season(d)])


def _frame(d: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"z_sched": d.z_sched.values, "occ": d.occ.values.astype(float), "t_occ_h": d.t_occ.fillna(0.0).values / 3600.0,
                         "hour": d.hour.values.astype(float), "winter": _season(d), "band": band(d.z_sched.values).astype(float),
                         "airline": d.airline.astype(str).values})


def logistic(t: pd.DataFrame, te: pd.DataFrame, copy_s: float = COPY) -> np.ndarray:
    """L8a: logistic regression on the measured separators, no airline."""
    c = is_copy(t, copy_s)
    if len(np.unique(c)) < 2:
        return np.full(len(te), float(c.mean()))
    return LogisticRegression(C=1.0, max_iter=2000).fit(_design(t), c).predict_proba(_design(te))[:, 1]


def logistic_airline(t: pd.DataFrame, te: pd.DataFrame, copy_s: float = COPY) -> np.ndarray:
    """L8b: L8a's odds times the odds ratio of L5a's shrunk airline x band table against the pooled rate."""
    c = is_copy(t, copy_s)
    if len(np.unique(c)) < 2:
        return np.full(len(te), float(c.mean()))
    o = _odds(logistic(t, te, copy_s)) * _odds(shrunk(t, te)) / _odds(float(c.mean()))
    return o / (1 + o)


def gbm(t: pd.DataFrame, te: pd.DataFrame, copy_s: float = COPY) -> np.ndarray:
    """L8c: a shallow CatBoost classifier on the same columns with the airline as a categorical."""
    c = is_copy(t, copy_s)
    if len(np.unique(c)) < 2:
        return np.full(len(te), float(c.mean()))
    cols = [*GBM_COLS, "airline"]
    m = CatBoostClassifier(**CB_KW).fit(Pool(_frame(t)[cols], c.astype(int), cat_features=["airline"]))
    return m.predict_proba(Pool(_frame(te)[cols], cat_features=["airline"]))[:, 1]


CELLS: dict[str, Prob] = {"L8a": logistic, "L8b": logistic_airline, "L8c": gbm}


def _pred(tr: pd.DataFrame, te: pd.DataFrame, prob: Prob) -> np.ndarray:
    return per_row(tr, te, L6_CELLS["L6a"], prob=prob)


def _champion(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    return per_row(tr, te, L6_CELLS["L6a"])


def main() -> None:
    judge(CELLS, _pred, "L8: lane-B learned copy probability",
          f"Regenerate with `python -m taxi.l8`. L6a's gated mixture throughout; only P(copy) changes, and it is fit per touched airport on that "
          f"airport's no-type rows of the fit months. Columns: band one-hot, the stand-occupancy flag, its age in hours (0 when free), hour of day "
          f"as sine and cosine, and the Nov-Apr flag ({WINTER}). L8a: logistic regression on those, no airline. L8b: L8a's odds times the odds "
          f"ratio of L5a's shrunk airline x band table ({PSEUDO:.0f} pseudo-rows) against the pooled rate. L8c: CatBoost classifier, depth 3, 300 "
          f"rounds, learning rate 0.05, l2 10, on z_sched, the occupancy flag and age, hour, the winter flag, the band and the airline as a "
          f"categorical. A cell whose fit rows are all one class returns their mean.",
          "docs/L8.md", champ=("L6a", _champion), k=4)


if __name__ == "__main__":
    main()
