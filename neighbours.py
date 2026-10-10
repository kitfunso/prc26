"""Neighbour donor predictions rebuilt inside each declared training scope."""
from __future__ import annotations

import hashlib
from collections.abc import Callable

import numpy as np
import pandas as pd

from campaign import prediction_windows
from features import ARR, CAT, DAY, NUM, OCC, WX, WZ
from train import fit_predict


def donor(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    cfg = {"family": "lgb", "cat": CAT, "num": NUM + DAY + OCC + WX + ARR + WZ,
           "params": {"objective": "regression", "learning_rate": .08, "num_leaves": 63,
                      "min_child_samples": 100, "n_estimators": 80, "n_jobs": 12,
                      "verbose": -1, "random_state": 0}}
    p, _ = fit_predict(cfg, tr.assign(y=tr.y - tr.z_aobt), te, iters=80)
    return np.maximum(p + te.z_aobt.values, 0)


def scoped_predictions(f: pd.DataFrame, fit_ids: pd.Index, predict: Callable = donor) -> tuple[pd.Series, list[dict]]:
    fit_ids = pd.Index(fit_ids)
    if not fit_ids.is_unique or not fit_ids.isin(f.id).all():
        raise ValueError("donor fit IDs must be unique and present")
    fit = f.id.isin(fit_ids) & (f.has_plan == 1)
    months = sorted(f.loc[fit, "month"].unique())
    if len(months) < 3:
        raise ValueError("month cross-fitting needs at least three training months")
    p = pd.Series(0., index=f.id.values)
    scopes = []
    for group in [months[i::3] for i in range(3)] + [None]:
        tr = f[fit & ~f.month.isin(group)] if group is not None else f[fit]
        query = f[fit & f.month.isin(group)] if group is not None else f[~fit & (f.has_plan == 1)]
        if query.empty:
            continue
        if tr.id.isin(query.id).any():
            raise ValueError("donor query IDs overlap fitted labels")
        p.loc[query.id] = predict(tr, query.assign(y=np.nan))
        scopes.append({"crossfit_months": None if group is None else [int(m) for m in group],
                       "fit_rows": len(tr), "query_rows": len(query),
                       "fit_ids_hash": hashlib.sha256(tr.id.to_numpy().tobytes()).hexdigest(),
                       "query_ids_hash": hashlib.sha256(query.id.to_numpy().tobytes()).hexdigest()})
    if not np.isfinite(p).all():
        raise ValueError("donor returned nonfinite predictions")
    return p, scopes


def with_scoped_pw(f: pd.DataFrame, fit_ids: pd.Index) -> tuple[pd.DataFrame, list[dict]]:
    p, scopes = scoped_predictions(f, fit_ids)
    return pd.concat([f, prediction_windows(f, p)], axis=1), scopes
