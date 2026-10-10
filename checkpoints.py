"""Validate and replay native tree checkpoints before reusing a completed fit."""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool


def tree_prediction(cfg: dict, query: pd.DataFrame, path: Path, classifier: bool,
                    iterations: int) -> np.ndarray:
    columns = cfg["cat"] + cfg["num"]
    family = cfg["clf"]["family"] if classifier else cfg.get("family", "cb")
    if family == "lgb":
        model = lgb.Booster(model_file=str(path))
        names, count = model.feature_name(), model.current_iteration()
        prediction = model.predict(query[columns])
        if classifier:
            prediction = np.column_stack((1 - prediction, prediction))
    elif family == "cb":
        model = (CatBoostClassifier() if classifier else CatBoostRegressor()).load_model(str(path))
        names, count = model.feature_names_, model.tree_count_
        pool = Pool(query[columns], cat_features=cfg["cat"])
        prediction = model.predict_proba(pool) if classifier else model.predict(pool)
    else:
        raise ValueError(f"checkpoint reuse is unsupported for {family}")
    if names != columns or count != iterations:
        raise ValueError(f"checkpoint features or tree budget changed: {path}")
    if not np.isfinite(prediction).all():
        raise ValueError(f"checkpoint returned nonfinite predictions: {path}")
    return prediction
