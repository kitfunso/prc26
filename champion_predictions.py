"""Raw checkpoint predictions preserve the original OOF blend arithmetic."""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

import nn
from campaign import file_hash
from champion import write_record


def raw_member_predictions(name: str, cfg: dict, query: pd.DataFrame, directory: Path) -> np.ndarray:
    query = query.assign(y=np.nan)
    if name.startswith("nn"):
        saved = torch.load(directory / "refit.pt", map_location=nn.DEV, weights_only=False)
        encoder = saved["encoder"]
        ci, xn = encoder(query)
        model = nn.Net(encoder, xn.shape[1], saved["config"]["cell"]).to(nn.DEV)
        model.load_state_dict(saved["state_dict"])
        p = nn._infer(model, ci, xn) * saved["sd"] + saved["mu"]
        del model, saved, ci, xn
        torch.cuda.empty_cache()
    else:
        columns = cfg["cat"] + cfg["num"]
        if cfg["clf"]["family"] == "lgb":
            gate, tail, body = [lgb.Booster(model_file=str(directory / "refit" / f"{head}.model"))
                                for head in ("gate", "tail", "body")]
            probability = gate.predict(query[columns])
            pt, pb = tail.predict(query[columns]), body.predict(query[columns])
        else:
            gate = CatBoostClassifier().load_model(str(directory / "refit/gate.model"))
            tail = CatBoostRegressor().load_model(str(directory / "refit/tail.model"))
            body = CatBoostRegressor().load_model(str(directory / "refit/body.model"))
            pool = Pool(query[columns], cat_features=cfg["cat"])
            probability, pt, pb = gate.predict_proba(pool)[:, 1], tail.predict(pool), body.predict(pool)
        p = probability * pt + (1 - probability) * pb
    raw = p + query.z_aobt.values
    saved_predictions = pd.read_parquet(directory / "predictions.parquet")
    if not np.array_equal(saved_predictions.id.values, query.id.values):
        raise ValueError("raw checkpoint query IDs changed")
    np.testing.assert_allclose(np.maximum(raw, 0), saved_predictions.p.values, rtol=1e-7, atol=1e-4)
    pd.DataFrame({"id": query.id.values, "p": raw}).to_parquet(directory / "raw_predictions.parquet")
    write_record(directory / "raw_predictions.json", {"prediction_floor": False, "floor_parity": "PASS",
                                                       "raw_hash": file_hash(directory / "raw_predictions.parquet"),
                                                       "floored_hash": file_hash(directory / "predictions.parquet"),
                                                       "replay_code_hash": file_hash(Path(__file__))})
    return raw
