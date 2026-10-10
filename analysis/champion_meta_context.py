"""One fixed residual-context recipe using preserved original blend arithmetic."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import recal
from analysis.champion_meta_weather import RUN, compare, load_block, read, require
from analysis.contender_errors import error_rows
from campaign import file_hash
from champion import frame_hash, ids_hash, write_record
from champion_neighbours import file_month
from champion_pipeline import MEMBERS, recal_frame

NAME = "CP4-context-residual"
EXTRA = ("w_ts", "w_sn", "w_fz", "w_vsby", "c_disagreement", "l_disagreement", "n_disagreement")
COLS = [*recal.COLS, *EXTRA]


def context_frame(rows: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    base = recal_frame(rows, rows[list(MEMBERS)].to_numpy(float) @ weights)
    require(np.isfinite(base.p).all(), "original blend is nonfinite")
    baseline = base[recal.COLS].copy()
    for column, member in zip(EXTRA[4:], MEMBERS):
        base[column] = base[member] - base.p
    pd.testing.assert_frame_equal(base[recal.COLS], baseline)
    return base


def predict(rows: pd.DataFrame, weights: np.ndarray, model: lgb.Booster) -> np.ndarray:
    query = context_frame(rows.assign(y=np.nan), weights)
    a = query.has_plan == 1
    output = query.p.to_numpy(float).copy()
    output[a] = np.maximum(output[a] + model.predict(query.loc[a, COLS]), 0)
    output[~a] = rows.loc[~a, "control"]
    require(np.isfinite(output).all() and (output >= 0).all(), "context prediction is invalid")
    return output


def fit_context(training: pd.DataFrame, rows: pd.DataFrame, weights: np.ndarray, directory: Path) -> tuple[np.ndarray, dict]:
    tr = context_frame(training, weights)
    a = tr.has_plan == 1
    require(not tr.id.isin(rows.id).any(), "outer labels entered context fitting")
    model = lgb.LGBMRegressor(**recal.PARAMS).fit(tr.loc[a, COLS], tr.loc[a, "y"] - tr.loc[a, "p"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "recal.model"
    model.booster_.save_model(str(path))
    saved = lgb.Booster(model_file=str(path))
    require(saved.feature_name() == COLS and saved.num_trees() == 300, "context feature or tree count changed")
    output = predict(rows, weights, model.booster_)
    np.testing.assert_allclose(output, predict(rows, weights, saved), rtol=1e-7, atol=1e-4)
    np.testing.assert_array_equal(output, predict(rows.assign(y=rows.y * -11 + 1e9), weights, model.booster_))
    np.testing.assert_array_equal(output[~(rows.has_plan == 1)], rows.loc[rows.has_plan == 0, "control"])
    return output, {"checkpoint_replay": "PASS", "query_label_mutation": "PASS", "laneB_parity": "PASS",
                    "baseline_constructor": "PASS", "weights": weights.tolist(), "params": recal.PARAMS,
                    "ordered_columns": COLS, "input_hash": frame_hash(tr.loc[a], ["id", "y", *COLS]),
                    "fit_ids": ids_hash(tr.loc[a, "id"]), "fit_rows": int(a.sum()), "model_hash": file_hash(path)}


def deadline(start: float) -> None:
    if time.monotonic() - start >= 900:
        raise TimeoutError("declared 15-minute CPU allocation reached")


def run_block(frame: pd.DataFrame, tag: str, start: float) -> dict:
    deadline(start)
    training, rows, provenance = load_block(frame, tag)
    proof = read(RUN / tag / "evaluation-audit.json")
    weights = np.array([proof["meta"]["applied_weights"][member] for member in MEMBERS])
    require(np.array_equal(weights, [read(RUN / tag / "meta/meta.json")["applied_weights"][m] for m in MEMBERS]), "original weights changed")
    original = pd.read_parquet(RUN / tag / "evaluation.parquet", columns=["id", "p", *[f"d_{z}" for z in recal.Z]])
    constructed = context_frame(rows, weights)
    require(np.array_equal(original.id, rows.id), "baseline constructor IDs changed")
    pd.testing.assert_frame_equal(constructed[original.columns].reset_index(drop=True), original.reset_index(drop=True))
    directory = RUN / "meta-context" / tag / NAME
    began = time.monotonic()
    rows[NAME], check = fit_context(training, rows, weights, directory)
    path = directory / "predictions.parquet"
    rows[["id", NAME]].to_parquet(path)
    deadline(start)
    metric_rows = rows.assign(month=file_month(rows))
    result = {"months": list(map(int, tag.split("-"))), "inputs": provenance,
              "recipes": {NAME: {"checks": check, "output_hash": file_hash(path), "metrics": compare(metric_rows, NAME),
                                   "errors": error_rows(metric_rows, NAME), "runtime_seconds": time.monotonic() - began}}}
    rows.to_parquet(directory.parent / "evaluation.parquet")
    result["output_hash"] = file_hash(directory.parent / "evaluation.parquet")
    write_record(directory.parent / "complete.json", result)
    logging.info("%s context recipe complete: %s", tag, result["recipes"][NAME]["metrics"]["clean_champion"]["gain"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    start = time.monotonic()
    source = ROOT / "data/runs/contender-20261003"
    require(file_hash(source / "cold_features.parquet") == read(source / "inputs.json")["feature_hash"], "cold input changed")
    columns = ["id", "file", "day", "month", "has_plan", "apt", "y", *recal.Z, *EXTRA[:4]]
    frame = pd.read_parquet(source / "cold_features.parquet", columns=columns)
    result = {"declared_plan_hash": file_hash(ROOT / "docs/META-CONTEXT-2026-10-05.html"), "recipes": [NAME],
              "script_hash": file_hash(Path(__file__)), "metric_hash": file_hash(ROOT / "campaign.py"),
              "cpu_limit_seconds": 900, "gpu_fit_seconds": 0, "status": "running", "blocks": []}
    for tag in ("1-7", "2-3"):
        result["blocks"].append(run_block(frame, tag, start))
        write_record(RUN / "meta-context/progress.json", result)
    deadline(start)
    result.update(status="complete", runtime_seconds=time.monotonic() - start, checked_utc=datetime.now(timezone.utc).isoformat())
    write_record(RUN / "meta-context/results.json", result)
    write_record(args.output / "prc26-meta-context-results.json", result)


if __name__ == "__main__":
    main()
