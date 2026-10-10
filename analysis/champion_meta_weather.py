"""Bounded training-only simplex recipes on checked original member tables."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import recal
from analysis.contender_errors import error_rows
from campaign import file_hash, outer_masks, paired_metrics
from champion import frame_hash, ids_hash, write_record
from champion_neighbours import file_month
from champion_pipeline import MEMBERS, recal_frame

RUN = ROOT / "data/runs/clean-champion-20261003"
RECIPES = ("CP4-global-simplex", "CP4-weather-simplex")
LIMIT_SECONDS = 2700


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def deadline(start: float) -> None:
    if time.monotonic() - start >= LIMIT_SECONDS:
        raise TimeoutError("declared 45-minute CPU allocation reached")


def hot(rows: pd.DataFrame) -> np.ndarray:
    return ((rows.w_ts > 0) | (rows.w_sn > 0) | (rows.w_fz > 0) | (rows.w_vsby <= 1)).to_numpy(bool)


def simplex(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict]:
    require(len(x) > 0 and x.shape == (len(y), 3), "simplex needs three complete member columns")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "simplex input is nonfinite")
    differences, target = (x[:, 1:] - x[:, :1]) / 1000, (y - x[:, 0]) / 1000
    h, g = differences.T @ differences / len(y), differences.T @ target / len(y)
    result = minimize(lambda z: .5 * z @ h @ z - g @ z, np.zeros(2),
                      jac=lambda z: h @ z - g, method="SLSQP", bounds=((0, 1), (0, 1)),
                      constraints={"type": "ineq", "fun": lambda z: 1 - z.sum(), "jac": lambda z: -np.ones(2)},
                      options={"ftol": 1e-12, "maxiter": 200})
    require(result.success, f"simplex optimization failed: {result.message}")
    weights = np.maximum(np.r_[1 - result.x.sum(), result.x], 0)
    weights /= weights.sum()
    gradient = np.r_[0, h @ weights[1:] - g]
    active = weights > 1e-8
    level = gradient[active].mean()
    residual = max(float(np.max(np.abs(gradient[active] - level))),
                   float(np.max(np.maximum(level - gradient[~active], 0))) if (~active).any() else 0)
    tolerance = 1e-6 * max(1., float(np.abs(h).max()), float(np.abs(g).max()))
    require(residual <= tolerance, "simplex KKT optimality check failed")
    return weights, {"kkt_residual": residual, "kkt_tolerance": tolerance,
                     "optimizer": "SLSQP, C-relative quadratic, fixed C start", "iterations": result.nit}


def applied(weights: np.ndarray) -> np.ndarray:
    rounded = np.round(weights, 6)
    rounded[np.argmax(weights)] += 1 - rounded.sum()
    require((rounded >= 0).all() and np.isclose(rounded.sum(), 1, atol=1e-12), "rounded weights violate simplex")
    return rounded


def weight_vectors(training: pd.DataFrame) -> dict:
    global_weights, check = simplex(training[list(MEMBERS)].to_numpy(float), training.y.to_numpy(float))
    output = {"global": {"exact": global_weights.tolist(), "applied": applied(global_weights).tolist(), **check}, "cells": {}}
    mask = hot(training)
    for label, selected in (("hot", mask), ("ordinary", ~mask)):
        cell = training[(training.has_plan == 1) & selected]
        supported = len(cell) >= 5000 and cell.day.nunique() >= 30
        fitted, detail = simplex(cell[list(MEMBERS)].to_numpy(float), cell.y.to_numpy(float)) if supported else (global_weights, {})
        exact = .5 * (fitted + global_weights) if supported else global_weights
        output["cells"][label] = {"rows": len(cell), "days": int(cell.day.nunique()), "supported": supported,
                                    "fit_ids": ids_hash(cell.id), "fitted": fitted.tolist(), "exact": exact.tolist(),
                                    "applied": applied(exact).tolist(), **detail}
    return output


def blend(rows: pd.DataFrame, vectors: dict, weather: bool) -> np.ndarray:
    values = rows[list(MEMBERS)].to_numpy(float)
    output = values @ np.array(vectors["global"]["applied"])
    if weather:
        mask = hot(rows)
        for label, selected in (("hot", mask), ("ordinary", ~mask)):
            output[selected] = values[selected] @ np.array(vectors["cells"][label]["applied"])
    return np.maximum(output, 0)


def load_block(frame: pd.DataFrame, tag: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    directory = RUN / tag
    proof = read(directory / "evaluation-audit.json")
    require(all(proof[key] == "PASS" for key in ("scope_checks", "meta_reproduction", "recording_replay", "laneB_parity")), "reference proof failed")
    require(proof["months"] == list(map(int, tag.split("-"))), "reference proof has wrong block")
    path = directory / "evaluation.parquet"
    require(file_hash(path) == proof["output_hash"], "reference predictions changed")
    rows = pd.read_parquet(path, columns=list(frame.columns) + [*MEMBERS, "control", "clean_champion"])
    fit, query = outer_masks(frame, tuple(proof["months"]))
    expected, training = frame[query], frame[fit]
    pd.testing.assert_frame_equal(rows[list(frame.columns)].reset_index(drop=True), expected.reset_index(drop=True),
                                  check_dtype=False, check_categorical=False)
    require(np.array_equal(rows.id, expected.id) and rows.id.is_unique, "outer complete-file coverage changed")
    pieces, hashes = [], {}
    for item in proof["meta"]["assembly_proofs"]:
        proof_path = ROOT / item["path"]
        require(file_hash(proof_path) == item["sha256"], "saved inner proof changed")
        inner = read(proof_path)
        path = directory / ("meta-" + "-".join(map(str, inner["meta_group"]))) / "members.parquet"
        require(file_hash(path) == inner["output_hash"], "checked raw inner predictions changed")
        pieces.append(pd.read_parquet(path))
        hashes[str(path.relative_to(ROOT))] = inner["output_hash"]
    values = pd.concat(pieces, ignore_index=True)
    require(values.id.is_unique and set(values.id) == set(training.id), "meta table scope changed")
    training = training.merge(values, on="id", validate="one_to_one")
    require(not training.id.isin(rows.id).any(), "outer IDs entered meta fitting")
    require(frame_hash(training, ["id", "y", *MEMBERS]) == proof["meta"]["meta_input_hash"], "meta targets or raw values changed")
    for table in (training, rows):
        require(np.isfinite(table[["y", *MEMBERS]]).all().all(), "candidate member or label input is nonfinite")
        b = table.loc[table.has_plan == 0, list(MEMBERS)].to_numpy(float)
        require(np.array_equal(b[:, 0], b[:, 1]) and np.array_equal(b[:, 0], b[:, 2]), "mixture B columns differ")
    return training, rows, {"proof_hash": file_hash(directory / "evaluation-audit.json"), "inner_hashes": hashes,
                            "reference_hash": proof["output_hash"], "fit_ids": ids_hash(training.id),
                            "query_ids": ids_hash(rows.id), "fit_rows": len(training), "scope_checks": "PASS"}


def predictions(query: pd.DataFrame, vectors: dict, weather: bool, model: lgb.Booster) -> np.ndarray:
    hidden = query.assign(y=np.nan)
    frame = recal_frame(hidden, blend(hidden, vectors, weather))
    a = frame.has_plan == 1
    output = frame.p.to_numpy(float).copy()
    output[a] = np.maximum(output[a] + model.predict(frame.loc[a, recal.COLS]), 0)
    output[~a] = query.loc[~a, "control"]
    require(np.isfinite(output).all() and (output >= 0).all(), "candidate output is invalid")
    return output


def fit_recipe(training: pd.DataFrame, rows: pd.DataFrame, vectors: dict, weather: bool, directory: Path) -> tuple[np.ndarray, dict]:
    tr = recal_frame(training, blend(training, vectors, weather))
    a = tr.has_plan == 1
    model = lgb.LGBMRegressor(**recal.PARAMS).fit(tr.loc[a, recal.COLS], tr.loc[a, "y"] - tr.loc[a, "p"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "recal.model"
    model.booster_.save_model(str(path))
    output = predictions(rows, vectors, weather, model.booster_)
    replay = predictions(rows, vectors, weather, lgb.Booster(model_file=str(path)))
    np.testing.assert_allclose(output, replay, rtol=1e-7, atol=1e-4)
    mutation = predictions(rows.assign(y=rows.y * -13 + 1e9), vectors, weather, model.booster_)
    np.testing.assert_array_equal(output, mutation)
    np.testing.assert_array_equal(output[rows.has_plan == 0], rows.loc[rows.has_plan == 0, "control"])
    return output, {"recalibration_model_hash": file_hash(path), "params": recal.PARAMS,
                    "fit_rows": int(a.sum()), "fit_ids": ids_hash(tr.loc[a, "id"]),
                    "checkpoint_replay": "PASS", "query_label_mutation": "PASS", "laneB_parity": "PASS"}


def compare(rows: pd.DataFrame, name: str) -> dict:
    result = {}
    for reference in ("control", "clean_champion"):
        metric = paired_metrics(rows, rows[reference].values, rows[name].values)
        mask = hot(rows)
        metric["hot"] = paired_metrics(rows[mask], rows.loc[mask, reference].values, rows.loc[mask, name].values)
        result[reference] = metric
    return result


def run_block(frame: pd.DataFrame, tag: str, start: float) -> dict:
    deadline(start)
    training, rows, provenance = load_block(frame, tag)
    vectors = weight_vectors(training)
    directory = RUN / "meta-weather" / tag
    write_record(directory / "weights.json", vectors)
    result = {"months": list(map(int, tag.split("-"))), "inputs": provenance, "weights": vectors, "recipes": {}}
    for name in RECIPES:
        deadline(start)
        began = time.monotonic()
        weather = name == RECIPES[1]
        rows[name], check = fit_recipe(training, rows, vectors, weather, directory / name)
        path = directory / name / "predictions.parquet"
        rows[["id", name]].to_parquet(path)
        metric_rows = rows.assign(month=file_month(rows))
        deadline(start)
        result["recipes"][name] = {"checks": check, "output_hash": file_hash(path),
                                     "metrics": compare(metric_rows, name), "errors": error_rows(metric_rows, name),
                                     "runtime_seconds": time.monotonic() - began}
        write_record(directory / "progress.json", result)
        logging.info("%s %s complete: %s", tag, name, result["recipes"][name]["metrics"]["control"]["gain"])
    rows.to_parquet(directory / "evaluation.parquet")
    result["output_hash"] = file_hash(directory / "evaluation.parquet")
    write_record(directory / "complete.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    start = time.monotonic()
    source = ROOT / "data/runs/contender-20261003"
    require(file_hash(source / "cold_features.parquet") == read(source / "inputs.json")["feature_hash"], "cold input changed")
    columns = ["id", "file", "day", "month", "has_plan", "apt", "y", *recal.Z, "w_ts", "w_sn", "w_fz", "w_vsby"]
    frame = pd.read_parquet(source / "cold_features.parquet", columns=columns)
    record = {"declared_plan_hash": file_hash(ROOT / "docs/META-WEATHER-2026-10-05.html"), "recipes": RECIPES,
              "script_hash": file_hash(Path(__file__)), "metric_hash": file_hash(ROOT / "campaign.py"),
              "cpu_limit_seconds": LIMIT_SECONDS, "gpu_fit_seconds": 0, "blocks": [], "status": "running"}
    for tag in ("1-7", "2-3"):
        record["blocks"].append(run_block(frame, tag, start))
        write_record(RUN / "meta-weather/progress.json", record)
    deadline(start)
    record.update(status="complete", checked_utc=datetime.now(timezone.utc).isoformat(), runtime_seconds=time.monotonic() - start)
    write_record(RUN / "meta-weather/results.json", record)
    args.output.mkdir(parents=True, exist_ok=True)
    write_record(args.output / "prc26-meta-weather-results.json", record)
    logging.info("Declared meta comparison complete; development qualification remains to be assessed")


if __name__ == "__main__":
    main()
