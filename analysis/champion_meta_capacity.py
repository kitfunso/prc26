"""One declared meta-source comparison using saved selection checkpoints."""
from __future__ import annotations

import argparse
import ast
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nn
import recal
from analysis.champion_checkpoint_audit import donor_windows
from analysis.champion_meta_weather import RUN, applied, compare, fit_recipe, load_block, read, require, simplex
from analysis.contender_errors import error_rows
from campaign import file_hash, outer_masks
from champion import configuration, frame_hash, ids_hash, replay_neural, replay_tree, write_record
from champion_neighbours import file_month
from champion_pipeline import MEMBERS
from checkpoints import tree_prediction
from features import CAT, PW
from train import stopping_split

RECIPE = "CP4-capacity-simplex"
NAMES = ("ra_cb_wz_t7", "ra_lgb_pw_t7", "nn3_pw_t7")
DEST = RUN / "meta-capacity"


def deadline(start: float, gpu_seconds: float) -> None:
    require(time.monotonic() - start < 1800, "declared 30-minute wall allocation reached")
    require(gpu_seconds < 900, "declared 15-minute neural inference allocation reached")


def metric_compatible(original: str) -> bool:
    archive = RUN / "metric-repair/campaign-before.py"
    require(file_hash(archive) == original, "archived metric source changed")
    def other(path: Path) -> str:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        tree.body = [node for node in tree.body if not isinstance(node, ast.FunctionDef) or node.name != "paired_metrics"]
        return ast.dump(tree, include_attributes=False)
    return other(archive) == other(ROOT / "campaign.py")


def source_check(directory: Path, selected: dict, complete: dict) -> dict:
    spec = selected["specification"]
    engine, recovery = spec["engine_hashes"], {}
    if (directory / "recovery.json").exists():
        require(directory == RUN / "1-7/outer/ra_cb_wz_t7", "undeclared source recovery")
        recovery = read(directory / "recovery.json")
        require(recovery["original_specification"] == spec, "original recovery specification changed")
        require({k: v for k, v in spec.items() if k != "engine_hashes"} ==
                {k: v for k, v in complete["specification"].items() if k != "engine_hashes"}, "recovered selection scope changed")
        for path, digest in engine.items():
            archive = directory / f"original-{path}.txt"
            original = archive if archive.exists() else (RUN / "metric-repair/campaign-before.py" if path == "campaign.py" else ROOT / path)
            require(file_hash(original) == digest, f"original recovery source changed: {path}")
            git_source = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{recovery['original_source_commit']}:{path}"])
            require(ast.dump(ast.parse(original.read_text(encoding="utf-8-sig")), include_attributes=False) ==
                    ast.dump(ast.parse(git_source.decode("utf-8-sig")), include_attributes=False), f"original recovery syntax changed: {path}")
        engine = recovery["resumed_engine_hashes"]
        compatibility = recovery["completed_engine_compatibility"]
        require(compatibility["execution_engine_hashes"] == complete["specification"]["engine_hashes"] and
                compatibility["resumed_engine_hashes"] == engine, "completed source compatibility changed")
    for path, digest in engine.items():
        require(file_hash(ROOT / path) == digest or (path == "campaign.py" and metric_compatible(digest)),
                f"selection source bytes changed: {path}")
    require(all(version(name) == value for name, value in spec["versions"].items()), "selection dependency version changed")
    return {"checks": "PASS", "selection_hash": file_hash(directory / "selection.json"),
            "complete_hash": file_hash(directory / "complete.json"),
            "recovery_hash": file_hash(directory / "recovery.json") if recovery else None,
            "original_engine_hashes": spec["engine_hashes"], "accepted_engine_hashes": engine}


def native_check(directory: Path, selected: dict, complete: dict) -> dict:
    names = ("selection.pt",) if directory.name.startswith("nn") else tuple(f"selection/{head}.model" for head in ("gate", "tail", "body"))
    hashes = {key.replace("\\", "/"): value for key, value in complete["artifact_hashes"].items()}
    for name in names:
        require(name in hashes and file_hash(directory / name) == hashes[name], f"selection native bytes changed: {name}")
    return {name: hashes[name] for name in names}


def selection_rows(tag: str, name: str, selected: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cfg, spec = selected["specification"]["config"], selected["specification"]
    require(cfg == configuration(name), "selection configuration changed")
    columns = list(dict.fromkeys(["id", "z_aobt", *cfg["cat"], *cfg["num"]]))
    stored = list(dict.fromkeys([c for c in columns if c not in PW] + ["file", "day", "has_plan", "y"]))
    frame = pd.read_parquet(ROOT / "data/runs/contender-20261003/cold_features.parquet", columns=stored)
    for col in CAT:
        if col in frame:
            frame[col] = frame[col].astype("category")
    months = tuple(map(int, tag.split("-")))
    fit_mask, query_mask = outer_masks(frame, months)
    fit_mask, query_mask = fit_mask & (frame.has_plan == 1), query_mask & (frame.has_plan == 1)
    require(ids_hash(frame.loc[fit_mask, "id"]) == spec["fit_ids"] and ids_hash(frame.loc[query_mask, "id"]) == spec["query_ids"], "selection outer scope changed")
    donor = {}
    if any(c in PW for c in columns):
        windows, donor = donor_windows(frame, frame[fit_mask], True, months)
        frame[PW] = windows[PW]
    fitting = frame[fit_mask]
    require(frame_hash(fitting, [*columns, "y"]) == spec["selection_input_hash"], "selection input fingerprint changed")
    gradient, stop = stopping_split(fitting)
    require(not gradient.id.isin(stop.id).any() and not gradient.day.isin(stop.day).any() and
            not frame.loc[query_mask, "id"].isin(fitting.id).any(), "gradient/stopping/outer scope overlap")
    if name.startswith("nn"):
        require(selected["selection"]["fit_rows"] == len(gradient) and selected["selection"]["fit_days"] == gradient.day.nunique(), "neural gradient scope changed")
    else:
        require(selected["inner_fit_ids_hash"] == ids_hash(gradient.id) and selected["stopping_ids_hash"] == ids_hash(stop.id), "tree stopping scope changed")
        require(selected["inner_fit_rows"] == len(gradient) and selected["stopping_rows"] == len(stop), "tree stopping coverage changed")
    return gradient, stop, {"input_hash": spec["selection_input_hash"], "gradient_ids": ids_hash(gradient.id),
                            "stopping_ids": ids_hash(stop.id), "gradient_rows": len(gradient), "stopping_rows": len(stop),
                            "stopping_days": int(stop.day.nunique()), "query_input_hash": frame_hash(stop, columns), "donor": donor}


def neural_prediction(directory: Path, selected: dict, gradient: pd.DataFrame, stop: pd.DataFrame) -> np.ndarray:
    saved = torch.load(directory / "selection.pt", map_location="cpu", weights_only=False)
    cfg = selected["specification"]["config"]
    require(saved["config"] == cfg and saved["run_info"] == selected["selection"] and
            saved["run_info"]["best_epoch"] == selected["epochs"], "neural selected epoch or configuration changed")
    expected, enc = nn.Enc(gradient, cfg["cat"], cfg["num"]), saved["encoder"]
    require((enc.cats, enc.nums, enc.maps, enc.card, enc.miss) ==
            (expected.cats, expected.nums, expected.maps, expected.card, expected.miss), "neural encoder scope changed")
    pd.testing.assert_series_equal(enc.mu, expected.mu)
    pd.testing.assert_series_equal(enc.sd, expected.sd)
    target = (gradient.y - gradient.z_aobt).values
    require(saved["mu"] == float(target.mean()) and saved["sd"] == max(float(target.std()), 1e-6), "neural target scale changed")
    ci, xn = enc(stop.assign(y=np.nan))
    model = nn.Net(enc, xn.shape[1], cfg["cell"]).to(nn.DEV)
    model.load_state_dict(saved["state_dict"])
    prediction = np.maximum(nn._infer(model, ci, xn) * saved["sd"] + saved["mu"] + stop.z_aobt.values, 0)
    del model, saved
    torch.cuda.empty_cache()
    replay = replay_neural(stop.assign(y=-1e9), directory / "selection.pt")
    np.testing.assert_allclose(prediction, replay, rtol=1e-7, atol=1e-4)
    return prediction


def selected_values(tag: str, start: float, gpu_seconds: float) -> tuple[pd.DataFrame, dict, float]:
    output, checks = None, {}
    for name, member in zip(NAMES, MEMBERS):
        deadline(start, gpu_seconds)
        directory = RUN / tag / "outer" / name
        selected, complete = read(directory / "selection.json"), read(directory / "complete.json")
        proof = source_check(directory, selected, complete)
        proof["native_hashes"] = native_check(directory, selected, complete)
        gradient, stop, proof["scope"] = selection_rows(tag, name, selected)
        began = time.monotonic()
        if name.startswith("nn"):
            values = neural_prediction(directory, selected, gradient, stop)
            gpu_seconds += time.monotonic() - began if nn.DEV == "cuda" else 0
        else:
            cfg = selected["specification"]["config"]
            query = stop.assign(y=np.nan)
            heads = [tree_prediction(cfg if i == 0 else {**cfg["reg"], "family": cfg["clf"]["family"]}, query,
                                     directory / "selection" / f"{head}.model", i == 0, selected["iterations"][i])
                     for i, head in enumerate(("gate", "tail", "body"))]
            values = np.maximum(heads[0][:, 1] * heads[1] + (1 - heads[0][:, 1]) * heads[2] + stop.z_aobt.values, 0)
            np.testing.assert_allclose(values, replay_tree(cfg, stop.assign(y=-1e9), directory / "selection"), rtol=1e-7, atol=1e-4)
        require(np.isfinite(values).all() and (values >= 0).all(), "selected prediction invalid")
        if output is None:
            output = stop[["id", "file", "day", "has_plan", "y"]].copy()
        require(np.array_equal(output.id, stop.id), "members do not share ordered stopping IDs")
        output[member] = values
        proof.update(checkpoint_replay="PASS", query_label_mutation="PASS", seconds=time.monotonic() - began)
        checks[name] = proof
        write_record(DEST / tag / "input-progress.json", checks)
        logging.info("%s selection %s replay checked", tag, name)
    deadline(start, gpu_seconds)
    output.to_parquet(DEST / tag / "stopping-members.parquet")
    return output, checks, gpu_seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    start, gpu_seconds = time.monotonic(), 0.
    source = ROOT / "data/runs/contender-20261003"
    require(file_hash(source / "cold_features.parquet") == read(source / "inputs.json")["feature_hash"], "cold input changed")
    frame = pd.read_parquet(source / "cold_features.parquet", columns=["id", "file", "day", "month", "has_plan", "apt", "y", *recal.Z, "w_ts", "w_sn", "w_fz", "w_vsby"])
    record = {"status": "running", "recipes": [RECIPE], "blocks": [], "gpu_fit_seconds": 0,
              "declared_plan_hash": file_hash(ROOT / "docs/META-CAPACITY-2026-10-05.html"),
              "script_hash": file_hash(Path(__file__)), "metric_hash": file_hash(ROOT / "campaign.py"),
              "wall_limit_seconds": 1800, "gpu_inference_limit_seconds": 900}
    for tag in ("1-7", "2-3"):
        training, rows, inputs = load_block(frame, tag)
        stopping, checks, gpu_seconds = selected_values(tag, start, gpu_seconds)
        weights, optimality = simplex(stopping[list(MEMBERS)].values, stopping.y.values)
        vectors = {"global": {"exact": weights.tolist(), "applied": applied(weights).tolist(), **optimality}}
        directory = DEST / tag
        write_record(directory / "weights.json", vectors)
        deadline(start, gpu_seconds)
        began = time.monotonic()
        rows[RECIPE], native = fit_recipe(training, rows, vectors, False, directory / RECIPE)
        path = directory / RECIPE / "predictions.parquet"
        rows[["id", RECIPE]].to_parquet(path)
        deadline(start, gpu_seconds)
        metric_rows = rows.assign(month=file_month(rows))
        block = {"months": list(map(int, tag.split("-"))), "inputs": inputs, "selection_checks": checks,
                 "stopping_prediction_hash": file_hash(directory / "stopping-members.parquet"),
                 "weight_fit_ids": ids_hash(stopping.id), "weight_input_hash": frame_hash(stopping, ["id", "y", *MEMBERS]),
                 "weights": vectors, "recipes": {RECIPE: {"checks": native, "output_hash": file_hash(path),
                 "metrics": compare(metric_rows, RECIPE), "errors": error_rows(metric_rows, RECIPE), "runtime_seconds": time.monotonic() - began}}}
        rows.to_parquet(directory / "evaluation.parquet")
        block["output_hash"] = file_hash(directory / "evaluation.parquet")
        write_record(directory / "complete.json", block)
        record["blocks"].append(block)
        write_record(DEST / "progress.json", record)
        logging.info("%s capacity source comparison measured", tag)
    deadline(start, gpu_seconds)
    record.update(status="complete", checked_utc=datetime.now(timezone.utc).isoformat(),
                  runtime_seconds=time.monotonic() - start, gpu_inference_seconds=gpu_seconds)
    write_record(DEST / "results.json", record)
    write_record(args.output / "prc26-meta-capacity-results.json", record)


if __name__ == "__main__":
    main()
