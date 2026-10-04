"""Check the saved full reference and describe complete-file errors."""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import laneb
import recal
from analysis.champion_donor_audit import read
from analysis.champion_screen import OUT, inputs
from analysis.contender_errors import error_rows
from campaign import file_hash, outer_masks, paired_metrics
from champion import frame_hash, ids_hash, write_record
from champion_laneb import component_spec, replay_recording
from champion_neighbours import file_month
from champion_pipeline import MEMBERS, recal_frame


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def checked_rows(frame: pd.DataFrame, directory: Path, complete: dict, months: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit, query = outer_masks(frame, months)
    train, expected = frame[fit], frame[query]
    rows = pd.read_parquet(directory / "evaluation.parquet")
    require(complete["status"] == "complete" and tuple(complete["months"]) == months, "block is not complete")
    require(complete["fit_ids"] == ids_hash(train.id) and complete["query_ids"] == ids_hash(expected.id), "block scope changed")
    require(complete["output_hash"] == file_hash(directory / "evaluation.parquet"), "evaluation bytes changed")
    require(rows.id.is_unique and np.array_equal(rows.id.values, expected.id.values), "evaluation file coverage changed")
    require(np.isfinite(rows[["y", "control", "clean_champion"]]).all().all(), "evaluation values are nonfinite")
    require((rows[["control", "clean_champion"]] >= 0).all().all(), "evaluation predictions are negative")
    pd.testing.assert_frame_equal(rows[frame.columns].reset_index(drop=True), expected.reset_index(drop=True),
                                  check_dtype=False, check_categorical=False)
    outer = pd.read_parquet(directory / "outer/members.parquet")
    require(np.array_equal(outer.id.values, expected.id.values), "outer member coverage changed")
    np.testing.assert_array_equal(rows[list(MEMBERS)].values, outer[list(MEMBERS)].values)
    return train, rows, outer


def checked_meta(train: pd.DataFrame, rows: pd.DataFrame, directory: Path, complete: dict) -> dict:
    pieces, proofs = [], []
    for group in complete["meta_groups"]:
        tag = "meta-" + "-".join(map(str, group))
        proof_path = OUT / f"{tag}-assembly-audit.json"
        proof = read(proof_path)
        path = directory / tag / "members.parquet"
        require(all(proof[key] == "PASS" for key in ("scope_checks", "native_lineage_checks", "raw_floor_parity", "member_assembly", "laneB_parity")), "inner assembly proof failed")
        require(proof["output_hash"] == file_hash(path), "checked inner assembly bytes changed")
        pieces.append(pd.read_parquet(path))
        proofs.append({"path": str(proof_path.relative_to(ROOT)), "sha256": file_hash(proof_path)})
    values = pd.concat(pieces, ignore_index=True)
    require(values.id.is_unique and set(values.id) == set(train.id), "NNLS training coverage changed")
    training = train.merge(values, on="id", validate="one_to_one")
    weights, _ = nnls(training[list(MEMBERS)].to_numpy(float), training.y.to_numpy(float))
    require(weights.sum() > 0, "NNLS returned zero weights")
    exact = weights / weights.sum()
    applied = np.array([float(f"{weight:.6f}") for weight in exact])
    meta = read(directory / "meta/meta.json")
    require(meta == complete["meta"] and meta["meta_fit_ids"] == ids_hash(train.id), "meta scope or record changed")
    require(meta["meta_fit_rows"] == len(train) and meta["laneA_recal_fit_rows"] == int((train.has_plan == 1).sum()), "meta fit coverage changed")
    require(meta["params"] == recal.PARAMS and meta["checkpoint_replay"] == "PASS", "recalibration recipe changed")
    np.testing.assert_allclose([meta["exact_weights"][name] for name in MEMBERS], exact, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal([meta["applied_weights"][name] for name in MEMBERS], applied)
    model_path = directory / "meta/recal.model"
    require(file_hash(model_path) == meta["model_hash"], "recalibration model bytes changed")
    model = lgb.Booster(model_file=str(model_path))
    require(model.num_trees() == recal.PARAMS["n_estimators"] and model.feature_name() == recal.COLS, "recalibration features or tree count changed")
    query = recal_frame(rows, rows[list(MEMBERS)].to_numpy(float) @ applied)
    lane_a = query.has_plan == 1
    prediction = np.maximum(query.loc[lane_a, "p"].values + model.predict(query.loc[lane_a, recal.COLS]), 0)
    np.testing.assert_allclose(rows.loc[lane_a, "clean_champion"], prediction, rtol=1e-7, atol=1e-4)
    np.testing.assert_array_equal(rows.loc[lane_a, "control"], rows.loc[lane_a, MEMBERS[0]])
    return {"assembly_proofs": proofs, "exact_weights": meta["exact_weights"], "applied_weights": meta["applied_weights"],
            "meta_input_hash": frame_hash(training, ["id", "y", *MEMBERS]), "recalibration_model_hash": meta["model_hash"]}


def checked_recording(frame: pd.DataFrame, train: pd.DataFrame, rows: pd.DataFrame, directory: Path, complete: dict) -> dict:
    path = directory / "outer/laneB"
    query = frame[frame.id.isin(rows.loc[rows.has_plan == 0, "id"])].assign(y=np.nan)
    record = read(path / "complete.json")
    require(record == complete["recording"] and record["checkpoint_replay"] == "PASS", "final recording record changed")
    require(record["specification"] == component_spec(train, query, "laneb"), "recording input scope or source changed")
    for name, digest in record["artifact_hashes"].items():
        require(file_hash(path / name) == digest, f"recording artifact changed: {name}")
    augmented = laneb.augment(frame)
    context = laneb.history_rows(augmented, pd.Index(train.id))
    expected_train = context[context.id.isin(train.id)]
    expected_query = context[context.id.isin(query.id)].assign(y=np.nan)
    saved_train = pd.read_parquet(path / "training_context.parquet")
    saved_query = pd.read_parquet(path / "query_context.parquet")
    pd.testing.assert_frame_equal(saved_train, expected_train, check_dtype=False, check_categorical=False)
    pd.testing.assert_frame_equal(saved_query, expected_query, check_dtype=False, check_categorical=False)
    _, stop = laneb.stopping_split(expected_train)
    selection = read(path / "selection.json")
    inner_ids = train.id[~train.day.isin(stop.day.unique())]
    require(selection["specification"] == record["specification"] and selection["iterations"] == record["iterations"], "recording stopping recipe changed")
    require(selection["history_ids"] == ids_hash(inner_ids) and selection["stop_ids"] == ids_hash(stop.id), "recording stopping or history scope changed")
    donor_path = directory / "outer/mixtureB"
    donor = read(donor_path / "complete.json")
    require(record["donor"] == donor == complete["outer"]["members"]["mixtureB"], "recording mixture lineage changed")
    for name, digest in donor["artifact_hashes"].items():
        require(file_hash(donor_path / name) == digest, f"recording donor artifact changed: {name}")
    donor_values = pd.read_parquet(donor_path / "predictions.parquet")
    require(np.array_equal(donor_values.id.values, query.id.values), "recording donor query order changed")
    replay, probability = replay_recording(expected_train, expected_query, path / "refit")
    replay = laneb.hull(expected_query, laneb.hull_prior(expected_train, expected_query,
                                                      laneb.blend_donor(replay, probability[:, 0], donor_values.p.values)))
    saved = pd.read_parquet(path / "predictions.parquet")
    require(np.array_equal(saved.id.values, query.id.values), "final recording query order changed")
    np.testing.assert_allclose(saved.p, replay, rtol=1e-7, atol=1e-4)
    for column in ("control", "clean_champion"):
        np.testing.assert_array_equal(rows.loc[rows.has_plan == 0, column], saved.p.values)
    return {"history_fit_ids": ids_hash(train.id), "stopping_history_ids": selection["history_ids"],
            "stopping_ids": selection["stop_ids"], "query_ids": ids_hash(query.id), "prediction_hash": record["artifact_hashes"]["predictions.parquet"]}


def component_costs(directory: Path) -> dict:
    components = []
    for path in sorted(directory.rglob("complete.json")):
        record = read(path)
        name = record.get("specification", {}).get("name", "")
        if name:
            components.append({"path": str(path.parent.relative_to(directory)), "name": name,
                               "seconds": record["runtime_seconds"], "gpu_family": name.startswith(("ra_cb", "nn", "mix_cb"))})
    return {"components": components, "component_wall_seconds": sum(item["seconds"] for item in components),
            "gpu_family_wall_seconds": sum(item["seconds"] for item in components if item["gpu_family"]),
            "accounting": "Saved component selection/refit/replay wall times. GPU-family subtotal is a conservative proxy; CPU work and concurrent stages are not GPU utilization."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", default="1-7")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    months = tuple(int(value) for value in args.months.split("-"))
    directory = OUT / args.months
    complete = read(directory / "complete.json")
    frame = inputs()
    train, rows, _ = checked_rows(frame, directory, complete, months)
    meta = checked_meta(train, rows, directory, complete)
    recording = checked_recording(frame, train, rows, directory, complete)
    clock_month = rows.month.copy()
    rows = rows.assign(month=file_month(rows))
    metrics = paired_metrics(rows, rows.control.values, rows.clean_champion.values)
    for key in ("n", "control_rmse", "candidate_rmse", "gain", "sse_gain"):
        np.testing.assert_allclose(metrics[key], complete["metrics"][key], rtol=1e-12, atol=1e-8)
    output = {"checked_utc": datetime.now(timezone.utc).isoformat(), "months": months, "scope_checks": "PASS",
              "meta_reproduction": "PASS", "recording_context_reproduction": "PASS", "recording_replay": "PASS",
              "laneB_parity": "PASS", "output_hash": complete["output_hash"], "query_files": sorted(rows.file.unique()),
              "meta": meta, "recording": recording, "metrics": metrics, "errors": error_rows(rows, "clean_champion"),
              "file_clock_month_mismatches": int((rows.month.values != clock_month.values).sum()),
              "costs": component_costs(directory), "audit_code_hash": file_hash(Path(__file__)),
              "pipeline_code_hash": file_hash(ROOT / "champion_pipeline.py"),
              "stages": {"reference_recipe": "checked for this block", "meta_and_evaluation": "checked for this block",
                         "candidate_and_confirmation": "pending"},
              "limit": "Reused development files. Clean recipe versus its scoped CatBoost control, not uploaded v53 or a measured board gain. Saved scopes and recipe inputs are checked; this audit does not rerun model fitting or establish an untouched test.",
              "command": f"python analysis/champion_evaluation_audit.py --months {args.months}"}
    write_record(directory / "evaluation-audit.json", output)
    logging.info("Full reference scope, replay, parity, errors and cost checks PASS")


if __name__ == "__main__":
    main()
