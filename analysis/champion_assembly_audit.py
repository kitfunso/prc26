"""Check saved inner member tables against their complete-file predictions."""
from __future__ import annotations

import argparse
import copy
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mix
from analysis.champion_donor_audit import BLOCKS, input_rows, member_values, proof_directory, read
from campaign import file_hash, outer_masks
from champion import configuration, ids_hash, write_record
from champion_neighbours import file_month
from champion_pipeline import MEMBERS
from train import stopping_split

RUN = ROOT / "data/runs/clean-champion-20261003"


def native_record(path: Path, detail: dict, config: dict) -> dict:
    record = read(path / "complete.json")
    if record != detail or record["checkpoint_replay"] != "PASS":
        raise ValueError("assembly member completion or replay record changed")
    spec = record["specification"]
    actual_config = record["config"] if path.name == "mixtureB" else spec["config"]
    if actual_config != config:
        raise ValueError("assembly member configuration changed")
    for name, digest in spec["engine_hashes"].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError(f"assembly member fitting source changed: {name}")
    for name, digest in record["artifact_hashes"].items():
        if file_hash(path / name) != digest:
            raise ValueError(f"assembly member native artifact changed: {name}")
    return record


def lane_a(path: Path, detail: dict, train: pd.DataFrame, query: pd.DataFrame,
           digest: str, raw_hash: str, replay_hash: str) -> np.ndarray:
    record = native_record(path, detail, configuration(path.name))
    values, _ = member_values(path, {"prediction_hash": raw_hash, "prediction_floor": False,
                                    "runtime_seconds": record["runtime_seconds"]}, train, query, digest, False)
    proof = read(path / "raw_predictions.json")
    if (proof["prediction_floor"] is not False or proof["floor_parity"] != "PASS"
            or proof["raw_hash"] != raw_hash or proof["replay_code_hash"] != replay_hash
            or proof["floored_hash"] != record["artifact_hashes"]["predictions.parquet"]):
        raise ValueError("assembly raw prediction lineage changed")
    floored = pd.read_parquet(path / "predictions.parquet")
    if not np.array_equal(floored.id.values, query.id.values):
        raise ValueError("assembly floored prediction IDs changed")
    np.testing.assert_allclose(np.maximum(values, 0), floored.p.values, rtol=1e-7, atol=1e-4)
    return values


def lane_b(path: Path, detail: dict, train: pd.DataFrame, query: pd.DataFrame, digest: str) -> np.ndarray:
    config = copy.deepcopy(mix.CONFIGS["mix_cb_wx"])
    config["clf"]["params"]["verbose"] = config["reg"]["params"]["verbose"] = 500
    record = native_record(path, detail, config)
    spec = record["specification"]
    if (spec["name"] != "mix_cb_wx_raw" or record["prediction_floor"] is not False
            or spec["fit_ids"] != ids_hash(train.id) or spec["query_ids"] != ids_hash(query.id)
            or spec["provenance"]["cold_feature_hash"] != digest):
        raise ValueError("assembly Lane B scope, provenance or output mode changed")
    selected = read(path / "selection.json")
    inner, stop = stopping_split(train)
    if (selected["specification"] != spec or selected["iterations"] != record["iterations"]
            or selected["inner_ids"] != ids_hash(inner.id) or selected["stopping_ids"] != ids_hash(stop.id)):
        raise ValueError("assembly Lane B stopping scope changed")
    values = pd.read_parquet(path / "predictions.parquet")
    if not values.id.is_unique or not np.array_equal(values.id.values, query.id.values) or not np.isfinite(values.p).all():
        raise ValueError("assembly Lane B prediction coverage changed")
    return values.p.to_numpy(float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", choices=BLOCKS, default="1-7")
    parser.add_argument("--meta-group", type=int, nargs="+", required=True)
    args = parser.parse_args()
    group = tuple(args.meta_group)
    months = tuple(int(value) for value in args.months.split("-"))
    frame, fit_a, digest, _ = input_rows(False, group, months)
    outer_fit, _ = outer_masks(frame, months)
    query_mask = outer_fit & file_month(frame).isin(group)
    fit_mask = outer_fit & ~file_month(frame).isin(group)
    train, query = frame[fit_mask], frame[query_mask]
    tag = "meta-" + "-".join(map(str, group))
    directory = RUN / args.months / tag
    detail = read(directory / "members.json")
    values = pd.read_parquet(directory / "members.parquet")
    if (detail["prediction_mode"] != "raw OOF" or detail["fit_ids"] != ids_hash(train.id)
            or detail["query_ids"] != ids_hash(query.id) or set(detail["members"]) != {*MEMBERS, "mixtureB"}
            or not values.id.is_unique or not np.array_equal(values.id.values, query.id.values)
            or not np.isfinite(values[list(MEMBERS)]).all().all()):
        raise ValueError("assembly complete-file scope or prediction coverage changed")
    replay_hash = file_hash(ROOT / "champion_predictions.py")
    if detail["replay_code_hash"] != replay_hash:
        raise ValueError("assembly raw replay source changed")
    query_a = query[query.has_plan == 1]
    for name in MEMBERS:
        raw = lane_a(directory / name, detail["members"][name], frame[fit_a], query_a,
                     digest, detail["raw_member_hashes"][name], replay_hash)
        np.testing.assert_array_equal(values.loc[query.has_plan.values == 1, name].values, raw)
    query_b = query[query.has_plan == 0]
    raw_b = lane_b(directory / "mixtureB", detail["members"]["mixtureB"], train, query_b, digest)
    for name in MEMBERS:
        np.testing.assert_array_equal(values.loc[query.has_plan.values == 0, name].values, raw_b)
    output = {"checked_utc": datetime.now(timezone.utc).isoformat(), "months": months, "meta_group": group,
              "command": f"python analysis/champion_assembly_audit.py --months {args.months} --meta-group {' '.join(map(str, group))}",
              "scope_checks": "PASS", "native_lineage_checks": "PASS", "raw_floor_parity": "PASS",
              "member_assembly": "PASS", "laneB_parity": "PASS", "fit_rows": len(train),
              "query_rows": len(query), "laneA_rows": len(query_a), "laneB_rows": len(query_b),
              "fit_ids_hash": ids_hash(train.id), "query_ids_hash": ids_hash(query.id),
              "query_files": sorted(query.file.unique().tolist()), "cold_input_hash": digest,
              "output_hash": file_hash(directory / "members.parquet"), "audit_code_hash": file_hash(Path(__file__)),
              "limit": "One inner raw prediction table checked. This audit does not reproduce fit/query feature fingerprints or donor windows. Donor window checks are recorded separately; full NNLS, residual recalibration, final recording-B override and whole-file errors remain pending."}
    write_record(proof_directory(months) / f"{tag}-assembly-audit.json", output)


if __name__ == "__main__":
    main()
