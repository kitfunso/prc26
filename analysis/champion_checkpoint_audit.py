"""Record scope, input fingerprints, output coverage and member cost."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campaign import file_hash, outer_masks
from analysis.champion_donor_audit import check_selection
from champion import configuration, frame_hash, ids_hash, write_record
from features import CAT, PW
from train import stopping_split

RUN = ROOT / "data/runs/clean-champion-20261003"
MEMBERS = ("ra_cb_wz_t7", "ra_lgb_pw_t7", "nn3_pw_t7")


def scoped_rows(record: dict, member: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = ROOT / "data/runs/contender-20261003"
    manifest = json.loads((source / "inputs.json").read_text(encoding="utf-8"))
    if manifest["feature_hash"] != record["specification"]["provenance"]["cold_feature_hash"]:
        raise ValueError("member cold-input lineage changed")
    if file_hash(source / "cold_features.parquet") != manifest["feature_hash"]:
        raise ValueError("member cold-input bytes changed")
    frame = pd.read_parquet(source / "cold_features.parquet", columns=["id", "file", "day", "apt", "has_plan", "y", "z_aobt"])
    if not frame.id.is_unique:
        raise ValueError("cold input IDs are not unique")
    fit_mask, query_mask = outer_masks(frame, (1, 7))
    fit, query = frame[fit_mask & (frame.has_plan == 1)], frame[query_mask & (frame.has_plan == 1)]
    spec = record["specification"]
    if ids_hash(fit.id) != spec["fit_ids"] or ids_hash(query.id) != spec["query_ids"] or fit.id.isin(query.id).any():
        raise ValueError("member permitted or query IDs changed")
    if len(fit) != record["fit_rows"] or fit.day.nunique() != record["fit_days"]:
        raise ValueError("member refit coverage changed")
    check_selection(record, fit)
    predictions = pd.read_parquet(member / "predictions.parquet")
    if file_hash(member / "predictions.parquet") != record["artifact_hashes"]["predictions.parquet"]:
        raise ValueError("member predictions changed")
    if not predictions.id.is_unique or not np.array_equal(predictions.id.values, query.id.values):
        raise ValueError("member output IDs do not cover the ordered query rows")
    if not np.isfinite(predictions.p).all() or (predictions.p < 0).any():
        raise ValueError("member predictions are not finite and nonnegative")
    if not np.isfinite(query.y).all() or not np.isfinite(query.z_aobt).all():
        raise ValueError("member diagnostic labels or AOBT anchors are nonfinite")
    return fit, query.assign(p=predictions.p.values)


def donor_windows(frame: pd.DataFrame, fit: pd.DataFrame, selection: bool) -> tuple[pd.DataFrame, dict]:
    permitted = stopping_split(fit)[0] if selection else fit
    target = RUN / "1-7/donors" / ids_hash(permitted.id)
    proof_path = RUN / ("selection-donor-audit.json" if selection else "first-donor-audit.json")
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if not all(proof[key] == "PASS" for key in ("scope_checks", "prediction_lineage_checks", "window_reproduction")):
        raise ValueError("donor scope and arithmetic proof did not pass")
    if proof["donor_path"].replace("\\", "/") != target.relative_to(ROOT).as_posix():
        raise ValueError("member donor scope differs from the checked donor")
    if file_hash(target / "pw.parquet") != proof["output_hash"]:
        raise ValueError("checked donor feature bytes changed")
    windows = pd.read_parquet(target / "pw.parquet")
    if not np.array_equal(windows.id.values, frame.id.values):
        raise ValueError("member donor feature order changed")
    return windows, {"path": str(target.relative_to(ROOT)), "output_hash": proof["output_hash"],
                     "audit_code_hash": proof["audit_code_hash"], "audit_checked_utc": proof["checked_utc"]}


def input_fingerprints(record: dict) -> dict:
    spec, cfg = record["specification"], record["specification"]["config"]
    columns = list(dict.fromkeys(["id", "z_aobt", *cfg["cat"], *cfg["num"]]))
    stored = list(dict.fromkeys([name for name in columns if name not in PW] + ["file", "day", "has_plan", "y"]))
    frame = pd.read_parquet(ROOT / "data/runs/contender-20261003/cold_features.parquet", columns=stored)
    for name in CAT:
        if name in frame:
            frame[name] = frame[name].astype("category")
    fit, query = outer_masks(frame, (1, 7))
    fit, query = fit & (frame.has_plan == 1), query & (frame.has_plan == 1)
    full, full_proof = donor_windows(frame, frame[fit], selection=False)
    frame[PW] = full[PW]
    actual = {"fit_input_hash": frame_hash(frame[fit], [*columns, "y"]),
              "query_input_hash": frame_hash(frame[query], columns)}
    del full
    selected, selection_proof = donor_windows(frame, frame[fit], selection=True)
    frame[PW] = selected[PW]
    actual["selection_input_hash"] = frame_hash(frame[fit], [*columns, "y"])
    if any(actual[key] != spec[key] for key in actual):
        raise ValueError("member fit/query/stopping feature fingerprints changed")
    return {"checks": "PASS", **actual, "full_donor": full_proof, "selection_donor": selection_proof}


def errors(rows: pd.DataFrame) -> dict:
    residual = rows.y.to_numpy(float) - rows.p.to_numpy(float)
    anchor = np.maximum(rows.z_aobt.to_numpy(float), 0)
    return {"rows": len(rows), "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "sse": float(np.sum(residual ** 2)), "mean_signed_error": float(np.mean(residual)),
            "aobt_only_rmse": float(np.sqrt(np.mean((rows.y.to_numpy(float) - anchor) ** 2)))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member", choices=MEMBERS, default=MEMBERS[0])
    name = parser.parse_args().member
    member = RUN / "1-7/outer" / name
    record = json.loads((member / "complete.json").read_text(encoding="utf-8"))
    if record["checkpoint_replay"] != "PASS":
        raise ValueError("member checkpoint replay did not pass")
    if configuration(name) != record["specification"]["config"]:
        raise ValueError("member configuration changed")
    if name != MEMBERS[0] and any(file_hash(ROOT / path) != digest
                                 for path, digest in record["specification"]["engine_hashes"].items()):
        raise ValueError("member fitting source bytes changed")
    for artifact, digest in record["artifact_hashes"].items():
        if file_hash(member / artifact) != digest:
            raise ValueError("saved member model or prediction bytes changed")
    fit, rows = scoped_rows(record, member)
    fingerprints = input_fingerprints(record) if name != MEMBERS[0] else {}
    frozen = json.loads((ROOT / "data/runs/cp0-laneb-scope-20261003/manifest.json").read_text(encoding="utf-8"))["champion"]
    if file_hash(Path(frozen["path"])) != frozen["sha256"] or file_hash(Path(frozen["frozen_path"])) != frozen["frozen_sha256"]:
        raise ValueError("scored champion or its frozen copy changed")
    budget = json.loads((RUN / "budget.json").read_text(encoding="utf-8"))
    audit = {"checked_utc": datetime.now(timezone.utc).isoformat(), "member": record["specification"]["name"],
        "scope_checks": "PASS", "prediction_checks": "PASS", "checkpoint_replay": record["checkpoint_replay"],
        "fit_rows": len(fit), "fit_days": int(fit.day.nunique()), "iterations": record.get("iterations"),
        "epochs": record.get("epochs"), "selection": record.get("selection"), "refit": record.get("refit"),
        "input_fingerprints": fingerprints, "native_artifact_checks": "PASS",
        "fit_files": sorted(fit.file.unique().tolist()), "query_files": sorted(rows.file.unique().tolist()),
        "runtime_seconds": record["runtime_seconds"], "runtime_includes_prior_attempt": bool(record.get("recovery")),
        "model_and_prediction_hashes": record["artifact_hashes"], "champion_preserved": "PASS",
        "budget_snapshot": budget, "laneA_descriptive_errors": errors(rows),
        "by_file": {str(name): errors(group) for name, group in rows.groupby("file", observed=True)},
        "by_airport": {str(name): errors(group) for name, group in rows.groupby("apt", observed=True)},
        "stages": {"reference_recipe": "partial: this Lane A member checked; Lane B and nested meta fits pending",
                   "meta_and_evaluation": "pending", "candidate_and_confirmation": "pending"},
        "limit": "Lane A member diagnostic on reused January/July files. No assembled whole-file ensemble, confirmation result, promotion evidence or new scorer gain.",
        "replay_limit": "Native replay PASS retained from the completion record; no new model fit or GPU replay.",
        "command": "python analysis/champion_checkpoint_audit.py" + (f" --member {name}" if name != MEMBERS[0] else ""),
        "audit_code_hash": file_hash(Path(__file__))}
    write_record(RUN / ("first-member-audit.json" if name == MEMBERS[0] else f"{name}-audit.json"), audit)
    print(json.dumps({"scope_checks": audit["scope_checks"], "prediction_checks": audit["prediction_checks"],
                      "runtime_seconds": audit["runtime_seconds"], "laneA_descriptive_errors": audit["laneA_descriptive_errors"]}))


if __name__ == "__main__":
    main()
