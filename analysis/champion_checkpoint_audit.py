"""Record scope, output coverage and cost of the first reconstructed member."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campaign import file_hash, outer_masks
from champion import ids_hash, write_record
from train import stopping_split

RUN = ROOT / "data/runs/clean-champion-20261003"
MEMBER = RUN / "1-7/outer/ra_cb_wz_t7"


def scoped_rows(record: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = ROOT / "data/runs/contender-20261003"
    manifest = json.loads((source / "inputs.json").read_text(encoding="utf-8"))
    if manifest["feature_hash"] != record["specification"]["provenance"]["cold_feature_hash"]:
        raise ValueError("member cold-input lineage changed")
    if file_hash(source / "cold_features.parquet") != manifest["feature_hash"]:
        raise ValueError("member cold-input bytes changed")
    frame = pd.read_parquet(source / "cold_features.parquet", columns=["id", "file", "day", "apt", "has_plan", "y", "z_aobt"])
    fit_mask, query_mask = outer_masks(frame, (1, 7))
    fit, query = frame[fit_mask & (frame.has_plan == 1)], frame[query_mask & (frame.has_plan == 1)]
    spec = record["specification"]
    if ids_hash(fit.id) != spec["fit_ids"] or ids_hash(query.id) != spec["query_ids"] or fit.id.isin(query.id).any():
        raise ValueError("member permitted or query IDs changed")
    if len(fit) != record["fit_rows"] or fit.day.nunique() != record["fit_days"]:
        raise ValueError("member refit coverage changed")
    inner, stop = stopping_split(fit)
    if (len(inner), len(stop)) != (record["inner_fit_rows"], record["stopping_rows"]):
        raise ValueError("stopping selection coverage changed")
    if ids_hash(inner.id) != record["inner_fit_ids_hash"] or ids_hash(stop.id) != record["stopping_ids_hash"]:
        raise ValueError("stopping selection IDs changed")
    predictions = pd.read_parquet(MEMBER / "predictions.parquet")
    if file_hash(MEMBER / "predictions.parquet") != record["artifact_hashes"]["predictions.parquet"]:
        raise ValueError("member predictions changed")
    if not predictions.id.is_unique or not np.array_equal(predictions.id.values, query.id.values):
        raise ValueError("member output IDs do not cover the ordered query rows")
    if not np.isfinite(predictions.p).all() or (predictions.p < 0).any():
        raise ValueError("member predictions are not finite and nonnegative")
    if not np.isfinite(query.y).all() or not np.isfinite(query.z_aobt).all():
        raise ValueError("member diagnostic labels or AOBT anchors are nonfinite")
    return fit, query.assign(p=predictions.p.values)


def errors(rows: pd.DataFrame) -> dict:
    residual = rows.y.to_numpy(float) - rows.p.to_numpy(float)
    anchor = np.maximum(rows.z_aobt.to_numpy(float), 0)
    return {"rows": len(rows), "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "sse": float(np.sum(residual ** 2)), "mean_signed_error": float(np.mean(residual)),
            "aobt_only_rmse": float(np.sqrt(np.mean((rows.y.to_numpy(float) - anchor) ** 2)))}


def main() -> None:
    record = json.loads((MEMBER / "complete.json").read_text(encoding="utf-8"))
    if record["checkpoint_replay"] != "PASS":
        raise ValueError("member checkpoint replay did not pass")
    fit, rows = scoped_rows(record)
    frozen = json.loads((ROOT / "data/runs/cp0-laneb-scope-20261003/manifest.json").read_text(encoding="utf-8"))["champion"]
    if file_hash(Path(frozen["path"])) != frozen["sha256"] or file_hash(Path(frozen["frozen_path"])) != frozen["frozen_sha256"]:
        raise ValueError("scored champion or its frozen copy changed")
    budget = json.loads((RUN / "budget.json").read_text(encoding="utf-8"))
    audit = {"checked_utc": datetime.now(timezone.utc).isoformat(), "member": record["specification"]["name"],
        "scope_checks": "PASS", "prediction_checks": "PASS", "checkpoint_replay": record["checkpoint_replay"],
        "fit_rows": len(fit), "fit_days": int(fit.day.nunique()), "iterations": record["iterations"],
        "fit_files": sorted(fit.file.unique().tolist()), "query_files": sorted(rows.file.unique().tolist()),
        "runtime_seconds": record["runtime_seconds"], "runtime_includes_prior_attempt": True,
        "model_and_prediction_hashes": record["artifact_hashes"], "champion_preserved": "PASS",
        "budget_snapshot": budget, "laneA_descriptive_errors": errors(rows),
        "by_file": {str(name): errors(group) for name, group in rows.groupby("file", observed=True)},
        "by_airport": {str(name): errors(group) for name, group in rows.groupby("apt", observed=True)},
        "stages": {"reference_recipe": "partial: tuned CatBoost member checked; original donor and remaining members fitting",
                   "meta_and_evaluation": "pending", "candidate_and_confirmation": "pending"},
        "limit": "Lane A member diagnostic on reused January/July files. No assembled whole-file ensemble, confirmation result, promotion evidence or new scorer gain.",
        "command": "python analysis/champion_checkpoint_audit.py", "audit_code_hash": file_hash(Path(__file__))}
    write_record(RUN / "first-member-audit.json", audit)
    print(json.dumps({"scope_checks": audit["scope_checks"], "prediction_checks": audit["prediction_checks"],
                      "runtime_seconds": audit["runtime_seconds"], "laneA_descriptive_errors": audit["laneA_descriptive_errors"]}))


if __name__ == "__main__":
    main()
