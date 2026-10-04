"""Audit saved neighbour donors' scopes, arithmetic and cost."""
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

from campaign import file_hash, outer_masks, prediction_windows
from champion import configuration, ids_hash, write_record
from champion_neighbours import file_month, month_groups
from features import PW, PW_BLEND
from train import stopping_split

RUN = ROOT / "data/runs/clean-champion-20261003"
BLOCKS = ("1-7", "2-3", "9-10")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def proof_directory(months: tuple[int, ...]) -> Path:
    block = "-".join(map(str, months))
    if block not in BLOCKS:
        raise ValueError("audit block is not declared")
    return RUN if months == (1, 7) else RUN / block


def input_rows(selection_donor: bool, meta_group: tuple[int, ...], months: tuple[int, ...]) -> tuple[pd.DataFrame, pd.Series, str, dict]:
    proof_directory(months)
    source = ROOT / "data/runs/contender-20261003"
    digest = read(source / "inputs.json")["feature_hash"]
    if file_hash(source / "cold_features.parquet") != digest:
        raise ValueError("cold input bytes changed")
    columns = ["id", "file", "day", "apt", "runway", "has_plan", "z_aobt", "t_off", "y"]
    frame = pd.read_parquet(source / "cold_features.parquet", columns=columns)
    if not frame.id.is_unique or not np.isfinite(frame.y).all():
        raise ValueError("cold input IDs or labels failed")
    fit, _ = outer_masks(frame, months)
    if meta_group:
        if meta_group not in month_groups(frame, pd.Index(frame.loc[fit, "id"])):
            raise ValueError("meta query months are not a declared complete-file group")
        fit &= ~file_month(frame).isin(meta_group)
    fit &= frame.has_plan == 1
    excluded = {}
    if selection_donor:
        inner, stop = stopping_split(frame[fit])
        if inner.id.isin(stop.id).any() or inner.day.isin(stop.day).any():
            raise ValueError("downstream stopping rows or days entered the donor fit")
        fit = frame.id.isin(inner.id)
        excluded = {"rows": len(stop), "days": int(stop.day.nunique()),
                    "ids_hash": ids_hash(stop.id), "rule": "Latest training-only stopping days excluded from every donor member fit."}
    return frame, fit, digest, excluded


def check_selection(record: dict, train: pd.DataFrame) -> None:
    inner, stop = stopping_split(train)
    if record["specification"]["name"].startswith("nn"):
        if (record["selection"]["fit_rows"], record["selection"]["fit_days"]) != (len(inner), inner.day.nunique()):
            raise ValueError("neural stopping fit coverage changed")
        if (record["refit"]["fit_rows"], record["refit"]["fit_days"]) != (len(train), train.day.nunique()):
            raise ValueError("neural full refit coverage changed")
    else:
        if (record["inner_fit_rows"], record["stopping_rows"]) != (len(inner), len(stop)):
            raise ValueError("tree stopping fit coverage changed")
        if record["inner_fit_ids_hash"] != ids_hash(inner.id) or record["stopping_ids_hash"] != ids_hash(stop.id):
            raise ValueError("tree stopping IDs changed")


def member_values(path: Path, detail: dict, train: pd.DataFrame, query: pd.DataFrame,
                  digest: str, floored: bool) -> tuple[np.ndarray, dict]:
    record = read(path / "complete.json")
    spec = record["specification"]
    if record["checkpoint_replay"] != "PASS":
        raise ValueError("member checkpoint replay did not pass")
    if spec["fit_ids"] != ids_hash(train.id) or spec["query_ids"] != ids_hash(query.id):
        raise ValueError("member permitted or query IDs changed")
    if spec["provenance"]["cold_feature_hash"] != digest:
        raise ValueError("member cold input lineage changed")
    if (record["fit_rows"], record["fit_days"]) != (len(train), train.day.nunique()):
        raise ValueError("member full fit coverage changed")
    check_selection(record, train)
    if spec["config"] != configuration(path.name):
        raise ValueError("donor member configuration changed")
    for source, digest_value in spec["engine_hashes"].items():
        if file_hash(ROOT / source) != digest_value:
            raise ValueError("donor member fitting source bytes changed")
    for artifact, digest_value in record["artifact_hashes"].items():
        if file_hash(path / artifact) != digest_value:
            raise ValueError("donor member native artifact bytes changed")
    name = "predictions.parquet" if floored else "raw_predictions.parquet"
    if file_hash(path / name) != detail["prediction_hash"]:
        raise ValueError("actual donor prediction bytes changed")
    values = pd.read_parquet(path / name)
    if not values.id.is_unique or not np.array_equal(values.id.values, query.id.values):
        raise ValueError("donor prediction IDs changed")
    if not np.isfinite(values.p).all() or (floored and (values.p < 0).any()):
        raise ValueError("donor predictions violate their output mode")
    if detail["prediction_floor"] != floored or detail["runtime_seconds"] != record["runtime_seconds"]:
        raise ValueError("donor floor or cost lineage changed")
    cost = {"member": path.name, "seconds": record["runtime_seconds"],
            "gpu_family": path.name.startswith(("ra_cb", "nn")), "checkpoint_replay": "PASS",
            "prediction_hash": detail["prediction_hash"], "model_hashes": record["artifact_hashes"]}
    return values.p.to_numpy(float), cost


def scope_values(frame: pd.DataFrame, fit: pd.Series, scope: dict, target: Path,
                 digest: str, months: tuple[int, ...]) -> tuple[pd.DataFrame, np.ndarray, dict]:
    group = scope["group"]
    allowed = fit & ~file_month(frame).isin(group) if group is not None else fit
    query_mask = fit & file_month(frame).isin(group) if group is not None else ~fit & (frame.has_plan == 1)
    train, query = frame[allowed], frame[query_mask]
    tag = "full" if group is None else "group-" + "-".join(map(str, group))
    if train.id.isin(query.id).any() or file_month(train).isin((1, 7, *months)).any():
        raise ValueError("held-out IDs or files entered a donor fit")
    if (scope["fit_rows"], scope["query_rows"]) != (len(train), len(query)):
        raise ValueError("donor scope coverage changed")
    if scope["fit_ids_hash"] != ids_hash(train.id) or scope["query_ids_hash"] != ids_hash(query.id):
        raise ValueError("donor scope IDs changed")
    if set(scope["members"]) != set(PW_BLEND):
        raise ValueError("donor member set changed")
    prediction, costs = np.zeros(len(query)), []
    for name, weight in PW_BLEND.items():
        detail = scope["members"][name]
        if detail["weight"] != weight:
            raise ValueError("donor blend weight changed")
        values, cost = member_values(target / tag / name, detail, train, query, digest, group is None)
        prediction += weight * values
        costs.append(cost)
    return query, np.maximum(prediction, 0), {"scope": tag, "fit_rows": len(train),
        "fit_days": int(train.day.nunique()), "query_rows": len(query), "members": costs,
        "component_wall_seconds": sum(cost["seconds"] for cost in costs),
        "gpu_family_wall_seconds": sum(cost["seconds"] for cost in costs if cost["gpu_family"])}


def reproduce_windows(frame: pd.DataFrame, fit: pd.Series, target: Path,
                      record: dict, digest: str, months: tuple[int, ...]) -> tuple[pd.DataFrame, list[dict]]:
    prediction, coverage, costs = pd.Series(0., index=frame.id.values), pd.Series(0, index=frame.id.values), []
    for scope in record["scopes"]:
        query, values, cost = scope_values(frame, fit, scope, target, digest, months)
        prediction.loc[query.id], coverage.loc[query.id] = values, coverage.loc[query.id] + 1
        costs.append(cost)
    if not coverage.loc[frame.loc[frame.has_plan == 1, "id"]].eq(1).all():
        raise ValueError("donor queries do not cover each Lane A row once")
    saved = pd.read_parquet(target / "pw.parquet")
    if file_hash(target / "pw.parquet") != record["output_hash"]:
        raise ValueError("saved neighbour feature bytes changed")
    if not saved.id.is_unique or not np.array_equal(saved.id.values, frame.id.values):
        raise ValueError("saved neighbour feature IDs changed")
    rebuilt = prediction_windows(frame, prediction)
    if np.isinf(saved[PW]).any().any():
        raise ValueError("saved neighbour features contain infinities")
    np.testing.assert_array_equal(saved[PW].isna().values, rebuilt[PW].isna().values)
    np.testing.assert_allclose(saved[PW].to_numpy(float), rebuilt[PW].to_numpy(float), rtol=1e-12, atol=1e-10)
    return saved, costs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", choices=BLOCKS, default="1-7")
    parser.add_argument("--selection-donor", action="store_true")
    parser.add_argument("--meta-group", nargs="+", type=int, default=())
    args = parser.parse_args()
    selection_donor, meta_group = args.selection_donor, tuple(args.meta_group)
    months = tuple(int(value) for value in args.months.split("-"))
    frame, fit, digest, excluded = input_rows(selection_donor, meta_group, months)
    target = RUN / args.months / "donors" / ids_hash(frame.loc[fit, "id"])
    record = read(target / "pw.json")
    groups = [list(group) for group in month_groups(frame, pd.Index(frame.loc[fit, "id"]))] + [None]
    if [scope["group"] for scope in record["scopes"]] != groups or record["weights"] != PW_BLEND:
        raise ValueError("donor cross-fit recipe changed")
    if record["fit_ids_hash"] != target.name or record["engine_hash"] != file_hash(ROOT / "champion_neighbours.py"):
        raise ValueError("donor scope or producer lineage changed")
    saved, costs = reproduce_windows(frame, fit, target, record, digest, months)
    audit = {"checked_utc": datetime.now(timezone.utc).isoformat(), "months": months, "scope_checks": "PASS",
        "prediction_lineage_checks": "PASS", "window_reproduction": "PASS", "cold_feature_hash": digest,
        "donor_path": str(target.relative_to(ROOT)), "fit_rows": int(fit.sum()),
        "fit_days": int(frame.loc[fit, "day"].nunique()), "scopes": costs, "weights": PW_BLEND,
        "donor_mode": "downstream_stopping_excluded" if selection_donor else ("full_meta_training" if meta_group else "full_outer_training"),
        "excluded_meta_months": meta_group,
        "excluded_downstream_stopping": excluded,
        "output_hash": record["output_hash"], "output_rows": len(saved),
        "missing_window_counts": {name: int(saved[name].isna().sum()) for name in PW},
        "missing_window_rule": "The original features.per returns NaN for an empty window. Missing positions match the rebuilt features; infinities are rejected.",
        "component_wall_seconds": sum(cost["component_wall_seconds"] for cost in costs),
        "gpu_family_wall_seconds": sum(cost["gpu_family_wall_seconds"] for cost in costs),
        "accounting": "Saved selection/refit/replay component wall time; GPU-family subtotal is a conservative proxy, not GPU utilization.",
        "stages": {"audited_donor": "checked", "reference_recipe": "partial: remaining members and nested meta fits pending",
                   "meta_and_evaluation": "pending", "candidate_and_confirmation": "pending"},
        "limit": "Selected donor only. Remaining nested meta donor chains and full reference evaluation remain pending. Native checkpoint replay is retained from completion records; no new model fit or GPU replay.",
        "command": f"python analysis/champion_donor_audit.py --months {args.months}" + (" --selection-donor" if selection_donor else "")
                   + (" --meta-group " + " ".join(map(str, meta_group)) if meta_group else ""),
        "audit_code_hash": file_hash(Path(__file__))}
    filename = ("meta-" + "-".join(map(str, meta_group)) + ("-selection" if selection_donor else "") + "-donor-audit.json"
                if meta_group else ("selection-donor-audit.json" if selection_donor else "first-donor-audit.json"))
    write_record(proof_directory(months) / filename, audit)
    print(json.dumps({key: audit[key] for key in ("scope_checks", "window_reproduction", "component_wall_seconds", "gpu_family_wall_seconds")}))


if __name__ == "__main__":
    main()
