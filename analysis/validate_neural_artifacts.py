"""Verify saved neural refit coverage, donor scopes and actual evaluation files."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nn
from campaign import file_hash, outer_masks, paired_metrics
from data import DATA
from features import CAT, PW
from train import stopping_split

OUT = DATA / "runs" / "contender-20261003"
FULL = OUT / "neural-full"


def verify_scopes(subset: pd.DataFrame, fit_ids: pd.Index, scopes: list[dict]) -> None:
    fit = subset.id.isin(fit_ids) & (subset.has_plan == 1)
    months = sorted(subset.loc[fit, "month"].unique())
    expected = []
    for group in [months[i::3] for i in range(3)] + [None]:
        tr = subset[fit & ~subset.month.isin(group)] if group is not None else subset[fit]
        query = subset[fit & subset.month.isin(group)] if group is not None else subset[~fit & (subset.has_plan == 1)]
        if query.empty:
            continue
        if tr.id.isin(query.id).any() or not tr.id.isin(fit_ids).all():
            raise ValueError("donor scope overlaps its queries or exceeds permitted labels")
        expected.append({"crossfit_months": None if group is None else [int(m) for m in group],
                         "fit_rows": len(tr), "query_rows": len(query),
                         "fit_ids_hash": hashlib.sha256(tr.id.to_numpy().tobytes()).hexdigest(),
                         "query_ids_hash": hashlib.sha256(query.id.to_numpy().tobytes()).hexdigest()})
    if expected != scopes:
        raise ValueError("saved donor provenance differs from reconstructed permitted label scope")


def verify_model(data: dict, training: pd.DataFrame, info: dict) -> None:
    expected = {"best_epoch": info["epochs"], "stopping_epoch": info["epochs"], "fit_rows": len(training),
                "fit_days": int(training.day.nunique()), "mode": "all_rows"}
    if data["run_info"] != expected or info["refit"] != expected:
        raise ValueError("checkpoint did not refit every permitted row/day at its selected best epoch")
    enc = data["encoder"]
    x = training[data["config"]["num"]].astype("float32")
    np.testing.assert_allclose(enc.mu, x.mean(), equal_nan=True, rtol=0, atol=0)
    np.testing.assert_allclose(enc.sd, x.std().replace(0., 1.).fillna(1.), rtol=0, atol=0)
    residual = (training.y - training.z_aobt).values
    np.testing.assert_allclose([data["mu"], data["sd"]], [residual.mean(), max(residual.std(), 1e-6)], rtol=0, atol=0)
    for col in enc.cats:
        levels = {v: i + 1 for i, v in enumerate(pd.unique(training[col].dropna().astype(object)))}
        if enc.maps[col] != levels:
            raise ValueError("checkpoint category encoder differs from the full training rows")


def verify_member(name: str, info: dict, training: pd.DataFrame, te: pd.DataFrame, rows: pd.DataFrame, tag: str, replay: bool) -> dict:
    path = FULL / f"{name}-{tag}.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    verify_model(data, training, info)
    if not np.isfinite(rows[name]).all() or (rows[name] < 0).any():
        raise ValueError("actual neural file has invalid predictions")
    np.testing.assert_array_equal(rows.loc[rows.has_plan == 0, name], rows.loc[rows.has_plan == 0, "control"])
    result = {"checkpoint_hash": file_hash(path), "coverage_and_encoder": "PASS", "donor_scopes": "PASS",
              "file_and_lane_parity": "PASS", "prediction_replay": "not requested", "hot": {}}
    hot = (rows.w_ts > 0) | (rows.w_sn > 0) | (rows.w_fz > 0) | (rows.w_vsby <= 1)
    result["hot"] = paired_metrics(rows[hot], rows.control.values[hot], rows[name].values[hot])
    if replay:
        enc = data["encoder"]
        ci, xn = enc(te)
        model = nn.Net(enc, xn.shape[1], data["config"]["cell"]).to(nn.DEV)
        model.load_state_dict(data["state_dict"])
        p = np.maximum(nn._infer(model, ci, xn) * data["sd"] + data["mu"] + te.z_aobt.values, 0)
        np.testing.assert_allclose(rows.id.map(pd.Series(p, index=te.id))[rows.has_plan == 1],
                                   rows.loc[rows.has_plan == 1, name], rtol=1e-7, atol=1e-4)
        result["prediction_replay"] = "PASS, rtol 1e-7 and atol 1e-4 seconds"
        del model, ci, xn
        torch.cuda.empty_cache()

    return result


def verify_block(f: pd.DataFrame, tag: str, replay: bool) -> dict | None:
    progress_path = FULL / f"progress-{tag}.json"
    if not progress_path.exists():
        return None
    record = json.loads(progress_path.read_text(encoding="utf-8"))
    fit, eva = outer_masks(f, tuple(record["months"]))
    training = f[fit & (f.has_plan == 1)]
    inner, _ = stopping_split(training)
    subset = f[fit | eva].copy().reset_index(drop=True)
    for label, ids in (("inner", pd.Index(inner.id)), ("outer", pd.Index(training.id))):
        verify_scopes(subset, ids, record[f"{label}_donor_scopes"])
    pw_path = FULL / f"pw-outer-{tag}.parquet"
    pw = pd.read_parquet(pw_path)
    if list(pw.columns) != ["id", *PW] or not pw.id.is_unique or set(pw.id) != set(subset.id):
        raise ValueError("saved outer PW feature IDs/schema differ from their scope")
    complete = subset.merge(pw, on="id", validate="one_to_one")
    train_full = complete[complete.id.isin(training.id)]
    te = complete[eva[fit | eva].to_numpy() & (complete.has_plan == 1)]
    rows_path = FULL / f"neural-{tag}.parquet"
    rows = pd.read_parquet(rows_path)
    control = pd.read_parquet(OUT / f"screen-{tag}.parquet")
    if not rows.id.is_unique or set(rows.id) != set(f.loc[eva, "id"]):
        raise ValueError("neural output is not a complete evaluation file")
    np.testing.assert_array_equal(rows.id, control.id)
    np.testing.assert_array_equal(rows.control, control.control)
    check = {"months": record["months"], "fit_rows": len(training), "fit_days": int(training.day.nunique()),
             "pw_hash": file_hash(pw_path), "output_hash": file_hash(rows_path), "models": {}}

    for name, info in record["models"].items():
        check["models"][name] = verify_member(name, info, train_full, te, rows, tag, replay)
    return check


def main() -> None:
    replay = "--replay" in sys.argv[1:]
    f = pd.read_parquet(OUT / "cold_features.parquet")
    for col in CAT:
        f[col] = f[col].astype("category")
    checks = [check for tag in ("2-3", "9-10") if (check := verify_block(f, tag, replay)) is not None]
    if not checks:
        raise ValueError("no completed full-capacity model artifact exists")
    (FULL / "artifact_checks.json").write_text(json.dumps({"blocks": checks}, indent=2), encoding="utf-8")
    print(json.dumps({"checked_blocks": len(checks), "completed_members": sum(len(b["models"]) for b in checks),
                      "replay": replay, "status": "PASS"}))

if __name__ == "__main__":
    main()
