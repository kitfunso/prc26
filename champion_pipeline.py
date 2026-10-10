"""Nested complete-file champion fitting and fixed residual recalibration."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import nnls

import recal
from campaign import file_hash, outer_masks, paired_metrics
from champion import fit_member, ids_hash, write_record
from champion_budget import budget_guard
from champion_laneb import mixture_fit, recording_fit
from champion_neighbours import file_month, month_groups, with_champion_pw
from champion_predictions import raw_member_predictions
from train import stopping_split

MEMBERS = ("ra_cb_wz_t7", "ra_lgb_pw_t7", "nn3_pw_t7")


def scope_members(frame: pd.DataFrame, fit_ids: pd.Index, query_ids: pd.Index,
                  directory: Path, donors: Path, raw: bool = False) -> tuple[pd.DataFrame, dict]:
    if fit_ids.isin(query_ids).any():
        raise ValueError("meta query labels overlap permitted training")
    fit = frame.id.isin(fit_ids)
    query = frame.id.isin(query_ids)
    ta, qa = frame[fit & (frame.has_plan == 1)], frame[query & (frame.has_plan == 1)]
    output = frame.loc[query, ["id"]].copy()
    budget_guard(directory)
    p, cat_record = fit_member(MEMBERS[0], ta, qa, directory / MEMBERS[0])
    if raw:
        p = raw_member_predictions(MEMBERS[0], cat_record["specification"]["config"], qa, directory / MEMBERS[0])
    output[MEMBERS[0]] = output.id.map(pd.Series(p, index=qa.id))
    complete, full_scopes = with_champion_pw(frame, pd.Index(ta.id), donors)
    inner, _ = stopping_split(ta)
    inside, inner_scopes = with_champion_pw(frame, pd.Index(inner.id), donors)
    fit_full = complete[complete.id.isin(ta.id)]
    fit_selection = inside[inside.id.isin(ta.id)]
    test = complete[complete.id.isin(qa.id)]
    records = {MEMBERS[0]: cat_record}
    for name in MEMBERS[1:]:
        budget_guard(directory)
        p, records[name] = fit_member(name, fit_full, test, directory / name, selection=fit_selection)
        if raw:
            p = raw_member_predictions(name, records[name]["specification"]["config"], test, directory / name)
        output[name] = output.id.map(pd.Series(p, index=test.id))
    qb = pd.Index(frame.loc[query & (frame.has_plan == 0), "id"])
    if len(qb):
        budget_guard(directory)
        p, records["mixtureB"] = mixture_fit(frame[fit], frame[frame.id.isin(qb)], directory / "mixtureB", raw=raw)
        for name in MEMBERS:
            output.loc[output.id.isin(qb), name] = output.loc[output.id.isin(qb), "id"].map(pd.Series(p, index=qb))
    if not np.isfinite(output[list(MEMBERS)]).all().all():
        raise ValueError("complete-file members have missing predictions")
    detail = {"fit_ids": ids_hash(fit_ids), "query_ids": ids_hash(query_ids), "members": records,
              "prediction_mode": "raw OOF" if raw else "floored outer", "replay_code_hash": file_hash(Path(__file__).parent / "champion_predictions.py"),
              "raw_member_hashes": {name: file_hash(directory / name / "raw_predictions.parquet") for name in MEMBERS} if raw else {},
              "full_donor_scopes": full_scopes, "selection_donor_scopes": inner_scopes}
    directory.mkdir(parents=True, exist_ok=True)
    output.to_parquet(directory / "members.parquet")
    write_record(directory / "members.json", detail)
    return output, detail


def recal_frame(rows: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    frame = rows.copy()
    frame["p"] = np.maximum(predictions, 0)
    frame["apt"] = frame.apt.astype(str).astype("category")
    for stamp in recal.Z:
        frame[f"d_{stamp}"] = frame[stamp] - frame.p
    return frame


def fit_meta(training: pd.DataFrame, outer: pd.DataFrame, directory: Path) -> tuple[pd.DataFrame, dict]:
    weights, _ = nnls(training[list(MEMBERS)].to_numpy(float), training.y.to_numpy(float))
    if weights.sum() <= 0:
        raise ValueError("champion NNLS returned zero total weight")
    exact = weights / weights.sum()
    applied = np.array([float(f"{weight:.6f}") for weight in exact])
    tr = recal_frame(training, training[list(MEMBERS)].to_numpy(float) @ applied)
    te = recal_frame(outer, outer[list(MEMBERS)].to_numpy(float) @ applied)
    train_a, test_a = tr[tr.has_plan == 1], te[te.has_plan == 1].assign(y=np.nan)
    model = lgb.LGBMRegressor(**recal.PARAMS).fit(train_a[recal.COLS], train_a.y - train_a.p)
    directory.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(directory / "recal.model"))
    predicted = model.predict(test_a[recal.COLS])
    replay = lgb.Booster(model_file=str(directory / "recal.model")).predict(test_a[recal.COLS])
    np.testing.assert_allclose(predicted, replay, rtol=1e-7, atol=1e-4)
    te["clean_champion"] = te.p
    te.loc[te.has_plan == 1, "clean_champion"] = np.maximum(test_a.p.values + predicted, 0)
    te.loc[te.has_plan == 0, "clean_champion"] = te.loc[te.has_plan == 0, MEMBERS[0]]
    if not np.array_equal(te.loc[te.has_plan == 0, "clean_champion"], te.loc[te.has_plan == 0, MEMBERS[0]]):
        raise ValueError("recalibration changed Lane B")
    record = {"method": "whole-file NNLS normalized, applied at six decimals, fixed residual recalibrator",
              "exact_weights": dict(zip(MEMBERS, exact.tolist())), "applied_weights": dict(zip(MEMBERS, applied.tolist())),
              "meta_fit_ids": ids_hash(training.id), "meta_fit_rows": len(training), "laneA_recal_fit_rows": len(train_a),
              "params": recal.PARAMS, "checkpoint_replay": "PASS", "model_hash": file_hash(directory / "recal.model")}
    write_record(directory / "meta.json", record)
    return te, record


def run_block(frame: pd.DataFrame, months: tuple[int, ...], directory: Path) -> dict:
    start = time.monotonic()
    fit, query = outer_masks(frame, months)
    fit_ids, query_ids = pd.Index(frame.loc[fit, "id"]), pd.Index(frame.loc[query, "id"])
    groups = month_groups(frame, fit_ids)
    progress = {"months": months, "status": "running", "fit_rows": int(fit.sum()), "query_rows": int(query.sum()),
                "meta_groups": groups, "fit_ids": ids_hash(fit_ids), "query_ids": ids_hash(query_ids), "completed_scopes": []}
    directory.mkdir(parents=True, exist_ok=True)
    write_record(directory / "progress.json", progress)
    outer, detail = scope_members(frame, fit_ids, query_ids, directory / "outer", directory / "donors")
    progress["outer"] = detail
    progress["completed_scopes"].append("outer")
    write_record(directory / "progress.json", progress)
    predictions = []
    for group in groups:
        eva = fit & file_month(frame).isin(group)
        inside = fit & ~file_month(frame).isin(group)
        tag = "meta-" + "-".join(map(str, group))
        logging.info("Champion meta scope %s", tag)
        values, detail = scope_members(frame, pd.Index(frame.loc[inside, "id"]), pd.Index(frame.loc[eva, "id"]),
                                       directory / tag, directory / "donors", raw=True)
        predictions.append(values)
        progress[tag] = detail
        progress["completed_scopes"].append(tag)
        write_record(directory / "progress.json", progress)
    meta = pd.concat(predictions, ignore_index=True)
    if not meta.id.is_unique or set(meta.id) != set(fit_ids):
        raise ValueError("meta predictions do not cover complete permitted training files")
    training = frame[fit].merge(meta, on="id", validate="one_to_one")
    outer = frame[query].merge(outer, on="id", validate="one_to_one")
    rows, detail = fit_meta(training, outer, directory / "meta")
    qb = pd.Index(frame.loc[query & (frame.has_plan == 0), "id"])
    b, recording = recording_fit(frame, fit_ids, qb, directory / "outer/laneB")
    rows["control"] = rows[MEMBERS[0]]
    for column in ("control", "clean_champion"):
        rows.loc[rows.has_plan == 0, column] = rows.loc[rows.has_plan == 0, "id"].map(pd.Series(b, index=qb))
    if not np.isfinite(rows[["control", "clean_champion"]]).all().all():
        raise ValueError("final complete-file assembly has invalid predictions")
    rows.to_parquet(directory / "evaluation.parquet")
    progress.update(status="complete", meta=detail, recording=recording, runtime_seconds=time.monotonic() - start,
                    output_hash=file_hash(directory / "evaluation.parquet"),
                    metrics=paired_metrics(rows, rows.control.values, rows.clean_champion.values),
                    limit="Reused 2025 block; clean recipe versus clean CatBoost member, not a measured board gain.")
    write_record(directory / "complete.json", progress)
    write_record(directory / "progress.json", progress)
    logging.info("Clean champion block complete: %s", progress["metrics"])
    return progress
