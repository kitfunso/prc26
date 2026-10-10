"""Scoped original mixture donor and recording model for the champion's Lane B."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

import laneb
import mix
from campaign import file_hash
from champion import frame_hash, ids_hash, write_record
from train import fit_predict, stopping_split


def mixture_stage(cfg: dict, fit: pd.DataFrame, query: pd.DataFrame, directory: Path,
                  stop: pd.DataFrame | None = None, iterations: tuple[int, ...] | None = None) -> tuple[np.ndarray, tuple[int, int]]:
    lab = mix.label(fit)
    stop_lab = mix.label(stop) if stop is not None else None
    gate, it_gate = mix.fit_clf(cfg, fit, lab, query, stop, stop_lab, iterations[0] if iterations else None,
                               model_path=directory / "gate.model")
    for k, stamp in enumerate(mix.Z):
        gate[query[stamp].isna().values, k] = 0
    gate /= gate.sum(1, keepdims=True)
    p, it_reg = fit_predict(cfg["reg"], fit[lab == 3], query, stop[stop_lab == 3] if stop is not None else None,
                            iterations[1] if iterations else None, model_path=directory / "physical.model")
    out = sum(gate[:, k] * np.nan_to_num(query[stamp].values) for k, stamp in enumerate(mix.Z)) + gate[:, 3] * p
    return out, (it_gate, it_reg)


def replay_mixture(cfg: dict, query: pd.DataFrame, directory: Path, floor: bool = True) -> np.ndarray:
    columns = cfg["cat"] + cfg["num"]
    gate = CatBoostClassifier().load_model(str(directory / "gate.model"))
    reg = CatBoostRegressor().load_model(str(directory / "physical.model"))
    pool = Pool(query[columns], cat_features=cfg["cat"])
    probability = gate.predict_proba(pool)
    for k, stamp in enumerate(mix.Z):
        probability[query[stamp].isna().values, k] = 0
    probability /= probability.sum(1, keepdims=True)
    p = sum(probability[:, k] * np.nan_to_num(query[stamp].values) for k, stamp in enumerate(mix.Z))
    out = p + probability[:, 3] * reg.predict(pool)
    return np.maximum(out, 0) if floor else out


def component_spec(fit: pd.DataFrame, query: pd.DataFrame, name: str) -> dict:
    return {"name": name, "fit_ids": ids_hash(fit.id), "query_ids": ids_hash(query.id),
            "fit_frame_hash": frame_hash(fit, list(fit.columns)),
            "query_frame_hash": frame_hash(query, [column for column in query if column != "y"]),
            "provenance": fit.attrs.get("provenance", {}),
            "engine_hashes": {name: file_hash(Path(__file__).parent / name) for name in
                              ("champion_laneb.py", "laneb.py", "mix.py", "train.py")}}


def completed(directory: Path, spec: dict, query: pd.DataFrame) -> tuple[np.ndarray, dict] | None:
    path = directory / "complete.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["specification"] != spec:
        raise ValueError(f"Lane B component scope changed: {directory}")
    for artifact, digest in record["artifact_hashes"].items():
        if file_hash(directory / artifact) != digest:
            raise ValueError(f"Lane B artifact changed: {artifact}")
    values = pd.read_parquet(directory / "predictions.parquet")
    if not np.array_equal(values.id.values, query.id.values):
        raise ValueError("Lane B component IDs changed")
    return values.p.values, record


def close_component(directory: Path, spec: dict, query: pd.DataFrame, p: np.ndarray, details: dict) -> dict:
    if not np.isfinite(p).all() or (details.get("prediction_floor", True) and (p < 0).any()):
        raise ValueError("Lane B component predictions are invalid")
    pd.DataFrame({"id": query.id.values, "p": p}).to_parquet(directory / "predictions.parquet")
    record = {"specification": spec, "checkpoint_replay": "PASS", **details,
              "artifact_hashes": {str(path.relative_to(directory)): file_hash(path)
                                  for path in sorted(directory.rglob("*")) if path.is_file() and path.suffix in (".model", ".parquet")}}
    write_record(directory / "complete.json", record)
    return record


def mixture_fit(fit: pd.DataFrame, query: pd.DataFrame, directory: Path, raw: bool = False) -> tuple[np.ndarray, dict]:
    if fit.id.isin(query.id).any():
        raise ValueError("mixture donor fit/query labels overlap")
    query = query.assign(y=np.nan)
    spec = component_spec(fit, query, "mix_cb_wx_raw" if raw else "mix_cb_wx")
    cached = completed(directory, spec, query)
    if cached is not None:
        return cached
    directory.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    logging.info("Lane B original mixture donor: fit %s, query %s", len(fit), len(query))
    write_record(directory / "started.json", {"specification": spec, "fit_rows": len(fit), "fit_days": int(fit.day.nunique())})
    cfg = json.loads(json.dumps(mix.CONFIGS["mix_cb_wx"]))
    cfg["clf"]["params"]["verbose"] = cfg["reg"]["params"]["verbose"] = 500
    inner, stop = stopping_split(fit)
    _, iterations = mixture_stage(cfg, inner, stop.assign(y=np.nan), directory / "selection", stop)
    write_record(directory / "selection.json", {"specification": spec, "iterations": iterations,
                                                "inner_ids": ids_hash(inner.id), "stopping_ids": ids_hash(stop.id)})
    p, _ = mixture_stage(cfg, fit, query, directory / "refit", iterations=iterations)
    p = p if raw else np.maximum(p, 0)
    np.testing.assert_allclose(p, replay_mixture(cfg, query, directory / "refit", floor=not raw), rtol=1e-7, atol=1e-4)
    detail = close_component(directory, spec, query, p, {"iterations": iterations, "config": cfg,
                                                         "prediction_floor": not raw, "runtime_seconds": time.monotonic() - start})
    return p, detail


def replay_recording(train: pd.DataFrame, query: pd.DataFrame, directory: Path) -> tuple[np.ndarray, np.ndarray]:
    lab, keep, _ = laneb._labels(train)
    present = sorted(set(lab[keep].tolist()))
    gate_cols = mix.B_CAT + mix.B_NUM + ["cross", "sched_hour"] + laneb.HIST + laneb.DAY_COLS
    tau_cols = laneb.TAU_CAT + laneb.tau_num(train) + laneb.DAY + laneb.OCC + laneb.WX + laneb.ARR + laneb.WZ
    gate = lgb.Booster(model_file=str(directory / "gate.model"))
    tau = lgb.Booster(model_file=str(directory / "tau.model")).predict(query[tau_cols])
    raw = gate.predict(query[gate_cols])
    probability = np.zeros((len(query), 3))
    for i, label in enumerate(present):
        probability[:, label] = raw[:, i]
    probability /= probability.sum(1, keepdims=True)
    rate, _ = laneb.shift_rate(train)
    shift = np.where((query.apt.values == laneb.APT_SHIFT) & (query.z_sched.values > laneb.Z_SHIFT), rate, 0.)
    p = (1 - shift) * (probability[:, 0] * query.z_sched.values + probability[:, 2] * tau) + shift * (laneb.SHIFT_ADD + tau)
    return np.maximum(p, 0), probability


def recording_fit(frame: pd.DataFrame, fit_ids: pd.Index, query_ids: pd.Index, directory: Path) -> tuple[np.ndarray, dict]:
    if fit_ids.isin(query_ids).any():
        raise ValueError("recording fit/query scopes overlap")
    fit = frame[frame.id.isin(fit_ids)]
    query = frame[frame.id.isin(query_ids)].assign(y=np.nan)
    if (query.has_plan != 0).any():
        raise ValueError("recording queries must be Lane B")
    spec = component_spec(fit, query, "laneb")
    cached = completed(directory, spec, query)
    if cached is not None:
        return cached
    directory.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    write_record(directory / "started.json", {"specification": spec, "fit_rows": int((fit.has_plan == 0).sum()),
                                              "fit_days": int(fit.loc[fit.has_plan == 0, "day"].nunique())})
    donor, donor_record = mixture_fit(fit, query, directory.parent / "mixtureB")
    augmented = laneb.augment(frame)
    rows = laneb.history_rows(augmented, fit_ids)
    train, test = rows[rows.id.isin(fit_ids)], rows[rows.id.isin(query_ids)].assign(y=np.nan)
    if not np.array_equal(test.id.values, query.id.values):
        raise ValueError("recording context changed query order")
    inner, stop = laneb.stopping_split(train)
    inner_ids = fit.id[~fit.day.isin(stop.day.unique())]
    inside = laneb.history_rows(augmented, pd.Index(inner_ids))
    _, iterations, *_ = laneb.predict(inside.loc[inner.index], inside.loc[stop.index].assign(y=np.nan), inside.loc[stop.index],
                                      model_dir=directory / "selection")
    write_record(directory / "selection.json", {"specification": spec, "iterations": iterations,
                                                "history_ids": ids_hash(inner_ids), "stop_ids": ids_hash(stop.id)})
    p, _, probability, _, _ = laneb.predict(train, test, iters=iterations, model_dir=directory / "refit")
    replay, replay_probability = replay_recording(train, test, directory / "refit")
    np.testing.assert_allclose(p, replay, rtol=1e-7, atol=1e-4)
    p = laneb.hull(test, laneb.hull_prior(train, test, laneb.blend_donor(p, probability[:, 0], donor)))
    replay = laneb.hull(test, laneb.hull_prior(train, test, laneb.blend_donor(replay, replay_probability[:, 0], donor)))
    np.testing.assert_allclose(p, replay, rtol=1e-7, atol=1e-4)
    train.to_parquet(directory / "training_context.parquet")
    test.to_parquet(directory / "query_context.parquet")
    detail = close_component(directory, spec, query, p, {"iterations": iterations, "donor": donor_record,
                                                         "runtime_seconds": time.monotonic() - start})
    return p, detail
