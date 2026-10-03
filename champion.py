"""Saved full-capacity members for the scoped champion reconstruction."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

import lanea
import nn
from campaign import file_hash
from train import stopping_split


def ids_hash(ids: pd.Series | pd.Index) -> str:
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


def write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def configuration(name: str) -> dict:
    source = nn.CONFIGS[name] if name.startswith("nn") else lanea.CONFIGS[name]
    cfg = json.loads(json.dumps(source))
    if name.startswith("nn"):
        cfg["cell"]["progress"] = True
    else:
        for params in (cfg["clf"]["params"], cfg["reg"]["params"]):
            if cfg["clf"]["family"] == "cb":
                params["verbose"] = 500
    return cfg


def frame_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    values = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def specification(name: str, fit: pd.DataFrame, query: pd.DataFrame, selection: pd.DataFrame) -> dict:
    cfg = configuration(name)
    columns = ["id", "z_aobt", *cfg["cat"], *cfg["num"]]
    columns = list(dict.fromkeys(columns))
    return {"name": name, "config": cfg, "fit_ids": ids_hash(fit.id), "query_ids": ids_hash(query.id),
            "fit_input_hash": frame_hash(fit, [*columns, "y"]), "query_input_hash": frame_hash(query, columns),
            "selection_input_hash": frame_hash(selection, [*columns, "y"]),
            "provenance": fit.attrs.get("provenance", {}),
            "versions": {name: version(name) for name in ("numpy", "pandas", "torch", "catboost", "lightgbm")},
            "engine_hashes": {p: file_hash(Path(__file__).parent / p) for p in
                              ("champion.py", "checkpoints.py", "train.py", "mix.py", "lanea.py", "nn.py", "features.py", "campaign.py", "weather.py")}}


def load_completed(directory: Path, spec: dict, query: pd.DataFrame) -> tuple[np.ndarray, dict] | None:
    path = directory / "complete.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["specification"] != spec:
        recovery_path = directory / "recovery.json"
        repair = json.loads(recovery_path.read_text(encoding="utf-8")).get("completed_engine_compatibility", {}) if recovery_path.exists() else {}
        saved = record["specification"]
        expected = {**saved, "engine_hashes": repair.get("resumed_engine_hashes")}
        if saved["engine_hashes"] != repair.get("execution_engine_hashes") or expected != spec:
            raise ValueError(f"saved component scope/configuration changed: {directory}")
    for name, digest in record["artifact_hashes"].items():
        if file_hash(directory / name) != digest:
            raise ValueError(f"saved component artifact changed: {directory / name}")
    predictions = pd.read_parquet(directory / "predictions.parquet")
    if not np.array_equal(predictions.id.values, query.id.values):
        raise ValueError("saved member prediction IDs changed")
    logging.info("Resume completed %s at %s", spec["name"], directory)
    return predictions.p.values, record


def save_neural(path: Path, trained: tuple, cfg: dict) -> None:
    model, enc, mu, sd, _ = trained
    torch.save({"state_dict": model.state_dict(), "encoder": enc, "mu": mu, "sd": sd,
                "config": cfg, "run_info": model.run_info}, path)


def neural_fit(cfg: dict, fit: pd.DataFrame, query: pd.DataFrame, selection: pd.DataFrame,
               directory: Path, spec: dict) -> tuple[np.ndarray, dict]:
    selected = nn.train(cfg, selection)
    epochs, selection_info = selected[4], selected[0].run_info.copy()
    save_neural(directory / "selection.pt", selected, cfg)
    write_record(directory / "selection.json", {"specification": spec, "epochs": epochs, "selection": selection_info})
    del selected
    torch.cuda.empty_cache()
    trained = nn.train(cfg, fit, epochs=epochs)
    model, enc, mu, sd, _ = trained
    if model.run_info["fit_rows"] != len(fit) or model.run_info["fit_days"] != fit.day.nunique():
        raise ValueError("neural refit omitted permitted rows or days")
    save_neural(directory / "refit.pt", trained, cfg)
    ci, xn = enc(query)
    p = nn._infer(model, ci, xn) * sd + mu + query.z_aobt.values
    info = {"epochs": epochs, "selection": selection_info, "refit": model.run_info.copy()}
    del trained, model, enc, ci, xn
    torch.cuda.empty_cache()
    return p, info


def tree_fit(cfg: dict, fit: pd.DataFrame, query: pd.DataFrame, selection: pd.DataFrame,
             directory: Path, spec: dict) -> tuple[np.ndarray, dict]:
    inner, stop = stopping_split(selection)
    if inner.id.isin(query.id).any() or stop.id.isin(query.id).any():
        raise ValueError("outer query labels entered tree selection")
    scope = {"inner_fit_rows": len(inner), "stopping_rows": len(stop),
            "inner_fit_ids_hash": ids_hash(inner.id), "stopping_ids_hash": ids_hash(stop.id)}
    if (directory / "selection.json").exists():
        selected = verified_selection(directory, spec)
        if any(selected[key] != value for key, value in scope.items()):
            raise ValueError("saved stopping scope changed")
        iterations = tuple(selected["iterations"])
        logging.info("Reuse verified stopping selection: %s", iterations)
    else:
        if any((directory / "refit").glob("*.model")):
            raise ValueError("refit models have no verified selection record")
        _, iterations = lanea.predict(cfg, inner, stop.assign(y=np.nan), stop, model_dir=directory / "selection")
        hashes = {str(path.relative_to(directory)): file_hash(path) for path in (directory / "selection").glob("*.model")}
        write_record(directory / "selection.json", {"specification": spec, "iterations": iterations, **scope,
                                                   "artifact_hashes": hashes})
    info = {"iterations": iterations, **scope}
    if (directory / "recovery.json").exists():
        info["recovery"] = json.loads((directory / "recovery.json").read_text(encoding="utf-8"))
    p, _ = lanea.predict(cfg, fit, query, iters=iterations, model_dir=directory / "refit", resume=True)
    return p, info


def verified_selection(directory: Path, spec: dict) -> dict:
    selected = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    saved_spec = selected["specification"]
    hashes = selected.get("artifact_hashes", {})
    if saved_spec != spec:
        path = directory / "recovery.json"
        if not path.exists():
            raise ValueError("saved selection scope/configuration changed")
        recovery = json.loads(path.read_text(encoding="utf-8"))
        expected = {**saved_spec, "engine_hashes": recovery["resumed_engine_hashes"]}
        if recovery["original_specification"] != saved_spec or expected != spec:
            raise ValueError("recovery does not match the original and resumed specifications")
        hashes = recovery["artifact_hashes"]
    hashes = {name.replace("\\", "/"): digest for name, digest in hashes.items()}
    if not hashes or not all(f"selection/{name}.model" in hashes for name in ("gate", "tail", "body")):
        raise ValueError("saved selection model hashes are missing")
    for name, digest in hashes.items():
        if file_hash(directory / name) != digest:
            raise ValueError(f"recovery checkpoint changed: {name}")
    return selected


def replay_tree(cfg: dict, query: pd.DataFrame, directory: Path) -> np.ndarray:
    columns = cfg["cat"] + cfg["num"]
    family = cfg["clf"]["family"]
    if family == "lgb":
        gate, tail, body = [lgb.Booster(model_file=str(directory / f"{name}.model"))
                            for name in ("gate", "tail", "body")]
        probability = gate.predict(query[columns])
        pt, pb = tail.predict(query[columns]), body.predict(query[columns])
    else:
        gate = CatBoostClassifier().load_model(str(directory / "gate.model"))
        tail = CatBoostRegressor().load_model(str(directory / "tail.model"))
        body = CatBoostRegressor().load_model(str(directory / "body.model"))
        pool = Pool(query[columns], cat_features=cfg["cat"])
        probability, pt, pb = gate.predict_proba(pool)[:, 1], tail.predict(pool), body.predict(pool)
    return np.maximum(probability * pt + (1 - probability) * pb + query.z_aobt.values, 0)


def replay_neural(query: pd.DataFrame, path: Path) -> np.ndarray:
    saved = torch.load(path, map_location=nn.DEV, weights_only=False)
    enc = saved["encoder"]
    ci, xn = enc(query)
    model = nn.Net(enc, xn.shape[1], saved["config"]["cell"]).to(nn.DEV)
    model.load_state_dict(saved["state_dict"])
    p = nn._infer(model, ci, xn) * saved["sd"] + saved["mu"] + query.z_aobt.values
    del model, saved, ci, xn
    torch.cuda.empty_cache()
    return np.maximum(p, 0)


def fit_member(name: str, fit: pd.DataFrame, query: pd.DataFrame, directory: Path,
               selection: pd.DataFrame | None = None) -> tuple[np.ndarray, dict]:
    if not fit.id.is_unique or not query.id.is_unique or fit.id.isin(query.id).any():
        raise ValueError("member fit/query IDs must be unique and disjoint")
    if not np.isfinite(fit.y).all() or (fit.has_plan != 1).any():
        raise ValueError("member training requires finite Lane A labels")
    if not np.isfinite(query.z_aobt).all():
        raise ValueError("member queries require a finite AOBT anchor")
    selection = fit if selection is None else selection
    if not np.array_equal(selection.id.values, fit.id.values):
        raise ValueError("selection feature frame must cover the same permitted rows")
    query = query.assign(y=np.nan)
    spec = specification(name, fit, query, selection)
    cached = load_completed(directory, spec, query)
    if cached is not None:
        return cached
    directory.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    logging.info("Fit %s: %s rows, %s days; query %s", name, len(fit), fit.day.nunique(), len(query))
    write_record(directory / "started.json", {"specification": spec, "fit_rows": len(fit), "fit_days": int(fit.day.nunique())})
    function = neural_fit if name.startswith("nn") else tree_fit
    p, info = function(spec["config"], fit, query, selection, directory, spec)
    p = np.maximum(p, 0)
    if not np.isfinite(p).all():
        raise ValueError("member returned nonfinite predictions")
    pd.DataFrame({"id": query.id.values, "p": p}).to_parquet(directory / "predictions.parquet")
    replay = replay_neural(query, directory / "refit.pt") if name.startswith("nn") else replay_tree(spec["config"], query, directory / "refit")
    np.testing.assert_allclose(p, replay, rtol=1e-7, atol=1e-4)
    elapsed = time.monotonic() - start
    recovered_seconds = info.get("recovery", {}).get("prior_fit_seconds", 0.)
    record = {"specification": spec, "fit_rows": len(fit), "fit_days": int(fit.day.nunique()),
              "runtime_seconds": elapsed + recovered_seconds, "checkpoint_replay": "PASS", **info,
              "artifact_hashes": {str(path.relative_to(directory)): file_hash(path)
                                  for path in sorted(directory.rglob("*")) if path.is_file() and path.suffix in (".pt", ".model", ".parquet")}}
    write_record(directory / "complete.json", record)
    logging.info("Saved %s: %.1f seconds", name, record["runtime_seconds"])
    return p, record
