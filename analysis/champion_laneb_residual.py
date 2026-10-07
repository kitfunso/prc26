"""One declared continuous Lane B correction on scoped recording predictions."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import laneb
import mix
from analysis.champion_meta_weather import RUN, compare, load_block, read, require
from analysis.champion_screen import inputs
from analysis.contender_errors import error_rows
from campaign import file_hash, outer_masks, paired_metrics
from champion import frame_hash, ids_hash, write_record
from champion_laneb import replay_mixture, replay_recording
from champion_neighbours import file_month

NAME = "CP1-B-residual"
OUT = RUN / "laneb-residual"
PLAN = ROOT / "docs/LANEB-RESIDUAL-2026-10-07.html"
WEATHER = ("w_ts", "w_sn", "w_fz", "w_vsby", "w_gust", "w_sn6h", "w_sn24h", "w_fz6h",
           "w_cold12h", "w_cold24h", "w_lovis6h")
ARRIVAL = ("ad_arr_a1", "ad_arr_sc", "ad_arr_tail", "ad_n_arr", "w_arr_a1", "w_arr_sc", "w_arr_tail", "w_n_arr")
COLS = list(dict.fromkeys([*mix.B_CAT, *mix.B_NUM, *laneb.DAY_COLS, *laneb.HIST,
                           *WEATHER, *ARRIVAL, "p_B", "p_B_minus_schedule"]))
PARAMS = {"objective": "regression", "learning_rate": .03, "num_leaves": 15, "max_depth": 4,
          "min_child_samples": 100, "lambda_l2": 20., "n_estimators": 300,
          "random_state": 0, "n_jobs": 8, "verbose": -1}
SHRINK = .5
LIMIT = 1800


def scope(fit: pd.DataFrame, query: pd.DataFrame) -> None:
    require(fit.id.is_unique and query.id.is_unique, "scope IDs must be unique")
    require(not fit.id.isin(query.id).any(), "fit and query IDs overlap")
    require(np.isfinite(fit.y).all(), "fit labels are nonfinite")


def features(rows: pd.DataFrame, baseline: np.ndarray) -> pd.DataFrame:
    require((rows.has_plan == 0).all(), "correction rows must all be Lane B")
    require(len(baseline) == len(rows) and np.isfinite(baseline).all(), "baseline coverage is invalid")
    result = rows.assign(p_B=baseline, p_B_minus_schedule=baseline - rows.z_sched.to_numpy(float))[COLS].copy()
    for column in mix.B_CAT:
        result[column] = result[column].astype("category")
    numeric = result.drop(columns=mix.B_CAT).to_numpy(float)
    require(not np.isinf(numeric).any(), "correction features contain infinity")
    return result


def correction(rows: pd.DataFrame, baseline: np.ndarray, model: lgb.Booster) -> np.ndarray:
    values = np.maximum(baseline + SHRINK * model.predict(features(rows.assign(y=np.nan), baseline)), 0)
    require(np.isfinite(values).all() and (values >= 0).all(), "correction output is invalid")
    return values


def fit_correction(training: pd.DataFrame, query: pd.DataFrame, directory: Path) -> tuple[np.ndarray, dict]:
    scope(training, query)
    x = features(training, training.p_B.to_numpy(float))
    model = lgb.LGBMRegressor(**PARAMS).fit(x, training.y - training.p_B)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "correction.model"
    model.booster_.save_model(str(path))
    saved = lgb.Booster(model_file=str(path))
    require(saved.feature_name() == COLS and saved.num_trees() == 300, "correction recipe changed")
    base = query.p_B.to_numpy(float)
    qx = features(query, base)
    prediction = correction(query, base, model.booster_)
    np.testing.assert_allclose(prediction, correction(query, base, saved), rtol=1e-7, atol=1e-4)
    np.testing.assert_array_equal(prediction, correction(query.assign(y=1e9), base, saved))
    record = {"params": PARAMS, "shrink": SHRINK, "columns": COLS, "model_hash": file_hash(path),
              "fit_ids": ids_hash(training.id), "query_ids": ids_hash(query.id), "fit_rows": len(training),
              "input_hash": frame_hash(training.assign(**{c: x[c] for c in COLS}), ["id", "y", *COLS]),
              "query_input_hash": frame_hash(query.assign(**{c: qx[c] for c in COLS}), ["id", *COLS]),
              "native_replay": "PASS", "query_label_mutation": "PASS"}
    write_record(directory / "fit.json", record)
    return prediction, record


def checked_donor(frame: pd.DataFrame, fit_ids: pd.Index, query_ids: pd.Index, directory: Path) -> tuple[np.ndarray, dict]:
    fit = frame[frame.id.isin(fit_ids)]
    query = frame[frame.id.isin(query_ids)].assign(y=np.nan)
    scope(fit, query)
    record = read(directory / "complete.json")
    spec = record["specification"]
    require(spec["name"] == "mix_cb_wx_raw" and record["prediction_floor"] is False, "inner donor must be raw original mixture")
    require(record["checkpoint_replay"] == "PASS", "inner donor replay was not checked")
    require(spec["fit_ids"] == ids_hash(fit.id) and spec["query_ids"] == ids_hash(query.id), "inner donor scope changed")
    require(spec["fit_frame_hash"] == frame_hash(fit, list(fit.columns)), "inner donor fit input changed")
    require(spec["query_frame_hash"] == frame_hash(query, [c for c in query if c != "y"]), "inner donor query input changed")
    for artifact, digest in record["artifact_hashes"].items():
        require(file_hash(directory / artifact) == digest, f"inner donor bytes changed: {artifact}")
    for engine, digest in spec["engine_hashes"].items():
        require(file_hash(ROOT / engine) == digest, f"recording engine changed: {engine}")
    raw = pd.read_parquet(directory / "predictions.parquet")
    require(np.array_equal(raw.id, query.id), "inner donor order changed")
    np.testing.assert_allclose(raw.p, replay_mixture(record["config"], query, directory / "refit", floor=False), rtol=1e-7, atol=1e-4)
    return np.maximum(raw.p.to_numpy(float), 0), {"path": str(directory.relative_to(ROOT)),
            "record_hash": file_hash(directory / "complete.json"), "artifacts": record["artifact_hashes"], "native_replay": "PASS"}


def original_output(train: pd.DataFrame, query: pd.DataFrame, donor: np.ndarray, directory: Path) -> np.ndarray:
    prediction, probability = replay_recording(train, query.assign(y=np.nan), directory)
    return laneb.hull(query, laneb.hull_prior(train, query, laneb.blend_donor(prediction, probability[:, 0], donor)))


def reconstruct(augmented: pd.DataFrame, fit_ids: pd.Index, query_ids: pd.Index,
                donor: np.ndarray, directory: Path) -> tuple[pd.DataFrame, dict]:
    scoped = laneb.history_rows(augmented, fit_ids)
    train = scoped[scoped.id.isin(fit_ids)]
    query = scoped[scoped.id.isin(query_ids)].assign(y=np.nan)
    scope(train, query)
    inner, stop = laneb.stopping_split(train)
    history_ids = pd.Index(augmented.loc[augmented.id.isin(fit_ids) & ~augmented.day.isin(stop.day.unique()), "id"])
    require(not history_ids.isin(stop.id).any() and not history_ids.isin(query_ids).any(), "stopping history labels leaked")
    selected = laneb.history_rows(augmented, history_ids)
    directory.mkdir(parents=True, exist_ok=True)
    _, iterations, *_ = laneb.predict(selected.loc[inner.index], selected.loc[stop.index].assign(y=np.nan),
                                      selected.loc[stop.index], model_dir=directory / "selection")
    base, _, probability, *_ = laneb.predict(train, query, iters=iterations, model_dir=directory / "refit")
    prediction = laneb.hull(query, laneb.hull_prior(train, query, laneb.blend_donor(base, probability[:, 0], donor)))
    replay = original_output(train, query, donor, directory / "refit")
    np.testing.assert_allclose(prediction, replay, rtol=1e-7, atol=1e-4)
    np.testing.assert_array_equal(replay, original_output(train, query.assign(y=1e9), donor, directory / "refit"))
    hidden = augmented.copy()
    hidden.loc[~hidden.id.isin(fit_ids), "y"] = 1e9
    mutated = laneb.history_rows(hidden, fit_ids)
    pd.testing.assert_frame_equal(scoped[laneb.HIST], mutated[laneb.HIST])
    query = query.assign(p_B=prediction)
    train.to_parquet(directory / "training_context.parquet")
    query.to_parquet(directory / "query_context.parquet")
    record = {"iterations": list(iterations), "fit_ids": ids_hash(train.id), "query_ids": ids_hash(query.id),
              "history_ids": ids_hash(fit_ids), "stopping_ids": ids_hash(stop.id), "selection_history_ids": ids_hash(history_ids),
              "fit_frame_hash": frame_hash(train, list(train.columns)), "native_replay": "PASS",
              "query_label_mutation": "PASS", "hidden_history_label_mutation": "PASS",
              "artifact_hashes": {str(p.relative_to(directory)): file_hash(p) for p in directory.rglob("*")
                                  if p.is_file() and p.suffix in (".model", ".parquet")}}
    write_record(directory / "complete.json", record)
    return query, record


def deadline(start: float, spent: float) -> None:
    if spent + time.monotonic() - start >= LIMIT:
        raise TimeoutError("declared 30-minute whole CPU allocation reached")


def inner_tables(frame: pd.DataFrame, augmented: pd.DataFrame, tag: str, start: float, spent: float) -> tuple[pd.DataFrame, list[dict]]:
    fit, _ = outer_masks(frame, tuple(map(int, tag.split("-"))))
    groups = read(RUN / tag / "complete.json")["meta_groups"]
    pieces, records = [], []
    for group in groups:
        deadline(start, spent)
        label = "meta-" + "-".join(map(str, group))
        inside = fit & ~file_month(frame).isin(group)
        query = fit & file_month(frame).isin(group) & (frame.has_plan == 0)
        fit_ids, query_ids = pd.Index(frame.loc[inside, "id"]), pd.Index(frame.loc[query, "id"])
        donor, donor_check = checked_donor(frame, fit_ids, query_ids, RUN / tag / label / "mixtureB")
        rows, check = reconstruct(augmented, fit_ids, query_ids, donor, OUT / tag / label)
        pieces.append(rows)
        records.append({"group": group, "donor": donor_check, "recording": check})
        logging.info("%s %s original inner B reconstructed and checked", tag, label)
        deadline(start, spent)
    rows = pd.concat(pieces, ignore_index=True)
    expected = frame.loc[fit & (frame.has_plan == 0), "id"]
    require(rows.id.is_unique and set(rows.id) == set(expected), "inner recording coverage changed")
    rows["y"] = rows.id.map(frame.set_index("id").y)
    require(np.isfinite(rows[["y", "p_B"]]).all().all(), "inner recording labels or predictions are nonfinite")
    return rows, records


def run_block(frame: pd.DataFrame, augmented: pd.DataFrame, tag: str, start: float, spent: float) -> dict:
    deadline(start, spent)
    _, rows, provenance = load_block(frame, tag)
    training, inner = inner_tables(frame, augmented, tag, start, spent)
    path = RUN / tag / "outer/laneB"
    record = read(path / "complete.json")
    for artifact, digest in record["artifact_hashes"].items():
        require(file_hash(path / artifact) == digest, f"outer recording bytes changed: {artifact}")
    query = pd.read_parquet(path / "query_context.parquet")
    original = rows.loc[rows.has_plan == 0, ["id", "clean_champion"]]
    require(np.array_equal(query.id, original.id), "outer B context order changed")
    query["p_B"] = original.clean_champion.to_numpy(float)
    directory = OUT / tag
    prediction, correction_check = fit_correction(training, query, directory)
    rows[NAME] = rows.clean_champion
    rows.loc[rows.has_plan == 0, NAME] = prediction
    np.testing.assert_array_equal(rows.loc[rows.has_plan == 1, NAME], rows.loc[rows.has_plan == 1, "clean_champion"])
    output = directory / "evaluation.parquet"
    rows.to_parquet(output)
    deadline(start, spent)
    metrics = rows.assign(month=file_month(rows))
    result = {"months": list(map(int, tag.split("-"))), "inputs": provenance, "inner_recording": inner,
              "correction": correction_check, "output_hash": file_hash(output), "laneA_parity": "PASS",
              "metrics": compare(metrics, NAME), "errors": error_rows(metrics, NAME)}
    result["laneB"] = paired_metrics(metrics[metrics.has_plan == 0], original.clean_champion.to_numpy(float), prediction)
    result["laneB_errors"] = error_rows(metrics[metrics.has_plan == 0].assign(control=original.clean_champion.to_numpy(float)), NAME)
    write_record(directory / "complete.json", result)
    logging.info("%s B residual measured: clean gain %.6f", tag, result["metrics"]["clean_champion"]["gain"])
    return result


def qualified(results: list[dict], pooled: dict) -> bool:
    if pooled["gain"] < 1.5 or min(pooled["day_ci"][0], pooled["moving5_ci"][0]) <= 0:
        return False
    for block in results:
        for reference in ("clean_champion", "control"):
            m = block["metrics"][reference]
            if min(m["gain"], m["leave_best_day_sse_gain"], m["leave_best_5day_episode_sse_gain"]) <= 0:
                return False
            if min(m["hot"]["day_ci"][1], m["hot"]["moving5_ci"][1]) < 0:
                return False
        if any(month["gain"] <= 0 for month in block["metrics"]["clean_champion"]["by_month"].values()):
            return False
    return True


def run(output: Path, start: float, spent: float) -> dict:
    deadline(start, spent)
    frame = inputs()
    augmented = laneb.augment(frame)
    reference_hashes = {tag: file_hash(RUN / tag / "evaluation-audit.json") for tag in ("1-7", "2-3")}
    result = {"recipe": NAME, "status": "running", "blocks": [], "plan_hash": file_hash(PLAN),
              "driver_hash": file_hash(Path(__file__)), "reference_proof_hashes": reference_hashes,
              "params": PARAMS, "shrink": SHRINK, "cpu_limit_seconds": LIMIT, "gpu_seconds": 0}
    for tag in ("1-7", "2-3"):
        result["blocks"].append(run_block(frame, augmented, tag, start, spent))
        write_record(OUT / "progress.json", result)
    deadline(start, spent)
    rows = pd.concat([pd.read_parquet(OUT / tag / "evaluation.parquet") for tag in ("1-7", "2-3")], ignore_index=True)
    pooled = paired_metrics(rows.assign(month=file_month(rows)), rows.clean_champion.to_numpy(float), rows[NAME].to_numpy(float))
    result.update(status="complete", pooled=pooled, verdict="SURVIVOR" if qualified(result["blocks"], pooled) else "REJECT")
    require(reference_hashes == {tag: file_hash(RUN / tag / "evaluation-audit.json") for tag in reference_hashes}, "reference proof bytes changed")
    result["reference_preservation"] = "PASS"
    result["runtime_seconds"] = time.monotonic() - start
    deadline(start, spent)
    write_record(OUT / "results.json", result)
    write_record(output / "prc26-laneb-residual-results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    attempts_path = OUT / "attempts.json"
    attempts = read(attempts_path) if attempts_path.exists() else {"attempts": []}
    require(not any(a["status"] == "running" for a in attempts["attempts"]), "an earlier attempt has no closed accounting")
    require(not (OUT / "results.json").exists(), "completed recipe must not be rerun")
    spent = sum(a["runtime_seconds"] for a in attempts["attempts"])
    start = time.monotonic()
    attempt = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running", "driver_hash": file_hash(Path(__file__))}
    attempts["attempts"].append(attempt)
    write_record(attempts_path, attempts)
    try:
        result = run(args.output, start, spent)
        attempt.update(status="complete", verdict=result["verdict"])
    except Exception as error:
        attempt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        attempt["runtime_seconds"] = time.monotonic() - start
        attempts["total_runtime_seconds"] = spent + attempt["runtime_seconds"]
        write_record(attempts_path, attempts)


if __name__ == "__main__":
    main()
