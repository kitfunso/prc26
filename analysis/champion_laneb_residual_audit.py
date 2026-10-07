"""Close actual saved Lane B correction bytes and export its measured decision."""
from __future__ import annotations

import argparse
import html
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.champion_laneb_residual import COLS, LIMIT, NAME, OUT, PARAMS, PLAN, SHRINK, features, original_output
from analysis.champion_meta_weather import RUN, read, require
from campaign import file_hash
from champion import frame_hash, ids_hash, write_record
from champion_pipeline import MEMBERS


def checked_table(rows: pd.DataFrame, reference: pd.DataFrame, cold_columns: list[str]) -> None:
    retained = [*cold_columns, *MEMBERS, "control", "clean_champion"]
    require(rows.columns.tolist() == [*retained, NAME], "candidate schema differs from declared table")
    require(np.array_equal(rows.id, reference.id), "whole-file IDs changed")
    pd.testing.assert_frame_equal(rows[retained], reference[retained])


def check_block(block: dict, truth: pd.Series, cold_columns: list[str]) -> dict:
    tag = "-".join(map(str, block["months"]))
    directory = OUT / tag
    require(file_hash(directory / "evaluation.parquet") == block["output_hash"], "candidate bytes changed")
    rows = pd.read_parquet(directory / "evaluation.parquet")
    reference = pd.read_parquet(RUN / tag / "evaluation.parquet")
    checked_table(rows, reference, cold_columns)
    np.testing.assert_array_equal(rows.loc[rows.has_plan == 1, NAME], reference.loc[reference.has_plan == 1, "clean_champion"])
    parts = []
    for item in block["inner_recording"]:
        inner = directory / ("meta-" + "-".join(map(str, item["group"])))
        record = read(inner / "complete.json")
        require(record == item["recording"], "inner recording record changed")
        for artifact, digest in record["artifact_hashes"].items():
            require(file_hash(inner / artifact) == digest, f"inner native/context bytes changed: {artifact}")
        train = pd.read_parquet(inner / "training_context.parquet")
        query = pd.read_parquet(inner / "query_context.parquet")
        require(not train.id.isin(query.id).any(), "inner query labels entered refit")
        require(ids_hash(train.id) == record["fit_ids"] and ids_hash(query.id) == record["query_ids"], "inner IDs changed")
        require(frame_hash(train, list(train.columns)) == record["fit_frame_hash"], "inner fitting context changed")
        donor = ROOT / item["donor"]["path"]
        require(file_hash(donor / "complete.json") == item["donor"]["record_hash"], "donor record changed")
        for artifact, digest in item["donor"]["artifacts"].items():
            require(file_hash(donor / artifact) == digest, "donor/native bytes changed")
        baseline = np.maximum(pd.read_parquet(donor / "predictions.parquet").p.to_numpy(float), 0)
        np.testing.assert_allclose(query.p_B, original_output(train, query, baseline, inner / "refit"), rtol=1e-7, atol=1e-4)
        parts.append(query.assign(y=query.id.map(truth)))
    training = pd.concat(parts, ignore_index=True)
    fit = read(directory / "fit.json")
    require(fit == block["correction"] and fit["params"] == PARAMS and fit["shrink"] == SHRINK, "fixed correction record changed")
    x = features(training, training.p_B.to_numpy(float))
    require(frame_hash(training.assign(**{c: x[c] for c in COLS}), ["id", "y", *COLS]) == fit["input_hash"], "residual labels/features changed")
    require(training.id.is_unique and not training.id.isin(rows.id).any(), "residual fitting/query scopes changed")
    require(file_hash(directory / "correction.model") == fit["model_hash"], "correction native bytes changed")
    model = lgb.Booster(model_file=str(directory / "correction.model"))
    require(model.feature_name() == COLS and model.num_trees() == PARAMS["n_estimators"], "native correction feature/tree count changed")
    query = pd.read_parquet(RUN / tag / "outer/laneB/query_context.parquet")
    base = reference.loc[reference.has_plan == 0, "clean_champion"].to_numpy(float)
    qx = features(query, base)
    require(frame_hash(query.assign(**{c: qx[c] for c in COLS}), ["id", *COLS]) == fit["query_input_hash"], "query inputs changed")
    prediction = np.maximum(base + SHRINK * model.predict(qx), 0)
    np.testing.assert_allclose(rows.loc[rows.has_plan == 0, NAME], prediction, rtol=1e-7, atol=1e-4)
    return {"block": tag, "whole_file_ids_labels": "PASS", "laneA_parity": "PASS", "native_bytes_and_replay": "PASS",
            "residual_scope_and_inputs": "PASS", "output_hash": block["output_hash"], "model_hash": fit["model_hash"]}


def failed_gates(result: dict) -> list[str]:
    reasons = []
    pooled = result["pooled"]
    if pooled["gain"] < 1.5:
        reasons.append("Pooled gain is below the declared 1.5-second floor.")
    if min(pooled["day_ci"][0], pooled["moving5_ci"][0]) <= 0:
        reasons.append("Pooled paired intervals do not both exclude zero.")
    for block in result["blocks"]:
        tag = "/".join(map(str, block["months"]))
        for reference, metric in block["metrics"].items():
            if metric["gain"] <= 0:
                reasons.append(f"{tag} loses against {reference}.")
            if min(metric["leave_best_day_sse_gain"], metric["leave_best_5day_episode_sse_gain"]) <= 0:
                reasons.append(f"{tag} lacks leave-largest-episode support against {reference}.")
            if min(metric["hot"]["day_ci"][1], metric["hot"]["moving5_ci"][1]) < 0:
                reasons.append(f"{tag} has clear weather harm against {reference}.")
        for month, metric in block["metrics"]["clean_champion"]["by_month"].items():
            if metric["gain"] <= 0:
                reasons.append(f"File month {month} loses against clean.")
    return reasons


def render(result: dict, audit: dict) -> str:
    entries = []
    for block in result["blocks"]:
        for reference, m in block["metrics"].items():
            entries.append(f"<tr><td>{'/'.join(map(str,block['months']))}</td><td>{html.escape(reference)}</td>"
                           f"<td>{m['gain']:+.4f}</td><td>{m['day_ci'][0]:+.3f} to {m['day_ci'][1]:+.3f}</td>"
                           f"<td>{m['moving5_ci'][0]:+.3f} to {m['moving5_ci'][1]:+.3f}</td>"
                           f"<td>{m['leave_best_5day_episode_sse_gain']/1e6:+.1f}M</td></tr>")
    pooled = result["pooled"]
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PRC26 missing-plan residual result</title><style>body{{max-width:1000px;margin:36px auto;padding:0 22px;font:17px/1.6 system-ui;color:#192a30;background:#f4f1e8}}h1,h2{{line-height:1.2}}section{{border-top:1px solid #bdc8c4;padding:15px 0}}.scroll{{overflow:auto}}table{{border-collapse:collapse;min-width:650px}}td,th{{padding:8px;border-bottom:1px solid #bdc8c4;text-align:left}}code{{overflow-wrap:anywhere;font-size:14px}}</style>
<h1>Missing-plan correction: {html.escape(result['verdict'])}</h1><p>Pooled whole-file gain against clean: {pooled['gain']:+.4f} seconds. Day interval [{pooled['day_ci'][0]:+.3f}, {pooled['day_ci'][1]:+.3f}]; five-day interval [{pooled['moving5_ci'][0]:+.3f}, {pooled['moving5_ci'][1]:+.3f}]. Positive gain means lower error. These adaptively reused development files do not establish a scored improvement.</p>
<section><h2>Complete files and episode support</h2><div class="scroll"><table><thead><tr><th>Files</th><th>Reference</th><th>Gain, seconds</th><th>Day interval</th><th>Five-day interval</th><th>Without best five-day episode, SSE</th></tr></thead><tbody>{''.join(entries)}</tbody></table></div><p>Failed gates: {html.escape(' '.join(audit['failed_gates']) or 'None.')}</p><p>The complete JSON records both months, both lanes, weather intervals, signed errors, airport and recorded-outcome slices, and influential rows. Original Lane A output is unchanged. This is one fixed continuous residual test, with original recording predictions reconstructed independently of their query labels. No retuning follows.</p></section>
<section><h2>Stages and actual artifacts</h2><p>Declaration and reviewed construction: done. Both outer development fits and comparisons: done. Saved donor/native/context bytes, original B replay, residual input/target coverage, complete-file IDs/labels, final native correction replay and unchanged Lane A/reference proofs: {audit['closure']}. Full screen and closure wall exposure: {audit['whole_stage_seconds']:.3f} seconds, inside the declared CPU allocation. GPU work: zero.</p><p>Further confirmation and ranking preparation are {'conditional on a separate priced declaration' if result['verdict']=='SURVIVOR' else 'skipped because this fixed recipe failed its gate'}. No submission of this screen has occurred. Keith has authorized genuine qualified submissions, but the scientific and actual-file requirements still apply. Exact scored v53 and the sealed prior local-readiness package remain preserved. Recipe freeze and campaign release checks remain open.</p></section>
<section><h2>Evidence</h2><p><a href="prc26-laneb-residual-results.json">Full measured results</a> · <a href="prc26-laneb-residual-audit.json">Actual artifact closure and attempt accounting</a> · <a href="prc26-laneb-residual-plan.html">Prospective fixed recipe</a>.</p><p>Regenerate the closure/report with <code>python analysis/champion_laneb_residual_audit.py --output &lt;local outputs directory&gt;</code>. The completed recipe must not be rerun. Checked {html.escape(audit['checked_utc'])}.</p></section></html>'''


def close(output: Path, start: float, prior: float) -> None:
    result, attempts = read(OUT / "results.json"), read(OUT / "attempts.json")
    require(result["status"] == "complete" and all(a["status"] != "running" for a in attempts["attempts"]), "screen has not closed")
    require(result["plan_hash"] == file_hash(PLAN), "prospective plan changed after fitting")
    cold = ROOT / "data/runs/contender-20261003/cold_features.parquet"
    require(file_hash(cold) == read(cold.with_name("inputs.json"))["feature_hash"], "cold labels/features changed")
    truth = pd.read_parquet(ROOT / "data/runs/contender-20261003/cold_features.parquet", columns=["id", "y"]).set_index("id").y
    checks = [check_block(block, truth, pq.read_schema(cold).names) for block in result["blocks"]]
    hashes = {tag: file_hash(RUN / tag / "evaluation-audit.json") for tag in result["reference_proof_hashes"]}
    require(hashes == result["reference_proof_hashes"], "preserved reference proofs changed")
    whole = attempts["total_runtime_seconds"] + prior + time.monotonic() - start
    require(whole < LIMIT, "screen plus closure exceeds declared cumulative allocation")
    audit = {"checked_utc": datetime.now(timezone.utc).isoformat(), "closure": "PASS", "blocks": checks,
             "reference_proof_hashes": hashes, "attempts": attempts, "whole_stage_seconds": whole,
             "cpu_limit_seconds": LIMIT, "gpu_seconds": 0, "result_hash": file_hash(OUT / "results.json"),
             "audit_driver_hash": file_hash(Path(__file__)), "verdict": result["verdict"], "failed_gates": failed_gates(result)}
    require(bool(audit["failed_gates"]) == (result["verdict"] == "REJECT"), "independent gate assessment disagrees")
    write_record(OUT / "completion-audit.json", audit)
    write_record(output / "prc26-laneb-residual-audit.json", audit)
    (output / "prc26-laneb-residual-plan.html").write_bytes(PLAN.read_bytes())
    page = render(result, audit)
    (output / "prc26-laneb-residual-result.html").write_text(page, encoding="utf-8")
    (ROOT / "docs/LANEB-RESIDUAL-RESULT-2026-10-07.html").write_text(page, encoding="utf-8")
    logging.info("Actual Lane B residual artifacts and full attempt exposure PASS; verdict %s", result["verdict"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    path = OUT / "audit-attempts.json"
    attempts = read(path) if path.exists() else {"attempts": []}
    prior = sum(a["runtime_seconds"] for a in attempts["attempts"])
    attempt = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "driver_hash": file_hash(Path(__file__))}
    start = time.monotonic()
    try:
        close(args.output, start, prior)
        attempt["status"] = "complete"
    except Exception as error:
        attempt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        attempt["runtime_seconds"] = time.monotonic() - start
        attempts["attempts"].append(attempt)
        attempts["total_runtime_seconds"] = prior + attempt["runtime_seconds"]
        write_record(path, attempts)


if __name__ == "__main__":
    main()
