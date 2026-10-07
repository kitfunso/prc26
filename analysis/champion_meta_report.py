"""Assess fixed meta experiments and export their actual complete-file evidence."""
from __future__ import annotations

import argparse
import html
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.champion_meta_weather import RUN, compare, read, require
from campaign import file_hash
from champion import write_record
from champion_neighbours import file_month


def qualifies(blocks: list[dict], name: str, pooled: dict) -> list[str]:
    failed = []
    primary = pooled["clean_champion"]
    if primary["gain"] < 1.5:
        failed.append("Pooled gain against clean recipe is below 1.5 seconds")
    if min(primary["day_ci"][0], primary["moving5_ci"][0]) <= 0:
        failed.append("Pooled paired intervals against clean recipe include zero")
    for block in blocks:
        tag = "-".join(map(str, block["months"]))
        for reference, values in block["recipes"][name]["metrics"].items():
            if values["gain"] <= 0:
                failed.append(f"{tag}: no gain against {reference}")
            if min(values["leave_best_day_sse_gain"], values["leave_best_5day_episode_sse_gain"]) <= 0:
                failed.append(f"{tag}: {reference} episode support fails")
            if values["hot"]["day_ci"][1] < 0 or values["hot"]["moving5_ci"][1] < 0:
                failed.append(f"{tag}: clear weather harm against {reference}")
    return failed


def assess(family: str) -> dict:
    source = RUN / family / "results.json"
    record = read(source)
    require(record["status"] == "complete", "meta experiment is not complete")
    pieces = []
    for block in record["blocks"]:
        tag = "-".join(map(str, block["months"]))
        path = RUN / family / tag / "evaluation.parquet"
        require(file_hash(path) == block["output_hash"], "candidate evaluation bytes changed")
        rows = pd.read_parquet(path)
        for name in record["recipes"]:
            check = block["recipes"][name]["checks"]
            require(all(check[key] == "PASS" for key in ("checkpoint_replay", "query_label_mutation", "laneB_parity")), "candidate check failed")
            actual = RUN / family / tag / name / "predictions.parquet"
            require(file_hash(actual) == block["recipes"][name]["output_hash"], "saved candidate bytes changed")
        pieces.append(rows.assign(month=file_month(rows)))
    rows = pd.concat(pieces, ignore_index=True)
    require(rows.id.is_unique, "pooled candidate IDs overlap")
    results = {}
    for name in record["recipes"]:
        metrics = compare(rows, name)
        failures = qualifies(record["blocks"], name, metrics)
        results[name] = {"verdict": "REJECT" if failures else "DEVELOPMENT SURVIVOR", "reasons": failures, "pooled": metrics}
    return {"checked_utc": datetime.now(timezone.utc).isoformat(), "family": family, "source_hash": file_hash(source),
            "results": results, "blocks": record["blocks"], "runtime_seconds": record["runtime_seconds"],
            "gpu_fit_seconds": record["gpu_fit_seconds"], "gpu_inference_seconds": record.get("gpu_inference_seconds", 0), "plan_hash": record["declared_plan_hash"],
            "limit": "Adaptive development comparisons. No untouched confirmation, actual ranking candidate, upload or board gain.",
            "command": "python analysis/champion_meta_report.py --output <local output directory>"}


def write_html(records: list[dict], path: Path) -> None:
    sections = []
    for record in records:
        table, detail = [], []
        for name, result in record["results"].items():
            for reference, metric in result["pooled"].items():
                table.append(f"<tr><td>{html.escape(name)}</td><td>{html.escape(reference)}</td><td>{metric['gain']:+.4f}</td><td>{metric['sse_gain']/1e6:+.2f}</td><td>{metric['day_ci']}</td><td>{metric['moving5_ci']}</td></tr>")
            reasons = "; ".join(result["reasons"]) or "Further confirmation must be declared before final preparation."
            detail.append(f"<p><strong>{html.escape(name)}: {result['verdict']}.</strong> {html.escape(reasons)}</p>")
        sections.append(f"<section><h2>{html.escape(record['family'])}</h2>{''.join(detail)}<div class='scroll'><table><thead><tr><th>Recipe</th><th>Reference</th><th>RMSE gain</th><th>SSE gain, million</th><th>Day interval</th><th>Five-day interval</th></tr></thead><tbody>{''.join(table)}</tbody></table></div><p>Stage wall time: {record['runtime_seconds']:.3f} seconds; GPU fits: {record['gpu_fit_seconds']}; neural GPU inference: {record['gpu_inference_seconds']:.3f} seconds. {html.escape(record['limit'])}</p></section>")
    path.write_text("""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PRC26 meta comparison</title><style>body{max-width:1080px;margin:36px auto;padding:0 22px;font:16px/1.6 system-ui;color:#192a30;background:#f4f1e8}h1,h2{line-height:1.2}section{border-top:1px solid #bdc8c4;padding:15px 0}.scroll{overflow-x:auto}table{border-collapse:collapse;white-space:nowrap}td,th{padding:8px;text-align:left;border-bottom:1px solid #bdc8c4}a{color:#164f46}</style><h1>Fixed meta experiments: measured verdicts</h1><p>Complete monthly files and two controls. Signed positive numbers mean lower error. Recipe promotion remains subject to independent coverage and actual candidate checks.</p>""" + "".join(sections) + "<p>Stage 1 inputs: checked. Stage 2 new residual fits: replay, query-label mutation and Lane B parity checked. Stage 3 errors and development gates: measured here. Final ranking file and submission: pending.</p><p><a href='prc26-meta-assessment.json'>Full block, weather, signed-error, airport, episode, lineage and decision record</a>. Regenerate with <code>python analysis/champion_meta_report.py --output &lt;local output directory&gt;</code>.</p></html>", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    records = [assess(family) for family in ("meta-weather", "meta-context", "meta-capacity") if (RUN / family / "results.json").exists()]
    verdict = {"experiments": records, "verdict_lanes": sum(len(record["results"]) for record in records),
               "alpha_times_lanes": .05 * sum(len(record["results"]) for record in records),
               "multiplicity_limit": "N times alpha is descriptive; intervals do not correct adaptive selection or year transfer."}
    write_record(RUN / "meta-assessment.json", verdict)
    write_record(args.output / "prc26-meta-assessment.json", verdict)
    write_html(records, args.output / "prc26-meta-assessment-2026-10-05.html")
    write_html(records, ROOT / "docs/META-ASSESSMENT-2026-10-05.html")
    logging.info("Meta evidence assessed: %s", {name: values["verdict"] for record in records for name, values in record["results"].items()})


if __name__ == "__main__":
    main()
