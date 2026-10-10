"""Price final preparation from saved fits without starting model training."""
from __future__ import annotations

import argparse
import html
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campaign import file_hash
from champion import write_record
from champion_neighbours import file_month, month_groups
from champion_pipeline import MEMBERS
from data import DATA
from features import CAT
from train import stopping_split
from weather import RAW

RUN = DATA / "runs/clean-champion-20261003"
INPUT = DATA / "runs/contender-20261003"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ranking_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest, support = read(INPUT / "inputs.json"), read(INPUT / "ranking_support.json")
    sources = {INPUT / "cold_features.parquet": manifest["feature_hash"],
               INPUT / "cold_ranking.parquet": support["feature_hash"],
               DATA / "ranking.parquet": support["ranking_hash"]}
    sources.update({RAW / f"{station}.csv": digest for station, digest in support["weather_hashes"].items()})
    for path, digest in sources.items():
        if file_hash(path) != digest:
            raise ValueError(f"preparation input changed: {path}")
    train = pd.read_parquet(INPUT / "cold_features.parquet", columns=["id", "file", "day", "has_plan"])
    rank = pd.read_parquet(INPUT / "cold_ranking.parquet")
    raw = pd.read_parquet(DATA / "ranking.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt"])
    dep = raw[raw.PHASE_mvt == "DEP"]
    template = pd.read_parquet(DATA / "submitting.parquet")
    if list(template.columns) != ["MVT_ID_mvt", "TAXITIME_SEC_mvt"] or not template.MVT_ID_mvt.is_unique:
        raise ValueError("ranking template contract changed")
    if not train.id.is_unique or not rank.id.is_unique or train.id.isin(rank.id).any():
        raise ValueError("training/ranking IDs are not unique and disjoint")
    if not np.array_equal(rank.id.values, dep.MVT_ID_mvt.values) or set(rank.id) != set(template.MVT_ID_mvt):
        raise ValueError("cold ranking IDs do not cover the actual ranking/template files")
    if not rank.y.isna().all() or not dep[["TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt"]].isna().all().all():
        raise ValueError("ranking departure labels are visible")
    if not np.isfinite(rank.loc[rank.has_plan == 1, "z_aobt"]).all() or not set(rank.has_plan).issubset({0, 1}):
        raise ValueError("ranking lane/anchor contract failed")
    if not set(CAT).issubset(rank.columns) or not rank.file.eq("ranking.parquet").all():
        raise ValueError("ranking feature/file metadata is incomplete")
    details = {"checks": "PASS", "training_rows": len(train), "ranking_rows": len(rank),
               "ranking_laneA_rows": int((rank.has_plan == 1).sum()), "ranking_laneB_rows": int((rank.has_plan == 0).sum()),
               "template_order_matches": bool(np.array_equal(rank.id.values, template.MVT_ID_mvt.values)),
               "source_hashes": {str(path.relative_to(ROOT)): digest for path, digest in sources.items()},
               "template_hash": file_hash(DATA / "submitting.parquet"),
               "limit": "Existing cold ranking features are reused; this does not rebuild their features or validate model inference.",
               "split_metadata": "The existing cold ranking cache says train; the final entry point must assign rank without changing the cache."}
    return train, rank, details


def observed_rates() -> tuple[dict, dict]:
    rates = defaultdict(list)
    observations = []
    proofs = {}
    for tag in ("1-7", "2-3"):
        block = RUN / tag
        proof_path = block / "evaluation-audit.json"
        if not proof_path.exists():
            continue
        proof = read(proof_path)
        if proof["scope_checks"] != "PASS":
            raise ValueError("cost source has no passing scope proof")
        proofs[tag] = {"hash": file_hash(proof_path), **{key: proof["costs"][key] for key in ("component_wall_seconds", "gpu_family_wall_seconds")}}
        for component in proof["costs"]["components"]:
            path = (block / component["path"] / "complete.json").resolve()
            if not path.is_relative_to(block.resolve()):
                raise ValueError("cost record points outside its measured block")
            saved = read(path)
            started = read(path.with_name("started.json"))
            if started["specification"] != saved["specification"]:
                raise ValueError(f"saved cost scope changed: {path}")
            if saved["runtime_seconds"] != component["seconds"] or saved["checkpoint_replay"] != "PASS":
                raise ValueError(f"saved cost disagrees with measured proof: {path}")
            rates[component["name"]].append(component["seconds"] / started["fit_rows"])
            observations.append({**component, "block": tag, "fit_rows": started["fit_rows"], "fit_days": started["fit_days"],
                                 "record_hash": file_hash(path), "start_record_hash": file_hash(path.with_name("started.json"))})
    seconds = sum(item["seconds"] for item in observations)
    gpu = sum(item["seconds"] for item in observations if item["gpu_family"])
    if not math.isclose(seconds, sum(p["component_wall_seconds"] for p in proofs.values())) or not math.isclose(gpu, sum(p["gpu_family_wall_seconds"] for p in proofs.values())):
        raise ValueError("saved component cost totals changed")
    return {name: {"n": len(values), "median": float(np.median(values)), "p90": float(np.quantile(values, .9))}
            for name, values in rates.items()}, {"proofs": proofs,
                                              "component_wall_seconds": seconds, "gpu_family_wall_seconds": gpu,
                                              "observations": observations}


def planned_components(train: pd.DataFrame, rank: pd.DataFrame) -> tuple[list[dict], tuple]:
    groups = month_groups(train, pd.Index(train.id))
    plans = []
    for group in (None, *groups):
        fit = train if group is None else train[~file_month(train).isin(group)]
        query = rank if group is None else train[file_month(train).isin(group)]
        tag = "ranking" if group is None else "meta-" + "-".join(map(str, group))
        fit_a = fit[fit.has_plan == 1]
        for name in MEMBERS:
            plans.append({"scope": tag, "name": name, "fit_rows": len(fit_a)})
        if (query.has_plan == 0).any():
            plans.append({"scope": tag, "name": "mix_cb_wx" if group is None else "mix_cb_wx_raw", "fit_rows": len(fit)})
        selection, _ = stopping_split(fit_a)
        for label, source in (("full-source", fit_a), ("stopping-excluded-source", selection)):
            donor_groups = (*month_groups(source, pd.Index(source.id)), None)
            for donor_group in donor_groups:
                rows = source if donor_group is None else source[~file_month(source).isin(donor_group)]
                donor_tag = "full" if donor_group is None else "group-" + "-".join(map(str, donor_group))
                for name in ("ra_cb_wz", "ra_lgb_wz", "nn3_wz"):
                    plans.append({"scope": f"{tag}/{label}/{donor_tag}", "name": name, "fit_rows": len(rows)})
    plans.append({"scope": "ranking/final-recording-B", "name": "laneb", "fit_rows": int((train.has_plan == 0).sum())})
    return plans, groups


def pricing(plans: list[dict], rates: dict) -> dict:
    for item in plans:
        rate = rates[item["name"]]
        item.update(gpu_family=item["name"].startswith(("ra_cb", "nn", "mix_cb")),
                    median_seconds=item["fit_rows"] * rate["median"], p90_seconds=item["fit_rows"] * rate["p90"])
    return {"component_count": len(plans), "family_counts": dict(Counter(item["name"] for item in plans)),
            **{f"{kind}_{quantile}_seconds": sum(item[f"{quantile}_seconds"] for item in plans if kind == "all" or item["gpu_family"])
               for kind in ("all", "gpu") for quantile in ("median", "p90")}, "rates": rates, "components": plans,
            "method": "Each planned fit-row count times its family's observed seconds per fit row; median and 90th-percentile rate sensitivity.",
            "limit": "Planning sensitivity, not a runtime interval or a guaranteed fit. Stopping budgets and query inference costs vary. Feature construction, residual fitting, validation and other CPU overhead are not priced."}


def write_html(record: dict, path: Path) -> None:
    costs, inputs, budget = record["pricing"], record["inputs"], record["budget_snapshot"]
    rows = "".join(f"<tr><td>{html.escape(name)}</td><td>{count}</td><td>{costs['rates'][name]['n']}</td></tr>"
                   for name, count in costs["family_counts"].items())
    gaps = "".join(f"<li>{html.escape(gap)}</li>" for gap in record["remaining_work"])
    path.write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PRC26 final preparation</title><link rel="icon" href="data:,"><style>body{{max-width:960px;margin:36px auto;padding:0 22px;background:#f4f1e8;color:#192a30;font:17px/1.6 system-ui}}h1,h2{{line-height:1.25}}section{{border-top:1px solid #bdc8c4;padding:16px 0}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:8px;border-bottom:1px solid #bdc8c4}}code{{overflow-wrap:anywhere;font-size:14px}}a{{color:#164f46}}</style>
<h1>Prepare the final ranking build</h1><p>Checked {html.escape(record['checked_utc'])}. Both reference blocks are measured and the unchanged recipe failed winter promotion. The four fixed original-member follow-ups are rejected. The exact scored fallback file is validated separately; no stronger recipe or new fitted ranking candidate is ready.</p>
<section><h2>Ranking inputs: checked for preparation</h2><p>Existing cold ranking features, actual ranking bytes and weather bytes match their recorded sources. Ranking targets and departure off-block labels are hidden. Training/ranking IDs are unique and disjoint. Ranking IDs cover the submission template.</p><p>Training departures: {inputs['training_rows']:,}. Ranking departures: {inputs['ranking_rows']:,}, including {inputs['ranking_laneA_rows']:,} Lane A and {inputs['ranking_laneB_rows']:,} Lane B. Template ordering matches: {inputs['template_order_matches']}.</p><p>{html.escape(inputs['limit'])} {html.escape(inputs['split_metadata'])}</p></section>
<section><h2>Complete-chain planning price</h2><p>Measured completed-reference GPU-family component cost: {record['observed']['gpu_family_wall_seconds']/3600:.2f} hours. Prospective all-training recipe: {costs['component_count']} saved components, with inner groups {html.escape(str(record['meta_groups']))}.</p><p>GPU-family planning estimate: {costs['gpu_median_seconds']/3600:.2f} hours at median observed rates; {costs['gpu_p90_seconds']/3600:.2f} hours at each family's 90th-percentile rate. All component wall work: {costs['all_median_seconds']/3600:.2f} / {costs['all_p90_seconds']/3600:.2f} hours. The declared final-preparation planning reserve is 16 GPU-family hours.</p><p>{html.escape(costs['method'])} {html.escape(costs['limit'])}</p><table><thead><tr><th>Family</th><th>Planned fits</th><th>Measured rate samples</th></tr></thead><tbody>{rows}</tbody></table></section>
<section><h2>Budget and remaining work</h2><p>Saved GPU-family cost at this snapshot: {budget['used_seconds']/3600:.2f} hours globally; {budget['block']['used_seconds']/3600:.2f} of 12 hours in February/March. The complete audit includes all original mixture-B components; its full subtotal governs planning. No allocation is expanded by this report.</p><ol>{gaps}</ol><p>Reference/evaluation: checked for January/July. Confirmation: measured for both declared blocks, with winter gate failure. Preparation inputs and planning price: checked. Ranking driver, fitted candidate, file/replay checks and upload: pending.</p></section>
<section><h2>Reproduce</h2><p><code>python analysis/champion_preparation.py --output &lt;local output directory&gt;</code></p><p><a href="prc26-final-preparation-2026-10-04.json">Input hashes, component costs and planning record</a>. This build uses the unchanged method only to price a possible final fit; it does not declare a new experiment, select a recipe or authorize an upload.</p></section></html>""", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    train, rank, input_record = ranking_inputs()
    rates, observed = observed_rates()
    plans, groups = planned_components(train, rank)
    try:
        file_month(rank[["file"]].iloc[:1])
    except ValueError:
        month_support = "Current training-file month parser rejects ranking.parquet; fix its input contract before ranking fitting."
    else:
        month_support = "Ranking file-month parser accepts the supplied filename; the final entry point still needs explicit training/query scope checks."
    record = {"checked_utc": datetime.now(timezone.utc).isoformat(), "inputs": input_record,
              "observed": observed, "meta_groups": groups, "pricing": pricing(plans, rates),
              "budget_snapshot": read(RUN / "budget.json"), "script_hash": file_hash(Path(__file__)),
              "remaining_work": ["Both reference blocks and four fixed CP4 follow-ups are measured and fail promotion. Preserve v53 through recipe freeze; no rejected recipe is retuned.",
                                 "Exact fallback file/template/byte checks pass in fallback-readiness.json. Keep its immutable copy and complete local release/write-up readiness without upload or publication.",
                                 "A stronger recipe still requires the declared qualification gates and supported historical comparison before a new ranking build.",
                                 month_support,
                                 "Add the ranking entry point, explicit rank split, all-eligible-training scopes, immutable source manifest and bounded component accounting after confirmation source checks close.",
                                 "Build and replay the full nested donor/member/meta/recording chain; preserve all completed confirmation proofs.",
                                 "Assemble the actual template IDs, run the submission-file checks, and present the exact candidate hash for per-upload sign-off."]}
    args.output.mkdir(parents=True, exist_ok=True)
    directory = RUN / "preparation"
    write_record(directory / "readiness.json", record)
    write_record(args.output / "prc26-final-preparation-2026-10-04.json", record)
    write_html(record, directory / "readiness.html")
    target = args.output / "prc26-final-preparation-2026-10-04.html"
    target.write_bytes((directory / "readiness.html").read_bytes())
    (ROOT / "docs/FINAL-PREPARATION-2026-10-04.html").write_bytes(target.read_bytes())
    logging.info("Preparation inputs PASS; %s planned components; GPU price %.2f / %.2f hours",
                 record["pricing"]["component_count"], record["pricing"]["gpu_median_seconds"] / 3600,
                 record["pricing"]["gpu_p90_seconds"] / 3600)


if __name__ == "__main__":
    main()
