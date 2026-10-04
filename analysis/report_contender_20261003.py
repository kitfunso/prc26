from __future__ import annotations

import hashlib
import html
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from campaign import paired_metrics

RUN = ROOT / "data/runs/contender-20261003"
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "docs"


def read(name: str) -> dict:
    path = RUN / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def aggregate(prefix: str, names: tuple[str, ...]) -> dict:
    if not all((RUN / f"{prefix}-{tag}.parquet").exists() for tag in ("2-3", "9-10")):
        return {}
    frames = [pd.read_parquet(RUN / f"{prefix}-{tag}.parquet") for tag in ("2-3", "9-10")]
    both = pd.concat(frames, ignore_index=True)
    return {name: paired_metrics(both, both.control.values, both[name].values) for name in names if name in both and both[name].notna().all()}


def render_donor(audit: dict, selection: bool = False) -> str:
    if not audit:
        return ""
    tag = "selection-donor" if selection else "first-donor"
    title = "Stopping-day neighbour donor" if selection else "First neighbour donor"
    meta_group = audit.get("excluded_meta_months", ())
    if meta_group:
        tag = "meta-" + "-".join(map(str, meta_group)) + ("-selection" if selection else "") + "-donor"
        title = "Nested " + ("stopping-day " if selection else "") + "neighbour donor: " + "/".join(map(str, meta_group))
    exclusion = audit.get("excluded_downstream_stopping", {})
    scope_note = (f"The downstream model's {exclusion['rows']:,} stopping rows over {exclusion['days']} days are excluded from every donor fit. "
                  if selection else "The fit uses the full permitted training scope. ")
    if meta_group:
        scope_note += "The inner query months " + ", ".join(map(str, meta_group)) + " are excluded from every donor fit. "
    cost_rows = "".join(f"<tr><td>{html.escape(scope['scope'])}</td><td>{scope['fit_rows']:,}</td><td>{scope['query_rows']:,}</td><td>{scope['component_wall_seconds']/60:.1f}</td><td>{scope['gpu_family_wall_seconds']/60:.1f}</td></tr>"
                        for scope in audit["scopes"])
    return f"""<section id="{tag}"><h2>{title}: saved and checked</h2><p>{scope_note}The original CatBoost, LightGBM and Huber neural blend has all three cross-fit groups and its full permitted-scope fit saved. Each native checkpoint passed prediction replay. The separate audit checks actual cold-input bytes, ordered permitted/query IDs, training-only stopping coverage, saved prediction bytes and the declared raw/floored arithmetic. January and July labels are excluded from donor fits. Its saved neighbour features reproduce from those actual predictions, including the original missing values for empty windows. Missing-value positions match; infinities are rejected.</p><p>Recorded component wall time: {audit['component_wall_seconds']/60:.1f} minutes. GPU-family subtotal: {audit['gpu_family_wall_seconds']/60:.1f} minutes. {html.escape(audit['accounting'])} This is one completed donor; the remaining donor chains add cost.</p><div class="scroll"><table><thead><tr><th>Scope</th><th>Fit rows</th><th>Query rows</th><th>Component minutes</th><th>GPU-family minutes</th></tr></thead><tbody>{cost_rows}</tbody></table></div><p>This donor component is checked. The full-block evaluation below records the later comparison; confirmation and candidate preparation remain open. No new scorer gain is established. Native replay is retained from member completion records; this audit does not run a new model fit or GPU replay.</p><p><a href="prc26-{tag}-audit.json">Donor scope, arithmetic and cost record</a>. Regenerate with <code>{html.escape(audit['command'])}</code>.</p></section>"""


def render_outer_members(members: tuple[dict, ...]) -> str:
    if len(members) != 3:
        return ""
    rows = []
    for member in members:
        budget = "/".join(map(str, member["iterations"])) if member["iterations"] else f"{member['epochs']} epochs"
        rows.append(f"<tr><td>{html.escape(member['member'])}</td><td>{member['fit_rows']:,} / {member['fit_days']}</td><td>{html.escape(budget)}</td><td>{member['laneA_descriptive_errors']['rmse']:.3f}</td><td>{member['runtime_seconds']/60:.1f}</td></tr>")
    return f"""<section id="outer-members"><h2>Outer Lane A members: saved and checked</h2>
<p>The tuned CatBoost, LightGBM and Huber neural members are saved. Their ordered January/July Lane A prediction IDs, training-only stopping coverage, every-row/day refit coverage and finite nonnegative outputs pass. Each native checkpoint passed replay before completion. The new LightGBM and neural audits also check actual saved-model bytes, the declared configurations and fitting source hashes, and reproduce fit/query/stopping input fingerprints from the cold features and both checked neighbour donors. The scored champion and its frozen copy still match their recorded hashes.</p>
<div class="scroll"><table><thead><tr><th>Member</th><th>Refit rows / days</th><th>Selected budget</th><th>Lane A RMSE, seconds</th><th>Component minutes</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>These errors are descriptive member diagnostics on reused development files. These component checks precede the full-block evaluation below, which records the later meta fits and final Lane B assembly. Confirmation and candidate preparation remain open. No promotion or scorer gain is established.</p>
<p>Native replay evidence is retained from completion records; the audits do not run new fits or GPU replays. LightGBM cost is CPU wall time; CatBoost and neural wall time provide a conservative GPU-family cost proxy. <a href="prc26-ra_lgb_pw_t7-audit.json">LightGBM scope/input/cost record</a> · <a href="prc26-nn3_pw_t7-audit.json">Neural scope/input/cost record</a>. Regenerate with <code>python analysis/champion_checkpoint_audit.py --member ra_lgb_pw_t7</code> and <code>python analysis/champion_checkpoint_audit.py --member nn3_pw_t7</code>.</p></section>"""


def render_assembly(audit: dict) -> str:
    tag = "meta-" + "-".join(map(str, audit["meta_group"]))
    return f"""<section id="{tag}-assembly"><h2>Inner complete-file predictions: {'/'.join(map(str, audit['meta_group']))}</h2><p>All three tuned Lane A members and the original mixture-B donor are saved with native replay PASS. The assembled table covers every ordered ID in its {len(audit['query_files'])} supplied monthly files: {audit['query_rows']:,} rows, including {audit['laneA_rows']:,} Lane A and {audit['laneB_rows']:,} Lane B rows. Permitted/stopping/query scopes, actual native model and prediction bytes, configurations and fitting source hashes pass.</p><p>The actual raw Lane A predictions match their saved floored predictions after the declared floor. The assembled columns equal the raw values exactly. Lane B is identical in all three columns and equals its separately fitted raw mixture prediction. Negative raw values remain permitted until the later blend floor. The audit runs no model fit or GPU replay.</p><p>This inner assembly is checked. The full-block evaluation below records the later NNLS, residual recalibration, final recording-B override and outer error analysis. Confirmation and candidate preparation remain open. {html.escape(audit['limit'])}</p><p><a href="prc26-{tag}-assembly-audit.json">Complete-file assembly proof</a>. Regenerate with <code>{html.escape(audit['command'])}</code>.</p></section>"""


def render_evaluation(audit: dict) -> str:
    tag = "-".join(map(str, audit["months"]))
    metric, costs = audit["metrics"], audit["costs"]
    rows = []
    for family in ("month", "lane", "hot"):
        for label, values in audit["errors"]["slices"][family].items():
            rows.append(f"<tr><td>{family}: {html.escape(label)}</td><td>{values['n']:,}</td><td>{values['control_rmse']:.3f}</td><td>{values['candidate_rmse']:.3f}</td><td>{values['gain']:+.3f}</td></tr>")
    return f"""<section id="reference-{tag}"><h2>Full reference {tag}: checked and measured</h2><p>Complete supplied files: {metric['n']:,} rows. Whole-file RMSE improves from {metric['control_rmse']:.3f} to {metric['candidate_rmse']:.3f} seconds, a gain of {metric['gain']:.3f}. Paired SSE gain: {metric['sse_gain']/1e9:.3f} billion seconds squared.</p><p>Calendar-day interval: [{metric['day_ci'][0]:.3f}, {metric['day_ci'][1]:.3f}]. Month-stratified five-day interval: [{metric['moving5_ci'][0]:.3f}, {metric['moving5_ci'][1]:.3f}]. Both use 2,000 paired resamples. Leave-largest-day SSE gain: {metric['leave_best_day_sse_gain']/1e6:.1f} million; leave-largest-five-day-episode: {metric['leave_best_5day_episode_sse_gain']/1e6:.1f} million. File/clock-month mismatches: {audit['file_clock_month_mismatches']}.</p><div class="scroll"><table><thead><tr><th>Slice</th><th>Rows</th><th>Control RMSE</th><th>Clean recipe RMSE</th><th>Gain, seconds</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p>Actual complete-file coverage, cold labels, raw inner tables, whole-file NNLS normalization and applied six-decimal weights pass. The saved 300-tree residual model reproduces Lane A output. Final Lane B context reproduces from permitted history; training-only stopping, native model/prediction bytes, final recording replay and identical B values in control/recipe pass. Airport, signed-error, recording-class and schedule-gap slices are in the proof.</p><p>Saved component wall time: {costs['component_wall_seconds']/3600:.2f} hours. GPU-family subtotal: {costs['gpu_family_wall_seconds']/3600:.2f} hours. {html.escape(costs['accounting'])}</p><p>Stage 1 reference: checked for this block. Stage 2 complete-file evaluation: checked for this block. Stage 3 confirmation and final candidate: pending. {html.escape(audit['limit'])}</p><p><a href="prc26-{tag}-evaluation-audit.json">Full scope, replay, error and cost proof</a>. Regenerate with <code>{html.escape(audit['command'])}</code>.</p></section>"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    template = OUT / "prc26-status-2026-10-03.html"
    old = (template if template.exists() else ROOT / "docs/STATUS-2026-10-03.html").read_text(encoding="utf-8")
    css = old.split("<style>")[1].split("</style>")[0]
    inputs, board = read("inputs.json"), json.loads((ROOT / "data/board.json").read_text(encoding="utf-8"))
    records = {"inputs": inputs, "scope_checks": read("scope_checks.json"), "neural_screen": read("neural.json"),
               "disruption": read("disruption.json"), "meta_screen": read("meta.json"),
               "neural_full": read("neural-full/neural.json"), "neural_full_progress": [read(f"neural-full/progress-{tag}.json") for tag in ("2-3", "9-10")],
               "neural_artifact_checks": read("neural-full/artifact_checks.json"), "meta_full": read("neural-full/meta.json"),
               "neural_full_decision": read("neural-full/decision.json"), "error_analysis": read("error_slices.json"),
               "mechanism_blocks": [read(f"screen-{tag}.json") for tag in ("2-3", "9-10")],
               "ranking_support": read("ranking_support.json"), "release_preflight": read("release_preflight.json"),
               "leaderboard": board, "recorded_utc": datetime.now(timezone.utc).isoformat(),
               "launch_base_commit": "9d0bc36bb8813b82f917f78e3810348fb70561f1",
               "implementation_checkpoints": {"validation": "1f2f3cb", "submission": "68645a0"},
               "source_limit": "Full neural fit modules stayed fixed; runner validation/stop guards were added after launch and artifacts checked independently.",
               "tests": {"command": "python -m pytest -q", "passed": 62, "seconds": 46.53,
                         "subsequent_focused_meta_tests": 2, "optimized_submission_tests": 26},
               "lint": "ruff check --select F on new research modules and changed model modules: PASS",
               "champion_checkpoint": json.loads((ROOT / "data/runs/cp0-laneb-scope-20261003/manifest.json").read_text(encoding="utf-8"))}
    records["pooled"] = {}
    for prefix, names in (("screen", ("CP1-hierarchy", "CP1-calibration", "CP1-outcomes")),
                          ("disruption", ("CP2-weather",)), ("neural", ("CP3-huber", "CP3-mse")),
                          ("meta", ("established", "simplex", "shrunk"))):
        records["pooled"].update(aggregate(prefix, names))
    records["full_pooled"] = {**aggregate("neural-full/neural", ("CP3-huber", "CP3-mse")),
                              **aggregate("neural-full/meta", ("established", "simplex", "shrunk"))}
    records["source_hashes"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [ROOT / n for n in ("campaign.py", "mechanisms.py", "disruption.py", "neighbours.py", "nn.py", "laneb.py",
                                                 "analysis/contender_screen.py", "analysis/neural_screen.py", "analysis/disruption_screen.py", "analysis/meta_screen.py",
                                                 "analysis/ranking_support.py", "analysis/validate_neural_artifacts.py", "analysis/contender_errors.py", "board.py")]}
    records["limits"] = ["Existing splits reused; no fresh 2026 labels.", "CPU control/proxy donor differs from clean-v53.",
                          "Original champion fitted-checkpoint lineage incomplete.", "No candidate uploaded; no board gain established."]
    active_path = ROOT / "data/runs/clean-champion-20261003/active.json"
    active = json.loads(active_path.read_text(encoding="utf-8")) if active_path.exists() else {}
    records["clean_champion_active"] = active
    first_path = ROOT / "data/runs/clean-champion-20261003/first-member-audit.json"
    first = json.loads(first_path.read_text(encoding="utf-8")) if first_path.exists() else {}
    records["clean_champion_first_member"] = first
    if first:
        shutil.copy2(first_path, OUT / "prc26-first-member-audit.json")
    other_members = []
    for name in ("ra_lgb_pw_t7", "nn3_pw_t7"):
        path = ROOT / f"data/runs/clean-champion-20261003/{name}-audit.json"
        if path.exists():
            other_members.append(json.loads(path.read_text(encoding="utf-8")))
            shutil.copy2(path, OUT / f"prc26-{name}-audit.json")
    records["clean_champion_other_members"] = other_members
    donor_path = ROOT / "data/runs/clean-champion-20261003/first-donor-audit.json"
    donor = json.loads(donor_path.read_text(encoding="utf-8")) if donor_path.exists() else {}
    records["clean_champion_first_donor"] = donor
    if donor:
        shutil.copy2(donor_path, OUT / "prc26-first-donor-audit.json")
    selection_path = ROOT / "data/runs/clean-champion-20261003/selection-donor-audit.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else {}
    records["clean_champion_selection_donor"] = selection
    if selection:
        shutil.copy2(selection_path, OUT / "prc26-selection-donor-audit.json")
    meta_donors = []
    for path in sorted((ROOT / "data/runs/clean-champion-20261003").glob("meta-*-donor-audit.json")):
        meta_donors.append(json.loads(path.read_text(encoding="utf-8")))
        shutil.copy2(path, OUT / f"prc26-{path.name}")
    records["clean_champion_meta_donors"] = meta_donors
    assemblies = []
    for path in sorted((ROOT / "data/runs/clean-champion-20261003").glob("meta-*-assembly-audit.json")):
        assemblies.append(json.loads(path.read_text(encoding="utf-8")))
        shutil.copy2(path, OUT / f"prc26-{path.name}")
    records["clean_champion_meta_assemblies"] = assemblies
    evaluations = []
    for path in sorted((ROOT / "data/runs/clean-champion-20261003").glob("*/evaluation-audit.json")):
        audit = json.loads(path.read_text(encoding="utf-8"))
        summary = {**audit, "errors": {"slices": audit["errors"]["slices"]}}
        evaluations.append(summary)
        shutil.copy2(path, OUT / f"prc26-{path.parent.name}-evaluation-audit.json")
    records["clean_champion_evaluations"] = evaluations
    allocation_path = ROOT / "data/runs/clean-champion-20261003/2-3/budget-limit.json"
    allocation = json.loads(allocation_path.read_text(encoding="utf-8")) if allocation_path.exists() else {}
    records["clean_champion_confirmation_allocation"] = allocation
    records["clean_champion_checks"] = {"full_suite_passed": 73, "full_suite_seconds": 57.17,
                                         "command": "python -m pytest tests/ -q",
                                         "full_model_comparison": "checked for 1-7" if evaluations else "pending",
                                         "confirmation": "pending", "block_budget_tests_passed": 4}
    (OUT / "prc26-run-record.json").write_text(json.dumps(records, indent=2, allow_nan=False), encoding="utf-8")
    rows = []
    for name, metric in records["pooled"].items():
        ci = metric["day_ci"]
        rows.append(f"<tr><td>{html.escape(name)}</td><td>{metric['gain']:+.3f}</td><td>[{ci[0]:+.3f}, {ci[1]:+.3f}]</td><td>{metric['sse_gain']/1e6:+.1f}M</td></tr>")
    raw = inputs["cold_audit"]
    full = records["neural_full"]
    decision = records["neural_full_decision"]
    status = "Full-data MSE lead rejected" if decision else ("Full-capacity comparison running" if len(full.get("blocks", [])) < 2 else "Full-capacity comparison measured")
    if active.get("status") == "running":
        status = "Full champion reconstruction running"
    elif active.get("status") == "failed":
        status = "Reference fit needs repair"
    elif active.get("status") == "complete":
        status = f"Reference {active.get('block', '1-7')} measured; campaign continues"
    model_rows = []
    for progress in records["neural_full_progress"]:
        for name, info in progress.get("models", {}).items():
            metric = progress["results"][name]
            model_rows.append(f"<tr><td>{progress['months']} {name}</td><td>{info['refit']['fit_rows']:,} / {info['refit']['fit_days']}</td><td>{info['epochs']}</td><td>{metric['gain']:+.3f}</td><td>{info['selection_seconds']+info['refit_seconds']:.0f}s</td></tr>")
    checkpoint = records["champion_checkpoint"]
    donor_section = render_donor(donor) + render_donor(selection, selection=True)
    donor_section += "".join(render_donor(audit, selection=audit["donor_mode"] == "downstream_stopping_excluded") for audit in meta_donors)
    outer_section = render_outer_members(tuple([first, *other_members])) if first else ""
    first_section = ""
    if first:
        file_rows = "".join(f"<tr><td>{html.escape(name)}</td><td>{value['rows']:,}</td><td>{value['rmse']:.3f}</td><td>{value['aobt_only_rmse']:.3f}</td></tr>"
                            for name, value in first["by_file"].items())
        first_section = f"""<section id="first-member"><h2>First reconstructed member: saved and checked</h2><p>The full tuned CatBoost Lane A member is complete. Saved-model replay, input bytes, permitted/stopping/query ID scopes, all-row/day refit coverage, output IDs, finite nonnegative predictions and unchanged scored champion pass. The controller reused the completed member and began the original cross-fitted neighbour models.</p><p>Fit: {first['fit_rows']:,} rows over {first['fit_days']} permitted days. Query: {first['laneA_descriptive_errors']['rows']:,} Lane A rows in the complete January/July files. Recorded selection/refit/replay cost is {first['runtime_seconds']/60:.1f} minutes, including the interrupted first attempt. It is a GPU-family wall-time proxy, not measured GPU utilization. The donor-chain price remains pending.</p><div class="scroll"><table><thead><tr><th>Supplied file</th><th>Lane A rows</th><th>Member RMSE, seconds</th><th>AOBT-only RMSE, seconds</th></tr></thead><tbody>{file_rows}</tbody></table></div><p>{html.escape(first['limit'])}</p><p>Stage 1 reference recipe: partial, first member checked. Stage 2 meta/evaluation: pending. Stage 3 candidate/confirmation: pending. <a href="prc26-first-member-audit.json">Scope, cost and descriptive error record</a>. Regenerate with <code>{html.escape(first['command'])}</code>.</p></section>"""
    if donor:
        price_note = "Both outer donor chains are checked and priced below; nested meta donor costs remain pending." if selection else "The first outer donor cost is recorded below; remaining stopping/meta donor costs are pending."
        groups = {tuple(map(int, scope["scope"].removeprefix("group-").split("-"))) for scope in donor["scopes"] if scope["scope"] != "full"}
        required = {(group, mode) for group in groups for mode in ("full_meta_training", "downstream_stopping_excluded")}
        checked = {(tuple(audit["excluded_meta_months"]), audit["donor_mode"]) for audit in meta_donors}
        if selection and required <= checked:
            price_note = "Both outer donor chains and all January/July nested meta donors are checked and priced below."
        first_section = first_section.replace("The donor-chain price remains pending.", price_note)
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PRC26 campaign status</title><link rel="icon" href="data:,"><style>{css}</style><main>
<div class="eyebrow">PRC26 · checked research checkpoint · {html.escape(records['recorded_utc'])}</div>
<h1>{status}.</h1><p class="lead">The frozen champion remains the scored best. The full original neighbour-model and recalibration chain is the current research task. No new scorer gain is established. The earlier MSE lead remains rejected.</p>
<section id="active-run"><h2>Current work</h2><p id="active-state">{html.escape(active.get('phase', 'No active job record yet.'))}</p><p id="active-detail" class="muted">Last job check: {html.escape(active.get('checked_utc', 'not recorded'))}</p><p>Implementation checks: done, with the earlier 73 full-suite tests and source review, plus four passing block-budget checks. January/July fitted reference, complete-file comparison and error census: checked. February/March confirmation: running; see the live job state above. Candidate preparation and submission: pending. Checkpoints and predictions are saved after each verified component. A failed candidate moves the campaign to its next supported step.</p><p>The first fit saved its full classifier, then stopped because CatBoost returned no best-iteration value without a validation set. Both fitting paths now accept that result. Checked checkpoint recovery preserves the selected tree budgets, original scope and source hashes, completed classifier and spent compute. Native checkpoint reuse, both Windows path formats and changed-scope rejection pass. The completed fit's actual source specification remains preserved alongside the recorded compatibility entry for the path-check correction.</p><p>Follow-up checks in this chat are set for every half hour, with updates on meaningful results or failures. The local controller updates this page's job record while the fits run.</p></section>
{first_section}
{outer_section}
{''.join(render_evaluation(audit) for audit in evaluations)}
{('<section id="confirmation-allocation"><h2>Current confirmation allocation</h2><p>' + html.escape(allocation['purpose']) + '</p><p>February/March GPU-family limit: ' + str(allocation['gpu_family_limit_seconds']/3600) + ' hours, checked between saved components. Existing campaign limit: 44 hours. Measured January/July: 12.28 hours. Reserve for final preparation: 16 hours, a planning allowance. Remaining margin: about 3.72 hours. September/October remains unrun; this is reduced coverage, not full CV. Original method and seed stay fixed.</p></section>') if allocation else ''}
{donor_section}
{''.join(render_assembly(audit) for audit in assemblies)}
<div class="cards"><div class="card"><strong>{board['us']['rank']} / {board['teams']}</strong><span>Leaderboard snapshot rank</span></div><div class="card"><strong>{board['us']['score']:.2f} s</strong><span>Our scored champion</span></div><div class="card"><strong>{board['leader']['score']:.2f} s</strong><span>Leader at snapshot</span></div></div>
<p class="muted">Live API snapshot: {board['fetched_utc']} UTC. <a href="https://prc-data-challenge-2026.netlify.app/">Official deadline</a>: 11 October 2026, 23:59:59 CET.</p>
<section><h2>Decision so far</h2><p>Keep v53 as the fallback. Hierarchical/copy/date-shift alternatives lost on the winter screen. Weather normalization has a small pooled gain with an interval spanning zero. The signed-residual recipe harms whole-file and hot-weather performance. The sampled MSE network beats its Huber counterpart in both two-month blocks, but standalone networks trail the CPU tree.</p>
<p>The sampled meta screen gives useful autumn gains, with weak winter results and adverse winter-weather effects. No sampled recipe clears the complete promotion gate. The first full-capacity block fits every permitted training row. MSE loses to Huber by 1.885 seconds, with calendar-day interval [-3.002, -0.815] and five-day interval [-2.826, -1.051]. Both winter months reverse. The run stops here; the second full block and conditional full-MSE meta comparison are skipped by the rejection rule.</p>
<div class="notice">The CPU control and small nested donor are research substitutes for the champion pipeline. These scores cannot establish a leaderboard improvement, complete clean-v53, or first-place potential.</div></section>
<section><h2>Stage checks</h2><div class="scroll"><table><thead><tr><th>Stage</th><th>Status</th><th>Evidence and remaining work</th></tr></thead><tbody>
<tr><td>Framing and target</td><td>Done</td><td>Recorded target, pooled squared error, frozen champion and bounded contender declarations.</td></tr>
<tr><td>Data audit and cold inputs</td><td>Done for training inputs</td><td>{len(raw)} monthly files; {sum(x['departures'] for x in raw):,} departures; {sum(x['missing_y'] for x in raw)} missing targets; {sum(x['negative_y'] for x in raw)} negative targets retained; {sum(x['over_12h_y'] for x in raw)} targets over 12 hours retained. Unique departure IDs and finite labels pass. Targets/block times hidden before features. January independent cold rebuild and hidden-field mutation agree exactly.</td></tr>
<tr><td>CP0 model scope and fit coverage</td><td>Partial</td><td>Training-only tree stopping, explicit Lane B history sources, scoped donor/PW builders, best/stopping NN epochs and full supplied-row refits implemented. Real Lane B prediction mutation, missing-query-target, stopping-call and control-reproduction checks pass. Recorded full suite: 73 passing tests. First full tuned CatBoost member is saved with replay, scope, input-byte and output-coverage checks. The original neighbour chain and remaining reference/meta fits are running. Full reference comparison and original checkpoint lineage remain open. Legacy global PW/tune/recal/blend paths are outside the scoped campaign proof.</td></tr>
<tr><td>CP1 mechanisms</td><td>Initial screen done; no survivor</td><td>Three declared recipes, two development blocks, complete files and paired stress/resampling checks. No recipe qualifies for expensive confirmation.</td></tr>
<tr><td>CP2 disruption</td><td>Initial screen done; no survivor</td><td>Normalized weather on both blocks. Signed recorder model stopped after winter whole-file/hot-slice harm. Weather gain falls below floor and is uncertain.</td></tr>
<tr><td>CP3 full-data neural comparison</td><td>First block measured; stopped on reversal</td><td>Two losses initially screened on every fifth fit day. The full block uses all permitted days and the original 150-epoch cap, with training-only budget selection and a complete outer refit. MSE reverses against Huber with both intervals below zero. Second full block and Jan/July confirmation are not completed. Coverage/provenance and saved-checkpoint prediction replay are checked separately.</td></tr>
<tr><td>CP4 nested ensemble</td><td>Sampled screen done; full MSE check skipped</td><td>Established, simplex and shrunk weights; weights fit on training-only inner dates. The conditional full MSE meta check is not run because the declared loss prerequisite failed.</td></tr>
<tr><td>Error analysis</td><td>Recorded screens done; transfer remains uncertain</td><td>Month/lane/hot results, 2,000 paired calendar-day resamples, five-day blocks and row/day/episode concentration recorded. Descriptive airport, signed-residual, recording-class and schedule-gap slices plus influential rows are saved for recorded files. Cold ranking support and day-blocked year separation checked. No hidden 2026 label confirmation exists.</td></tr>
<tr><td>CP5 candidate, release and write-up</td><td>Partial</td><td>Champion preserved and file parity checked. Explicit submission exceptions applied locally; 26 tests pass under optimized Python and the actual champion passes both modes. Scheduled-publication trigger/action and authenticated read access checked. Future execution and a complete candidate release remain open. No new upload. This page/run record is a checkpoint.</td></tr>
</tbody></table></div></section>
<section><h2>Initial screens, pooled complete files</h2><p>Positive gain means lower RMSE. Both blocks cover all movements in their supplied departure files. Model counts stay within eight initial recipes, including the CPU control; the meta screen has three declared weight methods. Signed CP2 is rejected from its first block and excluded from the two-block table.</p><div class="scroll"><table><thead><tr><th>Recipe</th><th>Gain over CPU, seconds</th><th>Calendar-day interval</th><th>Squared-error gain</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>These are reused 2025 development months, February/March and September/October. The neural initial screen samples fit days, not evaluation rows. Day intervals and month-stratified five-day sensitivity do not correct adaptive choice or year transfer. Small or hot-slice gains alone do not justify uploading.</p></section>
<section><h2>Full-capacity neural check</h2><p>Architecture, input feature groups, seed and optimizer stay fixed between Huber/MSE. The original epoch cap and patience apply. Each model records its selected best epoch, actual refit rows/days, runtime and peak GPU allocation. Both saved checkpoint predictions reproduce within 1e-4 seconds absolute plus 1e-7 relative tolerance. Encoder and target-scale coverage, donor hashes, complete-file IDs and unchanged Lane B pass.</p><div class="scroll"><table><thead><tr><th>Block/member</th><th>Full refit rows/days</th><th>Best epoch</th><th>Gain over CPU</th><th>Fit time</th></tr></thead><tbody>{''.join(model_rows) or '<tr><td colspan="5">No completed full-capacity member recorded yet.</td></tr>'}</tbody></table></div><p>Both networks improve ordinary-weather rows but lose on winter weather. Huber's hot-slice RMSE loss is 87.921 seconds and MSE's is 92.972 seconds versus the CPU control. The largest harmed rows include the LTFM snow episode on 19 February. Per-airport, recorded-mechanism, signed-residual and gap slices are descriptive diagnostics, not new features or selection rules. The first full block takes 1,933 seconds wall time; recorded selection/refit stages sum to 1,863 seconds. That earlier neural comparison was stopped under its rejection rule.</p></section>
<section><h2>Ranking support</h2><p>The cold ranking file covers 344,841 unique IDs with hidden departure targets and finite Lane A AOBT anchors. Weather observation coverage is complete; 15 rows use an observation older than one hour. In the matched January/July 2025 files, no row has 15 or more snow reports over one day; ranking has 8,028 such rows. The maximum is 13 in matched training, 41 across all training, and 47 in ranking. These are reporting counts, not elapsed snow duration.</p>
<p>Matched airport/month samples with five-day blocks kept together yield year-separation AUC 0.8073, January 0.6466 and July 0.9222. Larger AUC means easier discrimination between the years; it does not establish a useful taxi prediction or justify weighting. Daily movement/stamp and arrival summaries lead the classifier. Unseen categories are uncommon against the full training year. Raw/weather hashes, feature-range counts and all classifier scopes are saved in the run record.</p></section>
<section><h2>Submission and publication checks</h2><p>The validator rejects missing/extra/duplicate/null IDs, unexpected columns, nonnumeric/boolean/complex predictions, nonfinite values and negative predictions before S3 access. Missing credentials produce a clear error. Upload verification uses explicit byte/hash checks. Tests mock S3; no scorer upload occurred.</p>
<p>The existing publication task is Ready for 11 October at 21:30 UTC, with its reviewed action at <code>data/go-public.cmd</code>. It requires an interactive login and AC power. Authenticated read access reports a private repository, default branch <code>master</code>, admin permission and an existing <code>campaign2</code> branch. The script has no read-only mode, so its visibility-changing command was not executed. Local changes still need to reach the intended release branch before publication.</p></section>
<section><h2>Reproduction and limits</h2><pre>python -m pytest -q
python analysis/contender_screen.py
python analysis/validate_scope_run.py
python analysis/disruption_screen.py
python analysis/neural_screen.py
python analysis/meta_screen.py
python analysis/neural_screen.py full
python analysis/meta_screen.py full
python analysis/validate_neural_artifacts.py --replay
python analysis/ranking_support.py
python -O -m pytest tests/test_board.py -q</pre><p>Run files: <code>data/runs/contender-20261003/</code>. Original global prediction caches and scored champion outputs are preserved. Current work is local on <code>codex/validation-scope-20261003</code>. The run record includes raw/source hashes, model scopes, selected budgets, weights and detailed metrics. CPU peak memory was not instrumented; the neural screen records GPU allocation.</p>
<p>Outer membership now follows supplied-file month. Two UTC boundary cases have a different movement-clock month; this does not establish faulty clocks. The initial September/October control was rebuilt under the complete-file scope before subsequent comparisons. Failures during construction: monthly category union, NumPy JSON values, a consumer started before its second input existed, and a test-spy argument index. Each is recorded as a construction interruption, not a successful model result.</p>
<p>Frozen champion SHA-256: <code>{checkpoint['champion']['sha256']}</code>. Earlier fixed-budget Lane B history repair yielded a small diagnostic gain while historical Lane A/donor caches stayed fixed. The original fitted-model lineage remains partial. No new measured scorer gain has been claimed.</p>
<p><a href="prc26-run-record.json">Machine-readable run record</a> · <a href="https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard">Live board source</a></p></section></main><script>async function updateJob(){{try{{const response=await fetch('prc26-active-run.json',{{cache:'no-store'}});if(!response.ok)return;const state=await response.json();document.querySelector('h1').textContent=state.status==='running'?'Full champion reconstruction running.':state.status==='complete'?'Reference '+(state.block||'1-7')+' measured; campaign continues.':'Reference fit needs repair.';document.getElementById('active-state').textContent=state.phase;const item=state.active_component;document.getElementById('active-detail').textContent='Checked '+state.checked_utc+(item?' · '+item.component+' · '+item.stage:'')+' · '+state.completed_components.length+' saved members';}}catch(error){{document.getElementById('active-detail').textContent='Live job record could not be read. Reload to retry.';}}}}updateJob();setInterval(updateJob,15000);</script></html>'''
    (OUT / "prc26-status-2026-10-03.html").write_text(page, encoding="utf-8")
    shutil.copy2(OUT / "prc26-status-2026-10-03.html", ROOT / "docs/STATUS-2026-10-03.html")
    repo_record = {**records, "error_analysis": {"limit": records["error_analysis"].get("limit"),
                   "row_export": "Influential rows remain in the local run and user export; this repo copy stores aggregate slices.",
                   "files": {name: {"sha256": info["sha256"], "recipes": {recipe: {"slices": value["slices"]}
                             for recipe, value in info["recipes"].items()}} for name, info in records["error_analysis"].get("files", {}).items()}}}
    (ROOT / "docs/prc26-run-record.json").write_text(json.dumps(repo_record, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": status, "pooled": {k: v['gain'] for k, v in records['pooled'].items()}}))


if __name__ == "__main__":
    main()
