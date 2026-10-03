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
    records["clean_champion_checks"] = {"full_suite_passed": 73, "full_suite_seconds": 57.17,
                                         "command": "python -m pytest tests/ -q", "full_model_comparison": "pending"}
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
    model_rows = []
    for progress in records["neural_full_progress"]:
        for name, info in progress.get("models", {}).items():
            metric = progress["results"][name]
            model_rows.append(f"<tr><td>{progress['months']} {name}</td><td>{info['refit']['fit_rows']:,} / {info['refit']['fit_days']}</td><td>{info['epochs']}</td><td>{metric['gain']:+.3f}</td><td>{info['selection_seconds']+info['refit_seconds']:.0f}s</td></tr>")
    checkpoint = records["champion_checkpoint"]
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PRC26 campaign status</title><link rel="icon" href="data:,"><style>{css}</style><main>
<div class="eyebrow">PRC26 · checked research checkpoint · {html.escape(records['recorded_utc'])}</div>
<h1>{status}.</h1><p class="lead">The frozen champion remains the scored best. The full original neighbour-model and recalibration chain is the current research task. No new scorer gain is established. The earlier MSE lead remains rejected.</p>
<section id="active-run"><h2>Current work</h2><p id="active-state">{html.escape(active.get('phase', 'No active job record yet.'))}</p><p id="active-detail" class="muted">Last job check: {html.escape(active.get('checked_utc', 'not recorded'))}</p><p>Implementation checks: done, with 73 full-suite tests and the source review. Full fitted reference: partial; see the live job state above. Complete comparison and error census: pending. Candidate preparation and submission: pending. The completed reference will be evaluated before choosing the next distinct score experiment. Checkpoints and predictions are saved after each verified component. A failed candidate moves the campaign to its next supported step.</p><p>The first fit saved its full classifier, then stopped because CatBoost returned no best-iteration value without a validation set. Both fitting paths now accept that result. Checked checkpoint recovery preserves the selected tree budgets, original scope and source hashes, completed classifier and spent compute. Native checkpoint reuse, both Windows path formats and changed-scope rejection pass. A recorded compatibility entry covers this path-check correction while the resumed fit continues; the fit's actual source specification remains preserved.</p><p>Follow-up checks in this chat are set for every half hour, with updates on meaningful results or failures. The local controller updates this page's job record while the fits run.</p></section>
<div class="cards"><div class="card"><strong>{board['us']['rank']} / {board['teams']}</strong><span>Live leaderboard rank</span></div><div class="card"><strong>{board['us']['score']:.2f} s</strong><span>Our scored champion</span></div><div class="card"><strong>{board['leader']['score']:.2f} s</strong><span>Current leader</span></div></div>
<p class="muted">Live API snapshot: {board['fetched_utc']} UTC. <a href="https://prc-data-challenge-2026.netlify.app/">Official deadline</a>: 11 October 2026, 23:59:59 CET.</p>
<section><h2>Decision so far</h2><p>Keep v53 as the fallback. Hierarchical/copy/date-shift alternatives lost on the winter screen. Weather normalization has a small pooled gain with an interval spanning zero. The signed-residual recipe harms whole-file and hot-weather performance. The sampled MSE network beats its Huber counterpart in both two-month blocks, but standalone networks trail the CPU tree.</p>
<p>The sampled meta screen gives useful autumn gains, with weak winter results and adverse winter-weather effects. No sampled recipe clears the complete promotion gate. The first full-capacity block fits every permitted training row. MSE loses to Huber by 1.885 seconds, with calendar-day interval [-3.002, -0.815] and five-day interval [-2.826, -1.051]. Both winter months reverse. The run stops here; the second full block and conditional full-MSE meta comparison are skipped by the rejection rule.</p>
<div class="notice">The CPU control and small nested donor are research substitutes for the champion pipeline. These scores cannot establish a leaderboard improvement, complete clean-v53, or first-place potential.</div></section>
<section><h2>Stage checks</h2><div class="scroll"><table><thead><tr><th>Stage</th><th>Status</th><th>Evidence and remaining work</th></tr></thead><tbody>
<tr><td>Framing and target</td><td>Done</td><td>Recorded target, pooled squared error, frozen champion and bounded contender declarations.</td></tr>
<tr><td>Data audit and cold inputs</td><td>Done for training inputs</td><td>{len(raw)} monthly files; {sum(x['departures'] for x in raw):,} departures; {sum(x['missing_y'] for x in raw)} missing targets; {sum(x['negative_y'] for x in raw)} negative targets retained; {sum(x['over_12h_y'] for x in raw)} targets over 12 hours retained. Unique departure IDs and finite labels pass. Targets/block times hidden before features. January independent cold rebuild and hidden-field mutation agree exactly.</td></tr>
<tr><td>CP0 model scope and fit coverage</td><td>Partial</td><td>Training-only tree stopping, explicit Lane B history sources, nested CPU donor/PW builder, best/stopping NN epochs and full supplied-row refits implemented. Real Lane B prediction mutation, missing-query-target, stopping-call and control-reproduction checks pass. 62 tests pass; static F checks pass. Full champion donor/meta reconstruction and original checkpoint lineage remain open. Legacy global PW/tune/recal/blend paths are outside the scoped campaign proof.</td></tr>
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
<p><a href="prc26-run-record.json">Machine-readable run record</a> · <a href="https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard">Live board source</a></p></section></main><script>async function updateJob(){{try{{const response=await fetch('prc26-active-run.json',{{cache:'no-store'}});if(!response.ok)return;const state=await response.json();document.querySelector('h1').textContent=state.status==='running'?'Full champion reconstruction running.':state.status==='complete'?'January/July reference measured.':'Reference fit needs repair.';document.getElementById('active-state').textContent=state.phase;const item=state.active_component;document.getElementById('active-detail').textContent='Checked '+state.checked_utc+(item?' · '+item.component+' · '+item.stage:'')+' · '+state.completed_components.length+' saved members';}}catch(error){{document.getElementById('active-detail').textContent='Live job record could not be read. Reload to retry.';}}}}updateJob();setInterval(updateJob,15000);</script></html>'''
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
