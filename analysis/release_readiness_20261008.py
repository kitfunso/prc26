"""Read-only final-object and existing publication-task preflight; preserve the frozen package."""
from __future__ import annotations

import hashlib
import html
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import board

RUN = ROOT / 'data/runs/clean-champion-20261003'
OUT = Path('C:/Users/skf_s/Documents/Codex/2026-10-03/can-x20/outputs')
LOG = logging.getLogger(__name__)


def command(args: tuple[str, ...]) -> str:
    result = subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True, encoding='utf-8')
    return result.stdout.strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publication_preflight() -> dict:
    task_script = (
        "$task = Get-ScheduledTask -TaskName 'prc26-public-20261011'; "
        "$info = Get-ScheduledTaskInfo -TaskName 'prc26-public-20261011'; "
        "[pscustomobject]@{Name=$task.TaskName; State=[string]$task.State; "
        "Enabled=$task.Settings.Enabled; action=@{Execute=$task.Actions[0].Execute; "
        "Arguments=$task.Actions[0].Arguments; WorkingDirectory=$task.Actions[0].WorkingDirectory}; "
        "trigger_start=$task.Triggers[0].StartBoundary; logon_type=[string]$task.Principal.LogonType; "
        "run_level=[string]$task.Principal.RunLevel; settings=@{"
        "DisallowStartIfOnBatteries=$task.Settings.DisallowStartIfOnBatteries; "
        "StopIfGoingOnBatteries=$task.Settings.StopIfGoingOnBatteries; "
        "StartWhenAvailable=$task.Settings.StartWhenAvailable}; NextRun=$info.NextRunTime.ToString('o'); "
        "LastRun=$info.LastRunTime.ToString('o'); LastResult=$info.LastTaskResult} | ConvertTo-Json -Depth 5"
    )
    task = json.loads(command(('powershell', '-NoProfile', '-Command', task_script)))
    if (not task['Enabled'] or Path(task['action']['Execute']).resolve() != ROOT / 'data/go-public.cmd'
            or datetime.fromisoformat(task['trigger_start']) != datetime.fromisoformat('2026-10-11T22:30:00+01:00')):
        raise ValueError('existing publication task action, enablement or schedule differs')
    repo = json.loads(command(('gh', 'api', 'repos/kitfunso/prc26')))
    branch = json.loads(command(('gh', 'api', 'repos/kitfunso/prc26/branches/campaign2')))
    remote_sha = branch['commit']['sha']
    command(('git', 'branch'))
    head = command(('git', 'rev-parse', 'HEAD'))
    command(('git', 'merge-base', '--is-ancestor', remote_sha, head))
    ahead = int(command(('git', 'rev-list', '--count', f'{remote_sha}..{head}')))
    return {'task': task, 'action_sha256': sha(ROOT / 'data/go-public.cmd'),
            'action_executed': False, 'schedule_changed': False,
            'repository': {'visibility': repo['visibility'], 'default_branch': repo['default_branch'],
                           'license': (repo.get('license') or {}).get('spdx_id'),
                           'admin': repo['permissions'].get('admin'), 'remote_campaign2_sha': remote_sha},
            'local_source_head': head, 'local_branch': command(('git', 'branch', '--show-current')),
            'local_commits_ahead': ahead,
            'release_branch_status': 'VERIFIED' if ahead == 0 else 'PENDING',
            'remote_is_ancestor': True, 'proposed_push_after_authority': 'git push origin HEAD:campaign2',
            'push_performed': False, 'release_gap': 'Release branch lacks checked local work.' if ahead else 'Release branch contains the local checkpoint.'}


def final_verification() -> dict:
    receipt = json.loads((RUN / 'preparation/final-designation-20261008.json').read_text())
    board.check(receipt['final_path'])
    client = board._s3()
    obj = client.get_object(Bucket=board.BUCKET, Key='merry-octopus_final.parquet')
    payload = obj['Body'].read()
    if hashlib.sha256(payload).hexdigest() != receipt['sha256'] or len(payload) != receipt['bytes']:
        raise ValueError('stored final artifact changed')
    result = None
    try:
        result = json.loads(client.get_object(Bucket=board.BUCKET, Key='merry-octopus_final.parquet_result.json')['Body'].read())
    except ClientError as error:
        if error.response['Error']['Code'] not in ('NoSuchKey', '404'):
            raise
    if result is not None:
        if (result.get('file') != 'merry-octopus_final.parquet' or result.get('bucket') != board.BUCKET
                or result.get('status') not in ('Succeeded', 'Failed')):
            raise ValueError('final scorer result has an unexpected identity or status')
        if result['status'] == 'Succeeded' and (result.get('used_pairs') != receipt['rows']
                or result.get('score') != receipt['prior_scored_object']['scorer_result']['score']):
            raise ValueError('final scorer result differs from the unchanged scored artifact')
        if result['status'] == 'Failed' and (not result.get('error') or not result.get('error_type')):
            raise ValueError('failed final scorer result lacks its error evidence')
    return {'artifact_sha256': receipt['sha256'], 'rows': receipt['rows'], 'bytes': len(payload),
            'path': receipt['final_path'], 'remote_key': 'merry-octopus_final.parquet',
            'downloaded_bytes': 'PASS', 'file_template_checks': 'PASS',
            'result': result, 'scorer_status': (result or {}).get('status', 'PENDING'),
            'score_verified': bool(result and result['status'] == 'Succeeded'),
            'designation_receipt_sha256': sha(RUN / 'preparation/final-designation-20261008.json'),
            'score_improvement': False}


def preserved_bytes() -> dict:
    freeze = json.loads((RUN / 'preparation/recipe-freeze-20261008.json').read_text())
    records = freeze['preservation']['champion'] + [freeze['preservation']['package']]
    records += [record for block in freeze['preservation']['references'].values() for record in block.values()]
    checks = []
    for record in records:
        actual = sha(Path(record['path']))
        if actual != record['sha256']:
            raise ValueError(f"preserved bytes changed: {record['path']}")
        checks.append({'path': record['path'], 'sha256': actual, 'check': 'PASS'})
    return {'checks': checks, 'freeze_receipt_sha256': sha(RUN / 'preparation/recipe-freeze-20261008.json'),
            'local_readiness_receipt_sha256': sha(RUN / 'preparation/local-readiness.json'),
            'package_rebuilt': False, 'native_audits_rerun': False, 'model_fit': False}


def render(record: dict) -> str:
    final, release = record['final'], record['publication_preflight']
    rows = ''.join(f'<tr><td>{html.escape(stage)}</td><td>{html.escape(state)}</td></tr>'
                   for stage, state in record['stages'].items())
    task = release['task']
    branch_state = record['stages']['Private release branch update']
    result = json.dumps(final['result'], indent=2) if final['result'] else 'No final scorer result exists yet.'
    incident = record.get('final_incident')
    incident_note = '' if not incident else (
        '<section class="notice"><h2>Final eligibility: unresolved</h2><p>The scorer returned '
        'FileNotFoundError for the official dev_sub_template.parquet path. Read-only dataset checks return 404 '
        'for that key. The live final_submitting.parquet contains 670,790 IDs; the stored final contains '
        '344,841, leaving 325,949 additional final IDs without predictions. The earlier file check covers '
        'the original ranking template only. Final scoring and final-population eligibility have failed their checks.</p>'
        '<p>Organizer clarification and a replacement decision are required before any corrective submission. '
        'No file was overwritten, reuploaded or fitted. The recipe freeze and publication schedule remain unchanged.</p>'
        '<p><a href="prc26-final-scorer-incident-20261009.html">Evidence and unsent organizer report</a></p></section>'
    )
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PRC26 final designation and release readiness</title><link rel="icon" href="data:,"><style>
:root{{color-scheme:light;--ink:#17253a;--muted:#526278;--line:#d7e0ea}}*{{box-sizing:border-box}}body{{margin:0;background:#f3f6fa;color:var(--ink);font:17px/1.65 system-ui,sans-serif}}main{{max-width:1050px;margin:auto;padding:42px 24px}}h1{{font-size:clamp(30px,5vw,48px);line-height:1.15}}h2{{font-size:24px}}section{{background:white;border:1px solid var(--line);padding:24px;margin:22px 0;border-radius:10px}}p{{max-width:90ch}}a{{color:#124c99}}code,pre{{font-size:14px;overflow-wrap:anywhere}}pre{{white-space:pre-wrap;background:#edf2f8;padding:16px}}table{{width:100%;border-collapse:collapse}}td,th{{padding:12px;text-align:left;border-bottom:1px solid var(--line)}}.muted{{color:var(--muted)}}.notice{{border-left:5px solid #c8860e}}@media(max-width:600px){{main{{padding:20px 12px}}section{{padding:16px}}td,th{{padding:8px}}}}
</style><main><p class="muted">Checked {html.escape(record['checked_utc'])} · campaign OPEN</p>
<h1>Frozen v53 designated as the final submission</h1>
<p>No score improvement qualified. The final object contains exactly the previously scored v53 bytes. Final-object storage and downloaded-byte checks pass. Scorer status: <strong>{html.escape(final['scorer_status'])}</strong>.</p>
{incident_note}
<section><h2>Required final file</h2><p>The official <a href="https://prc-data-challenge-2026.netlify.app/ranking.html#final-submission">final-submission rule</a> requires a unique team final filename. Keith's existing submission authority covers this designation of the checked frozen selection. No versioned duplicate or rejected recipe was uploaded.</p>
<p>Object: <code>{final['remote_key']}</code>; {final['rows']:,} original-template rows; {final['bytes']:,} bytes. SHA-256: <code>{final['artifact_sha256']}</code>.</p>
<p>The strict final command creates only this exact filename, rejects an existing object through conditional creation, and checks the complete stored body. The parquet validation function is unchanged from the sealed validator.</p>
<pre>{html.escape(result)}</pre><p><a href="prc26-final-designation-20261008.json">Immutable upload and preservation receipt</a>.</p></section>
<section><h2>Deadline and frozen recipe</h2><p>The <a href="https://prc-data-challenge-2026.netlify.app/">official timeline</a> now ends on 14 October 2026 at 10:00 UTC, 11:00 London. The 8 October recipe freeze remains complete. Both clean reference blocks and all rejected recipes remain closed; no new search, fitting or retuning follows the extension.</p>
<p>Original, frozen and delivered champion files, both reference proofs/predictions, immutable freeze receipt and sealed local-readiness package retain their checked hashes. No package rebuild or native audit rerun occurred. The reconstruction allowance and final-preparation planning reserve remain unchanged.</p></section>
<section class="notice"><h2>Release branch: {html.escape(release['release_branch_status'])}</h2><p>The repository remains <strong>{html.escape(release['repository']['visibility'])}</strong>, with default branch <code>{release['repository']['default_branch']}</code>. The existing publication task will switch to <code>campaign2</code> and then make the repository public. {html.escape(release['release_gap'])}</p>
<p>Remote campaign2: <code>{release['repository']['remote_campaign2_sha']}</code>. Checked local source commit: <code>{release['local_source_head']}</code>. {html.escape(branch_state)}. This verification reads the remote and task without executing publication.</p>
<p>The task <code>{task['Name']}</code> is {task['State']}; next run {html.escape(task['NextRun'])}. The reviewed action remains <code>data/go-public.cmd</code>. It needs interactive login and AC power. Its visibility-changing command was not executed and its schedule was not changed. Actual publication/default-branch outcome remains a future check.</p></section>
<section><h2>Code, data and reproduction eligibility</h2><p>The <a href="https://prc-data-challenge-2026.netlify.app/eligibility.html">award rules</a> require public GPLv3 code, documented usable external data and sufficient reproduction instructions. Local source carries GPLv3. Public code eligibility remains pending until actual publication is verified.</p>
<p>External weather is the Iowa Environmental Mesonet ASOS/METAR archive, fetched by <code>weather.py</code> for the ten listed airport stations. IEM's <a href="https://mesonet.agron.iastate.edu/disclaimer.php">usage policy</a> places its materials in the public domain and permits lawful reuse. README documents the source, fetch command, date range and as-of feature construction. Challenge input files remain obtained through the challenge data process.</p>
<p>Exact scored-file and prediction lineage is preserved. Historical native-model/checkpoint lineage remains partial, and GPU refits need not be bit-identical. The recipe history documents the original method; corrected clean reconstruction is separately audited. This checkpoint does not claim complete historical fitted-model reproduction or a fresh validation result.</p></section>
<section><h2>Stage closures</h2><table><thead><tr><th>Stage</th><th>Check and status</th></tr></thead><tbody>{rows}</tbody></table>
<p>Verification commands: <code>python -m pytest -q</code>; <code>python -O -m pytest tests/test_board.py -q</code>; <code>python -m ruff check --select F board.py tests/test_board.py analysis/release_readiness_20261008.py analysis/report_contender_20261003.py</code>; <code>python analysis/release_readiness_20261008.py</code> for a due read-only verification. Do not rerun the final upload command.</p>
<p><a href="prc26-release-readiness-20261008.json">Current read-only release record</a> · <a href="prc26-status-2026-10-03.html">Campaign status</a> · <a href="prc26-recipe-freeze-20261008.html">Immutable freeze report</a>.</p></section></main></html>'''


def update_active(record: dict) -> None:
    path = RUN / 'active.json'
    active = json.loads(path.read_text())
    release = record['publication_preflight']['repository']
    changes = {'final_submission_status': 'STORED_BYTE_VERIFIED',
               'final_scorer_status': record['final']['scorer_status'], 'deadline_utc': record['deadline_utc'],
               'publication_visibility': release['visibility'], 'publication_campaign2_sha': release['remote_campaign2_sha'],
               'release_branch_status': record['publication_preflight']['release_branch_status'],
               'final_template_status': 'INCOMPLETE' if record.get('final_incident') else 'NOT_CHECKED'}
    if all(active.get(key) == value for key, value in changes.items()):
        return
    incident = record.get('final_incident')
    active.update(changes, checked_utc=record['checked_utc'], block='release-readiness',
                  headline='Final scoring failed; final-template coverage incomplete' if incident else 'Frozen v53 designated final; release checks open',
                  phase='Required unique final designation of exact scored v53 is stored and byte-checked. '
                        f"Final scorer: {record['final']['scorer_status']}. No improved recipe qualified. "
                        'The official deadline is extended to 14 October, 10:00 UTC. Recipe freeze and existing publication schedule remain. '
                        f"{record['publication_preflight']['release_gap']} Historical native-model lineage remains partial. "
                        + ('Live final template includes additional IDs absent from the stored final; organizer clarification and replacement authority are required.' if incident else ''),
                  next_stage=('Await the pending organizer-contact decision and official template/scorer clarification; never overwrite or reupload the final object. '
                              if incident else 'Inspect the named final scorer result at a due check without reuploading. ')
                             + 'Verify the release branch and actual outcome after the existing publication event. '
                             'No search, refit, retuning, package rebuild, regular duplicate, publication execution or schedule change.',
                  release_readiness_path='preparation/release-readiness-20261008.json',
                  final_designation_path='preparation/final-designation-20261008.json')
    path.write_text(json.dumps(active, indent=2), encoding='utf-8')
    shutil.copy2(path, OUT / 'prc26-active-run.json')


def main() -> None:
    publication = publication_preflight()
    final = final_verification()
    preservation = preserved_bytes()
    tests = json.loads((RUN / 'validation-release-20261008.json').read_text())
    incident_path = RUN / 'preparation/final-scorer-incident-20261009.json'
    incident = json.loads(incident_path.read_text()) if incident_path.exists() else None
    scorer_stage = ('FAILED: ' + final['result']['error_type'] if final['scorer_status'] == 'Failed'
                    else 'DONE: same scored outcome' if final['score_verified'] else 'PENDING: result object absent')
    record = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'campaign_status': 'OPEN',
              'deadline_utc': '2026-10-14T10:00:00+00:00', 'deadline_london': '2026-10-14T11:00:00+01:00',
              'official_sources': {'timeline': 'https://prc-data-challenge-2026.netlify.app/',
                                  'submission': 'https://prc-data-challenge-2026.netlify.app/ranking.html',
                                  'eligibility': 'https://prc-data-challenge-2026.netlify.app/eligibility.html',
                                  'weather_usage': 'https://mesonet.agron.iastate.edu/disclaimer.php'},
              'final': final, 'preservation': preservation, 'publication_preflight': publication,
              'validation': tests, 'historical_native_model_lineage': 'PARTIAL', 'final_incident': incident,
              'stages': {'Recipe comparison': 'DONE: no declared recipe qualifies; rejected artifacts retained',
                         'Reduced confirmation': 'PARTIAL: September/October deferred; reused development data',
                         'Recipe freeze': 'DONE: immutable receipt and exact scored v53 selection',
                         'New ranking fit': 'SKIPPED: no improved recipe qualified',
                         'Regular duplicate upload': 'SKIPPED: no scorer probe or new version of unchanged v53',
                         'Required unique final designation': 'DONE: stored bytes and original-template checks; expanded final eligibility separate',
                         'Final scorer readback': scorer_stage,
                         'Final-template coverage': 'FAILED: additional final IDs have no predictions' if incident else 'PENDING: live final template not checked',
                         'Local evidence package': 'DONE: preserved sealed package; no rebuild',
                         'Historical fitted-model lineage': 'PARTIAL: original native lineage incomplete',
                         'Release command validation': f"DONE: {tests['passed']} full-suite tests; optimized board tests and F lint pass",
                         'Private release branch update': ('DONE: remote campaign2 equals the checked local source commit'
                                                          if publication['release_branch_status'] == 'VERIFIED'
                                                          else 'PENDING: remote campaign2 lacks local commits'),
                         'Existing publication-task preflight': 'DONE: read-only actual task/action/repository checks',
                         'Public code and scheduled outcome': 'PENDING: verify after the existing event'},
              'command': 'python analysis/release_readiness_20261008.py (read-only remote/task checks; writes local evidence)'}
    receipt = RUN / 'preparation/release-readiness-20261008.json'
    receipt.write_text(json.dumps(record, indent=2), encoding='utf-8')
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(receipt, OUT / 'prc26-release-readiness-20261008.json')
    page = ROOT / 'docs/RELEASE-READINESS-2026-10-08.html'
    page.write_text(render(record), encoding='utf-8')
    shutil.copy2(page, OUT / 'prc26-release-readiness-20261008.html')
    update_active(record)
    if final['scorer_status'] == 'Failed':
        LOG.error('Final scoring failed: %s: %s', final['result']['error_type'], final['result']['error'])
    LOG.info('Final scorer: %s; campaign2 needs %s local commits; release remains open.',
             final['scorer_status'], publication['local_commits_ahead'])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
