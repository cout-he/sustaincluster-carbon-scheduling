"""Frozen finite 18-job training/evaluation dispatch. Never opens a Test payload."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from experiments.e1.contracts import content_hash, file_hash
from experiments.joint_v2.checkpoint import atomic_json
from .checkpoint import code_identity
from .config import ABLATIONS, defaults, load_config

SEEDS = (20261017, 20261018, 20261019)
METHODS = ('B0', 'B1', 'B2', 'B3', 'B4', 'B5')
SCHEMA = 'rac-mappo-finite-pipeline-v1'
ROOT = Path(__file__).resolve().parents[2]


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _pool_manifest(pool):
    value = _read(Path(pool)/'manifest.json')
    if value.get('test_access') is not False:
        raise ValueError('Pool must preserve the sealed Test rule')
    entries = value.get('scenarios', [])
    if any(e.get('role') not in ('train', 'development', 'validation') for e in entries):
        raise ValueError('Unexpected pool role; Test stays sealed')
    development = sorted(e['scenario_id'] for e in entries if e['role'] == 'development')
    if not development or not any(e['role'] == 'train' for e in entries):
        raise ValueError('Need frozen Train and Development metadata')
    return value, development


def _evaluation_ids(development):
    return [scene+f'-eval-r{pressure:g}-s{sla:g}' for scene in development
            for pressure in (.75, 1., 1.25) for sla in (1.5, 2.)]


def prepare(pool, output, *, v2_checkpoint=None):
    """Read metadata only and materialize the complete predeclared experiment."""
    pool, out = Path(pool).resolve(), Path(output).resolve()
    if out.exists():
        raise FileExistsError('Frozen protocol output already exists')
    _, development = _pool_manifest(pool)
    code = code_identity()
    jobs = []
    for variant in ABLATIONS:
        for seed in SEEDS:
            config = defaults(seed, variant)
            load_config(config)
            job_id = f'{variant}-{seed}'
            path = Path('configs')/(job_id+'.json')
            jobs.append(dict(job_id=job_id, name=job_id, variant=variant, seed=seed, config=path.as_posix(),
                             config_identity=content_hash(config), training=f'jobs/{job_id}/training'))
    protocol = dict(schema=SCHEMA, source_identity=code['sha256'], source_files=code['files'],
        pool=str(pool), pool_manifest_sha256=file_hash(pool/'manifest.json'),
        seeds=list(SEEDS), variants=list(ABLATIONS), jobs=jobs, job_count=18,
        budget=dict(total_physical_steps=65536, max_wall_seconds=21600., max_episodes=None),
        maximum_concurrency=3, cpu_threads_per_job=1, automatic_budget_expansion=False,
        evaluation=dict(role='development', scenario_ids=_evaluation_ids(development),
            selection_scenario_ids=development[:defaults()['validation']['max_development_scenarios']],
            full_methods=list(METHODS), ablation_methods=['B5'],
            selection='best_feasible_else_best_deadline_explicitly_infeasible',
            v2_checkpoint=str(Path(v2_checkpoint or ROOT/'research_notes/joint_v2/formal-001/training/checkpoints').resolve()),
            v2_completed_episode=52,
            baseline_cooling=dict(B0='fixed22_entire_operating_window', B1='fixed18_entire_operating_window',
                other_methods='original_trained_or_fixed_active_control_then_shared_idle18')),
        test_access=False)
    protocol['protocol_sha256'] = content_hash(protocol)
    out.mkdir(parents=True)
    for job in jobs:
        atomic_json(out/job['config'], defaults(job['seed'], job['variant']))
    atomic_json(out/'protocol.json', protocol)
    return out/'protocol.json'


def validate_protocol(path):
    """Validate all identities before creating status files or child processes."""
    path = Path(path).resolve()
    protocol = _read(path)
    unsigned = dict(protocol)
    digest = unsigned.pop('protocol_sha256', None)
    if digest != content_hash(unsigned) or protocol.get('schema') != SCHEMA:
        raise ValueError('Frozen protocol identity differs')
    expected = [(variant, seed) for variant in ABLATIONS for seed in SEEDS]
    if (protocol.get('test_access') is not False or protocol.get('seeds') != list(SEEDS)
            or protocol.get('variants') != list(ABLATIONS) or protocol.get('job_count') != 18
            or [(j['variant'], j['seed']) for j in protocol['jobs']] != expected
            or protocol.get('maximum_concurrency') != 3 or protocol.get('cpu_threads_per_job') != 1
            or protocol.get('automatic_budget_expansion') is not False):
        raise ValueError('Frozen protocol jobs or limits differ')
    for job in protocol['jobs']:
        job_id = f"{job['variant']}-{job['seed']}"
        if job['job_id'] != job_id or job.get('name') != job_id or job['training'] != f'jobs/{job_id}/training':
            raise ValueError('Frozen protocol job path differs')
        config_path = (path.parent/job['config']).resolve()
        if not config_path.is_relative_to(path.parent):
            raise ValueError('Frozen config escapes the protocol directory')
        config = _read(config_path)
        if config != defaults(job['seed'], job['variant']) or content_hash(config) != job['config_identity']:
            raise ValueError('Frozen config identity differs')
        load_config(config)
    if protocol['budget'] != defaults()['budget']:
        raise ValueError('Frozen protocol budget differs')
    _, development = _pool_manifest(protocol['pool'])
    if file_hash(Path(protocol['pool'])/'manifest.json') != protocol['pool_manifest_sha256']:
        raise ValueError('Frozen pool manifest identity differs')
    evaluation = protocol['evaluation']
    if (evaluation['role'] != 'development' or evaluation['scenario_ids'] != _evaluation_ids(development)
            or evaluation['selection_scenario_ids'] != development[:defaults()['validation']['max_development_scenarios']]
            or evaluation['full_methods'] != list(METHODS) or evaluation['ablation_methods'] != ['B5']):
        raise ValueError('Frozen Development evaluation protocol differs')
    if code_identity()['sha256'] != protocol['source_identity']:
        raise ValueError('Frozen source identity differs')
    return protocol


def _finished(result, config):
    stop = result.get('last_run', {}).get('stop_reason', '')
    return (result.get('physical_steps', 0) >= config['budget']['total_physical_steps']
            or result.get('elapsed_seconds', 0.) >= config['budget']['max_wall_seconds']
            or stop.startswith('wall_budget') or stop in ('step_budget', 'incomplete_episode_rolled_back', 'episode_budget'))


def _evaluation_complete(folder, job, protocol):
    path = Path(folder)/'results.json'
    if not path.exists():
        return False
    value = _read(path)
    if not value.get('complete'):
        return False
    methods = METHODS if job['variant'] == 'full' else ('B5',)
    expected = {(s, m) for s in protocol['evaluation']['scenario_ids'] for m in methods}
    rows = value['results']
    if (value.get('test_access') is not False or len(rows) != len(expected)
            or {(r['scenario_id'], r['comparison_method']) for r in rows} != expected
            or any(r.get('scenario_role') != 'development' or r.get('seed') != job['seed']
                   or r.get('variant') != job['variant'] for r in rows)):
        raise ValueError('Completed evaluation does not match its frozen job')
    return True


def run(protocol_path, *, jobs=3):
    """Run at most three CPU workers, at most once per job/phase per invocation."""
    if isinstance(jobs, bool) or not isinstance(jobs, int) or not 1 <= jobs <= 3:
        raise ValueError('jobs must be an integer between1 and3')
    protocol_path = Path(protocol_path).resolve()
    protocol = validate_protocol(protocol_path)
    root, status_path = protocol_path.parent, protocol_path.parent/'pipeline-status.json'
    previous = _read(status_path) if status_path.exists() else {}
    if previous and previous.get('protocol_sha256') != protocol['protocol_sha256']:
        raise ValueError('Saved pipeline status identity differs')
    for row in previous.get('jobs', {}).values():
        if row.get('stage') in ('training', 'evaluating') and row.get('pid') and os.name == 'posix':
            try:
                os.kill(row['pid'], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError('A prior pipeline worker is still running; refusing duplicate dispatch')
    records = {j['job_id']: dict(previous.get('jobs', {}).get(j['job_id'], {})) for j in protocol['jobs']}
    status = dict(schema=SCHEMA, protocol_sha256=protocol['protocol_sha256'], stage='running',
                  coordinator_pid=os.getpid(), jobs=records, error=None, test_access=False)
    active, queue = {}, []
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')

    def publish():
        status['updated_utc'] = datetime.now(timezone.utc).isoformat()
        status['completed_runs'] = sum(r.get('stage') == 'complete' for r in records.values())
        status['failed_runs'] = sum(r.get('stage') == 'failed' for r in records.values())
        atomic_json(status_path, status)

    def selected(job, row):
        training = root/job['training']
        for alias in ('best_feasible', 'best_deadline'):
            if (training/'checkpoints'/(alias+'.json')).exists():
                row.update(checkpoint_alias=alias, selected_feasible=alias == 'best_feasible')
                return alias
        raise ValueError('Training ended without a selected checkpoint; no evaluation can be claimed')

    def launch(job, phase):
        row, folder = records[job['job_id']], root/'jobs'/job['job_id']
        folder.mkdir(parents=True, exist_ok=True)
        training = root/job['training']
        if phase == 'training':
            args = [sys.executable, '-m', 'experiments.joint_v3.train', '--pool', protocol['pool'],
                    '--output', str(training), '--config', str(root/job['config'])]
            if training.exists():
                args.append('--resume')
        else:
            alias = selected(job, row)
            output = folder/'evaluation'
            if output.exists():
                for attempt in range(1, 101):
                    archived = folder/f'evaluation-incomplete-{attempt:03d}'
                    if not archived.exists():
                        if output.resolve().parent != folder.resolve() or archived.resolve().parent != folder.resolve():
                            raise ValueError('Evaluation archive escaped its job directory')
                        output.rename(archived)
                        break
                else:
                    raise ValueError('Finite evaluation-attempt inventory exhausted')
            row['evaluation_output'] = str(output.relative_to(root))
            args = [sys.executable, '-m', 'experiments.joint_v3.evaluate', '--pool', protocol['pool'],
                    '--checkpoint', str(training), '--output', str(output), '--role', 'development',
                    '--alias', alias, '--methods', *(METHODS if job['variant'] == 'full' else ('B5',))]
            if job['variant'] == 'full':
                args.extend(['--v2-checkpoint', protocol['evaluation']['v2_checkpoint']])
        log = (folder/(phase+'.log')).open('ab')
        try:
            process = subprocess.Popen(args, cwd=ROOT, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, shell=False)
        finally:
            log.close()
        row.update(stage='training' if phase == 'training' else 'evaluating', error=None,
                   pid=process.pid, command=args, started_utc=datetime.now(timezone.utc).isoformat())
        active[job['job_id']] = dict(job=job, phase=phase, process=process,
            started=time.monotonic(), previous_wall=float(row.get('training_wall_seconds', 0.)), forced=False)
        publish()

    try:
        # Recover from immutable artifacts rather than trusting a previous stage label.
        for job in protocol['jobs']:
            row, training = records[job['job_id']], root/job['training']
            if training.exists():
                meta = _read(training/'identity.json')
                if (meta['config_identity'] != job['config_identity']
                        or meta['code_identity'] != protocol['source_identity']):
                    raise ValueError('Existing training config/source identity differs')
            completed = [p for p in sorted((root/'jobs'/job['job_id']).glob('evaluation*'))
                         if p.is_dir() and _evaluation_complete(p, job, protocol)]
            result_path = training/'run-result.json'
            result = _read(result_path) if result_path.exists() else None
            if completed:
                if result is None or not _finished(result, _read(root/job['config'])):
                    raise ValueError('Completed evaluation has no completed frozen training budget')
                row.update(stage='complete', error=None, evaluation_output=str(completed[0].relative_to(root)))
                row['run_result'] = result
                continue
            if result:
                row['run_result'] = result
                row['training_wall_seconds'] = max(float(row.get('training_wall_seconds', 0.)),
                                                  float(result.get('elapsed_seconds', 0.)))
            done = ((result is not None and _finished(result, _read(root/job['config'])))
                    or row.get('training_wall_seconds', 0.) >= protocol['budget']['max_wall_seconds'])
            row.update(stage='queued', error=None)
            queue.append((job, 'evaluation' if done else 'training'))
        publish()
        last_publish = time.monotonic()
        while queue or active:
            while queue and len(active) < jobs:
                job, phase = queue.pop(0)
                try:
                    launch(job, phase)
                except Exception as exc:
                    records[job['job_id']].update(stage='failed', error=f'{type(exc).__name__}: {exc}')
                    publish()
            for job_id, worker in list(active.items()):
                process, row = worker['process'], records[job_id]
                elapsed = time.monotonic()-worker['started']
                if worker['phase'] == 'training':
                    row['training_wall_seconds'] = worker['previous_wall']+elapsed
                    if row['training_wall_seconds'] >= protocol['budget']['max_wall_seconds'] and process.poll() is None:
                        process.send_signal(signal.SIGINT) if os.name == 'posix' else process.terminate()
                        worker['forced'] = True
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill(); process.wait(timeout=10)
                code = process.poll()
                if code is None:
                    continue
                del active[job_id]
                row.update(returncode=code, pid=None)
                if code and not worker['forced']:
                    row.update(stage='failed', error=f"{worker['phase']} exited with code{code}; see local log")
                elif worker['phase'] == 'training':
                    result_path = root/worker['job']['training']/'run-result.json'
                    if result_path.exists():
                        row['run_result'] = _read(result_path)
                    if worker['forced']:
                        row['training_stop_reason'] = 'external_finite_wall_budget_stop; latest committed checkpoint retained'
                    result = row.get('run_result')
                    if not worker['forced'] and (result is None or not _finished(result, _read(root/worker['job']['config']))):
                        row.update(stage='failed', error='Training stopped before its frozen budget; explicit pipeline resume retains committed state')
                        publish()
                        continue
                    try:
                        selected(worker['job'], row)
                        queue.insert(0, (worker['job'], 'evaluation'))
                        row['stage'] = 'trained'
                    except ValueError as exc:
                        row.update(stage='failed', error=str(exc))
                elif _evaluation_complete(root/row['evaluation_output'], worker['job'], protocol):
                    row.update(stage='complete', error=None,
                               evaluation_sha256=file_hash(root/row['evaluation_output']/'results.json'))
                else:
                    row.update(stage='failed', error='Evaluation exited without complete frozen results')
                publish()
            if time.monotonic()-last_publish >= 5:
                publish(); last_publish = time.monotonic()
            if active:
                time.sleep(.25)
        status['stage'] = 'complete' if all(r.get('stage') == 'complete' for r in records.values()) else 'complete_with_errors'
    except BaseException as exc:
        status.update(stage='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                      error=f'{type(exc).__name__}: {exc}')
        for job_id, worker in active.items():
            process = worker['process']
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=10)
            row = records[job_id]
            if worker['phase'] == 'training':
                row['training_wall_seconds'] = worker['previous_wall']+time.monotonic()-worker['started']
            row.update(stage='interrupted', pid=None)
        publish()
        raise
    publish()
    from .analyze import analyze
    analyze(root)
    from .report import refresh
    refresh(root, root/'reports')
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--run', action='store_true')
    parser.add_argument('--pool'); parser.add_argument('--output'); parser.add_argument('--protocol')
    parser.add_argument('--v2-checkpoint'); parser.add_argument('--jobs', type=int, default=3)
    args = parser.parse_args(argv)
    if args.prepare:
        if not args.pool or not args.output:
            parser.error('--prepare requires --pool and --output')
        print(prepare(args.pool, args.output, v2_checkpoint=args.v2_checkpoint), flush=True)
    else:
        if not args.protocol:
            parser.error('--run requires --protocol')
        result = run(args.protocol, jobs=args.jobs)
        print(json.dumps(dict(stage=result['stage'], completed_runs=result['completed_runs'],
                              failed_runs=result['failed_runs'])), flush=True)
        if result['failed_runs']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
