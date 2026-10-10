"""Run shared E1 policies, import frozen inputs, or strictly replay a run."""
import argparse
import csv
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import SEMANTICS_VERSION
from .config import ROOT, load_config, resolve_path, validate_config
from .contracts import canonical_bytes, content_hash, file_hash, write_json, window
from .engine import LocalExecutor
from .environment import FrozenEnvironment, load_environment
from .power import UpstreamPowerModel
from .policies import POLICY_SCHEMA, make_policy
from .workload import FrozenWorkload, load_workload

LEGACY_FILES = ('resolved_config.json', 'workload.json', 'environment.json', 'dc_config.json',
                'ledger.jsonl', 'events.jsonl', 'intervals.csv', 'summary.json')
SCIENTIFIC_FILES = LEGACY_FILES + ('policy_ledger.jsonl', 'policy_events.jsonl',
                                  'policy_intervals.jsonl', 'policy_summary.json')


def _git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def source_hashes():
    # Upstream tracked Python plus the new (possibly uncommitted) experimental
    # source. Exact contents matter even when HEAD alone cannot identify a run.
    files = set(_git('ls-files', '*.py').splitlines())
    for directory, children, names in os.walk(ROOT / 'experiments/e1'):
        children[:] = [name for name in children if name not in ('artifacts', '__pycache__')
                       and not (Path(directory) / name / 'run_status.json').is_file()]
        files.update(str((Path(directory) / name).relative_to(ROOT)).replace('\\', '/')
                     for name in names if name.endswith('.py'))
    files = {name for name in files if not name.startswith('experiments/e1/artifacts/')}
    return {name: file_hash(ROOT / name) for name in sorted(files)}


def dependencies():
    # Inventory the actual reproduction environment, including transitive
    # dependencies. Do not require packages the reused model never imports.
    return dict(sorted((dist.metadata['Name'].lower().replace('_', '-'), dist.version)
                       for dist in importlib.metadata.distributions() if dist.metadata['Name']))


def _check_contract(cfg, workload, environment):
    expected = window(cfg['start'], cfg['end'])
    if (workload.start, workload.end) != expected or (environment.start, environment.end) != expected:
        raise ValueError('Frozen input/config window mismatch')
    w = cfg['workload']
    if (workload.resource_multiplier, workload.sla_multiplier, workload.trace_year) != (
            w['resource_multiplier'], w['sla_multiplier'], w['trace_year']):
        raise ValueError('Frozen workload/config multiplier or year mismatch')
    if any(t.sla_multiplier != w['sla_multiplier'] or t.dc_id != cfg['datacenter']['dc_id']
           for t in workload.tasks):
        raise ValueError('Frozen task/config SLA or origin mismatch')


def _write_result(out, result):
    for name, rows in (('ledger.jsonl', result.tasks), ('events.jsonl', result.events),
                       ('policy_ledger.jsonl', result.policy_tasks),
                       ('policy_events.jsonl', result.policy_events),
                       ('policy_intervals.jsonl', result.policy_steps)):
        with (out / name).open('wb') as stream:
            for row in rows:
                stream.write(canonical_bytes(row) + b'\n')
    with (out / 'intervals.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result.steps[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(result.steps)
    write_json(out / 'summary.json', result.summary)
    write_json(out / 'policy_summary.json', result.policy_summary)


def _execute(out, cfg, workload, environment, model_path, replay_of=None, frozen_from=None):
    _check_contract(cfg, workload, environment)
    hashes = source_hashes()
    git = dict(head=_git('rev-parse', 'HEAD'), branch=_git('branch', '--show-current'),
               status_porcelain=_git('status', '--porcelain', '--untracked-files=normal'))
    for name, digest in hashes.items():
        target = out / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        if file_hash(target) != digest:
            raise RuntimeError('Source changed during snapshot')
    write_json(out / 'resolved_config.json', cfg)
    workload.save(out / 'workload.json')
    environment.save(out / 'environment.json')
    shutil.copyfile(model_path, out / 'dc_config.json')
    dc = cfg['datacenter']
    model = UpstreamPowerModel(dc, out / 'dc_config.json')
    result = LocalExecutor(workload, environment,
                           (dc['total_cores'], dc['total_gpus'], dc['total_mem']), model,
                           policy=make_policy(cfg)).run()
    if source_hashes() != hashes:
        raise RuntimeError('Source changed during execution')
    _write_result(out, result)
    provenance = dict(semantics_version=SEMANTICS_VERSION, git=git,
                      config_sha256=content_hash(cfg), sources=hashes, source_sha256=content_hash(hashes),
                      python=sys.version, dependencies=dependencies(), platform=platform.platform(),
                      replay_of=str(replay_of.resolve()) if replay_of is not None else None,
                      frozen_from=str(frozen_from.resolve()) if frozen_from is not None else None,
                      frozen_parent_source_sha256=(json.loads((frozen_from / 'provenance.json').read_text(
                          encoding='utf-8'))['source_sha256'] if frozen_from is not None else None),
                      policy_schema=POLICY_SCHEMA,
                      input_hashes=dict(workload_source=workload.source_sha256,
                                        carbon_source=environment.carbon_source_sha256,
                                        weather_source=environment.weather_source_sha256,
                                        workload_snapshot=workload.sha256,
                                        environment_snapshot=environment.sha256,
                                        dc_config=file_hash(out / 'dc_config.json')),
                      output_hashes={name: file_hash(out / name) for name in SCIENTIFIC_FILES})
    write_json(out / 'provenance.json', provenance)
    return result


def _fresh_run(output_dir, action):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite any prior evidence.
    started = datetime.now(timezone.utc).isoformat()
    write_json(out / 'run_status.json', dict(state='running', started_utc=started))
    try:
        result = action(out)
    except Exception as exc:
        write_json(out / 'run_status.json', dict(state='failed', started_utc=started,
                   ended_utc=datetime.now(timezone.utc).isoformat(),
                   error_type=type(exc).__name__, error=str(exc)))
        raise
    write_json(out / 'run_status.json', dict(state='complete', started_utc=started,
               ended_utc=datetime.now(timezone.utc).isoformat()))
    return result


def run_experiment(config, output_dir, root=ROOT):
    def action(out):
        cfg = load_config(config) if isinstance(config, (str, Path)) else validate_config(config)
        w, e = cfg['workload'], cfg['environment']
        start, end = window(cfg['start'], cfg['end'])
        workload = load_workload(resolve_path(w['path'], root), start, end,
                                 w['resource_multiplier'], w['sla_multiplier'],
                                 cfg['datacenter']['dc_id'], w['trace_year'])
        environment = load_environment(resolve_path(e['carbon_path'], root),
                                       resolve_path(e['weather_path'], root), start, end)
        return _execute(out, cfg, workload, environment,
                        resolve_path(cfg['datacenter']['dc_config_file'], root))
    return _fresh_run(output_dir, action)


def verified_archive(previous_dir, strict_source=True):
    """Verify archived evidence. Cross-version input import is explicit, not replay."""
    previous = Path(previous_dir)
    status = json.loads((previous / 'run_status.json').read_text(encoding='utf-8'))
    if status['state'] != 'complete':
        raise ValueError('Only completed experiments can be imported or replayed')
    provenance = json.loads((previous / 'provenance.json').read_text(encoding='utf-8'))
    if provenance['semantics_version'] != SEMANTICS_VERSION:
        raise ValueError('Replay semantics version mismatch')
    if content_hash(provenance['sources']) != provenance['source_sha256']:
        raise ValueError('Archived source manifest hash mismatch')
    if strict_source and source_hashes() != provenance['sources']:
        raise ValueError('Replay source hash mismatch: use the archived source/version')
    for name, digest in provenance['sources'].items():
        if file_hash(previous / 'source' / name) != digest:
            raise ValueError('Archived source hash mismatch')
    if dependencies() != provenance['dependencies'] or sys.version != provenance['python']:
        raise ValueError('Replay dependency/Python version mismatch')
    current_archive = (strict_source or provenance.get('policy_schema') is not None
                       or 'experiments/e1/policies.py' in provenance['sources'])
    if current_archive and provenance.get('policy_schema') != POLICY_SCHEMA:
        raise ValueError('Archived policy schema mismatch')
    required_outputs = SCIENTIFIC_FILES if current_archive else LEGACY_FILES
    if not set(required_outputs) <= set(provenance['output_hashes']):
        raise ValueError('Incomplete archived scientific output manifest')
    for name, digest in provenance['output_hashes'].items():
        if file_hash(previous / name) != digest:
            raise ValueError(f'Replay input/output file hash mismatch: {name}')
    cfg = validate_config(json.loads((previous / 'resolved_config.json').read_text(encoding='utf-8')))
    if content_hash(cfg) != provenance['config_sha256']:
        raise ValueError('Replay config hash mismatch')
    workload = FrozenWorkload.load(previous / 'workload.json')
    environment = FrozenEnvironment.load(previous / 'environment.json')
    inputs = dict(workload_source=workload.source_sha256,
                  carbon_source=environment.carbon_source_sha256,
                  weather_source=environment.weather_source_sha256,
                  workload_snapshot=workload.sha256, environment_snapshot=environment.sha256,
                  dc_config=file_hash(previous / 'dc_config.json'))
    if inputs != provenance['input_hashes']:
        raise ValueError('Replay frozen input hash mismatch')
    _check_contract(cfg, workload, environment)
    return cfg, workload, environment, provenance


def shared_config(cfg):
    return {key: value for key, value in cfg.items() if key not in ('policy', 'policy_params')}


def run_frozen_experiment(config, previous_dir, output_dir):
    previous = Path(previous_dir)
    def action(out):
        cfg = load_config(config) if isinstance(config, (str, Path)) else validate_config(config)
        old_cfg, workload, environment, _ = verified_archive(previous, strict_source=False)
        if shared_config(cfg) != shared_config(old_cfg):
            raise ValueError('Frozen import shared config mismatch: only policy and policy_params may change')
        return _execute(out, cfg, workload, environment, previous / 'dc_config.json', frozen_from=previous)
    return _fresh_run(output_dir, action)


def replay_experiment(previous_dir, output_dir):
    previous = Path(previous_dir)
    def action(out):
        cfg, workload, environment, provenance = verified_archive(previous)
        result = _execute(out, cfg, workload, environment, previous / 'dc_config.json', previous)
        for name, digest in provenance['output_hashes'].items():
            if file_hash(out / name) != digest:
                raise RuntimeError(f'Replay scientific output mismatch: {name}')
        return result
    return _fresh_run(output_dir, action)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--config', type=Path)
    source.add_argument('--replay', type=Path)
    parser.add_argument('--frozen-from', type=Path, help='Import verified inputs; requires --config')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.frozen_from is not None and args.config is None:
        parser.error('--frozen-from requires --config and cannot accompany --replay')
    if args.replay is not None:
        result = replay_experiment(args.replay, args.output)
    elif args.frozen_from is not None:
        result = run_frozen_experiment(args.config, args.frozen_from, args.output)
    else:
        result = run_experiment(args.config, args.output)
    print(json.dumps(result.summary, indent=2, ensure_ascii=False, allow_nan=False))
    print(f'Artifacts: {args.output.resolve()}')


if __name__ == '__main__':
    main()
