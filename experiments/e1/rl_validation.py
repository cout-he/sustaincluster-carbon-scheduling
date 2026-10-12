"""E1-5A independent historical equivalence and seeded rollout evidence.

This importer preserves sealed provenance and the original strict replay tools.
Only the intentional shared-kernel refactor and explicitly declared appended
modules may differ from the historical source manifest. Scheduling is performed
solely by LocalExecutor, directly or through CarbonSchedulingEnv.
"""
import argparse
import copy
import csv
import math
import platform
import sys
from pathlib import Path, PurePosixPath
from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd

from . import cars_experiment as historical
from . import feasibility_experiment as feasibility
from .cars import CARS_SCHEMA, make_cars_policy
from .config import ROOT
from .contention import factor_folder, read_json
from .contracts import STEP, canonical_bytes, content_hash, file_hash, utc_time, write_json
from .drain_metrics import phase_summary
from .engine import LocalExecutor
from .power import UpstreamPowerModel
from .run import _fresh_run, _write_result, dependencies
from .selection import TemporalEnvironment
from .temporal import experiment_sources, validate_temporal_config, write_csv
from .workload import FrozenWorkload

VERSION = 'e1-5a-mode-selection-validation-v1'
ACTION_POLICIES = ('edf_backfill', 'edf_carbon_threshold', 'cars_risk_only')
BATCH_POLICIES = ('immediate_fifo', 'edf_backfill', 'carbon_threshold_fifo', 'cars_risk_only')
RESULT_FILES = ('ledger.jsonl', 'events.jsonl', 'intervals.csv', 'summary.json',
                'policy_ledger.jsonl', 'policy_events.jsonl', 'policy_intervals.jsonl',
                'policy_summary.json')
CHANGED_ORIGINAL_SOURCES = {'experiments/e1/engine.py'}
NEW_SOURCE_FILES = feasibility.NEW_SOURCE_FILES | {
    'experiments/e1/rl_env.py', 'experiments/e1/rl_validation.py',
    'tests/e1/test_stepwise.py', 'tests/e1/test_rl_env.py', 'tests/e1/test_rl_validation.py'}
EXCLUDED = ['runtime.json', 'run_status.json', 'provenance.json', 'source/**']


def _safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe archived manifest path')
    return name


def _inventory(path):
    return {p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file()
            and 'source' not in p.relative_to(path).parts
            and p.name not in ('runtime.json', 'run_status.json', 'provenance.json')}


def verify_sealed_archive(path, expected_files, *, current_sources=None, snapshot=True,
                          campaign=False):
    """Validate archived bytes independently of authorized current source changes."""
    path = Path(path)
    provenance = read_json(path/'provenance.json')
    current = experiment_sources() if current_sources is None else current_sources
    sources = provenance['sources']
    if read_json(path/'run_status.json').get('state') != 'complete':
        raise ValueError('Historical archive is incomplete')
    if content_hash(sources) != provenance['source_sha256']:
        raise ValueError('Historical source manifest hash mismatch')
    if any(name not in current for name in sources):
        raise ValueError('Historical original source is missing from current checkout')
    changed = sorted(name for name, sha in sources.items() if current[name] != sha)
    if set(changed)-CHANGED_ORIGINAL_SOURCES:
        raise ValueError('Historical non-kernel source changed: '+str(changed))
    added = sorted(set(current)-set(sources))
    if set(added)-NEW_SOURCE_FILES:
        raise ValueError('Undeclared new source: '+str(set(added)-NEW_SOURCE_FILES))
    if snapshot:
        actual = {p.relative_to(path/'source').as_posix()
                  for p in (path/'source').rglob('*') if p.is_file()}
        if actual != set(sources):
            raise ValueError('Historical source snapshot inventory mismatch')
        for name, sha in sources.items():
            if file_hash(path/'source'/_safe_name(name)) != sha:
                raise ValueError('Historical source snapshot hash mismatch: '+name)
    outputs = provenance['output_hashes']
    actual = ({p.name for p in path.iterdir() if p.is_file()
               and p.name not in ('runtime.json', 'run_status.json', 'provenance.json')}
              if campaign else _inventory(path))
    if set(outputs) != set(expected_files) or actual != set(expected_files):
        raise ValueError('Historical scientific manifest/inventory incomplete')
    if provenance.get('excluded_outputs') != EXCLUDED:
        raise ValueError('Historical excluded output manifest mismatch')
    if provenance.get('runtime_hashes') != {'runtime.json': file_hash(path/'runtime.json')}:
        raise ValueError('Historical runtime integrity mismatch')
    for name, sha in outputs.items():
        if file_hash(path/_safe_name(name)) != sha:
            raise ValueError('Historical scientific file hash mismatch: '+name)
    return dict(source_sha256=provenance['source_sha256'],
                provenance_sha256=content_hash(provenance),
                changed_original_sources=changed, new_sources=added,
                snapshot_files_checked=len(sources) if snapshot else 0,
                scientific_files_checked=len(outputs))


def _verify_reference_identity(identity, inputs, parameters, archived_sources):
    required = {'version', 'archive_kind', 'reference_provenance_sha256', 'source_sha256', 'sources',
        'git', 'python', 'dependencies', 'input_hashes', 'threshold_g_per_kwh', 'common_end',
        'scientific_hashes', 'complete_output_hashes', 'frozen_import', 'workload_regenerated'}
    if (set(identity) != required or identity['version'] != CARS_SCHEMA
        or identity['archive_kind'] != 'comparison' or identity['frozen_import'] is not True
        or identity['workload_regenerated'] is not False
        or set(identity['scientific_hashes']) != feasibility.CONTROL_FILES
        or set(identity['complete_output_hashes']) != historical.LEVEL_FILES
        or identity['scientific_hashes'] != {
            name: identity['complete_output_hashes'][name] for name in sorted(feasibility.CONTROL_FILES)}
        or identity['input_hashes'] != inputs
        or identity['threshold_g_per_kwh'] != parameters['threshold_g_per_kwh']
        or content_hash(identity['sources']) != identity['source_sha256']
        or any(archived_sources.get(name) != sha for name, sha in identity['sources'].items())):
        raise ValueError('Historical frozen reference identity differs')


def _load_level(path, kind, current):
    module = historical if kind == CARS_SCHEMA else feasibility
    evidence = verify_sealed_archive(path, module.LEVEL_FILES, current_sources=current)
    provenance = read_json(path/'provenance.json')
    keys = feasibility._OLD_LEVEL_KEYS | ({'model_source_version'} if kind != CARS_SCHEMA else set())
    if (set(provenance) != keys or provenance.get('version') != kind
        or provenance.get('archive_kind') != 'comparison'):
        raise ValueError('Historical comparison schema mismatch')
    cfg, parameters, selection = (read_json(path/name) for name in
                                  ('resolved_config.json', 'parameters.json', 'selection.json'))
    historical.validate_parameters(parameters)
    workload = FrozenWorkload.load(path/'workload.json')
    environment = TemporalEnvironment.load(path/'environment.json')
    if validate_temporal_config(cfg, workload, environment) != cfg:
        raise ValueError('Historical frozen configuration mismatch')
    hashes = historical._input_hashes(cfg, workload, environment, path/'dc_config.json', parameters, selection)
    if kind != CARS_SCHEMA:
        ablation = read_json(path/'ablation_config.json')
        feasibility.validate_ablation_config(ablation)
        identity = read_json(path/'reference.json')
        _verify_reference_identity(identity, hashes, parameters, provenance['sources'])
        hashes.update(ablation_config=content_hash(ablation), reference=content_hash(identity))
        if provenance['model_source_version'] != feasibility._model_source_version(identity):
            raise ValueError('Historical model source version mismatch')
    if hashes != provenance['input_hashes']:
        raise ValueError('Historical complete input manifest mismatch')
    fairness = read_json(path/'fairness.json')
    endpoint = utc_time(fairness['common_end'], True)
    if not workload.end <= endpoint <= environment.end:
        raise ValueError('Historical common endpoint outside frozen coverage')
    names = historical.CARS_POLICY_NAMES if kind == CARS_SCHEMA else feasibility.ABLATION_POLICY_NAMES
    if read_json(path/'design.json')['policies'] != list(names):
        raise ValueError('Historical policy declaration differs')
    endpoints = []
    for name in names:
        expected_cfg = feasibility._old_policy_config(cfg, parameters, name)
        if read_json(path/name/'resolved_config.json') != expected_cfg:
            raise ValueError('Historical policy parameters changed: '+name)
        endpoints.append(utc_time(read_json(path/name/'phases.json')['own']['observation_end'], True))
    if max(endpoints) != endpoint:
        raise ValueError('Historical common drain endpoint mismatch')
    return dict(path=path, cfg=cfg, workload=workload, environment=environment,
                parameters=parameters, selection=selection, model_path=path/'dc_config.json',
                common_end=endpoint, provenance=provenance, evidence=evidence)


def verify_historical_campaign(path, *, current_sources=None):
    """Verify every sealed level, including unexecuted policies and load levels."""
    path = Path(path)
    provenance, config = read_json(path/'provenance.json'), read_json(path/'config.json')
    kind = provenance.get('version')
    if kind not in (CARS_SCHEMA, feasibility.ABLATION_SCHEMA) or provenance.get('archive_kind') != 'campaign':
        raise ValueError('Historical campaign schema mismatch')
    module = historical if kind == CARS_SCHEMA else feasibility
    keys = feasibility._OLD_CAMPAIGN_KEYS | ({'git', 'model_source_version'} if kind != CARS_SCHEMA else set())
    if set(provenance) != keys:
        raise ValueError('Historical campaign provenance manifest incomplete')
    current = experiment_sources() if current_sources is None else current_sources
    evidence = verify_sealed_archive(path, module.CAMPAIGN_FILES, current_sources=current,
                                    snapshot=kind != CARS_SCHEMA, campaign=True)
    required_factors = [1.0, 1.25, 1.5, 2.0] if kind == CARS_SCHEMA else [1.5, 2.0]
    config_keys = {'version', 'archive_kind', 'factors', 'parameters', 'reference_parent_sha256',
                   'frozen_import', 'workload_regenerated'}
    if kind != CARS_SCHEMA:
        config_keys.add('ablation_config')
    if (set(config) != config_keys or config['version'] != kind or config['archive_kind'] != 'campaign'
        or config['factors'] != required_factors or config['frozen_import'] is not True
        or config['workload_regenerated'] is not False
        or config['parameters'] != read_json(path/'parameters.json')):
        raise ValueError('Historical campaign declaration differs')
    historical.validate_parameters(config['parameters'])
    identities = read_json(path/'input_identities.json')
    expected_inputs = dict(parameters=content_hash(config['parameters']), identities=content_hash(identities))
    if kind != CARS_SCHEMA:
        feasibility.validate_ablation_config(config['ablation_config'])
        if config['ablation_config'] != read_json(path/'ablation_config.json'):
            raise ValueError('Historical ablation declaration differs')
        expected_inputs.update(ablation_config=content_hash(config['ablation_config']), config=content_hash(config))
    if provenance['input_hashes'] != expected_inputs:
        raise ValueError('Historical campaign input manifest mismatch')
    indexed = {i['factor']: i for i in identities}
    children = {factor_folder(f) for f in required_factors}
    if (len(indexed) != len(identities) or set(indexed) != set(required_factors)
        or set(provenance['child_output_manifests']) != children
        or {p.name for p in path.iterdir() if p.is_dir()} != children | ({'source'} if kind != CARS_SCHEMA else set())):
        raise ValueError('Historical child manifest/inventory incomplete')
    levels = {}
    for factor in required_factors:
        name, child = factor_folder(factor), path/factor_folder(factor)
        child_prov = read_json(child/'provenance.json')
        if (child_prov != provenance['child_output_manifests'][name]
            or child_prov['sources'] != provenance['sources']):
            raise ValueError('Historical child provenance differs from sealed parent')
        item = _load_level(child, kind, current)
        identity, workload = indexed[factor], item['workload']
        if (identity['workload_sha256'] != workload.sha256
            or identity['environment_sha256'] != item['environment'].sha256
            or identity['dc_config_sha256'] != file_hash(item['model_path'])
            or identity['tasks'] != len(workload.tasks)
            or identity['common_end'] != item['common_end'].isoformat()
            or identity['reference'] != read_json(child/'reference.json')):
            raise ValueError('Historical child frozen identity mismatch')
        parameters = copy.deepcopy(config['parameters'])
        parameters['threshold_g_per_kwh'] = item['parameters']['threshold_g_per_kwh']
        if parameters != item['parameters']:
            raise ValueError('Historical child fixed parameters differ from parent')
        item['factor'] = float(factor)
        levels[float(factor)] = item
    if kind != CARS_SCHEMA and provenance['model_source_version'] != {
        factor_folder(factor): item['provenance']['model_source_version'] for factor, item in levels.items()}:
        raise ValueError('Historical campaign model source version mismatch')
    return dict(path=path, provenance=provenance, levels=levels, evidence=evidence)


def _csv_numerically_exact(first, second):
    with Path(first).open(encoding='utf-8', newline='') as stream:
        left = list(csv.reader(stream))
    with Path(second).open(encoding='utf-8', newline='') as stream:
        right = list(csv.reader(stream))
    if not left or not right or left[0] != right[0] or len(left) != len(right):
        return False
    fields = left[0]
    for a, b in zip(left[1:], right[1:]):
        if len(a) != len(fields) or len(b) != len(fields):
            return False
        for name, x, y in zip(fields, a, b):
            if name in ('interval_start', 'interval_end'):
                if x != y:
                    return False
            else:
                try:
                    one, two = Decimal(x), Decimal(y)
                except InvalidOperation:
                    return False
                if not one.is_finite() or not two.is_finite() or one != two:
                    return False
    return True


def compare_archived_result(result, archive_policy, output):
    """Pin seven files' bytes and every CSV field exactly, without float tolerance."""
    output, archive_policy = Path(output), Path(archive_policy)
    output.mkdir(parents=True, exist_ok=False)
    _write_result(output, result)
    hashes = {name: file_hash(output/name) for name in RESULT_FILES}
    byte_differences = [name for name in RESULT_FILES if hashes[name] != file_hash(archive_policy/name)]
    csv_exact = _csv_numerically_exact(output/'intervals.csv', archive_policy/'intervals.csv')
    differences = [name for name in byte_differences if name != 'intervals.csv']
    if not csv_exact:
        differences.append('intervals.csv')
    resource_hours = {resource: math.fsum(s['used_'+resource]*s['interval_hours']
                                        for s in result.steps)
                      for resource in ('cores', 'gpus', 'mem')}
    phase_path = archive_policy/'phases.json'
    archived_resource_hours = None
    if phase_path.exists():
        phase = read_json(phase_path)['common']
        archived_resource_hours = {resource: phase[resource+'_hours']
                                   for resource in ('cores', 'gpus', 'mem')}
        if resource_hours != archived_resource_hours:
            differences.append('phases.json resource hours')
    report = dict(files_checked=len(RESULT_FILES), all_files_byte_identical=not byte_differences,
                  all_numerical_fields_exact=not differences,
                  serialization_only_differences=['intervals.csv'] if csv_exact and 'intervals.csv' in byte_differences else [],
                  csv_comparison='same rows/columns/timestamps and exact Decimal values; no tolerance',
                  output_hashes=hashes, mismatched_files=differences,
                  resource_hours=resource_hours, archived_resource_hours=archived_resource_hours)
    write_json(output/'regression.json', report)
    if differences:
        raise RuntimeError('Historical numerical regression mismatch: '+', '.join(differences))
    return report


def _rollout(workload, environment, capacity, power_factory, parameters, action, *,
             seed, max_episode_steps=None, stop_at_drain=False):
    from .rl_env import CarbonSchedulingEnv
    env = CarbonSchedulingEnv(workload, environment, capacity, power_factory, parameters,
                              max_episode_steps=max_episode_steps, stop_at_drain=stop_at_drain)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    trace = []
    try:
        while True:
            selected = int(rng.integers(3)) if action is None else action
            _, reward, terminated, truncated, info = env.step(selected)
            interval = env.executor.steps[-1]
            trace.append(dict(step=len(trace), interval_start=interval['interval_start'], action=selected,
                              reward=float(reward), carbon_kg=interval['carbon_kg'],
                              reward_terms=copy.deepcopy(info['reward']),
                              terminated=bool(terminated), truncated=bool(truncated)))
            if terminated or truncated:
                return env.executor.result(), trace
    finally:
        env.close()


def run_fixed_action(workload, environment, capacity, power_factory, parameters, action, until):
    if isinstance(action, bool) or action not in (0, 1, 2):
        raise ValueError('Fixed action must be 0, 1, or 2')
    until = utc_time(until, True)
    if not workload.end <= until <= environment.end:
        raise ValueError('Accounting endpoint must include the finite arrival window')
    steps = int((until-workload.start)/STEP)
    return _rollout(workload, environment, capacity, power_factory, parameters, action,
                    seed=20261012, max_episode_steps=steps, stop_at_drain=False)


def run_random_actions(workload, environment, capacity, power_factory, parameters, *, seed=20261012):
    return _rollout(workload, environment, capacity, power_factory, parameters, None,
                    seed=seed, stop_at_drain=True)


def _write_trajectory(path, trace):
    rows = []
    for source in trace:
        row = {key: value for key, value in source.items() if key != 'reward_terms'}
        row.update({'reward_'+key: canonical_bytes(value).decode('utf-8')
                    if isinstance(value, (list, tuple, dict)) else value
                    for key, value in source['reward_terms'].items()})
        rows.append(row)
    write_csv(path, rows)


def _summary_row(result, factor, kind, policy, regression, action=None):
    phase = phase_summary(result, pd.Timestamp(result.summary['observation_end']))
    return dict(factor=factor, run_kind=kind, policy=policy, action=action,
                all_files_byte_identical=regression['all_files_byte_identical'],
                all_numerical_fields_exact=regression['all_numerical_fields_exact'],
                serialization_only_differences=canonical_bytes(
                    regression['serialization_only_differences']).decode('utf-8'), files_checked=8,
                **{key: phase[key] for key in ('arrived', 'completed', 'pending', 'running',
                    'deadline_missed_completed', 'deadline_missed_unfinished', 'observation_end',
                    'it_energy_kwh', 'hvac_energy_kwh', 'total_energy_kwh', 'carbon_kg',
                    'cores_hours', 'gpus_hours', 'mem_hours')})


def run_validation(e1_3, e1_4a, output, *, seed=20261012):
    """Run eight direct controls, six constant modes, and one seeded smoke episode."""
    def action(out):
        sources = experiment_sources()
        historical._snapshot_sources(out, sources)
        old = verify_historical_campaign(e1_3, current_sources=sources)
        ablations = verify_historical_campaign(e1_4a, current_sources=sources)
        linked = read_json(Path(e1_4a)/'config.json')['reference_parent_sha256']
        if linked != content_hash(old['provenance']):
            raise ValueError('E1-4A parent reference identity mismatch')
        archive_evidence = {'e1_3': old['evidence'], 'e1_4a': ablations['evidence'], 'levels': {}}
        rows = []
        for factor in (1.5, 2.0):
            item, later = old['levels'][factor], ablations['levels'][factor]
            identity = read_json(later['path']/'reference.json')
            expected_identity = dict(version=CARS_SCHEMA, archive_kind='comparison',
                reference_provenance_sha256=content_hash(item['provenance']),
                source_sha256=item['provenance']['source_sha256'], sources=item['provenance']['sources'],
                git=item['provenance']['git'], python=item['provenance']['python'],
                dependencies=item['provenance']['dependencies'], input_hashes=item['provenance']['input_hashes'],
                threshold_g_per_kwh=item['parameters']['threshold_g_per_kwh'],
                common_end=item['common_end'].isoformat(),
                scientific_hashes={name: item['provenance']['output_hashes'][name]
                                   for name in sorted(feasibility.CONTROL_FILES)},
                complete_output_hashes=item['provenance']['output_hashes'],
                frozen_import=True, workload_regenerated=False)
            if (identity != expected_identity or later['cfg'] != item['cfg']
                or later['selection'] != item['selection']
                or file_hash(later['model_path']) != file_hash(item['model_path'])
                or later['workload'].sha256 != item['workload'].sha256
                or later['environment'].sha256 != item['environment'].sha256
                or later['parameters'] != item['parameters']):
                raise ValueError('E1-4A linked frozen E1-3 identity mismatch')
            for name in feasibility.CONTROL_NAMES:
                for filename in (*historical.POLICY_FILES, 'diagnostics.jsonl'):
                    if file_hash(later['path']/'legacy-controls'/name/filename) != file_hash(item['path']/name/filename):
                        raise ValueError('E1-4A sealed control scientific mismatch: '+name+'/'+filename)
            archive_evidence['levels'][factor_folder(factor)] = {
                'e1_3': item['evidence'], 'e1_4a': later['evidence']}
            workload, environment, parameters = item['workload'], item['environment'], item['parameters']
            dc = item['cfg']['datacenter']
            capacity = dc['total_cores'], dc['total_gpus'], dc['total_mem']
            factory = lambda: UpstreamPowerModel(dc, item['model_path'])
            level = out/factor_folder(factor)
            level.mkdir()
            for name in BATCH_POLICIES:
                print(f'E1-5A {factor}: direct {name}', flush=True)
                executor = LocalExecutor(workload, environment, capacity, factory(),
                    policy=make_cars_policy(name, parameters, workload.start, capacity), drain=True)
                own = executor.run(stop_at_drain=True)
                result = executor.extend_idle(item['common_end']) if executor.drained_at is not None else own
                regression = compare_archived_result(result, item['path']/name, level/('batch_'+name))
                rows.append(_summary_row(result, factor, 'direct_policy', name, regression))
            for selected, name in enumerate(ACTION_POLICIES):
                print(f'E1-5A {factor}: fixed mode {selected} ({name})', flush=True)
                result, trace = run_fixed_action(workload, environment, capacity, factory,
                                                parameters, selected, item['common_end'])
                folder = level/('action_'+str(selected))
                regression = compare_archived_result(result, item['path']/name, folder)
                _write_trajectory(folder/'trajectory.csv', trace)
                rows.append(_summary_row(result, factor, 'fixed_mode', name, regression, selected))
        write_csv(out/'numerical_equivalence.csv', rows)
        item = old['levels'][1.5]
        dc = item['cfg']['datacenter']
        capacity = dc['total_cores'], dc['total_gpus'], dc['total_mem']
        print(f'E1-5A 1.5: random seeded rollout ({seed})', flush=True)
        result, trace = run_random_actions(item['workload'], item['environment'], capacity,
            lambda: UpstreamPowerModel(dc, item['model_path']), item['parameters'], seed=seed)
        if (result.summary['arrived'] != len(item['workload'].tasks)
            or result.summary['completed'] != result.summary['arrived']
            or result.summary['pending'] or result.summary['running']
            or not trace[-1]['terminated'] or trace[-1]['truncated']):
            raise RuntimeError('Random smoke rollout did not complete its finite cohort')
        random_dir = out/'random_load_1p50'
        random_dir.mkdir()
        _write_result(random_dir, result)
        _write_trajectory(random_dir/'trajectory.csv', trace)
        write_json(random_dir/'validation.json', dict(seed=seed, all_arrived_tasks_completed=True,
            mode='seeded_smoke_validation', performance_claim=False, training_performed=False,
            policy_summary_role='last selected mode plus episode-wide counts',
            action_counts={str(a): sum(r['action'] == a for r in trace) for a in range(3)},
            reward_sum=math.fsum(r['reward'] for r in trace)))
        write_json(out/'archive_integrity.json', archive_evidence)
        if experiment_sources() != sources:
            raise RuntimeError('Source changed during E1-5A validation')
        outputs = _inventory(out)
        write_json(out/'provenance.json', dict(version=VERSION,
            historical_archives={'e1_3': str(Path(e1_3).resolve()), 'e1_4a': str(Path(e1_4a).resolve())},
            historical_provenance_untouched=True, sources=sources, source_sha256=content_hash(sources),
            python=sys.version, dependencies=dependencies(), platform=platform.platform(), seed=seed,
            allowed_changed_original_sources=sorted(CHANGED_ORIGINAL_SOURCES),
            allowed_appended_sources=sorted(NEW_SOURCE_FILES),
            threshold_role='sealed 137.64999999999998 equivalence; independent of future training calibration',
            training_performed=False, performance_claim=False,
            output_hashes={name: file_hash(out/name) for name in sorted(outputs)}))
        return rows
    return _fresh_run(output, action)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--e1-3', default=ROOT/'research_notes/E1/E1-3/results-001', type=Path)
    parser.add_argument('--e1-4a', default=ROOT/'research_notes/E1/E1-4A/results-001', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', default=20261012, type=int)
    args = parser.parse_args()
    run_validation(args.e1_3, args.e1_4a, args.output, seed=args.seed)


if __name__ == '__main__':
    main()
