"""E1-4A five-policy feasibility ablations over sealed E1-3 frozen cohorts."""
import argparse
import copy
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from . import cars_experiment as legacy
from .cars_experiment import TimingPolicy, _input_hashes, _jsonl, _snapshot_sources
from .config import ROOT, resolve_path
from .contention import factor_folder, read_json
from .contracts import canonical_bytes, content_hash, file_hash, write_json
from .drain_metrics import phase_rows, phase_summary
from .engine import LocalExecutor
from .feasibility_ablations import ABLATION_POLICY_NAMES, ABLATION_SCHEMA, make_ablation_policy
from .policies import POLICY_NAMES
from .power import UpstreamPowerModel
from .run import _fresh_run, _git, _write_result, dependencies
from .selection import CI_BASIS, TemporalEnvironment
from .temporal import experiment_sources, validate_temporal_config, write_csv
from .workload import FrozenWorkload


POLICY_FILES = legacy.POLICY_FILES
CONTROL_NAMES = ABLATION_POLICY_NAMES[:3]
NEW_SOURCE_FILES = {
    'experiments/e1/feasibility_ablations.py', 'experiments/e1/feasibility_experiment.py',
    'experiments/e1/diagnose_feasibility.py', 'experiments/e1/feasibility_intervention.py',
    'tests/e1/test_feasibility_ablations.py', 'tests/e1/test_feasibility_experiment.py',
    'tests/e1/test_diagnose_feasibility.py', 'tests/e1/test_feasibility_intervention.py'}
PUBLICATION_COMMIT = 'ff09c06ebac789716416abc4771f238b629cf39d'
HISTORICAL_TASK_ID = '3ba8a8e0067288f8/2023-07-01T10:15:00+00:00/17475/189~e1-2b-20261010-copy1'
DEFAULT_CONFIG = ROOT/'experiments/e1/configs/e1_4a_feasibility.json'
REQUIRED_FACTORS = (1.5, 2.0)
DECLARED_FACTORS = (1.0, 1.25, 1.5, 2.0)
CONTROL_FILES = {f'{p}/{n}' for p in CONTROL_NAMES for n in (*POLICY_FILES, 'diagnostics.jsonl')}
LEVEL_FILES = {'resolved_config.json', 'parameters.json', 'ablation_config.json', 'design.json',
               'selection.json', 'workload.json', 'environment.json', 'dc_config.json',
               'reference.json', 'baseline_regression.json', 'fairness.json', 'comparison.csv',
               'timeseries.csv', 'paired_task_shifts.csv', 'task_diagnostics.jsonl', 'diagnostics.json'}
LEVEL_FILES.update(f'{p}/{n}' for p in ABLATION_POLICY_NAMES for n in (*POLICY_FILES, 'diagnostics.jsonl'))
LEVEL_FILES.update('legacy-controls/'+n for n in CONTROL_FILES)
CAMPAIGN_FILES = {'config.json', 'parameters.json', 'ablation_config.json', 'comparison.csv',
                  'input_identities.json', 'baseline_regression.json', 'diagnostics.json'}
_OLD_LEVEL_KEYS = {'version', 'archive_kind', 'git', 'command', 'python', 'dependencies',
    'sources', 'source_sha256', 'input_hashes', 'excluded_outputs', 'runtime_hashes', 'output_hashes'}
_OLD_CAMPAIGN_KEYS = _OLD_LEVEL_KEYS-{'git'} | {'child_output_manifests'}
_EXCLUDED = ['runtime.json', 'run_status.json', 'provenance.json', 'source/**']


def validate_ablation_config(config=None):
    """Keep experiment declarations distinct from unchanged E1-3 policy parameters."""
    config = read_json(DEFAULT_CONFIG) if config is None else (
        read_json(config) if isinstance(config, (str, Path)) else copy.deepcopy(config))
    expected = dict(version=ABLATION_SCHEMA, policies=list(ABLATION_POLICY_NAMES),
        required_factors=list(REQUIRED_FACTORS), optional_factors=[1.0, 1.25],
        reference_publication_commit=PUBLICATION_COMMIT, parameters_source='sealed_e1_3_archive',
        frozen_parameters=True, future_ci=False, cap_count_basis='current_candidate_pending',
        cap_condition='pending>max_pending', shadow_uncertain='risk_only_decision',
        representative_selection='historical_case_plus_first_chronological_per_guard',
        historical_case_task_id=HISTORICAL_TASK_ID)
    for key, value in expected.items():
        if key not in config or canonical_bytes(config[key]) != canonical_bytes(value):
            raise ValueError('E1-4A declaration differs: '+key)
    allowed = set(expected) | {'intervention'}
    if set(config) != allowed or not isinstance(config['historical_case_task_id'], str):
        raise ValueError('E1-4A declaration manifest incomplete')
    intervention = config['intervention']
    if (set(intervention) != {'factor', 'time', 'task_id', 'override', 'separate_from_baselines'}
        or intervention['factor'] != 1.5 or intervention['time'] != '2023-07-01T10:15:00+00:00'
        or intervention['task_id'] != config['historical_case_task_id']
        or intervention['override'] != 'defer_once' or intervention['separate_from_baselines'] is not True):
        raise ValueError('E1-4A separate intervention declaration differs')
    return config


def _scientific_inventory(path):
    return {str(p.relative_to(path)).replace('\\', '/') for p in path.rglob('*') if p.is_file()
        and 'source' not in p.relative_to(path).parts
        and p.name not in ('run_status.json', 'provenance.json', 'runtime.json')}


def _verify_campaign_inventory(path, root_files, child_files):
    path = Path(path)
    directories = {p.name for p in path.iterdir() if p.is_dir()}
    required = set(root_files)
    required.update(name+'/'+filename for name, files in child_files.items() for filename in files)
    if (not directories <= set(child_files) | {'source'}
        or _scientific_inventory(path) != required):
        raise ValueError('Campaign complete scientific inventory differs from required manifest')


def _verify_hash_manifest(path, provenance, required, label):
    if set(provenance['output_hashes']) != required or _scientific_inventory(path) != required:
        raise ValueError(label+' scientific manifest incomplete')
    if provenance.get('excluded_outputs') != _EXCLUDED:
        raise ValueError(label+' excluded output manifest differs')
    if provenance.get('runtime_hashes') != {'runtime.json': file_hash(path/'runtime.json')}:
        raise ValueError(label+' runtime integrity mismatch')
    for name, sha in provenance['output_hashes'].items():
        if file_hash(path/name) != sha:
            raise ValueError(label+' archived output mismatch: '+name)


def _verify_sources(path, provenance, *, cross_version=False, snapshot=True):
    current, sources = experiment_sources(), provenance['sources']
    if content_hash(sources) != provenance['source_sha256']:
        raise ValueError('Source manifest hash mismatch')
    if cross_version:
        if not set(current)-NEW_SOURCE_FILES <= set(sources):
            raise ValueError('E1-3 original source manifest incomplete')
        if any(current.get(name) != sha for name, sha in sources.items()):
            raise ValueError('E1-3 original source changed in current checkout')
    elif sources != current:
        raise ValueError('E1-4A replay source mismatch')
    if provenance['python'] != sys.version or provenance['dependencies'] != dependencies():
        raise ValueError('Python/dependency mismatch')
    if snapshot:
        actual = {str(p.relative_to(path/'source')).replace('\\', '/')
                  for p in (path/'source').rglob('*') if p.is_file()}
        if actual != set(sources):
            raise ValueError('Archived source snapshot manifest incomplete')
        for name, sha in sources.items():
            if file_hash(path/'source'/name) != sha:
                raise ValueError('Archived source mismatch: '+name)


def _old_policy_config(cfg, parameters, name):
    config = copy.deepcopy(cfg)
    config['policy'] = name
    config.pop('policy_params', None)
    if name == 'carbon_threshold_fifo':
        config['policy_params'] = {'threshold_g_per_kwh': parameters['threshold_g_per_kwh']}
    elif name not in POLICY_NAMES:
        config['policy_params'] = copy.deepcopy(parameters)
    return config


def _verify_import_level(path, *, require_approved=False):
    """Cross-version import verifies every original source; new modules may be appended."""
    path = Path(path)
    prov = read_json(path/'provenance.json')
    if (read_json(path/'run_status.json')['state'] != 'complete'
        or set(prov) != _OLD_LEVEL_KEYS or prov.get('version') != legacy.CARS_SCHEMA
        or prov.get('archive_kind') != 'comparison'):
        raise ValueError('E1-3 comparison status/schema/manifest mismatch')
    _verify_sources(path, prov, cross_version=True)
    _verify_hash_manifest(path, prov, legacy.LEVEL_FILES, 'E1-3')
    cfg, parameters, selection = (read_json(path/n) for n in
        ('resolved_config.json', 'parameters.json', 'selection.json'))
    legacy.validate_parameters(parameters)
    workload = FrozenWorkload.load(path/'workload.json')
    environment = TemporalEnvironment.load(path/'environment.json')
    if validate_temporal_config(cfg, workload, environment) != cfg:
        raise ValueError('E1-3 configuration is not resolved')
    if _input_hashes(cfg, workload, environment, path/'dc_config.json', parameters, selection) != prov['input_hashes']:
        raise ValueError('E1-3 complete input manifest mismatch')
    design = read_json(path/'design.json')
    expected_design = dict(version=legacy.CARS_SCHEMA, archive_kind='comparison',
        policies=list(legacy.CARS_POLICY_NAMES), arrival_end=workload.end.isoformat(),
        safety_end=environment.end.isoformat(), threshold_g_per_kwh=parameters['threshold_g_per_kwh'],
        ci_basis=CI_BASIS,
        common_end_rule='maximum seven observed grid drain boundaries, or safety cap if any incomplete',
        legacy_common_end_rule='maximum original-three observed grid drain boundaries',
        idle_rule='same persistent physical model, zero CPU/GPU/MEM, real corresponding weather/CI')
    if design != expected_design:
        raise ValueError('E1-3 design/parameter/configuration mismatch')
    endpoints = []
    for name in legacy.CARS_POLICY_NAMES:
        if read_json(path/name/'resolved_config.json') != _old_policy_config(cfg, parameters, name):
            raise ValueError('E1-3 policy fixed configuration mismatch: '+name)
        endpoints.append(pd.Timestamp(read_json(path/name/'phases.json')['own']['observation_end']))
    common = max(endpoints).isoformat()
    if read_json(path/'fairness.json')['common_end'] != common:
        raise ValueError('E1-3 common drain endpoint mismatch')
    previous = read_json(path/'reference.json')
    if previous is not None:
        for name in ('workload_snapshot', 'environment_snapshot', 'dc_config'):
            if previous['input_hashes'][name] != prov['input_hashes'][name]:
                raise ValueError('E1-3 parent input identity mismatch')
    if require_approved:
        dc = cfg['datacenter']
        expected_dc = dict(total_cores=50000, total_gpus=1000, total_mem=80000,
                           location='US-CAL-CISO', hvac_setpoint_c=22)
        if (prov['git']['head'] != legacy.APPROVED_ARCHIVE_HEAD
            or workload.start.isoformat() != legacy.FIXED_PARAMETERS['arrival_start']
            or workload.end.isoformat() != legacy.FIXED_PARAMETERS['arrival_end']
            or workload.resource_multiplier != 5 or workload.sla_multiplier != 1.5
            or any(dc[k] != v for k, v in expected_dc.items())
            or environment.sha256 != legacy.APPROVED_ENV_HASH
            or file_hash(path/'dc_config.json') != legacy.APPROVED_DC_HASH
            or parameters['threshold_g_per_kwh'] != 137.64999999999998):
            raise ValueError('Reference is not the approved E1-3 experiment')
        actual_sources = dict(workload_source=file_hash(resolve_path(cfg['workload']['path'])),
            carbon_source=file_hash(resolve_path(cfg['environment']['carbon_path'])),
            weather_source=file_hash(resolve_path(cfg['environment']['weather_path'])))
        if any(prov['input_hashes'][key] != sha for key, sha in actual_sources.items()):
            raise ValueError('Approved E1-3 original datafile hash mismatch')
    identity = dict(version=legacy.CARS_SCHEMA, archive_kind='comparison',
        reference_provenance_sha256=content_hash(prov), source_sha256=prov['source_sha256'],
        sources=prov['sources'], git=prov['git'], python=prov['python'], dependencies=prov['dependencies'],
        input_hashes=prov['input_hashes'], threshold_g_per_kwh=parameters['threshold_g_per_kwh'],
        common_end=common, scientific_hashes={n:prov['output_hashes'][n] for n in sorted(CONTROL_FILES)},
        complete_output_hashes=prov['output_hashes'], frozen_import=True, workload_regenerated=False)
    return dict(cfg=cfg, workload=workload, environment=environment, parameters=parameters,
                selection=selection, model_path=path/'dc_config.json', identity=identity)


def _model_source_version(identity):
    return dict(experiment_schema=ABLATION_SCHEMA,
        reference_research_head=identity['git']['head'], reference_publication_commit=PUBLICATION_COMMIT,
        reference_source_sha256=identity['source_sha256'],
        reused_original_sources=identity['sources'],
        reference_provenance_sha256=identity['reference_provenance_sha256'])


def _write_policy(folder, result, own, executor, config, workload, endpoint, diagnostics):
    folder.mkdir()
    _write_result(folder, result)
    write_json(folder/'resolved_config.json', config)
    phases = dict(fixed=phase_summary(result, workload.end),
        own=phase_summary(own, pd.Timestamp(own.summary['observation_end'])),
        common=phase_summary(result, endpoint))
    write_json(folder/'phases.json', phases)
    write_json(folder/'fixed_ledger.json', phase_rows(result, workload.end))
    _jsonl(folder/'diagnostics.jsonl', diagnostics)
    write_json(folder/'run_status.json', dict(state='complete'))
    return phases


def _paired_shifts(results, environment):
    ci = {s.time:s.ci_g_per_kwh for s in environment.samples}
    baseline = {t['task_id']:t for t in results[CONTROL_NAMES[0]].tasks}
    rows = []
    for name, result in results.items():
        for row in result.tasks:
            base = baseline[row['task_id']]
            bs = pd.Timestamp(base['start_time']) if base['start_time'] else None
            ps = pd.Timestamp(row['start_time']) if row['start_time'] else None
            rows.append(dict(policy=name, task_id=row['task_id'], reference_policy=CONTROL_NAMES[0],
                reference_start=base['start_time'], policy_start=row['start_time'],
                reference_start_ci=ci.get(bs), policy_start_ci=ci.get(ps),
                shift_minutes=(ps-bs).total_seconds()/60 if bs is not None and ps is not None else None,
                state=row['state'], deadline_outcome=row['deadline_outcome']))
    return rows


def _write_science(out, results, own, executors, configs, policies, workload, environment,
                   parameters, selection, common):
    phases, comparison, series, task_rows, aggregate = {}, [], [], [], {}
    reference = {t['task_id']:t for t in results[CONTROL_NAMES[0]].tasks}
    for name, result in results.items():
        ex, policy = executors[name], policies[name]
        details = getattr(policy, 'diagnostics', [])
        phases[name] = _write_policy(out/name, result, own[name], ex, configs[name], workload, common, details)
        for phase, summary in phases[name].items():
            comparison.append(dict(policy=name, phase=phase,
                drained_at=ex.drained_at.isoformat() if ex.drained_at else None,
                drain_hours=(ex.drained_at-workload.end).total_seconds()/3600 if ex.drained_at else None,
                termination='all_completed' if ex.drained_at else 'safety_cap_incomplete', **summary))
        cumulative = 0.0
        for step in result.steps:
            cumulative = math.fsum((cumulative, step['carbon_kg']))
            series.append(dict(policy=name, **step, cumulative_carbon_kg=cumulative,
                phase='arrival' if pd.Timestamp(step['interval_start']) < workload.end else 'drain'))
        candidates = {t['task_id']:t for t in result.tasks}
        events = {}
        for row in result.policy_events:
            events.setdefault(row['task_id'], Counter())[row['reason']] += 1
        missed = lambda row: row['deadline_outcome'] in ('missed', 'missed_unfinished')
        avoided = sorted(k for k in reference if missed(reference[k]) and candidates[k]['deadline_outcome'] == 'met')
        extra = sorted(k for k in reference if reference[k]['deadline_outcome'] == 'met' and missed(candidates[k]))
        aggregate[name] = dict(counters=getattr(policy, 'counters', {}), diagnostics_count=len(details),
            reference_policy=CONTROL_NAMES[0], avoided_misses_vs_edf_carbon=len(avoided),
            extra_misses_vs_edf_carbon=len(extra), avoided_task_ids=avoided, extra_task_ids=extra,
            reason_counts=dict(sorted(Counter(r['reason'] for r in result.policy_events).items())),
            attribution='paired association; not single-event causation')
        for row in result.policy_tasks:
            key = row['task_id']
            task_rows.append(dict(policy=name, **row, deadline_outcome=candidates[key]['deadline_outcome'],
                reference_policy=CONTROL_NAMES[0], reference_deadline_outcome=reference[key]['deadline_outcome'],
                reason_counts=dict(sorted(events.get(key, {}).items()))))
    completed = all(phases[p]['common']['completed'] == len(workload.tasks) for p in results)
    resources = {r:[phases[p]['common'][r+'_hours'] for p in results] for r in ('cores', 'gpus', 'mem')}
    conserved = all(all(math.isclose(v, values[0], rel_tol=1e-12, abs_tol=1e-8) for v in values)
                    for values in resources.values())
    if completed and not conserved:
        raise RuntimeError('Completed cohort resource-hours differ')
    keys = tuple(workload.tasks[0].to_dict()) if workload.tasks else ()
    expected_features = [{k:t[k] for k in keys} for t in results[CONTROL_NAMES[0]].tasks]
    if any([{k:t[k] for k in keys} for t in result.tasks] != expected_features for result in results.values()):
        raise RuntimeError('Policy altered frozen task features')
    write_json(out/'fairness.json', dict(version=ABLATION_SCHEMA, identical_frozen_inputs=True,
        identical_shared_config=True, all_tasks_completed=completed, equal_resource_hours=conserved,
        equal_workload_carbon_comparison_valid=completed and conserved, resource_hours=resources,
        arrived=len(workload.tasks), arrival_end=workload.end.isoformat(), common_end=common.isoformat(),
        safety_end=environment.end.isoformat(), threshold_g_per_kwh=parameters['threshold_g_per_kwh'],
        threshold_source='selection calibration' if selection else 'explicit fixture literal',
        deadline_denominator='same finite arrived cohort; censored tasks retained',
        idle_included=True, policy_future_ci_access=False))
    write_csv(out/'comparison.csv', comparison)
    write_csv(out/'timeseries.csv', series)
    write_csv(out/'paired_task_shifts.csv', _paired_shifts(results, environment))
    _jsonl(out/'task_diagnostics.jsonl', task_rows)
    write_json(out/'diagnostics.json', aggregate)
    return phases


def _verify_identity(identity, cfg, workload, environment, model_path, parameters, selection):
    required = {'version', 'archive_kind', 'reference_provenance_sha256', 'source_sha256', 'sources',
        'git', 'python', 'dependencies', 'input_hashes', 'threshold_g_per_kwh', 'common_end',
        'scientific_hashes', 'complete_output_hashes', 'frozen_import', 'workload_regenerated'}
    if (set(identity) != required or identity['version'] != legacy.CARS_SCHEMA
        or identity['archive_kind'] != 'comparison' or identity['frozen_import'] is not True
        or identity['workload_regenerated'] is not False
        or set(identity['scientific_hashes']) != CONTROL_FILES
        or set(identity['complete_output_hashes']) != legacy.LEVEL_FILES
        or identity['scientific_hashes'] != {n:identity['complete_output_hashes'][n] for n in sorted(CONTROL_FILES)}
        or identity['input_hashes'] != _input_hashes(cfg, workload, environment, model_path, parameters, selection)
        or identity['threshold_g_per_kwh'] != parameters['threshold_g_per_kwh']):
        raise ValueError('E1-3 frozen reference identity differs')
    # The E1-4A snapshot retains every original source as well as the appended
    # modules. This makes the cached cross-version identity verifiable offline.
    _verify_sources(None, identity, cross_version=True, snapshot=False)


def _execute(out, cfg, workload, environment, model_path, parameters, threshold=None,
             selection=None, reference=None, identity=None, ablation_config=None):
    clock = time.perf_counter()
    parameters = legacy.validate_parameters(parameters)
    ablation_config = validate_ablation_config(ablation_config)
    cfg = validate_temporal_config(cfg, workload, environment)
    if not isinstance(environment, TemporalEnvironment):
        raise ValueError('E1-4A requires causal TemporalEnvironment')
    if reference is not None:
        imported = _verify_import_level(reference)
        identity = imported['identity']
        if (cfg != imported['cfg'] or workload.sha256 != imported['workload'].sha256
            or environment.sha256 != imported['environment'].sha256
            or parameters != imported['parameters'] or selection != imported['selection']):
            raise ValueError('E1-4A import changed frozen E1-3 configuration/input/parameters')
    if identity is None:
        raise ValueError('E1-4A requires a verified E1-3 reference identity')
    _verify_identity(identity, cfg, workload, environment, model_path, parameters, selection)
    if threshold is not None and threshold != parameters['threshold_g_per_kwh']:
        raise ValueError('Threshold must preserve exact E1-3 binary value')
    hashes = experiment_sources()
    _snapshot_sources(out, hashes)
    for filename, value in (('resolved_config.json', cfg), ('parameters.json', parameters),
        ('ablation_config.json', ablation_config), ('selection.json', selection), ('reference.json', identity)):
        write_json(out/filename, value)
    write_json(out/'design.json', dict(version=ABLATION_SCHEMA, archive_kind='comparison',
        policies=list(ABLATION_POLICY_NAMES), arrival_end=workload.end.isoformat(),
        safety_end=environment.end.isoformat(), threshold_g_per_kwh=parameters['threshold_g_per_kwh'],
        ci_basis=CI_BASIS, common_end_rule='maximum five observed grid drain boundaries, or safety cap if any incomplete',
        legacy_control_end=identity['common_end'], legacy_control_regression='byte-identical policy outputs at original E1-3 common end',
        idle_rule='same persistent physical model, zero CPU/GPU/MEM, real corresponding weather/CI',
        frozen_e1_3_parameters=True, future_ci=False))
    workload.save(out/'workload.json')
    environment.save(out/'environment.json')
    shutil.copyfile(model_path, out/'dc_config.json')
    dc = cfg['datacenter']
    capacity = dc['total_cores'], dc['total_gpus'], dc['total_mem']
    executors, own, policies, configs, timing = {}, {}, {}, {}, {}
    for name in ABLATION_POLICY_NAMES:
        configs[name] = _old_policy_config(cfg, parameters, name)
        started = time.perf_counter()
        policy = TimingPolicy(make_ablation_policy(name, parameters, workload.start, capacity))
        ex = LocalExecutor(workload, environment, capacity, UpstreamPowerModel(dc, out/'dc_config.json'),
                           policy=policy, drain=True)
        policies[name], executors[name] = policy, ex
        own[name] = ex.run(stop_at_drain=True)
        timing[name] = dict(own_run_seconds=time.perf_counter()-started, idle_extension_seconds=0.0)
    common = max(ex.observed_end for ex in executors.values())
    control_end = pd.Timestamp(identity['common_end'])
    if not workload.end <= control_end <= environment.end:
        raise ValueError('E1-3 control cutoff outside frozen coverage')
    results, control_results = {}, {}
    # Capture immutable results in increasing endpoint order: extending a physical
    # model cannot undo its state, and the old seven-policy end can exceed ours.
    for name, ex in executors.items():
        endpoints = sorted({common, control_end} if name in CONTROL_NAMES else {common})
        if name in CONTROL_NAMES and ex.observed_end > control_end:
            raise RuntimeError('Control own drain changed from E1-3')
        for endpoint in endpoints:
            started = time.perf_counter()
            result = ex.extend_idle(endpoint) if ex.drained_at is not None else own[name]
            timing[name]['idle_extension_seconds'] += time.perf_counter()-started
            if endpoint == common:
                results[name] = result
            if name in CONTROL_NAMES and endpoint == control_end:
                control_results[name] = result
        timing[name]['counters'] = getattr(policies[name], 'counters', {})
        timing[name]['scheduler'] = policies[name].timings
    controls = out/'legacy-controls'
    controls.mkdir()
    for name in CONTROL_NAMES:
        _write_policy(controls/name, control_results[name], own[name], executors[name], configs[name],
                      workload, control_end, getattr(policies[name], 'diagnostics', []))
    differences = [n for n, sha in identity['scientific_hashes'].items() if file_hash(controls/n) != sha]
    if differences:
        raise RuntimeError('E1-3 control byte regression mismatch: '+str(differences))
    write_json(out/'baseline_regression.json', dict(matched=True, files_checked=len(CONTROL_FILES),
        controls=list(CONTROL_NAMES), reference_common_end=identity['common_end'],
        historical_provenance_untouched=True, rule='byte-identical scientific and diagnostics files; no metadata exclusions needed'))
    phases = _write_science(out, results, own, executors, configs, policies, workload, environment,
                            parameters, selection, common)
    write_json(out/'runtime.json', dict(excluded_from_scientific_hashes=True,
        total_seconds=time.perf_counter()-clock, policies=timing))
    if experiment_sources() != hashes:
        raise RuntimeError('Sources changed during E1-4A execution')
    if _scientific_inventory(out) != LEVEL_FILES:
        raise RuntimeError('E1-4A scientific output inventory differs from required manifest')
    inputs = _input_hashes(cfg, workload, environment, out/'dc_config.json', parameters, selection)
    inputs.update(ablation_config=content_hash(ablation_config), reference=content_hash(identity))
    write_json(out/'provenance.json', dict(version=ABLATION_SCHEMA, archive_kind='comparison',
        git=dict(head=_git('rev-parse', 'HEAD'), branch=_git('branch', '--show-current'), status=_git('status', '--porcelain')),
        command=sys.argv, python=sys.version, dependencies=dependencies(), sources=hashes,
        source_sha256=content_hash(hashes), model_source_version=_model_source_version(identity),
        input_hashes=inputs, excluded_outputs=_EXCLUDED,
        runtime_hashes={'runtime.json':file_hash(out/'runtime.json')},
        output_hashes={n:file_hash(out/n) for n in sorted(LEVEL_FILES)}))
    return phases


def run_ablation_comparison(cfg, workload, environment, model_path, output_dir, parameters,
                            threshold=None, selection=None, reference=None, ablation_config=None):
    return _fresh_run(output_dir, lambda out:_execute(out, cfg, workload, environment, model_path,
        parameters, threshold, selection, reference, ablation_config=ablation_config))


def _verify_level(path):
    path = Path(path)
    prov = read_json(path/'provenance.json')
    if (read_json(path/'run_status.json')['state'] != 'complete'
        or set(prov) != _OLD_LEVEL_KEYS | {'model_source_version'}
        or prov.get('version') != ABLATION_SCHEMA or prov.get('archive_kind') != 'comparison'):
        raise ValueError('E1-4A comparison status/schema/manifest mismatch')
    _verify_sources(path, prov)
    _verify_hash_manifest(path, prov, LEVEL_FILES, 'E1-4A')
    cfg, parameters, selection, config, identity = (read_json(path/n) for n in
        ('resolved_config.json', 'parameters.json', 'selection.json', 'ablation_config.json', 'reference.json'))
    legacy.validate_parameters(parameters)
    validate_ablation_config(config)
    w, e = FrozenWorkload.load(path/'workload.json'), TemporalEnvironment.load(path/'environment.json')
    if validate_temporal_config(cfg, w, e) != cfg:
        raise ValueError('E1-4A frozen configuration mismatch')
    _verify_identity(identity, cfg, w, e, path/'dc_config.json', parameters, selection)
    inputs = _input_hashes(cfg, w, e, path/'dc_config.json', parameters, selection)
    inputs.update(ablation_config=content_hash(config), reference=content_hash(identity))
    if prov['input_hashes'] != inputs or prov['model_source_version'] != _model_source_version(identity):
        raise ValueError('E1-4A complete input/model source version mismatch')
    return dict(cfg=cfg, workload=w, environment=e, parameters=parameters, selection=selection,
                identity=identity, model_path=path/'dc_config.json', ablation_config=config, provenance=prov)


def replay_comparison(previous_dir, output_dir):
    def action(out):
        item = _verify_level(previous_dir)
        prov = item.pop('provenance')
        result = _execute(out, **item)
        for name, sha in prov['output_hashes'].items():
            if file_hash(out/name) != sha:
                raise RuntimeError('E1-4A replay numerical mismatch: '+name)
        return result
    return _fresh_run(output_dir, action)


def _verify_import_campaign(path, factors, require_approved):
    path = Path(path)
    prov, config = read_json(path/'provenance.json'), read_json(path/'config.json')
    if (read_json(path/'run_status.json')['state'] != 'complete'
        or set(prov) != _OLD_CAMPAIGN_KEYS or prov.get('version') != legacy.CARS_SCHEMA
        or prov.get('archive_kind') != 'campaign'):
        raise ValueError('E1-3 campaign status/schema/manifest mismatch')
    _verify_sources(path, prov, cross_version=True, snapshot=False)
    # Campaign scientific inventory includes child scientific files, which have
    # their own mandatory sealed manifests. Verify the root's exact file set here.
    root_files = {p.name for p in path.iterdir() if p.is_file() and p.name not in
                  ('runtime.json', 'run_status.json', 'provenance.json')}
    if set(prov['output_hashes']) != legacy.CAMPAIGN_FILES or root_files != legacy.CAMPAIGN_FILES:
        raise ValueError('E1-3 campaign scientific manifest incomplete')
    if prov.get('excluded_outputs') != _EXCLUDED or prov.get('runtime_hashes') != {'runtime.json':file_hash(path/'runtime.json')}:
        raise ValueError('E1-3 campaign runtime/excluded output integrity mismatch')
    for name, sha in prov['output_hashes'].items():
        if file_hash(path/name) != sha:
            raise ValueError('E1-3 campaign output mismatch: '+name)
    if (set(config) != {'version', 'archive_kind', 'factors', 'parameters', 'reference_parent_sha256', 'frozen_import', 'workload_regenerated'}
        or config['version'] != legacy.CARS_SCHEMA or config['archive_kind'] != 'campaign'
        or config['frozen_import'] is not True or config['workload_regenerated'] is not False
        or config['factors'] != list(DECLARED_FACTORS)
        or config['parameters'] != read_json(path/'parameters.json')):
        raise ValueError('E1-3 fixed campaign declaration mismatch')
    legacy.validate_parameters(config['parameters'])
    identities = read_json(path/'input_identities.json')
    if prov['input_hashes'] != dict(parameters=content_hash(config['parameters']), identities=content_hash(identities)):
        raise ValueError('E1-3 campaign complete input manifest mismatch')
    indexed = {i['factor']:i for i in identities}
    children = {factor_folder(f) for f in config['factors']}
    if len(indexed) != len(identities) or set(indexed) != set(config['factors']) or set(prov['child_output_manifests']) != children:
        raise ValueError('E1-3 campaign children/identities manifest incomplete')
    _verify_campaign_inventory(path, legacy.CAMPAIGN_FILES,
        {name:legacy.LEVEL_FILES for name in children})
    descriptors = []
    for factor in config['factors']:
        name, child = factor_folder(factor), path/factor_folder(factor)
        child_prov = read_json(child/'provenance.json')
        if child_prov != prov['child_output_manifests'][name] or child_prov['sources'] != prov['sources']:
            raise ValueError('E1-3 campaign child complete manifest mismatch')
        item = _verify_import_level(child, require_approved=require_approved)
        identity = indexed[factor]
        if (set(identity) != {'factor', 'realized_factor', 'workload_sha256', 'environment_sha256',
            'dc_config_sha256', 'common_end', 'tasks', 'reference'}
            or identity['workload_sha256'] != item['workload'].sha256
            or identity['environment_sha256'] != item['environment'].sha256
            or identity['dc_config_sha256'] != file_hash(item['model_path'])
            or identity['common_end'] != item['identity']['common_end']
            or identity['tasks'] != len(item['workload'].tasks)
            or identity['reference'] != read_json(child/'reference.json')):
            raise ValueError('E1-3 campaign child frozen identity mismatch')
        expected = copy.deepcopy(config['parameters'])
        expected['threshold_g_per_kwh'] = item['parameters']['threshold_g_per_kwh']
        if item['parameters'] != expected:
            raise ValueError('E1-3 child fixed parameters differ from parent')
        if require_approved and (len(item['workload'].tasks) != 9943+math.floor(9943*(factor-1))
            or not math.isclose(identity['realized_factor'], len(item['workload'].tasks)/9943, rel_tol=0, abs_tol=1e-15)):
            raise ValueError('Approved E1-3 frozen replication identity mismatch')
        if factor in factors:
            item.update(factor=float(factor), realized_factor=identity['realized_factor'])
            descriptors.append(item)
    return descriptors, config, prov


def _execute_campaign(out, descriptors, config, verify_replay=None):
    clock = time.perf_counter()
    hashes = experiment_sources()
    _snapshot_sources(out, hashes)
    for name, value in (('config.json', config), ('parameters.json', config['parameters']),
                        ('ablation_config.json', config['ablation_config'])):
        write_json(out/name, value)
    rows, identities, regression, diagnostics, timing = [], [], {}, {}, {}
    for item in descriptors:
        factor = item['factor']
        name, child = factor_folder(factor), out/factor_folder(factor)
        print(f'E1-4A factor {factor}: {len(item["workload"].tasks)} frozen tasks', flush=True)
        arguments = {k:v for k,v in item.items() if k not in ('factor', 'realized_factor', 'provenance')}
        arguments['ablation_config'] = config['ablation_config']
        started = time.perf_counter()
        phases = _fresh_run(child, lambda path:_execute(path, **arguments))
        timing[name] = dict(total_seconds=time.perf_counter()-started, policies=read_json(child/'runtime.json')['policies'])
        fair, w = read_json(child/'fairness.json'), item['workload']
        for policy in ABLATION_POLICY_NAMES:
            drained = read_json(child/policy/'summary.json')['drained_at']
            for phase, summary in phases[policy].items():
                rows.append(dict(factor=factor, realized_factor=item['realized_factor'], policy=policy, phase=phase,
                    drain_hours=(pd.Timestamp(drained)-w.end).total_seconds()/3600 if drained else None,
                    termination='all_completed' if drained else 'safety_cap_incomplete',
                    equal_workload_carbon_comparison_valid=fair['equal_workload_carbon_comparison_valid'], **summary))
        identities.append(dict(factor=factor, realized_factor=item['realized_factor'], workload_sha256=w.sha256,
            environment_sha256=item['environment'].sha256, dc_config_sha256=file_hash(item['model_path']),
            common_end=fair['common_end'], tasks=len(w.tasks), reference=item['identity']))
        regression[name] = read_json(child/'baseline_regression.json')
        diagnostics[name] = read_json(child/'diagnostics.json')
    write_csv(out/'comparison.csv', rows)
    for name, value in (('input_identities.json', identities), ('baseline_regression.json', regression),
                        ('diagnostics.json', diagnostics)):
        write_json(out/name, value)
    write_json(out/'runtime.json', dict(excluded_from_scientific_hashes=True,
        total_seconds=time.perf_counter()-clock, levels=timing))
    if experiment_sources() != hashes:
        raise RuntimeError('Sources changed during E1-4A campaign')
    _verify_campaign_inventory(out, CAMPAIGN_FILES,
        {factor_folder(f):LEVEL_FILES for f in config['factors']})
    model_sources = {factor_folder(i['factor']):_model_source_version(i['identity']) for i in descriptors}
    write_json(out/'provenance.json', dict(version=ABLATION_SCHEMA, archive_kind='campaign',
        git=dict(head=_git('rev-parse', 'HEAD'), branch=_git('branch', '--show-current'), status=_git('status', '--porcelain')),
        command=sys.argv, python=sys.version, dependencies=dependencies(), sources=hashes, source_sha256=content_hash(hashes),
        model_source_version=model_sources,
        input_hashes=dict(parameters=content_hash(config['parameters']), identities=content_hash(identities),
                          ablation_config=content_hash(config['ablation_config']), config=content_hash(config)),
        excluded_outputs=_EXCLUDED, runtime_hashes={'runtime.json':file_hash(out/'runtime.json')},
        output_hashes={n:file_hash(out/n) for n in sorted(CAMPAIGN_FILES)},
        child_output_manifests={factor_folder(f):read_json(out/factor_folder(f)/'provenance.json') for f in config['factors']}))
    if verify_replay is not None:
        verify_replay(out)
    return rows


def run_reference_levels(reference_root, output_dir, factors=None, require_approved=False, config=None):
    def action(out):
        declaration = validate_ablation_config(config)
        selected = list(REQUIRED_FACTORS if factors is None else factors)
        if (not selected or selected != sorted(set(selected)) or any(isinstance(f, bool) or f not in DECLARED_FACTORS for f in selected)
            or require_approved and not set(REQUIRED_FACTORS) <= set(selected)):
            raise ValueError('E1-4A factors must be increasing declared levels, including both required levels')
        descriptors, old_config, old_prov = _verify_import_campaign(reference_root, selected, require_approved)
        campaign = dict(version=ABLATION_SCHEMA, archive_kind='campaign', factors=selected,
            parameters=old_config['parameters'], ablation_config=declaration,
            reference_parent_sha256=content_hash(old_prov), frozen_import=True, workload_regenerated=False)
        return _execute_campaign(out, descriptors, campaign)
    return _fresh_run(output_dir, action)


def replay_levels(previous_dir, output_dir):
    previous = Path(previous_dir)
    def action(out):
        prov = read_json(previous/'provenance.json')
        if (read_json(previous/'run_status.json')['state'] != 'complete'
            or set(prov) != _OLD_CAMPAIGN_KEYS | {'git', 'model_source_version'}
            or prov.get('version') != ABLATION_SCHEMA or prov.get('archive_kind') != 'campaign'):
            raise ValueError('E1-4A campaign status/schema/manifest mismatch')
        _verify_sources(previous, prov)
        config = read_json(previous/'config.json')
        children = {factor_folder(f) for f in config['factors']}
        if set(prov['output_hashes']) != CAMPAIGN_FILES or set(prov['child_output_manifests']) != children:
            raise ValueError('E1-4A campaign scientific/child manifest incomplete')
        _verify_campaign_inventory(previous, CAMPAIGN_FILES,
            {name:LEVEL_FILES for name in children})
        if prov.get('excluded_outputs') != _EXCLUDED or prov.get('runtime_hashes') != {'runtime.json':file_hash(previous/'runtime.json')}:
            raise ValueError('E1-4A campaign runtime/excluded output integrity mismatch')
        root_files = {p.name for p in previous.iterdir() if p.is_file() and p.name not in ('runtime.json', 'run_status.json', 'provenance.json')}
        if root_files != CAMPAIGN_FILES:
            raise ValueError('E1-4A campaign scientific inventory differs')
        for name, sha in prov['output_hashes'].items():
            if file_hash(previous/name) != sha:
                raise ValueError('E1-4A campaign output mismatch: '+name)
        validate_ablation_config(config['ablation_config'])
        legacy.validate_parameters(config['parameters'])
        identities = read_json(previous/'input_identities.json')
        expected = dict(parameters=content_hash(config['parameters']), identities=content_hash(identities),
                        ablation_config=content_hash(config['ablation_config']), config=content_hash(config))
        if (prov['input_hashes'] != expected or read_json(previous/'parameters.json') != config['parameters']
            or read_json(previous/'ablation_config.json') != config['ablation_config']):
            raise ValueError('E1-4A campaign complete input manifest mismatch')
        indexed = {i['factor']:i for i in identities}
        if len(indexed) != len(identities) or set(indexed) != set(config['factors']):
            raise ValueError('E1-4A campaign identities incomplete')
        descriptors = []
        for factor in config['factors']:
            name, child = factor_folder(factor), previous/factor_folder(factor)
            if read_json(child/'provenance.json') != prov['child_output_manifests'][name]:
                raise ValueError('E1-4A campaign child manifest changed')
            item = _verify_level(child)
            identity = indexed[factor]
            params = copy.deepcopy(config['parameters'])
            params['threshold_g_per_kwh'] = item['parameters']['threshold_g_per_kwh']
            if (item['parameters'] != params or item['ablation_config'] != config['ablation_config']
                or identity['workload_sha256'] != item['workload'].sha256
                or identity['environment_sha256'] != item['environment'].sha256
                or identity['dc_config_sha256'] != file_hash(item['model_path'])
                or identity['tasks'] != len(item['workload'].tasks) or identity['reference'] != item['identity']
                or identity['common_end'] != read_json(child/'fairness.json')['common_end']):
                raise ValueError('E1-4A campaign child frozen input identity mismatch')
            item.update(factor=factor, realized_factor=identity['realized_factor'])
            descriptors.append(item)
        if prov['model_source_version'] != {factor_folder(i['factor']):_model_source_version(i['identity']) for i in descriptors}:
            raise ValueError('E1-4A campaign model source version mismatch')
        def verify(candidate):
            for name, sha in prov['output_hashes'].items():
                if file_hash(candidate/name) != sha:
                    raise RuntimeError('E1-4A campaign replay mismatch: '+name)
            for name, child in prov['child_output_manifests'].items():
                for filename, sha in child['output_hashes'].items():
                    if file_hash(candidate/name/filename) != sha:
                        raise RuntimeError('E1-4A child replay mismatch: '+name+'/'+filename)
        return _execute_campaign(out, descriptors, config, verify)
    return _fresh_run(output_dir, action)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--factors', nargs='+', type=float)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay', type=Path)
    args = parser.parse_args()
    if args.replay:
        runner = replay_levels if (args.replay/'config.json').is_file() else replay_comparison
        runner(args.replay, args.output)
    else:
        if args.reference_root is None:
            parser.error('--reference-root is required for sealed E1-3 import')
        run_reference_levels(args.reference_root, args.output, args.factors, require_approved=True, config=args.config)
    print(f'Artifacts: {args.output.resolve()}')


if __name__ == '__main__':
    main()
