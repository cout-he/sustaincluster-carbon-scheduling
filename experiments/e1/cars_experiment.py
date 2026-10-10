"""E1-3 seven-policy frozen reference cohorts, historical regression and strict replay."""
import argparse
import copy
import json
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from .cars import CARS_POLICY_NAMES, CARS_SCHEMA, make_cars_policy
from .config import ROOT
from .contention import factor_folder, read_json, verify_archive
from .contracts import canonical_bytes, content_hash, file_hash, write_json
from .drain_metrics import phase_rows, phase_summary
from .engine import LocalExecutor
from .policies import POLICY_NAMES
from .power import UpstreamPowerModel
from .run import _fresh_run, _git, _write_result, dependencies
from .selection import CI_BASIS, TemporalEnvironment
from .temporal import VERSION as TEMPORAL_VERSION
from .temporal import experiment_sources, paired_shifts, validate_temporal_config, write_csv
from .workload import FrozenWorkload

POLICY_FILES = ('ledger.jsonl', 'events.jsonl', 'intervals.csv', 'summary.json',
                'policy_ledger.jsonl', 'policy_events.jsonl', 'policy_intervals.jsonl',
                'policy_summary.json', 'resolved_config.json', 'phases.json', 'fixed_ledger.json')
LEGACY_FILES = {'comparison.csv', 'timeseries.csv', 'paired_task_shifts.csv', 'fairness.json'}
LEGACY_FILES.update(f'{p}/{n}' for p in POLICY_NAMES for n in POLICY_FILES)
LEVEL_FILES = {'resolved_config.json', 'parameters.json', 'design.json', 'selection.json',
               'workload.json', 'environment.json', 'dc_config.json', 'reference.json',
               'baseline_regression.json', 'fairness.json', 'comparison.csv', 'timeseries.csv',
               'paired_task_shifts.csv', 'task_diagnostics.jsonl', 'diagnostics.json'}
LEVEL_FILES.update(f'{p}/{n}' for p in CARS_POLICY_NAMES for n in POLICY_FILES)
LEVEL_FILES.update(f'{p}/diagnostics.jsonl' for p in CARS_POLICY_NAMES)
LEVEL_FILES.update('legacy-three/'+n for n in LEGACY_FILES)
CAMPAIGN_FILES = {'config.json', 'parameters.json', 'comparison.csv', 'input_identities.json',
                  'baseline_regression.json', 'diagnostics.json'}
FIXED_PARAMETERS = dict(version='e1-3-cars-v1', threshold_g_per_kwh=137.65,
    history_minutes=120, risk_breakpoints=[0.5,1.0], risk_buffer_steps=[0,1,2],
    cold_start_buffer_steps=2, horizon_steps=8, max_pending=256, factors=[1.0,1.25,1.5,2.0],
    stress_seed=20261010, sla_multiplier=1.5, arrival_start='2023-07-01T05:00:00+00:00',
    arrival_end='2023-07-03T05:00:00+00:00',
    reference_source_commit='aae1a1337ec60706a6fcb6346d8ba8b607e5e929',
    future_ci=False, shadow_unknown='execute', step_minutes=15)
APPROVED_ARCHIVE_HEAD='3f6ea95cb835b89ba50b0ef76d66d14b8037643e'
APPROVED_ENV_HASH='cef88789e3a2db24e1d504a9d78ab232cc525d98747b93ccc6d6239352992825'
APPROVED_DC_HASH='bbe439176605512523a3a0ecd2bd22f18fdb9abf6ae6e782217157beedbf705b'


def validate_approved_reference(cfg,workload,identity,replication,parameters,archive_head):
    # The historical experiment ran in the uncommitted research checkout; the
    # independent publication commit is a different provenance role, not its HEAD.
    expected_dc=dict(total_cores=50000,total_gpus=1000,total_mem=80000,
                     location='US-CAL-CISO',hvac_setpoint_c=22)
    if (workload.start.isoformat()!=parameters['arrival_start']
        or workload.end.isoformat()!=parameters['arrival_end']
        or workload.sla_multiplier!=parameters['sla_multiplier'] or workload.resource_multiplier!=5
        or replication['base_tasks']!=9943
        or any(cfg['datacenter'][k]!=v for k,v in expected_dc.items())
        or archive_head!=APPROVED_ARCHIVE_HEAD
        or identity['input_hashes']['environment_snapshot']!=APPROVED_ENV_HASH
        or identity['input_hashes']['dc_config']!=APPROVED_DC_HASH
        or identity['threshold_g_per_kwh']!=137.64999999999998):
        raise ValueError('Reference is not the approved E1-2B experiment design')


def validate_parameters(parameters):
    """Reject undeclared choices; retain the equivalent reference threshold's bits."""
    if not isinstance(parameters, dict) or set(parameters) != set(FIXED_PARAMETERS):
        raise ValueError('CARS fixed parameter manifest differs from declared design')
    for key, expected in FIXED_PARAMETERS.items():
        value = parameters[key]
        if key == 'threshold_g_per_kwh':
            if isinstance(value, bool) or not isinstance(value, (float,int)) or not math.isclose(value, expected, rel_tol=0, abs_tol=1e-12):
                raise ValueError('CARS fixed parameter differs: '+key)
        elif canonical_bytes(value) != canonical_bytes(expected):
            raise ValueError('CARS fixed parameter differs: '+key)
    return copy.deepcopy(parameters)


class _TimedPlanner:
    def __init__(self, planner, timings):
        self.planner, self.timings = planner, timings

    def compare(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return self.planner.compare(*args, **kwargs)
        finally:
            self.timings['planner_seconds'] += time.perf_counter()-started
            self.timings['planner_compare_calls'] += 1

    def __getattr__(self, name):
        return getattr(self.planner, name)


class TimingPolicy:
    """Time real scheduling methods without inventing optional hooks or changing decisions."""
    _HOOKS = {'on_step_start':'step_state', 'on_candidate_resources':'candidate_state',
              'on_task_started':'start_notification', 'on_task_deferred':'defer_notification'}

    def __init__(self, policy):
        self.policy = policy
        self.timings = dict(order_seconds=0.0, order_calls=0, decision_seconds=0.0, decide_calls=0,
            planner_seconds=0.0, planner_compare_calls=0,
            planner_seconds_is_subset_of_decision_seconds=True)
        for metric in self._HOOKS.values():
            self.timings[metric+'_seconds'] = 0.0
            self.timings[metric+'_calls'] = 0
        if hasattr(policy, 'planner'):
            policy.planner = _TimedPlanner(policy.planner, self.timings)

    def _call(self, method, seconds, calls, *args):
        started = time.perf_counter()
        try:
            return method(*args)
        finally:
            self.timings[seconds] += time.perf_counter()-started
            self.timings[calls] += 1

    def order(self, waiting):
        return self._call(self.policy.order, 'order_seconds', 'order_calls', waiting)

    def decide(self, spec, context):
        return self._call(self.policy.decide, 'decision_seconds', 'decide_calls', spec, context)

    def __getattr__(self, name):
        value = getattr(self.policy, name)  # AttributeError preserves absent old-policy hooks.
        if name in self._HOOKS:
            metric = self._HOOKS[name]
            return lambda *args:self._call(value, metric+'_seconds', metric+'_calls', *args)
        return value


def _jsonl(path, rows):
    with Path(path).open('wb') as stream:
        for row in rows:
            stream.write(canonical_bytes(row)+b'\n')


def _snapshot_sources(out, hashes):
    for name, sha in hashes.items():
        target = out/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
        if file_hash(target) != sha:
            raise RuntimeError('Source changed during snapshot')


def _input_hashes(cfg, workload, environment, model_path, parameters, selection):
    return dict(workload_snapshot=workload.sha256, environment_snapshot=environment.sha256,
                workload_source=workload.source_sha256, carbon_source=environment.carbon_source_sha256,
                weather_source=environment.weather_source_sha256, dc_config=file_hash(model_path),
                resolved_config=content_hash(cfg), parameters=content_hash(parameters),
                selection=content_hash(selection))


def _load_reference(path):
    """Import verified cross-version evidence without executing the old source."""
    path = Path(path)
    workload, environment = verify_archive(path)
    cfg, design, selection = (read_json(path/n) for n in
                              ('resolved_config.json', 'design.json', 'selection.json'))
    prov = read_json(path/'provenance.json')
    expected = dict(workload_snapshot=workload.sha256, environment_snapshot=environment.sha256,
                    workload_source=workload.source_sha256, carbon_source=environment.carbon_source_sha256,
                    weather_source=environment.weather_source_sha256, dc_config=file_hash(path/'dc_config.json'))
    if prov['input_hashes'] != expected:
        raise ValueError('Reference complete input manifest mismatch')
    validate_temporal_config(cfg, workload, environment)
    identity = dict(version=prov['version'], source_sha256=prov['source_sha256'],
                    input_hashes=expected, threshold_g_per_kwh=design['threshold_g_per_kwh'],
                    scientific_hashes={n:prov['output_hashes'][n] for n in sorted(LEGACY_FILES)})
    return cfg, workload, environment, selection, identity


def _write_science(out, results, own, executors, configs, workload, environment,
                   threshold, selection, version, names):
    """The legacy-three branch intentionally preserves temporal output formatting."""
    phases, comparison, series = {}, [], []
    common = max(executors[p].observed_end for p in names)
    for policy in names:
        result, ex = results[policy], executors[policy]
        folder = out/policy
        folder.mkdir()
        _write_result(folder, result)
        write_json(folder/'resolved_config.json', configs[policy])
        phases[policy] = dict(fixed=phase_summary(result, workload.end),
            own=phase_summary(own[policy], pd.Timestamp(own[policy].summary['observation_end'])),
            common=phase_summary(result, common))
        for name, summary in phases[policy].items():
            comparison.append(dict(policy=policy, phase=name,
                drained_at=ex.drained_at.isoformat() if ex.drained_at else None,
                drain_hours=(ex.drained_at-workload.end).total_seconds()/3600 if ex.drained_at else None,
                termination='all_completed' if ex.drained_at else 'safety_cap_incomplete', **summary))
        write_json(folder/'phases.json', phases[policy])
        write_json(folder/'fixed_ledger.json', phase_rows(result, workload.end))
        write_json(folder/'run_status.json', dict(state='complete'))
        cumulative = 0.0
        for step in result.steps:
            cumulative = math.fsum((cumulative, step['carbon_kg']))
            series.append(dict(policy=policy, **step, cumulative_carbon_kg=cumulative,
                phase='arrival' if pd.Timestamp(step['interval_start']) < workload.end else 'drain'))
    completed = all(phases[p]['common']['completed'] == len(workload.tasks) for p in names)
    resource_hours = {r:[phases[p]['common'][r+'_hours'] for p in names] for r in ('cores','gpus','mem')}
    conserved = all(all(math.isclose(v, values[0], rel_tol=1e-12, abs_tol=1e-8) for v in values)
                    for values in resource_hours.values())
    if completed and not conserved:
        raise RuntimeError('Completed cohort resource-hours differ')
    keys = tuple(workload.tasks[0].to_dict()) if workload.tasks else ()
    features = [{k:t[k] for k in keys} for t in results['immediate_fifo'].tasks]
    for result in results.values():
        if [{k:t[k] for k in keys} for t in result.tasks] != features:
            raise RuntimeError('Policy altered frozen task features')
    write_json(out/'fairness.json', dict(version=version, identical_frozen_inputs=True,
        identical_shared_config=True, all_tasks_completed=completed, equal_resource_hours=conserved,
        equal_workload_carbon_comparison_valid=completed and conserved, resource_hours=resource_hours,
        arrived=len(workload.tasks), arrival_end=workload.end.isoformat(), common_end=common.isoformat(),
        safety_end=environment.end.isoformat(), threshold_g_per_kwh=threshold,
        threshold_source='selection calibration' if selection else 'explicit fixture literal',
        deadline_denominator='same finite arrived cohort; censored tasks retained',
        idle_included=True, policy_future_ci_access=False))
    write_csv(out/'comparison.csv', comparison)
    write_csv(out/'timeseries.csv', series)
    write_csv(out/'paired_task_shifts.csv', paired_shifts({p:results[p] for p in names}, environment))
    return phases


def _regression(legacy, identity):
    if identity is None:
        return None
    expected = identity['scientific_hashes']
    if set(expected) != LEGACY_FILES:
        raise ValueError('Reference legacy scientific manifest incomplete')
    differences = [n for n, sha in expected.items() if file_hash(legacy/n) != sha]
    if differences:
        raise RuntimeError('Historical three-policy numerical regression mismatch: '+str(differences))
    return dict(matched=True, files_checked=len(expected), historical_provenance_untouched=True,
                rule='byte-identical scientific results; new source/provenance not relabeled as old')


def _diagnostics(out, policies, results):
    task_rows, aggregate = [], {}
    fifo = {t['task_id']:t for t in results['immediate_fifo'].tasks}
    carbon = {t['task_id']:t for t in results['carbon_threshold_fifo'].tasks}
    missed = lambda row: row['deadline_outcome'] in ('missed', 'missed_unfinished')
    for name, policy in policies.items():
        details = getattr(policy, 'diagnostics', [])
        _jsonl(out/name/'diagnostics.jsonl', details)
        event_counts = {}
        for event in results[name].policy_events:
            event_counts.setdefault(event['task_id'], Counter())[event['reason']] += 1
        candidates = {t['task_id']:t for t in results[name].tasks}
        avoided = [k for k in fifo if missed(carbon[k]) and candidates[k]['deadline_outcome'] == 'met']
        extra = [k for k in fifo if fifo[k]['deadline_outcome'] == 'met' and missed(candidates[k])]
        aggregate[name] = dict(counters=getattr(policy, 'counters', {}),
            diagnostics_count=len(details), avoided_misses_vs_carbon=len(avoided),
            extra_misses_vs_fifo=len(extra), avoided_task_ids=sorted(avoided),
            extra_task_ids=sorted(extra), attribution='paired association; not single-event causation')
        for row in results[name].policy_tasks:
            task_id = row['task_id']
            task_rows.append(dict(policy=name, **row,
                deadline_outcome=candidates[task_id]['deadline_outcome'],
                fifo_deadline_outcome=fifo[task_id]['deadline_outcome'],
                carbon_deadline_outcome=carbon[task_id]['deadline_outcome'],
                reason_counts=dict(sorted(event_counts.get(task_id, {}).items()))))
    _jsonl(out/'task_diagnostics.jsonl', task_rows)
    write_json(out/'diagnostics.json', aggregate)


def _execute(out, cfg, workload, environment, model_path, parameters, threshold=None,
             selection=None, reference=None, reference_identity=None):
    clock = time.perf_counter()
    parameters = validate_parameters(parameters)
    cfg = validate_temporal_config(cfg, workload, environment)
    if not isinstance(environment, TemporalEnvironment):
        raise ValueError('CARS requires a causal TemporalEnvironment')
    if reference is not None:
        ref_cfg, ref_w, ref_e, ref_selection, reference_identity = _load_reference(reference)
        if cfg != ref_cfg or workload.sha256 != ref_w.sha256 or environment.sha256 != ref_e.sha256:
            raise ValueError('CARS import changed frozen reference configuration or inputs')
        if file_hash(model_path) != reference_identity['input_hashes']['dc_config'] or selection != ref_selection:
            raise ValueError('CARS import changed reference physics or selection')
    archived_threshold = reference_identity['threshold_g_per_kwh'] if reference_identity is not None else threshold
    if archived_threshold is None:
        archived_threshold = parameters['threshold_g_per_kwh']
    if threshold is not None and threshold != archived_threshold:
        raise ValueError('Threshold must preserve exact reference binary value')
    parameters['threshold_g_per_kwh'] = archived_threshold
    if parameters.get('version') != CARS_SCHEMA:
        raise ValueError('CARS parameter schema mismatch')
    hashes = experiment_sources()
    _snapshot_sources(out, hashes)
    write_json(out/'resolved_config.json', cfg)
    write_json(out/'parameters.json', parameters)
    write_json(out/'selection.json', selection)
    write_json(out/'reference.json', reference_identity)
    write_json(out/'design.json', dict(version=CARS_SCHEMA, archive_kind='comparison',
        policies=list(CARS_POLICY_NAMES), arrival_end=workload.end.isoformat(), safety_end=environment.end.isoformat(),
        threshold_g_per_kwh=archived_threshold, ci_basis=CI_BASIS,
        common_end_rule='maximum seven observed grid drain boundaries, or safety cap if any incomplete',
        legacy_common_end_rule='maximum original-three observed grid drain boundaries',
        idle_rule='same persistent physical model, zero CPU/GPU/MEM, real corresponding weather/CI'))
    workload.save(out/'workload.json')
    environment.save(out/'environment.json')
    shutil.copyfile(model_path, out/'dc_config.json')
    dc = cfg['datacenter']
    capacity = dc['total_cores'], dc['total_gpus'], dc['total_mem']
    executors, own, policies, configs, timings = {}, {}, {}, {}, {}
    for name in CARS_POLICY_NAMES:
        p = copy.deepcopy(cfg)
        p['policy'] = name
        p.pop('policy_params', None)
        if name == 'carbon_threshold_fifo':
            p['policy_params'] = {'threshold_g_per_kwh':archived_threshold}
        elif name not in POLICY_NAMES:
            p['policy_params'] = parameters
        configs[name] = p
        started = time.perf_counter()
        policy = TimingPolicy(make_cars_policy(name, parameters, workload.start, capacity))
        ex = LocalExecutor(workload, environment, capacity, UpstreamPowerModel(dc, out/'dc_config.json'),
                           policy=policy, drain=True)
        policies[name], executors[name] = policy, ex
        own[name] = ex.run(stop_at_drain=True)
        timings[name] = dict(own_run_seconds=time.perf_counter()-started)
    legacy_end = max(executors[p].observed_end for p in POLICY_NAMES)
    legacy_results = {}
    for name in POLICY_NAMES:
        ex = executors[name]
        legacy_results[name] = ex.extend_idle(legacy_end) if ex.drained_at is not None else own[name]
    legacy = out/'legacy-three'
    legacy.mkdir()
    _write_science(legacy, legacy_results, own, executors, configs, workload, environment,
                   archived_threshold, selection, TEMPORAL_VERSION, POLICY_NAMES)
    write_json(out/'baseline_regression.json', _regression(legacy, reference_identity))
    common = max(ex.observed_end for ex in executors.values())
    results = {}
    for name, ex in executors.items():
        started = time.perf_counter()
        results[name] = ex.extend_idle(common) if ex.drained_at is not None else own[name]
        timings[name]['idle_extension_seconds'] = time.perf_counter()-started
        timings[name]['counters'] = getattr(policies[name], 'counters', {})
        timings[name]['scheduler'] = policies[name].timings
    phases = _write_science(out, results, own, executors, configs, workload, environment,
                            archived_threshold, selection, CARS_SCHEMA, CARS_POLICY_NAMES)
    _diagnostics(out, policies, results)
    write_json(out/'runtime.json', dict(excluded_from_scientific_hashes=True,
        total_seconds=time.perf_counter()-clock, policies=timings))
    if experiment_sources() != hashes:
        raise RuntimeError('Sources changed during CARS execution')
    present = {str(p.relative_to(out)).replace('\\','/') for p in out.rglob('*') if p.is_file()
               and 'source' not in p.relative_to(out).parts and p.name not in
               ('run_status.json','provenance.json','runtime.json')}
    if present != LEVEL_FILES:
        raise RuntimeError('CARS scientific output inventory differs from required manifest')
    write_json(out/'provenance.json', dict(version=CARS_SCHEMA, archive_kind='comparison',
        git=dict(head=_git('rev-parse','HEAD'), branch=_git('branch','--show-current'), status=_git('status','--porcelain')),
        command=sys.argv, python=sys.version, dependencies=dependencies(), sources=hashes,
        source_sha256=content_hash(hashes),
        input_hashes=_input_hashes(cfg, workload, environment, out/'dc_config.json', parameters, selection),
        excluded_outputs=['runtime.json','run_status.json','provenance.json','source/**'],
        runtime_hashes={'runtime.json':file_hash(out/'runtime.json')},
        output_hashes={n:file_hash(out/n) for n in sorted(LEVEL_FILES)}))
    return phases


def run_cars_comparison(cfg, workload, environment, model_path, output_dir, parameters,
                        threshold=None, selection=None, reference=None):
    return _fresh_run(output_dir, lambda out:_execute(out, cfg, workload, environment,
        model_path, parameters, threshold, selection, reference))


def _verify_level(previous):
    previous = Path(previous)
    prov = read_json(previous/'provenance.json')
    if read_json(previous/'run_status.json')['state'] != 'complete':
        raise ValueError('CARS replay requires complete archive')
    if prov.get('version') != CARS_SCHEMA or prov.get('archive_kind') != 'comparison':
        raise ValueError('CARS replay schema/kind mismatch')
    if (prov['sources'] != experiment_sources() or content_hash(prov['sources']) != prov['source_sha256']
            or prov['python'] != sys.version or prov['dependencies'] != dependencies()):
        raise ValueError('CARS replay source/Python/dependency mismatch')
    if set(prov['output_hashes']) != LEVEL_FILES:
        raise ValueError('CARS replay scientific manifest incomplete')
    if prov.get('runtime_hashes')!={'runtime.json':file_hash(previous/'runtime.json')}:
        raise ValueError('CARS runtime integrity mismatch')
    for name, sha in prov['sources'].items():
        if file_hash(previous/'source'/name) != sha:
            raise ValueError('CARS archived source mismatch: '+name)
    for name, sha in prov['output_hashes'].items():
        if file_hash(previous/name) != sha:
            raise ValueError('CARS archived output mismatch: '+name)
    cfg, parameters, selection = (read_json(previous/n) for n in
                                  ('resolved_config.json','parameters.json','selection.json'))
    workload = FrozenWorkload.load(previous/'workload.json')
    environment = TemporalEnvironment.load(previous/'environment.json')
    if _input_hashes(cfg, workload, environment, previous/'dc_config.json', parameters, selection) != prov['input_hashes']:
        raise ValueError('CARS complete input manifest mismatch')
    identity = read_json(previous/'reference.json')
    if identity is not None and (identity['input_hashes']['workload_snapshot'] != workload.sha256
            or identity['input_hashes']['environment_snapshot'] != environment.sha256
            or identity['input_hashes']['dc_config'] != file_hash(previous/'dc_config.json')):
        raise ValueError('CARS reference identity mismatch')
    return cfg, workload, environment, parameters, selection, identity, prov


def replay_comparison(previous_dir, output_dir):
    previous = Path(previous_dir)
    def action(out):
        cfg, w, e, parameters, selection, identity, prov = _verify_level(previous)
        result = _execute(out, cfg, w, e, previous/'dc_config.json', parameters,
                          read_json(previous/'design.json')['threshold_g_per_kwh'], selection,
                          reference_identity=identity)
        for name, sha in prov['output_hashes'].items():
            if file_hash(out/name) != sha:
                raise RuntimeError('CARS replay numerical mismatch: '+name)
        return result
    return _fresh_run(output_dir, action)


def _execute_campaign(out, descriptors, config, verify_replay=None):
    clock = time.perf_counter()
    hashes = experiment_sources()
    write_json(out/'config.json', config)
    write_json(out/'parameters.json', config['parameters'])
    rows, identities, regression, diagnostics, timing = [], [], {}, {}, {}
    for item in descriptors:
        factor, cfg, w, e = item['factor'], item['cfg'], item['workload'], item['environment']
        name = factor_folder(factor)
        child = out/name
        print(f'CARS factor {factor}: {len(w.tasks)} frozen tasks', flush=True)
        started = time.perf_counter()
        phases = _fresh_run(child, lambda path:_execute(path, cfg, w, e, item['model_path'],
            config['parameters'], item['identity']['threshold_g_per_kwh'], item['selection'],
            reference_identity=item['identity']))
        timing[name] = dict(total_seconds=time.perf_counter()-started,
                           policies=read_json(child/'runtime.json')['policies'])
        fair = read_json(child/'fairness.json')
        for policy in CARS_POLICY_NAMES:
            drained = read_json(child/policy/'summary.json')['drained_at']
            for phase, summary in phases[policy].items():
                rows.append(dict(factor=float(factor), realized_factor=item['realized_factor'],
                    policy=policy, phase=phase,
                    drain_hours=(pd.Timestamp(drained)-w.end).total_seconds()/3600 if drained else None,
                    termination='all_completed' if drained else 'safety_cap_incomplete',
                    equal_workload_carbon_comparison_valid=fair['equal_workload_carbon_comparison_valid'], **summary))
        identities.append(dict(factor=float(factor), realized_factor=item['realized_factor'],
            workload_sha256=w.sha256, environment_sha256=e.sha256, dc_config_sha256=file_hash(item['model_path']),
            common_end=fair['common_end'], tasks=len(w.tasks), reference=item['identity']))
        regression[name] = read_json(child/'baseline_regression.json')
        diagnostics[name] = read_json(child/'diagnostics.json')
    write_csv(out/'comparison.csv', rows)
    write_json(out/'input_identities.json', identities)
    write_json(out/'baseline_regression.json', regression)
    write_json(out/'diagnostics.json', diagnostics)
    write_json(out/'runtime.json', dict(excluded_from_scientific_hashes=True,
        total_seconds=time.perf_counter()-clock, levels=timing))
    if hashes != experiment_sources():
        raise RuntimeError('Sources changed during CARS campaign')
    write_json(out/'provenance.json', dict(version=CARS_SCHEMA, archive_kind='campaign',
        command=sys.argv, python=sys.version, dependencies=dependencies(), sources=hashes,
        source_sha256=content_hash(hashes),
        input_hashes=dict(parameters=content_hash(config['parameters']), identities=content_hash(identities)),
        excluded_outputs=['runtime.json','run_status.json','provenance.json','source/**'],
        runtime_hashes={'runtime.json':file_hash(out/'runtime.json')},
        output_hashes={n:file_hash(out/n) for n in sorted(CAMPAIGN_FILES)},
        child_output_manifests={factor_folder(f):read_json(out/factor_folder(f)/'provenance.json') for f in config['factors']}))
    if verify_replay is not None:
        verify_replay(out)
    return rows


def run_reference_levels(reference_root, parameters, output_dir, factors=None, require_approved=False):
    """Import existing E1-2B child snapshots; never invoke stress_workload."""
    reference_root = Path(reference_root)
    if isinstance(parameters, (str, Path)):
        parameters = read_json(parameters)
    parameters = copy.deepcopy(parameters)
    factors = list(parameters['factors'] if factors is None else factors)
    def action(out):
        validate_parameters(parameters)
        if (not factors or factors != sorted(set(factors))
            or any(isinstance(f,bool) or f not in parameters['factors'] for f in factors)):
            raise ValueError('CARS factors must be unique increasing declared levels')
        parent = read_json(reference_root/'provenance.json')
        parent_config = read_json(reference_root/'config.json')
        if read_json(reference_root/'run_status.json')['state'] != 'complete':
            raise ValueError('Reference campaign incomplete')
        if content_hash(parent['sources']) != parent['source_sha256']:
            raise ValueError('Reference campaign source manifest corrupt')
        parent_children = {factor_folder(f) for f in parameters['factors']}
        required = {'config.json','comparison.csv','diagnostics.json','input_identities.json','baseline_regression.json'}
        required.update(n+'_replication.json' for n in parent_children)
        if (parent.get('version') != 'e1-2b-nested-arrival-replication-v1'
            or set(parent['output_hashes']) != required
            or set(parent['child_output_manifests']) != parent_children):
            raise ValueError('Reference parent campaign manifest incomplete')
        if (parent_config['seed'] != parameters['stress_seed']
            or parent_config['factors'] != parameters['factors'] or not parent_config['synthetic']):
            raise ValueError('Reference parent fixed replication design mismatch')
        for name, sha in parent['output_hashes'].items():
            if file_hash(reference_root/name) != sha:
                raise ValueError('Reference campaign output mismatch: '+name)
        descriptors = []
        for factor in factors:
            name = factor_folder(factor)
            path = reference_root/name
            if parent['child_output_manifests'].get(name) != read_json(path/'provenance.json'):
                raise ValueError('Reference child manifest mismatch')
            cfg, w, e, selection, identity = _load_reference(path)
            normalized_parent = validate_temporal_config(parent_config['reference_config'],w,e)
            if (cfg != normalized_parent or e.sha256 != parent_config['environment_sha256']
                or identity['input_hashes']['dc_config'] != parent_config['dc_config_sha256']):
                raise ValueError('Reference child shared configuration/environment/physics mismatch')
            if not math.isclose(parameters['threshold_g_per_kwh'], identity['threshold_g_per_kwh'], rel_tol=0, abs_tol=1e-12):
                raise ValueError('Predeclared threshold differs from reference')
            replication = read_json(reference_root/(name+'_replication.json'))
            if (replication['seed'] != parameters['stress_seed'] or replication['requested_factor'] != factor
                or replication['base_snapshot_sha256'] != parent_config['base_workload_sha256']
                or replication['stressed_snapshot_sha256'] != w.sha256 or replication['total_tasks'] != len(w.tasks)):
                raise ValueError('Reference frozen replication identity mismatch')
            if require_approved:
                validate_approved_reference(cfg,w,identity,replication,parameters,
                                            read_json(path/'provenance.json')['git']['head'])
            descriptors.append(dict(factor=factor, cfg=cfg, workload=w, environment=e,
                model_path=path/'dc_config.json', selection=selection, identity=identity,
                realized_factor=replication['realized_factor']))
        config = dict(version=CARS_SCHEMA, archive_kind='campaign', factors=factors, parameters=parameters,
                      reference_parent_sha256=content_hash(parent), frozen_import=True, workload_regenerated=False)
        return _execute_campaign(out, descriptors, config)
    return _fresh_run(output_dir, action)


def replay_levels(previous_dir, output_dir):
    previous = Path(previous_dir)
    def action(out):
        prov = read_json(previous/'provenance.json')
        if (read_json(previous/'run_status.json')['state'] != 'complete' or prov.get('version') != CARS_SCHEMA
            or prov.get('archive_kind') != 'campaign' or prov['sources'] != experiment_sources()
            or content_hash(prov['sources']) != prov['source_sha256'] or prov['python'] != sys.version
            or prov['dependencies'] != dependencies()):
            raise ValueError('CARS campaign status/source/dependency mismatch')
        config = read_json(previous/'config.json')
        children = {factor_folder(f) for f in config['factors']}
        if set(prov['output_hashes']) != CAMPAIGN_FILES or set(prov['child_output_manifests']) != children:
            raise ValueError('CARS campaign manifest incomplete')
        if prov.get('runtime_hashes')!={'runtime.json':file_hash(previous/'runtime.json')}:
            raise ValueError('CARS campaign runtime integrity mismatch')
        for name, sha in prov['output_hashes'].items():
            if file_hash(previous/name) != sha:
                raise ValueError('CARS campaign output mismatch: '+name)
        identities = read_json(previous/'input_identities.json')
        if prov['input_hashes'] != dict(parameters=content_hash(config['parameters']), identities=content_hash(identities)):
            raise ValueError('CARS campaign input manifest mismatch')
        indexed = {i['factor']:i for i in identities}
        if set(indexed) != set(config['factors']):
            raise ValueError('CARS campaign identities incomplete')
        descriptors = []
        for factor in config['factors']:
            name, path = factor_folder(factor), previous/factor_folder(factor)
            if read_json(path/'provenance.json') != prov['child_output_manifests'][name]:
                raise ValueError('CARS campaign child manifest changed')
            cfg, w, e, params, selection, identity, _ = _verify_level(path)
            item = indexed[factor]
            if (item['workload_sha256'] != w.sha256 or item['environment_sha256'] != e.sha256
                or item['dc_config_sha256'] != file_hash(path/'dc_config.json') or item['reference'] != identity):
                raise ValueError('CARS campaign child input identity mismatch')
            resolved = copy.deepcopy(config['parameters'])
            resolved['threshold_g_per_kwh'] = identity['threshold_g_per_kwh']
            if params != resolved:
                raise ValueError('CARS child parameters differ from campaign')
            descriptors.append(dict(factor=factor, cfg=cfg, workload=w, environment=e,
                model_path=path/'dc_config.json', selection=selection, identity=identity,
                realized_factor=item['realized_factor']))
        def verify(out):
            for name, sha in prov['output_hashes'].items():
                if file_hash(out/name) != sha:
                    raise RuntimeError('CARS campaign replay mismatch: '+name)
            for name, child in prov['child_output_manifests'].items():
                for filename, sha in child['output_hashes'].items():
                    if file_hash(out/name/filename) != sha:
                        raise RuntimeError('CARS child replay mismatch: '+name+'/'+filename)
        return _execute_campaign(out, descriptors, config, verify)
    return _fresh_run(output_dir, action)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--parameters', type=Path, default=ROOT/'experiments/e1/configs/e1_3_cars.json')
    parser.add_argument('--factors', nargs='+', type=float)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay', type=Path)
    args = parser.parse_args()
    if args.replay:
        runner = replay_levels if (args.replay/'config.json').is_file() else replay_comparison
        runner(args.replay, args.output)
    else:
        if args.reference_root is None:
            parser.error('--reference-root is required for frozen E1-2B import')
        run_reference_levels(args.reference_root, args.parameters, args.output, args.factors, require_approved=True)
    print(f'Artifacts: {args.output.resolve()}')


if __name__ == '__main__':
    main()
