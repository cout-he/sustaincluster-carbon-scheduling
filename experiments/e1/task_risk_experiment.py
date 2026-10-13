"""E1-6A deterministic eight-policy ablations, Train gate and strict replay.

The existing executor and physical adapter remain the only execution path.
This driver consumes immutable Train/Validation bundles, never PPO or Test.
"""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys
import time

import pandas as pd

from .cars import make_cars_policy
from .cars_experiment import TimingPolicy, _snapshot_sources
from .config import ROOT
from .contracts import canonical_bytes, content_hash, file_hash, write_json
from .drain_metrics import phase_summary
from .engine import LocalExecutor
from .run import _fresh_run, _git, _write_result, dependencies
from .temporal import experiment_sources, write_csv

VERSION = 'e1-6a-task-risk-ablation-v1'
BASELINES = ('edf_backfill', 'edf_carbon_threshold', 'cars_risk_only')
GROUP_POLICIES = BASELINES + ('task_risk_full', 'task_risk_no_history',
    'task_risk_no_urgent', 'task_risk_no_utilization', 'task_risk_max_required')
PARAMETERS = dict(threshold_g_per_kwh=161.075, a_u=1, a_q=1, a_h=1,
    rho0=.5, k=1, K=8, cold_buffer_steps=2, history_minutes=120)
CONFIG_SCHEMA = 'e1-6a-frozen-campaign-v1'


def _development(split):
    if split not in ('train', 'validation'):
        raise ValueError('Test is sealed; only Train/Validation are allowed')


def _parameters(value):
    selected = dict(PARAMETERS if value is None else value)
    if selected != PARAMETERS or any(isinstance(v, bool) for v in selected.values()):
        raise ValueError('E1-6A preregistered parameters differ')
    return selected


def _safe_relative(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Unsafe relative artifact path')
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Artifact path escapes root')
    return path


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _jsonlines(path):
    with Path(path).open(encoding='utf8') as stream:
        return tuple(json.loads(line) for line in stream if line.strip())


def _jsonl(path, rows):
    with Path(path).open('wb') as stream:
        for row in rows:
            stream.write(canonical_bytes(row) + b'\n')


def _verify_hashes(root, manifest):
    for name, digest in manifest.items():
        if file_hash(_safe_relative(root, name)) != digest:
            raise ValueError('Artifact hash mismatch: ' + name)


def _input_identity(bundle):
    return dict(split=bundle.split, factor=bundle.factor, stress_seed=bundle.stress_seed,
        workload_sha256=bundle.workload.sha256, environment_sha256=bundle.environment.sha256,
        capacity=list(bundle.capacity), legacy_parameters=bundle.parameters,
        bundle_identity=bundle.identity)


def _policy(name, bundle, parameters):
    if name in BASELINES:
        return make_cars_policy(name, bundle.parameters, bundle.workload.start, bundle.capacity)
    from .task_risk import make_task_risk_policy
    return make_task_risk_policy(name, parameters, bundle.workload.start, bundle.capacity)


def _distribution(values):
    if not values:
        return dict(count=0, minimum=None, median=None, p95=None, maximum=None, mean=None)
    series = pd.Series(values, dtype='float64')
    return dict(count=len(values), minimum=float(series.min()), median=float(series.median()),
        p95=float(series.quantile(.95)), maximum=float(series.max()),
        mean=math.fsum(values)/len(values))


def _diagnostics(policy, result):
    details = getattr(policy, 'diagnostics', [])
    eligible = [row for row in details if row.get('budget') is not None]
    first = {}; grouped = {}; bins = Counter()
    for row in eligible:
        budget = row['budget']; value = float(budget['buffer_minutes'])
        if not math.isfinite(value) or not 0 <= value <= 150:
            raise RuntimeError('Task risk budget is outside preregistered bounds')
        first.setdefault(row['task_id'], row)
        category = str(budget['dominant_resource'])
        grouped.setdefault(category, []).append(value)
        bins[str(int(round(value/15)))] += 1
    first_bins = Counter(str(int(round(float(row['budget']['buffer_minutes'])/15)))
                         for row in first.values())
    reasons = Counter(event['reason'] for event in result.policy_events)
    return dict(policy=policy.name, diagnostics_count=len(details),
        counters=dict(getattr(policy, 'counters', {})), reason_counts=dict(sorted(reasons.items())),
        eligible_risk_checks=len(eligible), first_eligible_tasks=len(first),
        eligible_defer_checks=sum(bool(row.get('defer')) for row in eligible),
        eligible_execute_checks=sum(not row.get('defer') for row in eligible),
        budget_distribution=_distribution([float(row['budget']['buffer_minutes']) for row in eligible]),
        buffer_steps_histogram=dict(sorted(bins.items())),
        first_eligible_buffer_steps_histogram=dict(sorted(first_bins.items())),
        budget_by_dominant_resource={key:_distribution(values) for key,values in sorted(grouped.items())},
        distribution_denominator='only high-CI, one-grid-step-eligible policy candidate checks; not capacity failures',
        capacity_wait_intervals=sum(e['event']=='capacity_waited' for e in result.policy_events),
        policy_defer_intervals=sum(e['event']=='policy_deferred' for e in result.policy_events))


def _baseline_regression(reference_dir, out, results, workload_sha, environment_sha):
    if reference_dir is None:
        return dict(matched=True, checked=False, policies_checked=0,
                    reason='microfixture: no immutable historical group supplied')
    from .sla_diagnostics import audit_historical_group
    reference = Path(reference_dir)
    audited = audit_historical_group(reference, workload_sha, environment_sha)
    historic_end = pd.Timestamp(audited['fairness']['common_end'])
    evidence = {}
    for name in BASELINES:
        current = out/name; old = reference/name
        for file in ('ledger.jsonl', 'events.jsonl', 'policy_ledger.jsonl', 'policy_events.jsonl'):
            if current.joinpath(file).read_bytes() != old.joinpath(file).read_bytes():
                raise RuntimeError('Old baseline numerical regression mismatch: '+name+'/'+file)
        endpoint = min(pd.Timestamp(results[name].summary['observation_end']), historic_end)
        with (old/'intervals.csv').open(encoding='utf8', newline='') as stream:
            historic = [row for row in csv.DictReader(stream)
                        if pd.Timestamp(row['interval_end']) <= endpoint]
        with (current/'intervals.csv').open(encoding='utf8', newline='') as stream:
            actual = [row for row in csv.DictReader(stream)
                      if pd.Timestamp(row['interval_end']) <= endpoint]
        if actual != historic:
            raise RuntimeError('Old baseline energy/resource prefix regression mismatch: '+name)
        past = [row for row in _jsonlines(old/'policy_intervals.jsonl')
                if pd.Timestamp(row['time']) < endpoint]
        present = [row for row in results[name].policy_steps if pd.Timestamp(row['time']) < endpoint]
        if past != present:
            raise RuntimeError('Old baseline queue prefix regression mismatch: '+name)
        evidence[name] = dict(task_and_event_files_exact=True, interval_prefix_exact=True,
            policy_queue_prefix_exact=True, prefix_end=endpoint.isoformat(), intervals=len(actual))
    return dict(matched=True, checked=True, policies_checked=len(BASELINES), evidence=evidence,
        historic_common_end=historic_end.isoformat(), historical_archive_untouched=True,
        rule='full task/event identity; exact interval and queue prefix; common-tail energy recalculated')


def run_group(bundle, output_dir, *, parameters=None, reference_dir=None, input_path=None):
    """Run eight immutable policies to a shared physical observation endpoint."""
    _development(bundle.split)
    parameters = _parameters(parameters)
    if bundle.parameters.get('threshold_g_per_kwh') != 161.075:
        raise ValueError('Frozen legacy CI threshold differs')
    sources = experiment_sources()
    input_identity = _input_identity(bundle)
    def action(out):
        _snapshot_sources(out, sources)
        write_json(out/'parameters.json', parameters)
        write_json(out/'input-identity.json', input_identity)
        write_json(out/'source-freeze.json', dict(sources=sources, source_sha256=content_hash(sources)))
        provenance = dict(version=VERSION, python=sys.version, dependencies=dependencies(),
            command=sys.argv, git=dict(head=_git('rev-parse','HEAD'), branch=_git('branch','--show-current')),
            input_path=str(Path(input_path).resolve()) if input_path is not None else None,
            reference_path=str(Path(reference_dir).resolve()) if reference_dir is not None else None,
            input_sha256=content_hash(input_identity), parameters_sha256=content_hash(parameters),
            source_sha256=content_hash(sources), formal_test_evaluated=False, training=False)
        write_json(out/'provenance.json', provenance)
        live = {}; own = {}; policies = {}; runtime = {}; models = []
        try:
            for name in GROUP_POLICIES:
                model = bundle.power_factory(); models.append(model)
                policies[name] = TimingPolicy(_policy(name, bundle, parameters))
                executor = LocalExecutor(bundle.workload, bundle.environment, bundle.capacity,
                    model, policies[name], drain=True)
                live[name] = executor
                began = time.perf_counter()
                own[name] = executor.run(stop_at_drain=True)
                runtime[name] = dict(run_seconds=time.perf_counter()-began)
            common = max(executor.observed_end for executor in live.values())
            results = {name:executor.extend_idle(common) if executor.drained_at is not None else own[name]
                       for name,executor in live.items()}
            rows = []; common_rows = []; diagnostics = {}
            expected = {spec.task_id:spec.to_dict() for spec in bundle.workload.tasks}
            for name in GROUP_POLICIES:
                result = results[name]; executor = live[name]; folder = out/name; folder.mkdir()
                _write_result(folder, result)
                if (len(result.tasks) != len(expected) or {row['task_id'] for row in result.tasks} != set(expected)
                    or any({key:row[key] for key in expected[row['task_id']]} != expected[row['task_id']]
                           for row in result.tasks)):
                    raise RuntimeError('Frozen task features/conservation changed')
                executor.check_ownership(); executor.pool.inventory()
                phases = dict(fixed=phase_summary(result, bundle.workload.end),
                    own=phase_summary(own[name], pd.Timestamp(own[name].summary['observation_end'])),
                    common=phase_summary(result, common))
                write_json(folder/'phases.json', phases)
                _jsonl(folder/'diagnostics.jsonl', getattr(policies[name], 'diagnostics', []))
                diagnostics[name] = _diagnostics(policies[name], result)
                write_json(folder/'diagnostics-summary.json', diagnostics[name])
                runtime[name].update(policies[name].timings)
                for phase_name, summary in phases.items():
                    row = dict(policy=name, phase=phase_name, split=bundle.split, factor=bundle.factor,
                        stress_seed=bundle.stress_seed, own_observation_end=own[name].summary['observation_end'],
                        drain_hours=(executor.drained_at-bundle.workload.end).total_seconds()/3600
                            if executor.drained_at is not None else None,
                        termination='all_completed' if executor.drained_at is not None else 'safety_cap_incomplete',
                        **summary)
                    rows.append(row)
                    if phase_name == 'common':common_rows.append(row)
            all_complete = all(row['completed'] == len(expected) and row['pending'] == row['running'] == 0
                               for row in common_rows)
            resource_hours = {key:[row[key+'_hours'] for row in common_rows] for key in ('cores','gpus','mem')}
            conserved = all(all(math.isclose(value, values[0], rel_tol=1e-12, abs_tol=1e-8)
                               for value in values) for values in resource_hours.values())
            if all_complete and not conserved:
                raise RuntimeError('Completed cohort resource-hours differ')
            fairness = dict(version=VERSION, split=bundle.split, factor=bundle.factor, policies=list(GROUP_POLICIES),
                workload_sha256=bundle.workload.sha256, environment_sha256=bundle.environment.sha256,
                same_frozen_input_sha256=content_hash(input_identity), identical_frozen_inputs=True,
                capacity=list(bundle.capacity), parameters=parameters, common_end=common.isoformat(),
                arrival_end=bundle.workload.end.isoformat(), safety_end=bundle.environment.end.isoformat(),
                arrived=len(expected), all_tasks_completed=all_complete, equal_resource_hours=conserved,
                equal_workload_carbon_comparison_valid=all_complete and conserved, resource_hours=resource_hours,
                idle_included=True, policy_future_ci_access=False, formal_test_evaluated=False, training=False)
            regression = _baseline_regression(reference_dir, out, results, bundle.workload.sha256,
                                              bundle.environment.sha256)
            write_json(out/'fairness.json', fairness)
            write_json(out/'baseline-regression.json', regression)
            write_json(out/'diagnostics-summary.json', diagnostics)
            write_csv(out/'comparison.csv', rows); write_csv(out/'common_summary.csv', common_rows)
            write_csv(out/'runtime.csv', [dict(policy=name, **runtime[name]) for name in GROUP_POLICIES])
            paired = []
            for name in GROUP_POLICIES:
                candidate = {row['task_id']:row for row in results[name].tasks}
                for reference_name in ('edf_backfill', 'cars_risk_only'):
                    reference = {row['task_id']:row for row in results[reference_name].tasks}
                    missed = lambda row:row['deadline_outcome'] in ('missed','missed_unfinished')
                    paired.append(dict(policy=name, reference=reference_name,
                        extra_misses=sum(missed(candidate[key]) and not missed(reference[key]) for key in expected),
                        avoided_misses=sum(not missed(candidate[key]) and missed(reference[key]) for key in expected),
                        both_missed=sum(missed(candidate[key]) and missed(reference[key]) for key in expected),
                        started_differently=sum(candidate[key]['start_time'] != reference[key]['start_time'] for key in expected),
                        attribution='paired trajectory association; not single-event causation'))
            write_csv(out/'paired-miss-summary.csv', paired)
            if experiment_sources() != sources:
                raise RuntimeError('Source changed during E1-6A group; preserve this failed run')
            scientific = {path.relative_to(out).as_posix():file_hash(path) for path in sorted(out.rglob('*'))
                if path.is_file() and path.name not in ('run_status.json', 'provenance.json', 'runtime.csv')}
            write_json(out/'scientific-hashes.json', scientific)
            return dict(common_rows=common_rows, fairness=fairness, baseline_regression=regression,
                        diagnostics=diagnostics, runtime=runtime, results=results)
        finally:
            for model in reversed(models):
                close = getattr(model, 'close', None)
                if close is not None:close()
    return _fresh_run(output_dir, action)


def _completed_group(path):
    path = Path(path)
    if _read(path/'run_status.json')['state'] != 'complete':
        raise ValueError('Group is not complete')
    _verify_hashes(path, _read(path/'scientific-hashes.json'))
    return _read(path/'fairness.json')


def write_train_gate(group_paths, output_path):
    """Freeze a correctness-only Train gate; performance cannot select parameters."""
    output = Path(output_path)
    if output.exists():raise FileExistsError('Refusing to overwrite Train gate')
    paths = tuple(map(Path, group_paths))
    if len(paths) != 3 or len(set(path.resolve() for path in paths)) != 3:
        raise ValueError('Train gate requires three distinct complete Train groups')
    sources = experiment_sources(); entries = []
    for path in paths:
        fairness = _completed_group(path)
        if (fairness['split'] != 'train' or not fairness['all_tasks_completed']
                or not fairness['equal_resource_hours'] or fairness['policies'] != list(GROUP_POLICIES)
                or fairness['parameters'] != PARAMETERS
                or _read(path/'source-freeze.json')['sources'] != sources
                or not _read(path/'baseline-regression.json')['matched']):
            raise ValueError('Train correctness/input/source/regression gate failed')
        entries.append(dict(path=str(path.resolve()), factor=fairness['factor'],
            scientific_manifest_sha256=file_hash(path/'scientific-hashes.json'),
            workload_sha256=fairness['workload_sha256'], environment_sha256=fairness['environment_sha256']))
    if sorted(entry['factor'] for entry in entries) != [1., 1.5, 2.]:
        raise ValueError('Train gate requires all three registered factors')
    document = dict(version=VERSION, functional_passed=True, not_performance_selected=True,
        parameters=PARAMETERS, parameters_sha256=content_hash(PARAMETERS), sources=sources,
        source_sha256=content_hash(sources), groups=entries, formal_test_evaluated=False,
        guard='task/resource conservation, complete Drain, parameter/source freeze and legacy execution regression; no DMR/carbon selection')
    write_json(output, document)
    return document


def validate_train_gate(gate_path, parameters=None):
    gate = _read(gate_path)
    selected = _parameters(parameters)
    if (gate.get('version') != VERSION or gate.get('functional_passed') is not True
            or gate.get('not_performance_selected') is not True or gate.get('formal_test_evaluated') is not False):
        raise ValueError('Train functional gate invalid')
    if gate.get('parameters') != selected or gate.get('parameters_sha256') != content_hash(selected):
        raise ValueError('Train parameters changed after gate')
    if gate.get('sources') != experiment_sources() or gate.get('source_sha256') != content_hash(gate['sources']):
        raise ValueError('Train source changed after gate')
    if len(gate.get('groups', [])) != 3 or sorted(entry['factor'] for entry in gate['groups']) != [1.,1.5,2.]:
        raise ValueError('Incomplete Train group gate')
    for entry in gate['groups']:
        path = Path(entry['path'])
        if file_hash(path/'scientific-hashes.json') != entry['scientific_manifest_sha256']:
            raise ValueError('Train gate artifact manifest changed')
        fairness = _completed_group(path)
        if (fairness['split'] != 'train' or fairness['factor'] != entry['factor']
                or fairness['workload_sha256'] != entry['workload_sha256']
                or fairness['environment_sha256'] != entry['environment_sha256']
                or not fairness['all_tasks_completed'] or not fairness['equal_resource_hours']):
            raise ValueError('Train gate input/completion changed')
    return gate


def replay_group(previous_dir, output_dir, *, bundle=None):
    """Replay one group with identical source, input and scientific bytes."""
    previous = Path(previous_dir)
    if Path(output_dir).exists():raise FileExistsError('Refusing to overwrite replay')
    fairness = _completed_group(previous); _development(fairness['split'])
    provenance = _read(previous/'provenance.json')
    frozen = _read(previous/'source-freeze.json')
    if (frozen['sources'] != experiment_sources() or frozen['source_sha256'] != content_hash(frozen['sources'])
            or provenance['python'] != sys.version or provenance['dependencies'] != dependencies()):
        raise ValueError('Replay source/Python/dependency mismatch')
    if bundle is None:
        from .rl_training_utils import load_bundle
        if not provenance['input_path']:raise ValueError('Replay requires the referenced frozen bundle')
        bundle = load_bundle(provenance['input_path'], allowed_split=fairness['split'])
    if content_hash(_input_identity(bundle)) != provenance['input_sha256']:
        raise ValueError('Replay frozen input identity mismatch')
    result = run_group(bundle, output_dir, parameters=_read(previous/'parameters.json'),
        reference_dir=provenance['reference_path'], input_path=provenance['input_path'])
    same = _read(previous/'scientific-hashes.json') == _read(Path(output_dir)/'scientific-hashes.json')
    if not same:raise RuntimeError('Replay scientific output mismatch')
    result['replay'] = dict(scientific_outputs_exact=True, runtime_excluded=True)
    write_json(Path(output_dir)/'replay-check.json', result['replay'])
    return result


def _campaign_config(config):
    if (config.get('schema') != CONFIG_SCHEMA or config.get('test_evaluation') is not False
            or config.get('new_training') is not False or config.get('parameters') != PARAMETERS
            or tuple(config.get('policies', [])) != GROUP_POLICIES or config.get('sla_target') != .05):
        raise ValueError('E1-6A campaign configuration differs from frozen design')
    entries = config.get('cohorts', [])
    if [(entry['split'], entry['factor']) for entry in entries] != [
            ('train',1.),('train',1.5),('train',2.),('validation',1.),('validation',1.5),('validation',2.)]:
        raise ValueError('Six registered Train/Validation cohorts required')
    return config


def run_campaign(config, output_dir, *, split, factors=None, gate_path=None):
    """Resume only complete groups; no group/result file can be overwritten."""
    _development(split); config = _campaign_config(config)
    if split == 'validation':
        if gate_path is None:raise ValueError('Validation requires an immutable Train functional gate')
        validate_train_gate(gate_path, config['parameters'])
    selected = [1.,1.5,2.] if factors is None else list(factors)
    if not selected or len(set(selected)) != len(selected) or any(f not in (1.,1.5,2.) for f in selected):
        raise ValueError('Only registered factors 1/1.5/2 are allowed')
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    frozen_path = out/'campaign-freeze.json'
    sources = experiment_sources()
    frozen = dict(version=VERSION, config=config, config_sha256=content_hash(config),
        sources=sources, source_sha256=content_hash(sources),
        parameters_sha256=content_hash(PARAMETERS), formal_test_evaluated=False, training=False)
    if frozen_path.exists():
        if _read(frozen_path) != frozen:raise ValueError('Campaign source/config changed; use a new directory')
    else:write_json(frozen_path, frozen)
    from .rl_training_utils import load_bundle
    all_bundles = []
    for entry in config['cohorts']:
        input_path = _safe_relative(ROOT, entry['input'])
        bundle = load_bundle(input_path, allowed_split=entry['split'])
        if (bundle.factor != entry['factor'] or bundle.workload.sha256 != entry['workload_sha256']
                or bundle.identity['bundle_manifest_sha256'] != entry['bundle_manifest_sha256']
                or bundle.environment.sha256 != entry['environment_sha256']):
            raise ValueError('Registered immutable cohort identity changed')
        all_bundles.append((entry,bundle,input_path))
    manifest = dict(all_inputs_frozen_before_train=True, cohorts=[dict(entry, identity=bundle.identity)
        for entry,bundle,_ in all_bundles], formal_test_evaluated=False, training=False)
    if (out/'input-manifest.json').exists():
        if _read(out/'input-manifest.json') != manifest:raise ValueError('Campaign input identity changed')
    else:write_json(out/'input-manifest.json', manifest)
    rows = []
    for entry,bundle,input_path in all_bundles:
        if entry['split'] != split or entry['factor'] not in selected:continue
        folder = out/f"{split}-{entry['factor']:g}"
        if folder.exists():
            fairness = _completed_group(folder)
            if (fairness['workload_sha256'] != bundle.workload.sha256 or fairness['parameters'] != PARAMETERS
                    or _read(folder/'source-freeze.json')['sources'] != frozen['sources']):
                raise ValueError('Existing group differs from frozen campaign')
            with (folder/'common_summary.csv').open(encoding='utf8') as stream:
                rows.extend(csv.DictReader(stream))
            continue
        result = run_group(bundle, folder, parameters=config['parameters'], input_path=input_path,
            reference_dir=_safe_relative(ROOT,entry['reference']))
        rows.extend(result['common_rows'])
        print(json.dumps(dict(state='group_complete', split=split, factor=entry['factor'],
            common_end=result['fairness']['common_end'], all_completed=result['fairness']['all_tasks_completed'])), flush=True)
    if split == 'train' and all((out/f'train-{factor:g}'/'run_status.json').is_file() for factor in (1.,1.5,2.)):
        gate = out/'train-functional-gate.json'
        if gate.exists():validate_train_gate(gate, PARAMETERS)
        else:write_train_gate([out/f'train-{factor:g}' for factor in (1.,1.5,2.)], gate)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'experiments/e1/configs/e1_6a.json')
    parser.add_argument('--split', choices=('train','validation'))
    parser.add_argument('--factor', type=float, action='append')
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.replay is not None:
        if args.split is not None or args.factor is not None or args.gate is not None:
            parser.error('--replay must not be mixed with campaign options')
        replay_group(args.replay, args.output)
    else:
        if args.split is None:parser.error('Campaign requires --split')
        run_campaign(_read(args.config), args.output, split=args.split, factors=args.factor, gate_path=args.gate)


if __name__ == '__main__':main()
