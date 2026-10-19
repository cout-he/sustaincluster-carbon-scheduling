"""Train-only A1 logical-capacity replays, with no IT/HVAC energy estimates.

Observed-usage equivalents are resource accounting units, not job reservations.
The unchanged LocalExecutor supplies all task and allocation time semantics.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import csv
import json
import math
import multiprocessing
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from experiments.e1.contracts import content_hash, file_hash
from experiments.e1.engine import LocalExecutor
from experiments.e1.policies import CarbonThresholdFIFO, EDFBackfill, ImmediateFIFO
from experiments.joint_v2.scenarios import load_pool
from experiments.object_model_a1.config import adapt_scenario, load_candidate
from experiments.object_validity.a0_resources import (
    DEMAND_NAMES, ENGINE_NAMES, RESOURCE_NAMES, ZeroPower, _window_metrics,
    distribution, longest_run, task_static_rows, write_csv,
)

POLICIES = ('fifo', 'edf', 'carbon_threshold_fifo')
MAPPINGS = ('legacy_usage_x5_pressure_v1', 'observed_usage_x1_pressure_v1')
BASE_SHA = 'a754c5ee4bd3922cecd6993d9abb6bfbacdda7e0'


def diagnose(scenario, policy_name, capacity, threshold=161.075):
    """Replay all tasks; export resource/Deadline and explicitly labelled CI proxies."""
    if scenario.role != 'train':
        raise ValueError('Only Train scenarios are authorized')
    if policy_name not in POLICIES:
        raise ValueError('Unsupported deterministic policy')
    capacity = tuple(float(v) for v in capacity)
    if len(capacity) != 3 or not all(math.isfinite(v) and v > 0 for v in capacity):
        raise ValueError('Candidate requires three positive finite capacities')
    for spec in scenario.workload.tasks:
        if any(getattr(spec, field) > cap for field, cap in zip(DEMAND_NAMES, capacity)):
            # Never remove a difficult job or silently fabricate a completion.
            raise ValueError('Task exceeds candidate capacity: ' + spec.task_id)
    policy = (ImmediateFIFO() if policy_name == 'fifo' else EDFBackfill()
              if policy_name == 'edf' else CarbonThresholdFIFO(threshold))
    executor = LocalExecutor(scenario.workload, scenario.environment, capacity,
                             ZeroPower(), policy, drain=True)
    result = executor.run(stop_at_drain=True)
    if result.summary['completed'] != len(scenario.workload.tasks):
        raise RuntimeError('Original frozen Train horizon did not drain every task; report missing coverage')
    static = task_static_rows(SimpleNamespace(scenario_id=scenario.scenario_id,
                               workload=scenario.workload, capacity=capacity))
    by_static = {r['task_id']: r for r in static}
    by_policy = {r['task_id']: r for r in result.policy_tasks}
    tasks = []
    # Piecewise-constant, original 15-minute Train CI. Actual task runtime is
    # integrated exactly; the separate reservation integral uses ceil releases.
    ci = np.array([s.ci_g_per_kwh for s in scenario.environment.samples], dtype=float)
    prefix = np.r_[0.0, np.cumsum(ci * .25)]
    start = scenario.workload.start
    exact_exposure = np.zeros(3)
    resource_hours = np.zeros(3)
    for original in result.tasks:
        task = dict(by_static[original['task_id']])
        task.update({k: original[k] for k in ('state', 'start_time', 'finish_time',
                    'completion_observed_at', 'deadline_outcome', 'sla_deadline')})
        waits = by_policy[task['task_id']]
        task.update(capacity_wait_minutes=waits['capacity_wait_intervals'] * 15,
                    policy_defer_minutes=waits['policy_defer_intervals'] * 15,
                    observed_wait_minutes=waits['observed_pending_minutes'])
        task['deadline_start_margin_minutes'] = task['initial_slack_minutes'] - task['observed_wait_minutes']
        task['wait_over_initial_slack'] = (task['observed_wait_minutes']/task['initial_slack_minutes']
                                          if task['initial_slack_minutes'] > 0 else None)
        task['jct_minutes'] = ((pd.Timestamp(task['finish_time'])-pd.Timestamp(task['arrival_time'])).total_seconds()/60)
        offset = int((pd.Timestamp(task['start_time'])-start).total_seconds()/900)
        whole, remainder = divmod(task['duration_minutes'], 15)
        finish_slot = offset + int(whole)
        ci_integral = prefix[finish_slot]-prefix[offset]
        if remainder > 1e-10:
            ci_integral += ci[finish_slot] * remainder/60
        task['execution_ci_integral_g_per_kwh_hours'] = float(ci_integral)
        task['task_type'] = 'gpu_positive' if task['gpu_demand'] > 0 else 'cpu_only'
        task['duration_group'] = ('lt60min' if task['duration_minutes'] < 60 else
                                  '60to240min' if task['duration_minutes'] < 240 else 'ge240min')
        for i, name in enumerate(RESOURCE_NAMES):
            exact_exposure[i] += task[name+'_demand'] * ci_integral
            resource_hours[i] += task[name+'_demand'] * task['duration_minutes']/60
        tasks.append(task)
    n = len(tasks)
    arrived = n
    row = dict(scenario_id=scenario.scenario_id, role='train', policy=policy_name,
               original_scenario_id=scenario.identity.get('source_scenario_id',scenario.scenario_id),
               original_scenario_identity_sha256=scenario.identity.get('source_scenario_identity_sha256',content_hash(scenario.identity)),
               derived_scenario_identity_sha256=content_hash(scenario.identity),
               source_fragment_id=scenario.identity.get('source_fragment_id', ''),
               mapped_workload_sha256=scenario.identity.get('derived_workload_sha256') or scenario.workload.sha256,
               original_workload_sha256=scenario.identity.get('source_workload_sha256',scenario.identity.get('workload_sha256', '')),
               original_environment_sha256=scenario.identity.get('source_environment_sha256',scenario.identity.get('environment_sha256', '')),
               resource_pressure_factor=scenario.identity.get('resource_pressure_factor'),
               sla_multiplier=scenario.workload.sla_multiplier,
               arrival_start=start.isoformat(), arrival_end=scenario.workload.end.isoformat(),
               frozen_observation_end=scenario.environment.end.isoformat(),
               active_observation_end=executor.observed_end.isoformat(), arrived=n,
               completed=result.summary['completed'], completed_fraction=result.summary['completed']/n if n else 1,
               unfinished=0, structurally_infeasible=result.summary['structurally_infeasible'],
               deadline_misses=result.summary['deadline_missed_completed'],
               capacity_waited_tasks=sum(t['capacity_wait_minutes'] > 0 for t in tasks),
               policy_deferred_tasks=sum(t['policy_defer_minutes'] > 0 for t in tasks),
               both_capacity_and_policy_waited_tasks=sum(t['capacity_wait_minutes'] > 0 and
                                                        t['policy_defer_minutes'] > 0 for t in tasks),
               capacity_wait_minutes_total=sum(t['capacity_wait_minutes'] for t in tasks),
               policy_defer_minutes_total=sum(t['policy_defer_minutes'] for t in tasks),
               observed_wait_minutes_total=sum(t['observed_wait_minutes'] for t in tasks),
               negative_start_margin_tasks=sum(t['deadline_start_margin_minutes'] < -1e-8 for t in tasks))
    row['dmr'] = row['deadline_misses']/n if n else 0
    row['dmr_le_5_percent'] = row['dmr'] <= .05
    row['capacity_waited_fraction'] = row['capacity_waited_tasks']/n if n else 0
    row['policy_deferred_fraction'] = row['policy_deferred_tasks']/n if n else 0
    for key in ('duration_minutes', 'initial_slack_minutes', 'capacity_wait_minutes',
                'policy_defer_minutes', 'observed_wait_minutes', 'deadline_start_margin_minutes',
                'wait_over_initial_slack', 'jct_minutes'):
        row.update(distribution([t[key] for t in tasks if t[key] is not None], key))
    row['mean_jct_minutes'] = row['jct_minutes_mean']
    for slots in (0, 1, 2, 3):
        row[f'legal_wait_slots_{slots}_fraction'] = sum(t['legal_wait_slots'] == slots for t in tasks)/n if n else 0
    row['legal_wait_slots_ge4_fraction'] = sum(t['legal_wait_slots'] >= 4 for t in tasks)/n if n else 0
    for minutes in (15, 30, 60):
        row[f'can_wait_ge{minutes}_tasks'] = sum(t['legal_wait_slots']*15 >= minutes for t in tasks)
    remaining_slots = [(-1 if t['deadline_start_margin_minutes'] < -1e-8 else
                        int(math.floor((t['deadline_start_margin_minutes']+1e-9)/15))) for t in tasks]
    row['observed_remaining_legal_wait_slots_negative_tasks'] = sum(v<0 for v in remaining_slots)
    for slots in (0,1,2,3):
        row[f'observed_remaining_legal_wait_slots_{slots}_tasks'] = sum(v==slots for v in remaining_slots)
    row['observed_remaining_legal_wait_slots_ge4_tasks'] = sum(v>=4 for v in remaining_slots)
    long_slack = [t for t in tasks if t['initial_slack_minutes'] >= 60]
    row.update(long_slack_tasks=len(long_slack),
        long_slack_capacity_waited_tasks=sum(t['capacity_wait_minutes'] > 0 for t in long_slack),
        long_slack_capacity_missed_tasks=sum(t['capacity_wait_minutes'] > 0 and t['deadline_outcome']=='missed'
                                             for t in long_slack),
        long_slack_all_missed_tasks=sum(t['deadline_outcome']=='missed' for t in long_slack))
    arrival_steps = [s for s in result.steps if pd.Timestamp(s['interval_start']) < scenario.workload.end]
    for suffix, steps in (('arrival', arrival_steps), ('through_drain', result.steps)):
        row.update(_window_metrics(steps, suffix))
    event_counts, unique_ids, exclusive_counts = Counter(), defaultdict(set), Counter()
    insufficient_times = defaultdict(set)
    unable_at_arrival = set()
    capacity_events = [e for e in result.policy_events if e['event'] == 'capacity_waited']
    unexplained = 0
    for event in capacity_events:
        task = by_static[event['task_id']]
        if event['time'] == task['arrival_time']:
            unable_at_arrival.add(task['task_id'])
        insufficient = [name for name, field in zip(RESOURCE_NAMES, ENGINE_NAMES)
                        if task[name+'_demand'] > event['available_before'][field] + 1e-9]
        if not insufficient:
            unexplained += 1
        for name in insufficient:
            event_counts[name] += 1
            unique_ids[name].add(task['task_id'])
            insufficient_times[name].add(event['time'])
        if len(insufficient) == 1:
            exclusive_counts[insufficient[0]] += 1
    row.update(capacity_wait_attempts=len(capacity_events),
               unclassified_capacity_wait_attempts=unexplained,
               unable_to_start_at_arrival_tasks=len(unable_at_arrival),
               unable_to_start_at_arrival_fraction=len(unable_at_arrival)/n if n else 0)
    competing_flags = [bool(s['capacity_waiting_ids']) for s in result.policy_steps]
    active_flags = [bool(s['policy_deferred_ids']) for s in result.policy_steps]
    row.update(capacity_competition_minutes=sum(competing_flags)*15,
               capacity_competition_longest_minutes=longest_run(competing_flags)*15,
               policy_defer_active_minutes=sum(active_flags)*15,
               policy_defer_longest_minutes=longest_run(active_flags)*15,
               capacity_waiting_tasks_peak=max((len(s['capacity_waiting_ids']) for s in result.policy_steps), default=0))
    for i, (name, engine_name, cap) in enumerate(zip(RESOURCE_NAMES, ENGINE_NAMES, capacity)):
        row[name+'_capacity'] = cap
        row[name+'_bottleneck_attempts'] = event_counts[name]
        row[name+'_bottleneck_unique_tasks'] = len(unique_ids[name])
        row[name+'_exclusive_bottleneck_attempts'] = exclusive_counts[name]
        row[name+'_bottleneck_active_minutes'] = len(insufficient_times[name])*15
        row[name+'_single_task_capacity_fraction_max'] = max((t[name+'_demand']/cap for t in tasks), default=0)
        row.update(distribution([t[name+'_demand'] for t in tasks], name+'_task_demand'))
        row[name+'_resource_hours'] = float(resource_hours[i])
        row[name+'_ci_exposure_proxy'] = float(exact_exposure[i])
        row[name+'_mean_execution_ci_g_per_kwh'] = (float(exact_exposure[i]/resource_hours[i])
                                                    if resource_hours[i] > 0 else None)
        row[name+'_reserved_ci_exposure_proxy'] = math.fsum(s['used_'+engine_name] * s['ci_g_per_kwh'] * .25
                                                            for s in result.steps)
        row[name+'_arrival_work_to_capacity_time'] = resource_hours[i]/(cap * len(arrival_steps)*.25)
    # Do not leak ZeroPower's placeholders or apparently physical emissions.
    clean_steps = [{k: s[k] for k in ('interval_start', 'interval_end', 'pending', 'running',
                    'used_cores', 'used_gpus', 'used_mem', 'cores_fraction', 'gpus_fraction', 'mem_fraction')}
                   for s in result.steps]
    return row, dict(tasks=tasks, steps=clean_steps,
                     arrival_slots=len(arrival_steps))


def _summarize_group(rows, task_values, utilization, type_members):
    first = rows[0]
    group = {k: first[k] for k in ('candidate_id', 'mapping_version', 'policy',
             'resource_pressure_factor', 'sla_multiplier', 'candidate_config_sha256')}
    n = sum(r['arrived'] for r in rows)
    group.update(scenarios=len(rows), arrived=n, completed=sum(r['completed'] for r in rows),
                 completion_fraction=sum(r['completed'] for r in rows)/n if n else 1,
                 fragments_with_dmr_gt5_percent=sum(r['dmr'] > .05 for r in rows),
                 worst_fragment_dmr=max(r['dmr'] for r in rows),
                 worst_fragment_id=max(rows,key=lambda r:r['dmr'])['scenario_id'],
                 queue_peak=max(r['queue_length_max_through_drain'] for r in rows),
                 longest_competition_minutes=max(r['capacity_competition_longest_minutes'] for r in rows))
    count_fields = ('deadline_misses', 'capacity_waited_tasks', 'policy_deferred_tasks',
                    'both_capacity_and_policy_waited_tasks', 'capacity_wait_attempts',
                    'capacity_wait_minutes_total', 'policy_defer_minutes_total', 'observed_wait_minutes_total',
                    'capacity_competition_minutes', 'policy_defer_active_minutes',
                    'unable_to_start_at_arrival_tasks', 'can_wait_ge15_tasks', 'can_wait_ge30_tasks',
                    'can_wait_ge60_tasks', 'long_slack_tasks', 'long_slack_capacity_waited_tasks',
                    'long_slack_capacity_missed_tasks', 'long_slack_all_missed_tasks', 'structurally_infeasible')
    for field in count_fields:
        group[field] = sum(r[field] for r in rows)
    for label in ('negative','0','1','2','3','ge4'):
        field = 'observed_remaining_legal_wait_slots_'+label+'_tasks'
        group[field] = sum(r[field] for r in rows)
    group['dmr'] = group['deadline_misses']/n if n else 0
    group['capacity_waited_fraction'] = group['capacity_waited_tasks']/n if n else 0
    group['policy_deferred_fraction'] = group['policy_deferred_tasks']/n if n else 0
    for field, values in task_values.items():
        group.update(distribution(values, field))
    for label in ('0','1','2','3','ge4'):
        field = 'legal_wait_slots_'+label+'_fraction'
        group[field] = sum(r[field]*r['arrived'] for r in rows)/n if n else 0
    positive = [v for v in task_values['capacity_wait_minutes'] if v > 0]
    group['capacity_wait_positive_mean_minutes'] = float(np.mean(positive)) if positive else 0
    utils = []
    for name in RESOURCE_NAMES:
        group[name+'_capacity'] = first[name+'_capacity']
        for suffix in ('bottleneck_attempts', 'bottleneck_unique_tasks', 'exclusive_bottleneck_attempts',
                       'bottleneck_active_minutes', 'resource_hours', 'ci_exposure_proxy', 'reserved_ci_exposure_proxy'):
            group[name+'_'+suffix] = sum(r[name+'_'+suffix] for r in rows)
        group[name+'_mean_execution_ci_g_per_kwh'] = (group[name+'_ci_exposure_proxy']/group[name+'_resource_hours']
                                                     if group[name+'_resource_hours'] else None)
        group[name+'_single_task_capacity_fraction_max'] = max(r[name+'_single_task_capacity_fraction_max'] for r in rows)
        util = {k: group[k] for k in ('candidate_id','mapping_version','policy','resource_pressure_factor','sla_multiplier')}
        util.update(resource=name, scenarios=len(rows), slots=len(utilization[name]),
                    window='original 24h arrival slots', unit='fraction of candidate logical capacity')
        util.update(distribution(utilization[name],'utilization'))
        utils.append(util)
        group[name+'_fraction_mean_arrival'] = util['utilization_mean']
        group[name+'_fraction_p99_arrival'] = util['utilization_p99']
        group[name+'_fraction_peak_arrival'] = util['utilization_max']
        group[name+'_fraction_peak_through_drain'] = max(r[name+'_fraction_max_through_drain'] for r in rows)
    types = []
    for (task_type,duration_group), values in sorted(type_members.items()):
        t = {k: group[k] for k in ('candidate_id','mapping_version','policy','resource_pressure_factor','sla_multiplier')}
        count = len(values['duration_minutes'])
        t.update(task_type=task_type,duration_group=duration_group,tasks=count,
            capacity_waited_tasks=values['capacity_waited_tasks'],
            deadline_misses=values['deadline_misses'],policy_deferred_tasks=values['policy_deferred_tasks'])
        t['dmr'] = t['deadline_misses']/count
        for field in ('duration_minutes','initial_slack_minutes','observed_wait_minutes',
                      'execution_ci_integral_g_per_kwh_hours'):
            t.update(distribution(values[field],field))
        types.append(t)
    return group,utils,types


_SCENARIOS = None


def _run_group(specification):
    candidate, pressure, sla, policy = specification
    capacity = candidate['capacity']
    rows, task_values, utilization = [], defaultdict(list), defaultdict(list)
    type_members = defaultdict(lambda: defaultdict(list))
    for original in _SCENARIOS:
        if not math.isclose(float(original.identity['resource_pressure_factor']), pressure) or not math.isclose(original.workload.sla_multiplier, sla):
            continue
        scenario = adapt_scenario(original,candidate)
        row, detail = diagnose(scenario,policy,capacity,candidate['threshold'])
        row.update(candidate_id=candidate['candidate_id'],mapping_version=candidate['mapping_version'],
                   candidate_config_sha256=candidate['config_sha256'])
        rows.append(row)
        for task in detail['tasks']:
            for field in ('duration_minutes','initial_slack_minutes','capacity_wait_minutes','policy_defer_minutes',
                          'observed_wait_minutes','deadline_start_margin_minutes','wait_over_initial_slack','jct_minutes'):
                if task[field] is not None:
                    task_values[field].append(task[field])
            values = type_members[(task['task_type'],task['duration_group'])]
            for field in ('duration_minutes','initial_slack_minutes','observed_wait_minutes',
                          'execution_ci_integral_g_per_kwh_hours'):
                values[field].append(task[field])
            for field in ('capacity_waited_tasks','deadline_misses','policy_deferred_tasks'):
                if isinstance(values[field],list):
                    values[field] = 0
            values['capacity_waited_tasks'] += task['capacity_wait_minutes'] > 0
            values['deadline_misses'] += task['deadline_outcome']=='missed'
            values['policy_deferred_tasks'] += task['policy_defer_minutes'] > 0
        for step in detail['steps'][:detail['arrival_slots']]:
            for name, engine_name in zip(RESOURCE_NAMES,ENGINE_NAMES):
                utilization[name].append(step[engine_name+'_fraction'])
    if not rows:
        raise ValueError('No authorized Train scenarios matched candidate group')
    group,utils,types = _summarize_group(rows,task_values,utilization,type_members)
    return rows,group,utils,types


def _read_candidate(path):
    raw = load_candidate(path)
    cap = tuple(raw['logical_capacity'][name] for name in RESOURCE_NAMES)
    mapping = raw['mapping_version']
    if mapping not in MAPPINGS:
        raise ValueError('Unknown config mapping version')
    policies = raw.get('policies',POLICIES if mapping==MAPPINGS[1] else ('fifo','edf'))
    if any(policy not in POLICIES for policy in policies):
        raise ValueError('Unknown candidate policy')
    return dict(raw,capacity=tuple(cap),threshold=raw.get('ci_threshold_g_per_kwh',161.075),
                policies=policies,config_path=str(path))


def repair_fragment_ids(output, source_csv, diagnostic_source_commit):
    """Correct only the identity column, using already public A0 Train metadata.

    Producer commit and original file hash remain recorded; every non-fragment
    CSV cell is hashed before/after to demonstrate no simulation metric changes.
    No pool, original dataset, environment, simulator or physical model is read.
    """
    output, source_csv = Path(output), Path(source_csv)
    with source_csv.open(encoding='utf8',newline='') as stream:
        source_rows = list(csv.DictReader(stream))
    by_start = {}
    for row in source_rows:
        start = row['arrival_start']
        if not row['source_fragment_id']:
            raise ValueError('Source Train fragment identity is empty')
        if start in by_start and by_start[start]['source_fragment_id']!=row['source_fragment_id']:
            raise ValueError('Conflicting source Train fragment identity')
        by_start[start] = row
    path = output/'scenario-summary.csv'
    original_hash = file_hash(path)
    with path.open(encoding='utf8',newline='') as stream:
        rows = list(csv.DictReader(stream))
    other_cells_before = content_hash([{k:v for k,v in row.items() if k!='source_fragment_id'} for row in rows])
    repaired = 0
    for row in rows:
        source = by_start.get(row['arrival_start'])
        if source is None or row['original_scenario_id'].split('-r')[0]!=source['scenario_id'].split('-r')[0]:
            raise ValueError('A1 scenario does not match public A0 Train fragment identity')
        wanted = source['source_fragment_id']
        if row['source_fragment_id'] not in ('',wanted):
            raise ValueError('Refusing to replace a conflicting nonempty fragment identity')
        if row['source_fragment_id']!=wanted:
            row['source_fragment_id'] = wanted
            repaired += 1
    other_cells_after = content_hash([{k:v for k,v in row.items() if k!='source_fragment_id'} for row in rows])
    if other_cells_before!=other_cells_after:
        raise RuntimeError('Identity correction attempted to change a simulation metric')
    write_csv(path,rows)
    result = dict(schema='a1-source-fragment-identity-repair-v1',rows=len(rows),repaired_rows=repaired,
        source='already published A0 Train ci_by_fragment.csv, matched by original scenario date and arrival_start',
        source_metadata_sha256=file_hash(source_csv),input_csv_sha256=original_hash,output_csv_sha256=file_hash(path),
        non_fragment_columns_sha256_before=other_cells_before,non_fragment_columns_sha256_after=other_cells_after,
        diagnostic_source_commit=diagnostic_source_commit,postprocess_script_sha256=file_hash(__file__),
        metrics_replayed=False,simulation_metrics_changed=False,heldout_payloads_read=False)
    (output/'identity-repair.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    metadata_path=output/'metadata.json'
    metadata=json.loads(metadata_path.read_text(encoding='utf8'))
    metadata.update(diagnostic_source_commit=diagnostic_source_commit,
        producer_run_rules_sha256=metadata['script_sha256'],
        current_reproduction_rules_sha256=file_hash(__file__),source_fragment_identity_repair=result)
    metadata_path.write_text(json.dumps(metadata,indent=2),encoding='utf8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool')
    parser.add_argument('--configs',nargs='+')
    parser.add_argument('--output',required=True)
    parser.add_argument('--source-sha',default=BASE_SHA)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--max-seconds',type=float,default=1200)
    parser.add_argument('--repair-source-fragments',help='Identity-only correction using public A0 Train CI metadata; no pool payload read')
    parser.add_argument('--diagnostic-source-commit',help='Commit containing the exact original producer rules/config adapter')
    args = parser.parse_args(argv)
    if args.repair_source_fragments:
        if not args.diagnostic_source_commit:
            parser.error('--diagnostic-source-commit is required for identity correction')
        print(json.dumps(repair_fragment_ids(args.output,args.repair_source_fragments,args.diagnostic_source_commit)),flush=True)
        return
    if not args.pool or not args.configs:
        parser.error('--pool and --configs are required for candidate replays')
    started=time.monotonic()
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    candidates=[_read_candidate(path) for path in args.configs]
    if len({c['candidate_id'] for c in candidates})!=len(candidates):
        raise ValueError('Candidate IDs must be unique')
    global _SCENARIOS
    _SCENARIOS=load_pool(args.pool,('train',))
    if len(_SCENARIOS)!=96:
        raise ValueError('Expected the unchanged 96-scenario Train inventory')
    pool_path=Path(args.pool)
    manifest_path=pool_path/'manifest.json' if pool_path.is_dir() else pool_path
    planned=[(candidate,pressure,sla,policy) for candidate in candidates
             for pressure in (.75,1.,1.25) for sla in (1.5,2.) for policy in candidate['policies']]
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    meta=dict(schema='a1-logical-resource-rules-v1',source_sha=args.source_sha,
        script_sha256=file_hash(__file__),a0_resource_helpers_sha256=file_hash(Path(__file__).parents[1]/'object_validity'/'a0_resources.py'),
        base_source_sha=BASE_SHA,pool_manifest_sha256=file_hash(manifest_path),
        original_source_hashes=manifest['source_hashes'],started_utc=datetime.now(timezone.utc).isoformat(),
        allowed_roles=['train'],heldout_payloads_read=False,physics_model_called=False,trained_models=False,
        power_adapter='ZeroPower bookkeeping only; ALL power/energy/carbon-kg fields excluded',
        candidates=candidates,planned_runs=len(planned)*16,workers=args.workers,
        units=dict(cpu='observed vCPU-equivalent x declared scale x pressure, not reservation',
                   gpu='observed GPU-equivalent x declared scale x pressure, not integer devices/reservation',
                   mem='observed GB x declared scale x pressure',time='minutes',utilization='fraction of candidate capacity',
                   ci_exposure_proxy='resource accounting units * hours * gCO2e/kWh; NOT grams or physical emissions'),
        methods=dict(quantiles='numpy linear quantiles; pooled tasks or equal 15min arrival slots, as named',
            wait_attribution='unchanged engine: capacity first at task turn; active defer only if currently fits',
            bottleneck='task demand exceeds available resource at task turn; resource-specific counts overlap',
            competition_duration='union of 15min intervals with >=1 capacity-waited task; distinct from active defer and queue',
            deadline='actual finish <= original arrival + SLA * original Duration',
            completion='all tasks retained and observed to actual drain, with original frozen safety-tail coverage',
            ci_proxy='exact original duration integrates piecewise-constant original CI; separate rounded reservation exposure',
            threshold='fixed inherited 161.075, uses only current CI and solo latest-start; no capacity reservation or future CI',
            grouping='16 distinct original fragments per pressure/SLA; repeated augmentations not independent observations'),
        scenario_identities={s.scenario_id:content_hash(s.identity) for s in _SCENARIOS})
    (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    rows,groups,utils,types=[],[],[],[]
    complete=True
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork')) as pool:
        for result in pool.map(_run_group,planned,chunksize=1):
            current,group,current_utils,current_types=result
            rows.extend(current);groups.append(group);utils.extend(current_utils);types.extend(current_types)
            write_csv(out/'scenario-summary.csv',rows)
            write_csv(out/'candidate-summary.csv',groups)
            write_csv(out/'utilization-pooled.csv',utils)
            write_csv(out/'task-types.csv',types)
            progress=dict(completed_runs=len(rows),planned_runs=meta['planned_runs'],
                elapsed_seconds=time.monotonic()-started,last_candidate=group['candidate_id'],
                last_pressure=group['resource_pressure_factor'],last_sla=group['sla_multiplier'],last_policy=group['policy'],complete=False)
            (out/'progress.json').write_text(json.dumps(progress,indent=2),encoding='utf8')
            print(json.dumps(progress),flush=True)
            if time.monotonic()-started>=args.max_seconds:
                complete=False
                for process in pool._processes.values():
                    process.terminate()
                break
    meta.update(elapsed_seconds=time.monotonic()-started,completed_runs=len(rows),
                complete=complete and len(rows)==meta['planned_runs'],
                completed_utc=datetime.now(timezone.utc).isoformat())
    (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    (out/'progress.json').write_text(json.dumps(dict(completed_runs=len(rows),
        planned_runs=meta['planned_runs'],elapsed_seconds=meta['elapsed_seconds'],complete=meta['complete']),indent=2),encoding='utf8')
    if not meta['complete']:
        raise SystemExit('Budget reached: partial aggregate results saved, no missing result fabricated')


if __name__=='__main__':
    main()
