"""A0 Train-only deterministic resource/Deadline diagnosis; no learned model.

Uses the unchanged LocalExecutor with its complete frozen environmental window.
ZeroPower is ONLY a resource-statistics adapter, never an energy estimate.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from experiments.e1.contracts import content_hash, file_hash
from experiments.e1.engine import LocalExecutor
from experiments.e1.policies import EDFBackfill, ImmediateFIFO
from experiments.e1.power import PowerReading
from experiments.joint_v2.scenarios import load_pool

RESOURCE_NAMES = ('cpu', 'gpu', 'mem')
ENGINE_NAMES = ('cores', 'gpus', 'mem')
DEMAND_NAMES = ('cores_req', 'gpu_req', 'mem_req')
QUANTILES = ((50, .50), (90, .90), (95, .95), (99, .99))


class ZeroPower:
    """No physics called: returned zeros are expressly excluded from outputs."""
    def measure(self, cpu, gpu, mem, sample):
        return PowerReading(0, 0, 0)


def distribution(values, prefix):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {prefix + '_' + key: None for key in ('mean', 'p50', 'p90', 'p95', 'p99', 'max')}
    out = {prefix + '_mean': float(a.mean()), prefix + '_max': float(a.max())}
    out.update({prefix + '_p' + str(label): float(np.quantile(a, q, method='linear'))
                for label, q in QUANTILES})
    return out


def longest_run(flags):
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def task_static_rows(scenario):
    rows = []
    for spec in scenario.workload.tasks:
        slack = (spec.sla_multiplier - 1) * spec.duration
        row = dict(scenario_id=scenario.scenario_id, task_id=spec.task_id,
                   source_row=spec.source_row, source_task_index=spec.source_task_index,
                   arrival_time=spec.arrival_time.isoformat(), duration_minutes=spec.duration,
                   sla_multiplier=spec.sla_multiplier, initial_slack_minutes=slack,
                   legal_wait_slots=int(math.floor((slack + 1e-9) / 15)),
                   legal_wait_slots_sla1_5=int(math.floor((.5 * spec.duration + 1e-9) / 15)),
                   legal_wait_slots_sla2_0=int(math.floor((spec.duration + 1e-9) / 15)))
        for name, field, capacity in zip(RESOURCE_NAMES, DEMAND_NAMES, scenario.capacity):
            row[name + '_demand'] = getattr(spec, field)
            row[name + '_single_task_capacity_fraction'] = getattr(spec, field) / capacity
        rows.append(row)
    return rows


def unconstrained_demand(scenario):
    """All original tasks start on arrival, WITHOUT capacity enforcement.

    This is required capacity demand, not a feasible schedule or energy model.
    Arrival and actual-finish timestamps remain original; release is ceil15min.
    """
    if scenario.role != 'train':
        raise ValueError('Only Train demand diagnosis is authorized')
    start = scenario.workload.start
    arrival_slots = int((scenario.workload.end-start).total_seconds()/900)
    intervals = []
    for spec in scenario.workload.tasks:
        first = int((spec.arrival_time-start).total_seconds()/900)
        last = first + math.ceil(spec.duration/15)
        intervals.append((first,last,spec))
    end = max([arrival_slots] + [last for _,last,_ in intervals])
    curves, rows = {}, []
    for name, field, capacity in zip(RESOURCE_NAMES, DEMAND_NAMES, scenario.capacity):
        diff = np.zeros(end+1, dtype=float)
        for first,last,spec in intervals:
            diff[first] += getattr(spec,field)
            diff[last] -= getattr(spec,field)
        curve = np.cumsum(diff)[:-1]
        curve[np.abs(curve)<1e-8] = 0
        curves[name] = curve.tolist()
        for window,values in (('arrival',curve[:arrival_slots]), ('through_unconstrained_drain',curve)):
            row = dict(scenario_id=scenario.scenario_id,
                       scenario_identity_sha256=content_hash(scenario.identity),
                       resource_pressure_factor=scenario.identity.get('resource_pressure_factor'),
                       sla_multiplier=scenario.workload.sla_multiplier,
                       window=window, resource=name, current_logical_capacity=capacity,
                       slots=len(values), feasible_schedule=False)
            row.update(distribution(values,'demand'))
            row.update(distribution(values/capacity,'fraction_of_current_capacity'))
            row['over_current_capacity_slots'] = int(np.sum(values>capacity+1e-8))
            row['over_current_capacity_minutes'] = row['over_current_capacity_slots']*15
            rows.append(row)
    return rows,curves


def diagnose_demand_only(scenarios, output, source_sha):
    started=time.monotonic()
    rows=[]
    pooled=defaultdict(list)
    for scenario in scenarios:
        current,curves=unconstrained_demand(scenario)
        rows.extend(current)
        # One SLA copy per original fragment; original duration is unchanged by SLA.
        if math.isclose(scenario.workload.sla_multiplier,1.5):
            slots=int((scenario.workload.end-scenario.workload.start).total_seconds()/900)
            for name,capacity in zip(RESOURCE_NAMES,scenario.capacity):
                key=(scenario.identity.get('resource_pressure_factor'),name,capacity)
                pooled[key].extend(curves[name][:slots])
    summaries=[]
    for (pressure,name,capacity),values in sorted(pooled.items()):
        row=dict(resource_pressure_factor=pressure,resource=name,current_logical_capacity=capacity,
                 window='pooled arrival slots, one SLA=1.5 copy per original fragment',
                 slots=len(values), feasible_schedule=False)
        row.update(distribution(values,'demand'))
        row.update(distribution(np.asarray(values)/capacity,'fraction_of_current_capacity'))
        row['over_current_capacity_slots']=sum(v>capacity+1e-8 for v in values)
        summaries.append(row)
    write_csv(output/'unconstrained-demand.csv',rows)
    write_csv(output/'unconstrained-demand-pooled.csv',summaries)
    meta=dict(schema='a0-unconstrained-demand-v1',source_sha=source_sha,
              script_sha256=file_hash(Path(__file__)),scenario_count=len(scenarios),
              allowed_roles=['train'],no_heldout_payload_access=True,
              assumptions='every task starts at original arrival; no capacity enforcement; original demands/durations; release at arrival+ceil(duration/15)*15',
              limitation='required concurrent demand only; NOT feasible scheduling, NOT optimal result, NO power/carbon inference',
              pooled='equal 15min arrival slots; 16 original fragments at each pressure; one SLA1.5 copy, since SLA does not change offered demand',
              units='logical current-model CPU/GPU/MEM demand units; source physical units remain unaudited',
              quantiles='numpy linear quantiles, time weighted by equal 15min slots',
              elapsed_seconds=time.monotonic()-started,
              scenario_identities={s.scenario_id:content_hash(s.identity) for s in scenarios})
    (output/'unconstrained-demand-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    print(json.dumps(dict(mode='demand-only',scenario_count=len(scenarios),elapsed_seconds=meta['elapsed_seconds'])),flush=True)


def _window_metrics(steps, suffix):
    out = {}
    for name, engine_name in zip(RESOURCE_NAMES, ENGINE_NAMES):
        values = [s[engine_name + '_fraction'] for s in steps]
        out.update({k + '_' + suffix: v for k, v in distribution(values, name + '_fraction').items()})
        out[name + '_above80_minutes_' + suffix] = sum(v >= .8 for v in values) * 15
        out[name + '_above80_longest_minutes_' + suffix] = longest_run(v >= .8 for v in values) * 15
    pending = [s['pending'] for s in steps]
    out.update({k + '_' + suffix: v for k, v in distribution(pending, 'queue_length').items()})
    peak = max(pending, default=0)
    high = max(1, math.ceil(float(np.quantile(pending, .95)))) if pending else 1
    out.update({f'queue_nonempty_minutes_{suffix}': sum(v > 0 for v in pending) * 15,
                f'queue_nonempty_longest_minutes_{suffix}': longest_run(v > 0 for v in pending) * 15,
                f'queue_p95_high_threshold_{suffix}': high,
                f'queue_high_total_minutes_{suffix}': sum(v >= high for v in pending) * 15,
                f'queue_high_longest_minutes_{suffix}': longest_run(v >= high for v in pending) * 15,
                f'queue_at_peak_longest_minutes_{suffix}': longest_run(v == peak and peak > 0 for v in pending) * 15})
    return out


def diagnose_scenario(scenario, policy_name):
    if scenario.role != 'train':
        raise ValueError('A0 resource diagnosis accepts only Train payloads')
    if policy_name not in ('fifo', 'edf'):
        raise ValueError('Only immediate FIFO/EDF in resource diagnosis')
    policy = ImmediateFIFO() if policy_name == 'fifo' else EDFBackfill()
    executor = LocalExecutor(scenario.workload, scenario.environment, scenario.capacity,
                             ZeroPower(), policy, drain=True)
    # Execute ALL tasks to actual drain; the remaining original horizon is known
    # exactly empty. Append zero resource slots analytically, not shortened data.
    result = executor.run(stop_at_drain=True)
    full_steps = list(result.steps)
    if executor.drained_at is not None:
        for sample in scenario.environment.samples:
            if sample.time >= executor.observed_end:
                full_steps.append(dict(interval_start=sample.time.isoformat(),
                    interval_end=(sample.time + pd.Timedelta(minutes=15)).isoformat(),
                    pending=0, running=0, used_cores=0, used_gpus=0, used_mem=0,
                    cores_fraction=0, gpus_fraction=0, mem_fraction=0))
    static = task_static_rows(scenario)
    by_static = {r['task_id']: r for r in static}
    by_policy = {r['task_id']: r for r in result.policy_tasks}
    tasks = []
    for original in result.tasks:
        row = dict(by_static[original['task_id']])
        row.update({k: original[k] for k in ('state', 'start_time', 'finish_time', 'completion_observed_at',
                                            'deadline_outcome', 'sla_deadline', 'structurally_infeasible')})
        waits = by_policy[row['task_id']]
        row.update(capacity_wait_minutes=waits['capacity_wait_intervals'] * 15,
                   policy_defer_minutes=waits['policy_defer_intervals'] * 15,
                   observed_wait_minutes=waits['observed_pending_minutes'])
        row['deadline_start_margin_minutes'] = row['initial_slack_minutes'] - row['observed_wait_minutes']
        row['wait_over_initial_slack'] = (row['observed_wait_minutes'] / row['initial_slack_minutes']
                                          if row['initial_slack_minutes'] > 0 else None)
        row['jct_minutes'] = ((pd.Timestamp(row['finish_time']) - pd.Timestamp(row['arrival_time'])).total_seconds() / 60
                              if row['state'] == 'completed' else None)
        tasks.append(row)
    n = len(tasks)
    row = dict(scenario_id=scenario.scenario_id, role='train', policy=policy_name,
               scenario_identity_sha256=content_hash(scenario.identity),
               workload_sha256=scenario.workload.sha256,
               resource_pressure_factor=scenario.identity.get('resource_pressure_factor'),
               sla_multiplier=scenario.workload.sla_multiplier,
               arrival_start=scenario.workload.start.isoformat(), arrival_end=scenario.workload.end.isoformat(),
               observation_end=scenario.environment.end.isoformat(), arrived=n,
               completed=result.summary['completed'], completed_fraction=result.summary['completed'] / n if n else 1,
               unfinished=result.summary['pending'] + result.summary['running'],
               deadline_misses=result.summary['deadline_missed_completed'] + result.summary['deadline_missed_unfinished'],
               capacity_waited_tasks=sum(t['capacity_wait_minutes'] > 0 for t in tasks),
               policy_deferred_tasks=sum(t['policy_defer_minutes'] > 0 for t in tasks),
               capacity_wait_minutes_total=sum(t['capacity_wait_minutes'] for t in tasks),
               policy_defer_minutes_total=sum(t['policy_defer_minutes'] for t in tasks),
               structurally_infeasible=result.summary['structurally_infeasible'],
               negative_start_margin_tasks=sum(t['deadline_start_margin_minutes'] < -1e-8 for t in tasks))
    row['dmr'] = row['deadline_misses'] / n if n else 0
    row['capacity_waited_fraction'] = row['capacity_waited_tasks'] / n if n else 0
    row['mean_jct_minutes'] = float(np.mean([t['jct_minutes'] for t in tasks if t['jct_minutes'] is not None]))
    for key in ('duration_minutes', 'initial_slack_minutes', 'capacity_wait_minutes', 'observed_wait_minutes',
                'deadline_start_margin_minutes', 'wait_over_initial_slack'):
        row.update(distribution([t[key] for t in tasks if t[key] is not None], key))
    for slots in (0, 1, 2, 3):
        row[f'legal_wait_slots_{slots}_fraction'] = sum(t['legal_wait_slots'] == slots for t in tasks) / n if n else 0
    row['legal_wait_slots_ge4_fraction'] = sum(t['legal_wait_slots'] >= 4 for t in tasks) / n if n else 0
    arrival_steps = [s for s in result.steps if pd.Timestamp(s['interval_start']) < scenario.workload.end]
    last_release = max((pd.Timestamp(t['completion_observed_at']) for t in tasks if t['completion_observed_at']),
                       default=scenario.workload.end)
    active_end = max(scenario.workload.end, last_release)
    active_steps = [s for s in result.steps if pd.Timestamp(s['interval_start']) < active_end]
    row['active_observation_end'] = active_end.isoformat()
    for suffix, steps in (('arrival', arrival_steps), ('through_drain', active_steps), ('full', full_steps)):
        row.update(_window_metrics(steps, suffix))
    capacity_events = [e for e in result.policy_events if e['event'] == 'capacity_waited']
    resource_ids, exclusive_ids = defaultdict(set), defaultdict(set)
    initial_ids = set()
    counts, exclusive_counts = Counter(), Counter()
    multi_count, multi_ids, unexplained = 0, set(), 0
    for event in capacity_events:
        task = by_static[event['task_id']]
        if event['time'] == task['arrival_time']:
            initial_ids.add(event['task_id'])
        insufficient = [name for name, field in zip(RESOURCE_NAMES, ENGINE_NAMES)
                        if task[name + '_demand'] > event['available_before'][field] + 1e-9]
        if not insufficient:
            unexplained += 1
        for name in insufficient:
            counts[name] += 1
            resource_ids[name].add(event['task_id'])
        if len(insufficient) == 1:
            exclusive_counts[insufficient[0]] += 1
            exclusive_ids[insufficient[0]].add(event['task_id'])
        elif len(insufficient) > 1:
            multi_count += 1
            multi_ids.add(event['task_id'])
    row.update(unable_to_start_at_arrival_tasks=len(initial_ids),
               unable_to_start_at_arrival_fraction=len(initial_ids) / n if n else 0,
               capacity_wait_attempts=len(capacity_events), multiple_bottleneck_attempts=multi_count,
               multiple_bottleneck_unique_tasks=len(multi_ids), unclassified_capacity_wait_attempts=unexplained)
    arrival_minutes = (scenario.workload.end - scenario.workload.start).total_seconds() / 60
    by_arrival = defaultdict(lambda: [0, 0, 0])
    ideal_diff = defaultdict(lambda: [0, 0, 0])
    for task in tasks:
        release = pd.Timestamp(task['arrival_time']) + pd.Timedelta(minutes=15 * math.ceil(task['duration_minutes'] / 15))
        for i, name in enumerate(RESOURCE_NAMES):
            demand = task[name + '_demand']
            by_arrival[task['arrival_time']][i] += demand
            ideal_diff[pd.Timestamp(task['arrival_time'])][i] += demand
            ideal_diff[release][i] -= demand
    ideal_used, ideal_peak = [0, 0, 0], [0, 0, 0]
    for stamp in sorted(ideal_diff):
        for i in range(3):
            ideal_used[i] += ideal_diff[stamp][i]
            ideal_peak[i] = max(ideal_peak[i], ideal_used[i])
    for i, (name, capacity) in enumerate(zip(RESOURCE_NAMES, scenario.capacity)):
        row[name + '_capacity'] = capacity
        row[name + '_bottleneck_attempts'] = counts[name]
        row[name + '_bottleneck_unique_tasks'] = len(resource_ids[name])
        row[name + '_exclusive_bottleneck_attempts'] = exclusive_counts[name]
        row[name + '_exclusive_bottleneck_unique_tasks'] = len(exclusive_ids[name])
        row.update(distribution([t[name + '_demand'] for t in tasks], name + '_task_demand'))
        row[name + '_single_task_capacity_fraction_max'] = max((t[name + '_demand'] / capacity for t in tasks), default=0)
        work = sum(t[name + '_demand'] * t['duration_minutes'] for t in tasks)
        rounded_work = sum(t[name + '_demand'] * 15 * math.ceil(t['duration_minutes'] / 15) for t in tasks)
        row[name + '_arrival_resource_minutes'] = work
        row[name + '_arrival_work_to_capacity_time'] = work / (capacity * arrival_minutes)
        row[name + '_rounded_arrival_work_to_capacity_time'] = rounded_work / (capacity * arrival_minutes)
        row[name + '_arrival_bucket_demand_fraction_peak'] = max((v[i] / capacity for v in by_arrival.values()), default=0)
        row[name + '_unconstrained_immediate_fraction_peak'] = ideal_peak[i] / capacity
    # Exclude deliberately fake energy/power fields from the exported resource time series.
    steps = [{k: s[k] for k in ('interval_start', 'interval_end', 'pending', 'running',
                                'used_cores', 'used_gpus', 'used_mem', 'cores_fraction',
                                'gpus_fraction', 'mem_fraction')} for s in full_steps]
    return row, dict(tasks=tasks, steps=steps)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def grouped_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['policy'], row['resource_pressure_factor'], row['sla_multiplier'])].append(row)
    output = []
    for (policy, pressure, sla), members in sorted(groups.items()):
        n = sum(r['arrived'] for r in members)
        group = dict(policy=policy, resource_pressure_factor=pressure, sla_multiplier=sla,
                     scenarios=len(members), arrived=n,
                     completed=sum(r['completed'] for r in members),
                     deadline_misses=sum(r['deadline_misses'] for r in members),
                     capacity_waited_tasks=sum(r['capacity_waited_tasks'] for r in members),
                     unable_to_start_at_arrival_tasks=sum(r['unable_to_start_at_arrival_tasks'] for r in members),
                     capacity_wait_attempts=sum(r['capacity_wait_attempts'] for r in members),
                     policy_defer_minutes_total=sum(r['policy_defer_minutes_total'] for r in members),
                     mean_jct_minutes=sum(r['mean_jct_minutes'] * r['completed'] for r in members) / sum(r['completed'] for r in members))
        for field in ('deadline_misses', 'capacity_waited_tasks', 'unable_to_start_at_arrival_tasks'):
            group[field + '_fraction'] = group[field] / n
        for name in RESOURCE_NAMES:
            for field in ('bottleneck_attempts', 'bottleneck_unique_tasks', 'exclusive_bottleneck_attempts'):
                group[name + '_' + field] = sum(r[name + '_' + field] for r in members)
            for field in ('fraction_mean_arrival', 'arrival_work_to_capacity_time', 'rounded_arrival_work_to_capacity_time'):
                group[name + '_' + field + '_scenario_mean'] = float(np.mean([r[name + '_' + field] for r in members]))
            group[name + '_fraction_peak_any_scenario'] = max(r[name + '_fraction_max_full'] for r in members)
            group[name + '_single_task_capacity_fraction_max'] = max(r[name + '_single_task_capacity_fraction_max'] for r in members)
        group['capacity_wait_minutes_mean'] = sum(r['capacity_wait_minutes_total'] for r in members) / n
        group['capacity_wait_minutes_max'] = max(r['capacity_wait_minutes_max'] for r in members)
        group['queue_length_peak_any_scenario'] = max(r['queue_length_max_full'] for r in members)
        group['queue_nonempty_minutes_sum'] = sum(r['queue_nonempty_minutes_full'] for r in members)
        for label in ('0', '1', '2', '3', 'ge4'):
            key = 'legal_wait_slots_' + label + '_fraction'
            group[key] = sum(r[key] * r['arrived'] for r in members) / n
        output.append(group)
    return output


def summarize_existing_details(output):
    """Exact pooled task quantiles from local resource runs, including zero waits."""
    with (output/'scenario-summary.csv').open(encoding='utf8',newline='') as stream:
        index=list(csv.DictReader(stream))
    groups=defaultdict(list)
    for row in index:
        groups[(row['policy'],row['resource_pressure_factor'],row['sla_multiplier'])].append(row)
    result=[]
    util_result=[]
    numeric=('duration_minutes','initial_slack_minutes','capacity_wait_minutes','observed_wait_minutes',
             'deadline_start_margin_minutes','jct_minutes','wait_over_initial_slack',
             'cpu_demand','gpu_demand','mem_demand')
    input_hashes={}
    resource_input_hashes={}
    for (policy,pressure,sla),members in sorted(groups.items()):
        values=defaultdict(list)
        utilization=defaultdict(list)
        slots=[]
        for member in members:
            path=output/'details'/member['scenario_id']/policy/'tasks.csv'
            input_hashes[path.relative_to(output).as_posix()]=file_hash(path)
            with path.open(encoding='utf8',newline='') as stream:
                for task in csv.DictReader(stream):
                    slots.append(int(task['legal_wait_slots']))
                    for field in numeric:
                        if task[field]!='':
                            values[field].append(float(task[field]))
            resource_path=output/'details'/member['scenario_id']/policy/'resources.csv'
            resource_input_hashes[resource_path.relative_to(output).as_posix()]=file_hash(resource_path)
            arrival_slots=int((pd.Timestamp(member['arrival_end'])-pd.Timestamp(member['arrival_start'])).total_seconds()/900)
            with resource_path.open(encoding='utf8',newline='') as stream:
                for interval,resource_row in enumerate(csv.DictReader(stream)):
                    if interval>=arrival_slots:
                        break
                    for name,engine_name in zip(RESOURCE_NAMES,ENGINE_NAMES):
                        utilization[name].append(float(resource_row[engine_name+'_fraction']))
        n=len(slots)
        current=dict(policy=policy,resource_pressure_factor=float(pressure),sla_multiplier=float(sla),
                     tasks=n,scenario_count=len(members))
        for field in numeric:
            current.update(distribution(values[field],field))
        for slot in (0,1,2,3):
            current[f'legal_wait_slots_{slot}_fraction']=sum(v==slot for v in slots)/n
        current['legal_wait_slots_ge4_fraction']=sum(v>=4 for v in slots)/n
        positive=[v for v in values['capacity_wait_minutes'] if v>0]
        current['capacity_wait_positive_mean_minutes']=float(np.mean(positive)) if positive else 0
        current['wait_exceeds_initial_slack_tasks']=sum(v< -1e-8 for v in values['deadline_start_margin_minutes'])
        current['duration_lt15_fraction']=sum(v<15 for v in values['duration_minutes'])/n
        result.append(current)
        for name in RESOURCE_NAMES:
            util=dict(policy=policy,resource_pressure_factor=float(pressure),sla_multiplier=float(sla),
                      resource=name,scenario_count=len(members),slots=len(utilization[name]),
                      window='original arrival window only',unit='fraction of logical capacity',
                      weighting='equal 15min intervals; exact pooled samples')
            util.update(distribution(utilization[name],'utilization'))
            util_result.append(util)
    write_csv(output/'task-distribution-pooled.csv',result)
    write_csv(output/'utilization-pooled.csv',util_result)
    (output/'task-distribution-pooled-metadata.json').write_text(json.dumps(dict(
        schema='a0-pooled-task-distribution-v1',script_sha256=file_hash(Path(__file__)),
        method='all tasks, including zero capacity waits; numpy linear quantiles; one scenario per fragment/pressure/SLA/policy',
        task_files_sha256=content_hash(input_hashes),input_files=input_hashes,
        no_heldout_payload_access=True),indent=2),encoding='utf8')
    (output/'utilization-pooled-metadata.json').write_text(json.dumps(dict(
        schema='a0-pooled-utilization-v1',script_sha256=file_hash(Path(__file__)),
        method='read original arrival-window prefix of each resource curve only; exact pooled equal 15min slots; numpy linear quantiles',
        resource_files_sha256=content_hash(resource_input_hashes),input_files=resource_input_hashes,
        no_heldout_payload_access=True),indent=2),encoding='utf8')
    print(json.dumps(dict(mode='summarize-existing-details',groups=len(result),task_files=len(input_hashes))),flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-seconds', type=float, default=900)
    parser.add_argument('--source-sha', help='Externally verified source SHA for Windows-created worktrees under WSL')
    parser.add_argument('--demand-only', action='store_true', help='No executor: unconstrained immediate concurrent capacity demand only')
    parser.add_argument('--summarize-existing-details', action='store_true', help='No pool read: exact pooled task quantiles from saved local diagnostic CSVs')
    args = parser.parse_args(argv)
    started = time.monotonic()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.summarize_existing_details:
        summarize_existing_details(output)
        return
    if args.demand_only:
        source_sha=args.source_sha or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        diagnose_demand_only(load_pool(args.pool,('train',)),output,source_sha)
        return
    pool_root = Path(args.pool)
    manifest_path = pool_root / 'manifest.json' if pool_root.is_dir() else pool_root
    meta = dict(schema='a0-resources-v1', started_utc=datetime.now(timezone.utc).isoformat(),
                source_sha=args.source_sha or subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                pool_manifest_sha256=file_hash(manifest_path),
                script_sha256=file_hash(Path(__file__)),
                allowed_roles=['train'], june_development_accessed=False,
                july_validation_accessed=False, september_test_accessed=False,
                power_adapter='zero only for resource bookkeeping; no energy/carbon values exported',
                policies=['FIFO immediate one-pass backfill', 'EDF immediate one-pass backfill'],
                selection='all 96 Train scenarios in frozen pool order; no filtering or task deletion',
                budget_seconds=args.max_seconds,
                units=dict(utilization='fraction of logical capacity', duration='minutes',
                           cpu='current logical CPU-equivalent', gpu='current logical GPU-equivalent',
                           mem='current code memory units (raw source unit not independently validated)'),
                methods=dict(quantiles='numpy linear quantiles; equal 15min slots => time weighted',
                    utilization_windows='arrival, through_drain=max(arrival_end,last observed release), full frozen environmental window',
                    idle_tail='after ALL tasks drain in unchanged LocalExecutor, append exact zero resource slots through original frozen environment.end; no energy claim',
                    wait_attribution='unchanged LocalExecutor capacity-first per task turn; active policy defer is always zero',
                    capacity_waited='unique task with >=1 capacity_wait event',
                    unable_at_arrival='unique task with capacity_wait at its arrival boundary',
                    bottleneck='candidate demand > available_before + 1e-9; attempts and unique/exclusive tasks separate',
                    high_queue='>=max(1,ceil(P95 pending after scheduling)); peak sustained separate',
                    resource_high='utilization >=0.8; no significance claim attached',
                    deadline='actual finish <= arrival + SLA_multiplier*unchanged duration; release ceil15min separate',
                    pressure='sum demand*duration/(logical capacity*arrival-window minutes); rounded variant ceil(duration/15)*15',
                    independence='96 augmentations derive from 16 original arrival fragments; not 96 independent traces'))
    (output / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf8')
    scenarios = load_pool(args.pool, ('train',))
    rows = []
    complete = True
    for scenario in scenarios:
        for policy in ('fifo', 'edf'):
            if time.monotonic() - started >= args.max_seconds:
                complete = False
                break
            row, details = diagnose_scenario(scenario, policy)
            rows.append(row)
            local = output / 'details' / scenario.scenario_id / policy
            write_csv(local / 'tasks.csv', details['tasks'])
            write_csv(local / 'resources.csv', details['steps'])
            write_csv(output / 'scenario-summary.csv', rows)
            groups = grouped_summary(rows)
            write_csv(output / 'pressure-policy-summary.csv', groups)
            progress = dict(completed_runs=len(rows), planned_runs=len(scenarios) * 2,
                            elapsed_seconds=time.monotonic() - started, complete=False,
                            last_scenario=scenario.scenario_id, last_policy=policy)
            (output / 'progress.json').write_text(json.dumps(progress, indent=2), encoding='utf8')
            print(json.dumps(progress), flush=True)
        if not complete:
            break
    meta.update(completed_utc=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=time.monotonic() - started,
                completed_runs=len(rows), planned_runs=len(scenarios) * 2,
                all_requested_runs_complete=complete,
                scenario_ids=[s.scenario_id for s in scenarios],
                scenario_identities={s.scenario_id:content_hash(s.identity) for s in scenarios})
    (output / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf8')
    (output / 'overview.json').write_text(json.dumps(dict(metadata=meta, groups=grouped_summary(rows)), indent=2), encoding='utf8')
    (output / 'progress.json').write_text(json.dumps(dict(completed_runs=len(rows),
        planned_runs=len(scenarios)*2, elapsed_seconds=time.monotonic()-started, complete=complete), indent=2), encoding='utf8')
    if complete:
        summarize_existing_details(output)


if __name__ == '__main__':
    main()
