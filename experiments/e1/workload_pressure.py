"""Offline workload/EDF diagnostics, with the existing E1 holding semantics.

This module never schedules tasks and is not an RL observation or policy input.
The CLI accepts only frozen Train/Validation bundles; Test remains sealed.
"""
import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import STEP, finite, interval_times, utc_time, write_json
from .policies import latest_start

VERSION = 'e1-5c-workload-pressure-v1'
RESOURCES = ('cores', 'gpus', 'mem')
DEMAND_FIELDS = ('cores_req', 'gpu_req', 'mem_req')
MINUTES = STEP.total_seconds() / 60


@dataclass(frozen=True)
class PressureAnalysis:
    summary: dict
    arrivals: tuple
    task_details: tuple


def distribution(values):
    """Population quantiles, linear interpolation; empty sets explicitly zero."""
    data = np.asarray(tuple(values), dtype=float)
    if data.size and not np.isfinite(data).all():
        raise ValueError('Distribution requires finite data')
    levels = (0, .25, .5, .75, .9, .95, .99, 1)
    quantiles = np.quantile(data, levels, method='linear') if data.size else np.zeros(8)
    result = dict(zip(('min', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'max'),
                      (float(v) for v in quantiles)))
    result.update(count=int(data.size), mean=float(data.mean()) if data.size else 0.,
                  std_population=float(data.std()) if data.size else 0.)
    return result


def occupied_minutes(duration):
    """Resource release at first grid boundary >= raw finish (not SLA finish)."""
    nanoseconds = pd.Timedelta(minutes=finite(duration, 'duration', True)).value
    return float(((nanoseconds + STEP.value - 1) // STEP.value) * MINUTES)


def _rolling(rows, capacity, history_minutes):
    width = int(history_minutes / MINUTES)
    for index, row in enumerate(rows):
        recent = rows[max(0, index - width + 1):index + 1]
        row['history_observed_minutes'] = min((index + 1) * MINUTES, history_minutes)
        for name, limit in zip(RESOURCES, capacity):
            for suffix in ('raw', 'occupied'):
                total = math.fsum(r[f'{name}_work_minutes_{suffix}'] for r in recent)
                row[f'{name}_pressure_{history_minutes}m' + ('_raw' if suffix == 'raw' else '')] = (
                    total / (limit * history_minutes))


def _empty_arrivals(time):
    row = dict(interval_start=time.isoformat(), interval_end=(time + STEP).isoformat(),
               arriving_tasks=0, urgent_arriving_tasks=0)
    for name in RESOURCES:
        row.update({f'{name}_demand': 0., f'{name}_work_minutes_raw': 0.,
                    f'{name}_work_minutes_occupied': 0., f'{name}_pressure_15m': 0.,
                    f'{name}_pressure_15m_raw': 0.})
    return row


def summarize(workload, capacity, history_minutes=120):
    """Describe every [t,t+15min) arrival bucket, including zero-arrival buckets.

    Instant pressure = arriving occupied-resource-minutes/(capacity*15min).
    History pressure uses current + seven previous buckets for 120min; before
    the frozen cohort start the contribution is zero, with coverage recorded.
    Both raw-duration and held-resource variants are retained, without claiming
    either diagnostic is a feasible schedule or predicting future arrivals.
    """
    capacity = tuple(finite(v, 'capacity', True) for v in capacity)
    if len(capacity) != 3:
        raise ValueError('CPU/GPU/MEM capacity requires three values')
    if (isinstance(history_minutes, bool) or not isinstance(history_minutes, int)
            or history_minutes <= 0 or history_minutes % MINUTES):
        raise ValueError('history_minutes must be a positive 15-minute multiple')
    groups = {time: [] for time in interval_times(workload.start, workload.end)}
    details = []
    for spec in workload.tasks:
        if spec.arrival_time not in groups:
            raise ValueError('Task arrival outside frozen interval grid')
        groups[spec.arrival_time].append(spec)
        bound = latest_start(spec)
        solo_slack = (bound - spec.arrival_time).total_seconds() / 60
        latest_grid = bound.floor('15min')
        row = dict(task_id=spec.task_id, arrival_time=spec.arrival_time.isoformat(),
                   duration_minutes=spec.duration,
                   rounded_occupancy_minutes=occupied_minutes(spec.duration),
                   sla_deadline=spec.sla_deadline.isoformat(),
                   solo_slack_minutes=solo_slack,
                   deadline_window_minutes=(spec.sla_deadline - spec.arrival_time).total_seconds()/60,
                   relative_solo_slack=solo_slack/spec.duration,
                   latest_raw_start=bound.isoformat(), latest_grid_start=latest_grid.isoformat(),
                   latest_start_minutes_from_cohort=(bound-workload.start).total_seconds()/60,
                   usable_grid_slack_minutes=(latest_grid-spec.arrival_time).total_seconds()/60,
                   urgent_on_arrival=spec.arrival_time + STEP > bound,
                   solo_infeasible_on_arrival=bound < spec.arrival_time)
        row.update({field: getattr(spec, field) for field in DEMAND_FIELDS})
        details.append(row)
    rows = []
    for time, tasks in groups.items():
        row = _empty_arrivals(time)
        row['arriving_tasks'] = len(tasks)
        row['urgent_arriving_tasks'] = sum(s.arrival_time + STEP > latest_start(s) for s in tasks)
        for name, field, limit in zip(RESOURCES, DEMAND_FIELDS, capacity):
            row[f'{name}_demand'] = math.fsum(getattr(s, field) for s in tasks)
            raw = math.fsum(getattr(s, field)*s.duration for s in tasks)
            held = math.fsum(getattr(s, field)*occupied_minutes(s.duration) for s in tasks)
            row[f'{name}_work_minutes_raw'] = raw
            row[f'{name}_work_minutes_occupied'] = held
            row[f'{name}_pressure_15m_raw'] = raw/(limit*MINUTES)
            row[f'{name}_pressure_15m'] = held/(limit*MINUTES)
        rows.append(row)
    _rolling(rows, capacity, history_minutes)
    window_minutes = (workload.end-workload.start).total_seconds()/60
    urgent = sum(r['urgent_on_arrival'] for r in details)
    summary = dict(schema=VERSION, workload_sha256=workload.sha256,
        arrival_start=workload.start.isoformat(), arrival_end=workload.end.isoformat(),
        window_minutes=window_minutes, arrival_intervals=len(rows), tasks=len(details),
        capacity=list(capacity), history_minutes=history_minutes,
        history_cold_start='zero before arrival cohort; fixed denominator; observed coverage saved',
        duration_semantics='SLA uses raw finish; resources held for ceil(duration/15)*15 minutes',
        urgent_definition='arrival + 15 minutes > raw deadline - raw duration',
        quantile_rule='numpy linear population quantiles, zero buckets included',
        memory_unit='original workload avg_mem unit; no bytes/GiB conversion asserted',
        analysis_only=True, online_policy_input=False,
        arrivals=distribution(r['arriving_tasks'] for r in rows),
        urgent_on_arrival_tasks=urgent,
        urgent_on_arrival_fraction=urgent/len(details) if details else 0.,
        solo_infeasible_on_arrival_tasks=sum(r['solo_infeasible_on_arrival'] for r in details))
    for field in ('duration_minutes', 'rounded_occupancy_minutes', 'solo_slack_minutes',
                  'usable_grid_slack_minutes', 'relative_solo_slack',
                  'deadline_window_minutes', 'latest_start_minutes_from_cohort'):
        summary[field] = distribution(r[field] for r in details)
    for name, limit in zip(RESOURCES, capacity):
        total_raw = math.fsum(r[f'{name}_work_minutes_raw'] for r in rows)
        total_held = math.fsum(r[f'{name}_work_minutes_occupied'] for r in rows)
        summary[name] = dict(window_work_minutes_raw=total_raw,
            window_work_minutes_occupied=total_held,
            arrival_window_load_raw=total_raw/(limit*window_minutes),
            arrival_window_load=total_held/(limit*window_minutes),
            demand_per_arrival_interval=distribution(r[f'{name}_demand'] for r in rows),
            work_minutes_per_arrival_interval_raw=distribution(r[f'{name}_work_minutes_raw'] for r in rows),
            work_minutes_per_arrival_interval_occupied=distribution(r[f'{name}_work_minutes_occupied'] for r in rows),
            pressure_15m=distribution(r[f'{name}_pressure_15m'] for r in rows),
            pressure_history=distribution(r[f'{name}_pressure_{history_minutes}m'] for r in rows),
            arrival_intervals_pressure_above_one=sum(r[f'{name}_pressure_15m'] > 1 for r in rows),
            history_intervals_pressure_above_one=sum(r[f'{name}_pressure_{history_minutes}m'] > 1 for r in rows))
    return PressureAnalysis(summary, tuple(rows), tuple(details))


def align_edf(analysis, steps, ledger, policy_steps):
    """Join sealed actual EDF interval/event logs; reconstruct only known times.

    `new_latest_start_misses_known` means the first observed grid boundary at
    which an unstarted task cannot meet its raw deadline even if started now.
    It is distinct from misses recorded when a completed task releases resources.
    Terminal-boundary releases get their own row, never moved into prior energy.
    This offline join uses final ledger to locate events; no output feeds PPO.
    """
    steps, ledger, policy_steps = tuple(steps), tuple(ledger), tuple(policy_steps)
    specs = {r['task_id']: r for r in analysis.task_details}
    tasks = {r['task_id']: r for r in ledger}
    if len(tasks) != len(ledger) or set(tasks) != set(specs):
        raise ValueError('EDF ledger task identities do not match frozen workload')
    if not steps or len(steps) != len(policy_steps):
        raise ValueError('EDF interval/event counts mismatch')
    capacity = tuple(analysis.summary['capacity'])
    all_ids = set(specs)
    times = tuple(utc_time(r['interval_start'], True) for r in steps)
    start = utc_time(analysis.summary['arrival_start'], True)
    endpoint = utc_time(steps[-1]['interval_end'], True)
    if times != interval_times(start, endpoint):
        raise ValueError('EDF intervals must be continuous from arrival cohort start')
    counters = {name: Counter() for name in ('completed', 'completed_miss', 'known_solo_miss')}
    for task_id, row in tasks.items():
        spec = specs[task_id]
        for field in ('arrival_time', 'sla_deadline', *DEMAND_FIELDS):
            if row[field] != spec[field]:
                raise ValueError('EDF task ledger modified frozen task fields')
        if row['duration'] != spec['duration_minutes']:
            raise ValueError('EDF task ledger modified frozen duration')
        arrival = utc_time(row['arrival_time'], True)
        bound = utc_time(spec['latest_raw_start'])
        known = max(arrival, bound.floor('15min') + STEP)
        if bound < arrival:
            known = arrival
        started = utc_time(row['start_time'], True) if row['start_time'] is not None else None
        if known <= endpoint and (started is None or started >= known):
            counters['known_solo_miss'][known] += 1
        if row['state'] == 'completed':
            observed = utc_time(row['completion_observed_at'], True)
            if not start <= observed <= endpoint:
                raise ValueError('Completion boundary outside observed EDF window')
            counters['completed'][observed] += 1
            raw_met = utc_time(row['finish_time']) <= utc_time(row['sla_deadline'])
            if row['deadline_outcome'] != ('met' if raw_met else 'missed'):
                raise ValueError('EDF deadline classification disagrees with raw finish')
            if not raw_met:
                counters['completed_miss'][observed] += 1
    arrival_lookup = {utc_time(r['interval_start'], True): r for r in analysis.arrivals}
    rows = []
    for time, physical, detail in zip(times, steps, policy_steps):
        if utc_time(detail['time'], True) != time:
            raise ValueError('EDF policy events have mismatched interval time')
        for field in ('decision_pending_ids', 'ordered_waiting_ids', 'policy_deferred_ids',
                      'capacity_waiting_ids', 'started_ids', 'pending_end_ids', 'running_ids'):
            ids = detail[field]
            if len(ids) != len(set(ids)) or not set(ids) <= all_ids:
                raise ValueError('EDF events contain duplicate or unknown task IDs')
        ordered_ids = set(detail['ordered_waiting_ids'])
        decisions = detail['started_ids'] + detail['policy_deferred_ids'] + detail['capacity_waiting_ids']
        if (ordered_ids != set(detail['decision_pending_ids'])
                or len(decisions) != len(set(decisions)) or set(decisions) != ordered_ids):
            raise ValueError('EDF candidate events do not form a partition of waiting tasks')
        if set(detail['pending_end_ids']) != ordered_ids - set(detail['started_ids']):
            raise ValueError('EDF candidate decisions do not match pending queue')
        if set(detail['pending_end_ids']) & set(detail['running_ids']):
            raise ValueError('EDF pending/running ownership overlap')
        for name, field, limit in zip(RESOURCES, DEMAND_FIELDS, capacity):
            allocated, free = physical[f'used_{name}'], physical[f'available_{name}']
            if allocated > limit or free < -1e-9 or not math.isclose(allocated+free, limit, abs_tol=1e-9):
                raise ValueError('EDF resource capacity conservation failed')
            actual = math.fsum(specs[i][field] for i in detail['running_ids'])
            if (not math.isclose(actual, allocated, rel_tol=1e-12, abs_tol=1e-9)
                    or not math.isclose(allocated/limit, physical[f'{name}_fraction'], abs_tol=1e-12)):
                raise ValueError('EDF running task occupancy disagrees with resource logs')
        if physical['pending'] != len(detail['pending_end_ids']) or physical['running'] != len(detail['running_ids']):
            raise ValueError('EDF queue/event sizes disagree')
        row = dict(arrival_lookup.get(time, _empty_arrivals(time)))
        row.update({key: physical[key] for key in ('pending', 'running', 'ci_g_per_kwh',
                    'carbon_kg', 'it_energy_kwh', 'hvac_energy_kwh', 'total_energy_kwh')})
        row.update({f'{name}_fraction': physical[f'{name}_fraction'] for name in RESOURCES})
        pending = [specs[i] for i in detail['decision_pending_ids']]
        row.update(decision_pending_tasks=len(pending),
                   urgent_pending_before_decision=sum(time + STEP > utc_time(s['latest_raw_start']) for s in pending),
                   past_latest_start_pending_before_decision=sum(time > utc_time(s['latest_raw_start']) for s in pending),
                   started_tasks=len(detail['started_ids']),
                   capacity_waiting_tasks=len(detail['capacity_waiting_ids']),
                   policy_deferred_tasks=len(detail['policy_deferred_ids']),
                   new_completed_observed=counters['completed'][time],
                   new_completed_misses_observed=counters['completed_miss'][time],
                   new_latest_start_misses_known=counters['known_solo_miss'][time])
        rows.append(row)
    _rolling(rows, capacity, analysis.summary['history_minutes'])
    terminal = dict(time=endpoint.isoformat(), new_completed_observed=counters['completed'][endpoint],
        new_completed_misses_observed=counters['completed_miss'][endpoint],
        new_latest_start_misses_known=counters['known_solo_miss'][endpoint])
    summary = dict(arrived=len(tasks), completed=sum(r['state']=='completed' for r in ledger),
        completed_misses=sum(r['deadline_outcome']=='missed' for r in ledger),
        unfinished_misses=sum(r['deadline_outcome']=='missed_unfinished' for r in ledger),
        right_censored=sum(r['deadline_outcome']=='right_censored' for r in ledger),
        capacity_wait_intervals=sum(r['capacity_waiting_tasks'] for r in rows),
        policy_defer_intervals=sum(r['policy_deferred_tasks'] for r in rows),
        pending=distribution(r['pending'] for r in rows),
        capacity_waiting=distribution(r['capacity_waiting_tasks'] for r in rows))
    for name in RESOURCES:
        summary[name+'_fraction'] = distribution(r[name+'_fraction'] for r in rows)
        summary[name+'_near_saturated_intervals'] = sum(r[name+'_fraction'] >= .99 for r in rows)
    return dict(series=tuple(rows), terminal_boundary=terminal, summary=summary,
        audit=dict(all_task_ids_match=True, resource_capacity_conserved=True,
                   intervals_continuous=True, no_new_drain_arrivals=all(
                       r['arriving_tasks']==0 for r in rows
                       if utc_time(r['interval_start']) >= utc_time(analysis.summary['arrival_end']))),
        event_definitions=dict(new_completed_misses_observed='missed raw finish, observed resource-release boundary',
            new_latest_start_misses_known='first grid boundary when unstarted solo deadline feasibility is lost; offline reconstructed'))


def write_csv(path, rows):
    rows = tuple(rows)
    if not rows:
        raise ValueError('CSV rows may not be empty')
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'validation'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Pressure output already exists; historical outputs are never overwritten')
    from .rl_training_utils import load_bundle
    bundle = load_bundle(args.bundle, allowed_split=args.split)
    result = summarize(bundle.workload, bundle.capacity)
    args.output.mkdir(parents=True)
    write_json(args.output/'pressure-summary.json', result.summary)
    write_csv(args.output/'arrival-pressure.csv', result.arrivals)
    # Complete task-level diagnostics stay local; not part of curated publication.
    if result.task_details:
        write_csv(args.output/'task-pressure-local.csv', result.task_details)
    return result


if __name__ == '__main__':
    main()
