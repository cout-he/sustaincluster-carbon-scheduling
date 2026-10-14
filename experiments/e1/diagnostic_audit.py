"""Full-workload/energy identity checks for the new offline diagnostic drivers."""
from dataclasses import asdict
import math

import pandas as pd

from .contracts import content_hash
from .counterfactual_audit import audit_result


def core_scientific_hash(result):
    """Real ledger/events/resources/power, excluding policy annotation changes."""
    return content_hash(dict(tasks=result.tasks, events=result.events, steps=result.steps,
                             summary=result.summary, policy_tasks=result.policy_tasks))


def full_scientific_hash(result):
    return content_hash(asdict(result))


def audit_fair_results(results, workload, environment, capacity):
    if not results:
        raise ValueError('At least one real result is required')
    windows = {tuple((step['interval_start'], step['interval_end']) for step in result.steps)
               for result in results.values()}
    if len(windows) != 1:
        raise RuntimeError('All paired results require the same common observation window')
    audited = {name: audit_result(result, workload, environment, capacity)
               for name, result in results.items()}
    hours = [tuple(math.fsum(step['used_'+key]*step['interval_hours'] for step in result.steps)
                   for key in ('cores','gpus','mem')) for result in results.values()]
    if any(not math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-9)
           for value in hours[1:] for first, second in zip(hours[0], value)):
        raise RuntimeError('Mandatory task occupied resource hours differ')
    return dict(passed=True, equal_resource_hours=True, methods=audited,
                resource_hours_by_method=dict(zip(results, hours)),
                all_tasks_completed=True, same_common_window=True,
                full_scientific_hashes={name: full_scientific_hash(result) for name,result in results.items()})


def diagnostic_summary(result, arrival_end):
    rows, steps = result.tasks, result.steps
    if not rows or not steps:
        raise ValueError('Summary requires actual arrived work and accounted intervals')
    n = len(rows)
    completed = sum(row['state']=='completed' for row in rows)
    misses = sum(row['deadline_outcome'] in ('missed','missed_unfinished') for row in rows)
    def mean(values):
        return math.fsum(values)/len(values) if values else None
    def minutes(end, start):
        return (pd.Timestamp(end)-pd.Timestamp(start)).total_seconds()/60
    summary = dict(arrived=n, completed=completed, unfinished=n-completed,
        censored=sum(row['deadline_outcome']=='right_censored' for row in rows), misses=misses, dmr=misses/n,
        mean_jct_minutes=mean([minutes(row['finish_time'],row['arrival_time']) for row in rows
                              if row['state']=='completed']),
        mean_wait_minutes=mean([minutes(row['start_time'],row['arrival_time']) for row in rows
                               if row['start_time'] is not None]),
        pending_mean=mean([step['pending'] for step in steps]),
        pending_max=max(step['pending'] for step in steps),
        policy_wait_intervals=sum(row['policy_defer_intervals'] for row in result.policy_tasks),
        capacity_wait_intervals=sum(row['capacity_wait_intervals'] for row in result.policy_tasks),
        common_end=steps[-1]['interval_end'], drained_at=result.summary.get('drained_at'),
        drain_hours=minutes(result.summary['drained_at'], arrival_end)/60
                    if result.summary.get('drained_at') else None)
    for key in ('it_energy_kwh','hvac_energy_kwh','total_energy_kwh','carbon_kg'):
        summary[key]=math.fsum(step[key] for step in steps)
    for resource in ('cores','gpus','mem'):
        summary[resource+'_hours']=math.fsum(step['used_'+resource]*step['interval_hours'] for step in steps)
        summary[resource+'_fraction_mean']=mean([step[resource+'_fraction'] for step in steps])
        summary[resource+'_fraction_max']=max(step[resource+'_fraction'] for step in steps)
    return summary
