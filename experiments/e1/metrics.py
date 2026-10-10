"""Explicit kW -> kWh and gCO2e/kWh -> kgCO2e integration."""
import math
from collections import Counter

from .contracts import STEP


def interval_record(time, sample, power, capacity, used, pending, running):
    hours = STEP.total_seconds() / 3600
    row = dict(interval_start=time.isoformat(), interval_end=(time + STEP).isoformat(),
               interval_hours=hours, pending=pending, running=running,
               ci_g_per_kwh=sample.ci_g_per_kwh, temperature_c=sample.temperature_c,
               wet_bulb_c=sample.wet_bulb_c, it_power_kw=power.it_kw,
               hvac_power_kw=power.hvac_kw, total_power_kw=power.total_kw,
               it_energy_kwh=power.it_kw * hours, hvac_energy_kwh=power.hvac_kw * hours,
               total_energy_kwh=power.total_kw * hours,
               carbon_kg=power.total_kw * hours * sample.ci_g_per_kwh / 1000,
               transmission_energy_kwh=0.0)
    for name, demand, limit in zip(('cores', 'gpus', 'mem'), used, capacity):
        row[f'used_{name}'] = demand
        row[f'available_{name}'] = limit - demand
        row[f'{name}_fraction'] = demand / limit
    return row


def summarize(tasks, steps):
    states = Counter(t['state'] for t in tasks)
    outcomes = Counter(t['deadline_outcome'] for t in tasks)
    result = dict(arrived=len(tasks), completed=states['completed'],
                  pending=states['pending'], running=states['running'],
                  structurally_infeasible=sum(t['structurally_infeasible'] for t in tasks),
                  deadline_met=outcomes['met'], deadline_missed_completed=outcomes['missed'],
                  deadline_missed_unfinished=outcomes['missed_unfinished'],
                  deadline_right_censored=outcomes['right_censored'], intervals=len(steps),
                  interval_load='left_boundary_hold', horizon='truncate',
                  carbon_unit='kgCO2e', energy_unit='kWh', power_unit='kW')
    for name in ('it_energy_kwh', 'hvac_energy_kwh', 'total_energy_kwh', 'carbon_kg',
                 'transmission_energy_kwh'):
        result[name] = math.fsum(s[name] for s in steps)
    return result


def summarize_policy(policy, tasks, policy_steps, steps):
    """Additional diagnostics: preserve the E1-1A summary schema and numbers."""
    from .policies import POLICY_SCHEMA
    result = dict(policy_schema=POLICY_SCHEMA, policy=policy.name,
                  threshold_g_per_kwh=getattr(policy, 'threshold_g_per_kwh', None),
                  waiting_attribution='capacity_first_per_task_turn',
                  policy_defer_intervals=sum(t['policy_defer_intervals'] for t in tasks),
                  capacity_wait_intervals=sum(t['capacity_wait_intervals'] for t in tasks),
                  policy_deferred_tasks=sum(t['policy_defer_intervals'] > 0 for t in tasks),
                  capacity_waited_tasks=sum(t['capacity_wait_intervals'] > 0 for t in tasks),
                  observed_pending_task_minutes=math.fsum(t['observed_pending_minutes'] for t in tasks))
    for name, field in (('pending', 'pending_end_ids'), ('policy_deferred', 'policy_deferred_ids'),
                        ('capacity_waiting', 'capacity_waiting_ids')):
        sizes = [len(s[field]) for s in policy_steps]
        result[f'{name}_mean'] = math.fsum(sizes) / len(sizes)
        result[f'{name}_max'] = max(sizes)
    for resource in ('cores', 'gpus', 'mem'):
        values = [s[f'{resource}_fraction'] for s in steps]
        result[f'{resource}_fraction_mean'] = math.fsum(values) / len(values)
        result[f'{resource}_fraction_max'] = max(values)
    return result
