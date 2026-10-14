"""Pre-outcome deterministic selection from authorized Train EDF observations."""
from collections import defaultdict
import math

from .cars_state import ArrivalHistory
from .contracts import utc_time
from .policies import latest_start

VERSION = 'e1-7a-selection-v1'
A_CATEGORIES = ('cpu_high', 'gpu_high', 'queue_long', 'low_pressure')
C_CATEGORIES = ('gpu_tight', 'cpu_tight', 'history_high', 'history_low',
                'short_slack', 'long_slack')


def build_selection_records(workload, capacity, policy_events, policy_steps, *, split='train'):
    """Only current/past fields inform selection; future outcomes are not consumed.

    Selection is offline on a pre-existing EDF reference, not an online policy.
    A snapshot is before the first candidate; C uses the exact candidate turn.
    """
    if split != 'train':
        raise ValueError('E1-7A selection permits Train only')
    capacity = tuple(capacity)
    if len(capacity) != 3 or any(not math.isfinite(x) or x <= 0 for x in capacity):
        raise ValueError('CPU/GPU/MEM positive capacity required')
    specs = {spec.task_id: spec for spec in workload.tasks}
    incoming, grouped = defaultdict(list), defaultdict(list)
    for spec in workload.tasks:
        incoming[spec.arrival_time].append(spec)
    for event in policy_events:
        now = utc_time(event['time'], True)
        if workload.start <= now < workload.end:
            grouped[now].append(event)
    history = ArrivalHistory(workload.start, capacity)
    snapshots, candidates = [], []
    for step in sorted(policy_steps, key=lambda row: row['time']):
        now = utc_time(step['time'], True)
        if not workload.start <= now < workload.end:
            continue
        history.observe(now, incoming.get(now, ()))
        pressure = history.pressure(now)
        pending = tuple(step['decision_pending_ids'])
        if len(set(pending)) != len(pending) or any(key not in specs for key in pending):
            raise ValueError('Unknown or duplicate pending task')
        if any(specs[key].arrival_time > now for key in pending):
            raise ValueError('Reference exposes a future pending task')
        events = sorted(grouped.get(now, ()), key=lambda row: row['decision_rank'])
        common = dict(time=now.isoformat(), pending_count=len(pending),
                      old_pending_count=sum(specs[key].arrival_time < now for key in pending),
                      history_cpu_rho=pressure.rho_resources[0],
                      history_gpu_rho=pressure.rho_resources[1],
                      history_mem_rho=pressure.rho_resources[2],
                      history_rho=pressure.rho, history_buffer_steps=pressure.buffer_steps,
                      history_effective_minutes=pressure.effective_minutes,
                      history_observed_tasks=pressure.observed_tasks)
        if events:
            free = tuple(events[0]['available_before'][key] for key in ('cores', 'gpus', 'mem'))
            utilization = tuple(1 - value / total for value, total in zip(free, capacity))
            snapshots.append(dict(common, pending_ids=list(pending),
                cpu_utilization=utilization[0], gpu_utilization=utilization[1],
                mem_utilization=utilization[2], max_utilization=max(utilization)))
        for event in events:
            if not event['can_fit_at_decision'] or event['event'] != 'started':
                continue
            spec = specs[event['task_id']]
            if spec.arrival_time > now or spec.task_id not in pending:
                raise ValueError('Candidate not currently known')
            slack = (latest_start(spec).floor('15min') - now).total_seconds() / 60
            if slack < 15:
                continue
            free = tuple(event['available_before'][key] for key in ('cores', 'gpus', 'mem'))
            utilization = tuple(1 - value / total for value, total in zip(free, capacity))
            candidates.append(dict(common, task_id=spec.task_id, candidate_occurrence=1,
                decision_rank=event['decision_rank'], ci_g_per_kwh=event['ci_g_per_kwh'],
                cpu_utilization=utilization[0], gpu_utilization=utilization[1],
                mem_utilization=utilization[2], max_utilization=max(utilization),
                solo_slack_minutes=slack, duration_minutes=spec.duration,
                cpu_demand=spec.cores_req, gpu_demand=spec.gpu_req, mem_demand=spec.mem_req))
    return dict(snapshots=snapshots, candidates=candidates)


def select_diagnostic_cases(records, *, min_tasks=10, max_tasks=40):
    """One snapshot/class; six non-repeated candidates/class for a single factor.

    Run independently on each frozen factor. Missing classes are final evidence,
    never replaced after solver/counterfactual outcomes become available.
    """
    if not 1 <= min_tasks <= max_tasks <= 50:
        raise ValueError('Invalid predeclared finite-case size')
    snapshots = [row for row in records['snapshots'] if row['pending_count'] >= min_tasks]
    a_rules = {
        'cpu_high': (lambda r: r['cpu_utilization'] >= .8, lambda r: -r['cpu_utilization']),
        'gpu_high': (lambda r: r['gpu_utilization'] >= .8, lambda r: -r['gpu_utilization']),
        'queue_long': (lambda r: r['old_pending_count'] >= 10, lambda r: -r['old_pending_count']),
        'low_pressure': (lambda r: r['max_utilization'] <= .5 and r['old_pending_count'] == 0,
                         lambda r: r['max_utilization']),
    }
    c_rules = {
        'gpu_tight': (lambda r: r['gpu_utilization'] >= .8, lambda r: -r['gpu_utilization']),
        'cpu_tight': (lambda r: r['cpu_utilization'] >= .8, lambda r: -r['cpu_utilization']),
        'history_high': (lambda r: r['history_rho'] is not None and r['history_rho'] >= 1,
                         lambda r: -r['history_rho']),
        'history_low': (lambda r: r['history_rho'] is not None and r['history_rho'] <= .5,
                        lambda r: r['history_rho']),
        'short_slack': (lambda r: 15 <= r['solo_slack_minutes'] <= 30,
                        lambda r: r['solo_slack_minutes']),
        'long_slack': (lambda r: r['solo_slack_minutes'] >= 60,
                       lambda r: -r['solo_slack_minutes']),
    }
    selected_a, selected_c, missing, used_times, used_ids = [], [], [], set(), set()
    # Copy only registered explanatory features, never arbitrary outcome columns.
    a_fields = ('time', 'pending_count', 'old_pending_count', 'pending_ids', 'cpu_utilization',
                'gpu_utilization', 'mem_utilization', 'max_utilization', 'history_cpu_rho',
                'history_gpu_rho', 'history_mem_rho', 'history_rho', 'history_buffer_steps',
                'history_effective_minutes', 'history_observed_tasks')
    c_fields = tuple(key for key in a_fields if key != 'pending_ids') + (
        'task_id', 'candidate_occurrence', 'decision_rank', 'ci_g_per_kwh',
        'solo_slack_minutes', 'duration_minutes', 'cpu_demand', 'gpu_demand', 'mem_demand')
    for category, (eligible, score) in a_rules.items():
        pool = sorted((row for row in snapshots if eligible(row) and row['time'] not in used_times),
                      key=lambda row: (score(row), row['time']))
        if not pool:
            missing.append(dict(experiment='A', category=category, reason='no eligible distinct snapshot'))
            continue
        row = pool[0]
        used_times.add(row['time'])
        selected_a.append(dict({key: row[key] for key in a_fields}, category=category,
            eligible_count=len(pool), selected_task_count=min(max_tasks, row['pending_count'])))
    for category, (eligible, score) in c_rules.items():
        pool = sorted((row for row in records['candidates'] if eligible(row) and row['task_id'] not in used_ids),
                      key=lambda row: (score(row), row['time'], row['task_id'], row['decision_rank']))
        if not pool:
            missing.append(dict(experiment='C', category=category, reason='no eligible distinct can-fit task'))
            continue
        row = pool[0]
        used_ids.add(row['task_id'])
        selected_c.append(dict({key: row[key] for key in c_fields}, category=category, eligible_count=len(pool)))
    return dict(A=selected_a, C=selected_c, missing=missing)
