"""Outcome-blind eligible events from an already frozen Train EDF reference.

Full reference replay is an offline sampling tool. Current event features select
cases; realized task finishes, misses, future CI and branch outcomes are unused.
"""
from collections import Counter
import math

import pandas as pd

from .cars_state import demand
from .contracts import utc_time
from .task_risk_state import latest_grid_start

VERSION = 'e1-7a-reference-candidates-v1'
RELEASE_SCREEN_VERSION = 'e1-7a-known-release-screen-v1'


def known_release_screen(spec, now, capacity, running, predecessors, horizon_minutes=120):
    """Single known predecessor at its latest grid start, under two phases.

    This is an online-observable time conflict screen, not a reservation system
    or a sufficient feasibility test. It ignores allocation among other pending
    tasks and all unknown arrivals. Existing running tasks retain grid releases.
    No screen result feeds the intervention decision or the EDF continuation.
    """
    now = utc_time(now, True)
    predecessors, running = tuple(predecessors), tuple(running)
    if spec.arrival_time > now or any(item.arrival_time > now for item in predecessors):
        raise ValueError('Cannot screen future task')
    capacity = tuple(capacity)
    if len(capacity) != 3 or any(not math.isfinite(value) or value <= 0 for value in capacity):
        raise ValueError('Screen requires positive CPU/GPU/MEM capacity')
    if not math.isfinite(horizon_minutes) or horizon_minutes <= 0:
        raise ValueError('Screen horizon must be positive')
    key = lambda item: (item.sla_deadline, item.arrival_time, item.task_id)
    chosen = sorted((item for item in predecessors if item.task_id != spec.task_id
        and key(item) < key(spec) and now <= latest_grid_start(item) <= now + pd.Timedelta(minutes=horizon_minutes)), key=key)
    if len({item.task_id for item in chosen}) != len(chosen):
        raise ValueError('Duplicate known screen predecessor')
    conflicts = {}
    for name, start in (('now', now), ('defer', now + pd.Timedelta(minutes=15))):
        release = (start + pd.Timedelta(minutes=spec.duration)).ceil('15min')
        blocked = []
        for predecessor in chosen:
            slot = latest_grid_start(predecessor)
            active = [item.demand for item in running if slot < item.release_time]
            if start <= slot < release:
                active.append(demand(spec))
            active.append(demand(predecessor))
            if any(math.fsum(requirements[index] for requirements in active) > limit
                   for index, limit in enumerate(capacity)):
                blocked.append(predecessor.task_id)
        conflicts[name] = blocked
    return dict(known_release_screen_version=RELEASE_SCREEN_VERSION,
        known_release_screen_checked_pred_count=len(chosen),
        known_release_screen_conflicts_now=len(conflicts['now']),
        known_release_screen_conflicts_defer=len(conflicts['defer']),
        known_release_screen_conflict_delta=len(conflicts['defer']) - len(conflicts['now']),
        known_release_screen_conflict_ids_now=conflicts['now'],
        known_release_screen_conflict_ids_defer=conflicts['defer'],
        existing_reservations=0, reservation_conflict_assessed=False,
        known_release_screen_horizon_minutes=horizon_minutes)


def candidate_records(workload, reference, capacity):
    """Return every current can-fit EDF event with >=15min grid solo slack.

    The caller must freeze deterministic selection criteria before running any
    paired outcomes. Sorting is time, EDF rank and stable task ID. This helper
    does not access task-ledger outcomes or any future environmental samples.
    """
    specs = {spec.task_id: spec for spec in workload.tasks}
    occurrences, records = Counter(), []
    for event in reference.policy_events:
        task_id, now = event['task_id'], utc_time(event['time'], True)
        if task_id not in specs or specs[task_id].arrival_time > now:
            raise ValueError('Reference candidate absent or before arrival')
        if event.get('policy') != 'edf_backfill':
            raise ValueError('Candidate sampling requires the fixed EDF reference')
        if not event['can_fit_at_decision']:
            continue
        if event['event'] != 'started':
            raise ValueError('Can-fit EDF candidate must start in the reference')
        occurrences[now, task_id] += 1
        spec = specs[task_id]
        slack = (latest_grid_start(spec) - now).total_seconds() / 60
        if slack < 15:
            continue
        available = tuple(float(event['available_before'][key]) for key in ('cores', 'gpus', 'mem'))
        if any(not math.isfinite(value) or value < 0 or value > total for value, total in zip(available, capacity)):
            raise ValueError('Invalid candidate available-resource record')
        records.append(dict(time=now.isoformat(), task_id=task_id,
            candidate_occurrence=occurrences[now, task_id], decision_rank=event['decision_rank'],
            ci_g_per_kwh=event['ci_g_per_kwh'], available=list(available),
            utilization=[1 - value / total for value, total in zip(available, capacity)],
            demand=[spec.cores_req, spec.gpu_req, spec.mem_req], duration_minutes=spec.duration,
            solo_slack_minutes=slack, deadline=spec.sla_deadline.isoformat()))
    return sorted(records, key=lambda row: (row['time'], row['decision_rank'], row['task_id']))
