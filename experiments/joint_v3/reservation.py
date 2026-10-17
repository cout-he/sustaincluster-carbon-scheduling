"""Bounded, causal nonpreemptive EDF backfill over a known queue snapshot.

The candidate is started before continuing the queue in the start trajectory.
In a wait trajectory it is ineligible for ``delay_slots`` grid intervals, then
joins the same EDF continuation. Existing within-slot wait commitments are
preserved in both trajectories. Physical finishes retain fractional minutes;
resources remain held through the next 15-minute executor release boundary.
"""
import math
import time
from dataclasses import dataclass
from itertools import islice

import numpy as np
import pandas as pd

from experiments.e1.contracts import STEP, finite, utc_time


@dataclass(frozen=True)
class Schedule:
    valid: bool
    reason: str
    starts: dict
    finishes: dict
    loads: np.ndarray
    settled: bool
    capacity: tuple = ()
    unresolved_task_ids: tuple = ()
    horizon_minutes: int = 0
    unresolved_duration_minutes: object = None


def _positive_bound(value, name, upper):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 < value <= upper:
        raise ValueError(f'{name} must be a positive integer at most {upper}')
    return int(value)


def _bounded(items, maximum):
    """Bound consumption too, so an oversized iterable cannot exhaust memory."""
    return tuple(islice(iter(items), maximum + 1))


def _resources(values, name, positive=False):
    result = tuple(finite(value, name, positive) for value in values)
    if len(result) != 3:
        raise ValueError(f'{name} requires CPU/GPU/MEM')
    return result


def _job_key(job):
    return (job.sla_deadline, job.arrival_time, job.task_id)


def _task_signature(job):
    return (job.task_id, utc_time(job.arrival_time).value, float(job.duration),
            float(job.cores_req), float(job.gpu_req), float(job.mem_req),
            utc_time(job.sla_deadline).value)


def _commitments(blocked_until):
    pairs = tuple(blocked_until.items()) if hasattr(blocked_until, 'items') else tuple(blocked_until)
    result = {}
    for task_id, until in pairs:
        if task_id in result:
            raise ValueError('duplicate wait commitment')
        result[task_id] = utc_time(until, True)
    return result


def schedule_known(candidate, now, capacity, pending, running, delay_slots=0,
                   max_tasks=256, max_slots=192, *, blocked_until=(), _deadline=None):
    """Schedule every known job or return an explicitly invalid bounded result.

    No unseen arrivals or environmental data are accepted. ``loads`` contains
    raw CPU/GPU/MEM totals, including running jobs, under left-boundary hold.
    ``starts`` for pre-existing running jobs denotes the snapshot boundary
    unless the input supplies its actual ``start_time``. An invalid plan must
    never be used as evidence of safety or as a complete carbon trajectory.
    """
    starts, finishes, release_slots = {}, {}, {}
    known_ids = []
    loads = np.zeros((0, 3), dtype=float)
    capacity_values = ()
    horizon = 0
    occupancy_minutes, fully_validated = {}, False

    def result(valid, reason, settled=False, stop=None):
        profile = loads if stop is None else loads[:stop]
        profile.setflags(write=False)
        unresolved = (() if settled else tuple(sorted(task_id for task_id in known_ids
                       if task_id not in release_slots or release_slots[task_id] > horizon)))
        tail_minutes = None
        if fully_validated:
            tail_minutes = math.fsum(
                max(0, release_slots[task_id] - horizon) * 15 if task_id in release_slots
                else occupancy_minutes[task_id] for task_id in unresolved)
        return Schedule(valid, reason, dict(starts), dict(finishes), profile,
                        settled, capacity_values, unresolved, horizon * 15, tail_minutes)

    try:
        max_tasks = _positive_bound(max_tasks, 'max_tasks', 4096)
        horizon = _positive_bound(max_slots, 'max_slots', 4096)
        if isinstance(delay_slots, bool) or not isinstance(delay_slots, (int, np.integer)) or delay_slots < 0:
            raise ValueError('delay_slots must be a nonnegative integer')
        now = utc_time(now, True)
        capacity_values = _resources(capacity, 'capacity', True)
        waiting_input, running = _bounded(pending, max_tasks), _bounded(running, max_tasks)
        known_ids = [candidate.task_id] + [job.task_id for job in waiting_input
                                         if job.task_id != candidate.task_id]
        known_ids.extend(job.task_id for job in running)
        candidate_copies = [job for job in waiting_input if job.task_id == candidate.task_id]
        if (len(waiting_input) > max_tasks or len(running) > max_tasks or
                len(known_ids) > max_tasks):
            return result(False, 'queue_limit_exceeded')
        if len(candidate_copies) > 1 or len(known_ids) != len(set(known_ids)):
            return result(False, 'duplicate_known_task_id')
        # Check arrival metadata before inspecting the rest of any queued job.
        if any(utc_time(job.arrival_time) > now for job in (candidate,) + waiting_input):
            return result(False, 'future_arrival_rejected')
        if candidate_copies and _task_signature(candidate_copies[0]) != _task_signature(candidate):
            return result(False, 'candidate_identity_mismatch')
        waiting = [candidate] + [job for job in waiting_input if job.task_id != candidate.task_id]
        commitments = _commitments(blocked_until)
        waiting_ids = {job.task_id for job in waiting}
        if not set(commitments) <= waiting_ids:
            return result(False, 'unknown_wait_commitment')
        if commitments.get(candidate.task_id, now) > now and delay_slots == 0:
            return result(False, 'candidate_already_deferred')
        demands, eligibility = {}, {}
        for job in waiting:
            finite(job.duration, 'duration', True)
            occupancy_minutes[job.task_id] = math.ceil(job.duration / 15) * 15
            utc_time(job.sla_deadline)
            demands[job.task_id] = _resources((job.cores_req, job.gpu_req, job.mem_req), 'demand')
            if any(value > limit for value, limit in zip(demands[job.task_id], capacity_values)):
                return result(False, 'task_exceeds_capacity')
            eligibility[job.task_id] = max(now, commitments.get(job.task_id, now))
        eligibility[candidate.task_id] = max(eligibility[candidate.task_id], now + STEP * int(delay_slots))
        loads = np.zeros((horizon, 3), dtype=float)
        for job in running:
            task_id = job.task_id
            demand = _resources(job.demand, 'running demand')
            physical_finish = utc_time(getattr(job, 'finish_time', job.release_time))
            release = max(utc_time(job.release_time, True), physical_finish.ceil('15min'))
            release_slot = max(0, int((release - now) / STEP))
            starts[task_id] = utc_time(getattr(job, 'start_time', now))
            finishes[task_id], release_slots[task_id] = physical_finish, release_slot
            if release_slot:
                loads[:min(horizon, release_slot)] += demand
        if np.any(loads > np.asarray(capacity_values)):
            return result(False, 'running_capacity_violation')
        fully_validated = True

        def fits(job, slot):
            return all(math.fsum((loads[slot, index], value)) <= capacity_values[index]
                       for index, value in enumerate(demands[job.task_id]))

        def start(job, slot):
            boundary = now + slot * STEP
            finish = boundary + pd.Timedelta(minutes=job.duration)
            release_slot = int((finish.ceil('15min') - now) / STEP)
            starts[job.task_id], finishes[job.task_id] = boundary, finish
            release_slots[job.task_id] = release_slot
            loads[slot:min(horizon, release_slot)] += demands[job.task_id]

        if _deadline is not None and time.perf_counter() >= _deadline:
            return result(False, 'time_budget_exceeded')
        if delay_slots == 0:
            if not fits(candidate, 0):
                return result(False, 'candidate_start_capacity_unavailable')
            start(candidate, 0)
            waiting.remove(candidate)
        waiting.sort(key=_job_key)
        for slot in range(horizon):
            if _deadline is not None and time.perf_counter() >= _deadline:
                return result(False, 'time_budget_exceeded')
            boundary = now + STEP * slot
            remaining = []
            for job in waiting:
                if eligibility[job.task_id] <= boundary and fits(job, slot):
                    start(job, slot)
                else:
                    remaining.append(job)
            waiting = remaining
            if not waiting:
                final_slot = max(release_slots.values(), default=0)
                if final_slot <= horizon:
                    return result(True, 'known_queue_settled', True, final_slot)
                return result(False, 'planning_horizon_exceeded')
        return result(False, 'planning_horizon_exceeded')
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
        return result(False, f'invalid_snapshot:{exc}')
