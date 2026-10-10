"""Bounded, nonpreemptive known-task EDF counterfactuals on read-only state.

This is a scheduling heuristic, not an admission proof or an SLA guarantee.
Only already arrived pending work and current running releases are inspected.
"""
import math
from dataclasses import dataclass

import pandas as pd

from .contracts import STEP

_STEP_NS = STEP.value


@dataclass(frozen=True)
class PlannerResult:
    status: str
    missed_now: tuple
    missed_defer: tuple
    unknown_now: tuple
    unknown_defer: tuple
    candidate_start_now: object
    candidate_start_defer: object
    checked_tasks: int
    horizon_end: object
    reason: str
    planner_calls: int = 2


@dataclass(frozen=True)
class _Job:
    task_id: str
    demand: tuple
    duration_ns: int
    deadline_ns: int
    arrival_ns: int

    @property
    def latest_start_ns(self):
        return self.deadline_ns - self.duration_ns


@dataclass(frozen=True)
class _Prepared:
    now_ns: int
    end_ns: int
    capacity: tuple
    jobs: tuple
    running: tuple
    blocked: dict
    running_missed: tuple


def _release_ns(finish_ns):
    return ((finish_ns + _STEP_NS - 1) // _STEP_NS) * _STEP_NS


def _timestamp(time_ns):
    return pd.Timestamp(time_ns, tz='UTC') if time_ns is not None else None


def _used(active):
    return tuple(math.fsum(demand[d] for _, demand in active) for d in range(3))


def _fits(demand, used, active, capacity):
    # Usually the precomputed sums are sufficient. At a float boundary, repeat
    # the executor's fsum over individual holds, never accept with an epsilon.
    for d in range(3):
        projected = math.fsum((used[d], demand[d]))
        # The sum is updated incrementally between release grids. The guard
        # bounds accumulated addition error; it only selects exact fsum, never
        # relaxes the capacity comparison itself.
        if abs(projected - capacity[d]) <= (len(active) + 2) * math.ulp(capacity[d]):
            projected = math.fsum([demand[d]] + [hold[d] for _, hold in active])
        if projected > capacity[d]:
            return False
    return True


class ShadowPlanner:
    def __init__(self, horizon_steps=8, max_pending=256):
        for name, value in (('horizon_steps', horizon_steps), ('max_pending', max_pending)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        self.horizon_steps = horizon_steps
        self.max_pending = max_pending
        self._job_cache = {}
        self._state_key = None
        self._prepared = None
        self._pending_key = None
        self._ordered_jobs = ()
        self._running_key = None
        self._running_holds = ()
        self._running_missed = ()

    def _job(self, spec):
        # Hold the immutable spec reference so a recycled Python object ID can
        # never refer to another task. This avoids repeated pandas conversions.
        cached = self._job_cache.get(id(spec))
        if cached is None or cached[0] is not spec:
            job = _Job(spec.task_id, (spec.cores_req, spec.gpu_req, spec.mem_req),
                pd.Timedelta(minutes=spec.duration).value, spec.sla_deadline.value,
                spec.arrival_time.value)
            self._job_cache[id(spec)] = (spec, job)
            return job
        return cached[1]

    def _prepare(self, snapshot):
        # Policies may construct a fresh frozen snapshot for every candidate.
        # Fingerprint immutable known state, not snapshot identity; any start,
        # changed capacity, release, pending membership or block invalidates it.
        running = tuple((r.task_id, r.demand, r.finish_time.value,
            r.deadline.value if r.deadline is not None else None) for r in snapshot.running)
        blocked = tuple((key, until.value) for key, until in snapshot.blocked_until)
        pending_key = tuple(id(s) for s in snapshot.pending)
        key = (snapshot.now.value, snapshot.capacity, snapshot.available,
            pending_key, running, blocked)
        if key != self._state_key:
            # A new prior-defer block changes planning state but not EDF order
            # or running releases. Share these separate preparations too.
            if pending_key != self._pending_key:
                self._ordered_jobs = tuple(sorted((self._job(s) for s in snapshot.pending),
                    key=lambda j: (j.deadline_ns, j.arrival_ns, j.task_id)))
                self._pending_key = pending_key
            if running != self._running_key:
                self._running_holds = tuple(sorted((_release_ns(finish), demand)
                    for _, demand, finish, _ in running))
                self._running_missed = tuple(sorted(task_id for task_id, _, finish, deadline in running
                    if deadline is not None and finish > deadline))
                self._running_key = running
            self._prepared = _Prepared(snapshot.now.value,
                snapshot.now.value + self.horizon_steps * _STEP_NS, snapshot.capacity,
                self._ordered_jobs, self._running_holds, dict(blocked), self._running_missed)
            self._state_key = key
        return self._prepared

    def _uncertain(self, snapshot, reason):
        ids = tuple(sorted(s.task_id for s in snapshot.pending))
        common = tuple(sorted(r.task_id for r in snapshot.running
            if r.deadline is not None and r.finish_time > r.deadline))
        return PlannerResult('uncertain', common, common, ids, ids, None, None, 0,
            snapshot.now + self.horizon_steps * STEP, reason, planner_calls=0)

    def compare(self, spec, snapshot):
        """Force candidate NOW versus forbid it until the next scheduling grid.

        All prior real deferrals remain blocked in both plans. Any unresolved
        pending task makes the bounded comparison uncertain and conservative.
        Already missed running outcomes are retained in both plans as common
        known pressure; their resource holds participate throughout the horizon.
        """
        if len(snapshot.pending) > self.max_pending:
            return self._uncertain(snapshot, 'pending_limit')
        if not any(s is spec for s in snapshot.pending):
            return self._uncertain(snapshot, 'candidate_missing_from_pending')
        state = self._prepare(snapshot)
        candidate = self._job(spec)
        if state.blocked.get(candidate.task_id, state.now_ns) > state.now_ns:
            return self._uncertain(snapshot, 'candidate_already_blocked')
        active = [(release, demand) for release, demand in state.running if release > state.now_ns]
        if not _fits(candidate.demand, _used(active), active, state.capacity):
            return self._uncertain(snapshot, 'candidate_capacity_unavailable')
        missed_now, unknown_now, start_now = self._plan(candidate, state, force_now=True)
        missed_defer, unknown_defer, start_defer = self._plan(candidate, state, force_now=False)
        if unknown_now or unknown_defer:
            status, reason = 'uncertain', 'bounded_horizon_unknown'
        elif set(missed_defer) - set(missed_now):
            status, reason = 'risk', 'additional_known_deadline_miss'
        else:
            status, reason = 'safe', 'no_additional_known_deadline_miss'
        return PlannerResult(status, missed_now, missed_defer, unknown_now, unknown_defer,
            _timestamp(start_now), _timestamp(start_defer), len(state.jobs),
            _timestamp(state.end_ns), reason)

    def _plan(self, candidate, state, force_now):
        pending = list(state.jobs)
        active = [(release, demand) for release, demand in state.running if release > state.now_ns]
        used = _used(active)
        missed = list(state.running_missed)
        candidate_start = None

        def start(job, time_ns):
            nonlocal used, candidate_start
            finish_ns = time_ns + job.duration_ns
            active.append((_release_ns(finish_ns), job.demand))
            used = tuple(math.fsum((total, demand)) for total, demand in zip(used, job.demand))
            if finish_ns > job.deadline_ns:
                missed.append(job.task_id)
            if job.task_id == candidate.task_id:
                candidate_start = time_ns

        if force_now:
            start(candidate, state.now_ns)
            pending = [j for j in pending if j.task_id != candidate.task_id]
        for step in range(self.horizon_steps):
            time_ns = state.now_ns + step * _STEP_NS
            if step:
                active = [(release, demand) for release, demand in active if release > time_ns]
                used = _used(active)
            remaining = []
            for job in pending:
                until_ns = state.blocked.get(job.task_id, state.now_ns)
                if not force_now and job.task_id == candidate.task_id:
                    until_ns = max(until_ns, state.now_ns + _STEP_NS)
                if time_ns < until_ns or not _fits(job.demand, used, active, state.capacity):
                    remaining.append(job)
                    continue
                start(job, time_ns)
            pending = remaining
            if not pending:
                break
        unknown = []
        for job in pending:
            if job.latest_start_ns < state.end_ns:
                missed.append(job.task_id)
            else:
                unknown.append(job.task_id)
        return tuple(sorted(missed)), tuple(sorted(unknown)), candidate_start
