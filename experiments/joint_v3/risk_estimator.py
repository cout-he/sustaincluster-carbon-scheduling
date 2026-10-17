"""Conditional known-queue deadline risk with prefix-safe grid wait budgets.

Confidence one denotes an exact settled deterministic continuation of the
observed snapshot. It is not a probability or a promise about future arrivals.
Invalid estimates carry zero wait budget and explicitly uncomputed deltas.
"""
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, replace

from experiments.e1.contracts import STEP, finite, utc_time

from .reservation import (_bounded, _commitments, _positive_bound, _resources,
                          _task_signature, schedule_known)


@dataclass(frozen=True)
class RiskEstimate:
    safe_wait_budget_minutes: float = 0.
    delta_deadline_misses: int = 0
    extra_tardiness_minutes: float = 0.
    risk_estimation_valid: bool = False
    risk_confidence: float = 0.
    reason: str = 'uncomputed'
    start_schedule: object = None
    wait_schedule: object = None
    runtime_seconds: float = 0.
    cache_hit: bool = False
    examined_wait_slots: int = 0
    candidate_delta_deadline_misses: int = 0
    other_delta_deadline_misses: int = 0
    candidate_extra_tardiness_minutes: float = 0.
    other_extra_tardiness_minutes: float = 0.

    @property
    def valid(self):
        return self.risk_estimation_valid

    @property
    def confidence(self):
        return self.risk_confidence


class RiskEstimator:
    """Bound both work per estimate and memory; cache identical causal states."""

    def __init__(self, config=None):
        config = config or {}
        cfg = config.get('risk', config)
        self.max_tasks = _positive_bound(cfg.get('max_tasks', 256), 'max_tasks', 4096)
        self.max_slots = _positive_bound(cfg.get('max_slots', 192), 'max_slots', 4096)
        self.max_wait_slots = _positive_bound(cfg.get('max_wait_slots', 16), 'max_wait_slots', self.max_slots)
        self.time_budget_ms = finite(cfg.get('time_budget_ms', 100.), 'time_budget_ms', True)
        self.cache_size = _positive_bound(cfg.get('cache_size', 128), 'cache_size', 4096)
        self._cache = OrderedDict()

    def estimate(self, candidate, now, capacity, pending, running, *, blocked_until=()):
        started = time.perf_counter()
        deadline = started + self.time_budget_ms / 1000
        baseline, first_wait, examined = None, None, 0

        def invalid(reason):
            return RiskEstimate(reason=reason, start_schedule=baseline, wait_schedule=first_wait,
                                runtime_seconds=time.perf_counter() - started,
                                examined_wait_slots=examined)

        try:
            now = utc_time(now, True)
            if utc_time(candidate.arrival_time) > now:
                return invalid('future_arrival_rejected')
            latest = utc_time(candidate.sla_deadline) - STEP * (finite(candidate.duration, 'duration', True) / 15)
            if now + STEP > latest:
                return invalid('own_latest_start_no_wait')
            pending, running = _bounded(pending, self.max_tasks), _bounded(running, self.max_tasks)
            if (len(pending) > self.max_tasks or len(running) > self.max_tasks or
                    len(pending) + len(running) + (candidate.task_id not in {j.task_id for j in pending}) > self.max_tasks):
                return invalid('queue_limit_exceeded')
            if any(utc_time(job.arrival_time) > now for job in pending):
                return invalid('future_arrival_rejected')
            capacity = _resources(capacity, 'capacity', True)
            commitments = _commitments(blocked_until)
            key = (now.value, capacity, _task_signature(candidate),
                   tuple(sorted(_task_signature(job) for job in pending)),
                   tuple(sorted((job.task_id, tuple(job.demand), utc_time(job.release_time).value,
                                 utc_time(getattr(job, 'finish_time', job.release_time)).value,
                                 None if getattr(job, 'deadline', None) is None else utc_time(job.deadline).value)
                                for job in running)),
                   tuple(sorted((task_id, until.value) for task_id, until in commitments.items())))
            if key in self._cache:
                cached = self._cache.pop(key)
                self._cache[key] = cached
                return replace(cached, cache_hit=True, runtime_seconds=time.perf_counter() - started)
            if time.perf_counter() >= deadline:
                return invalid('time_budget_exceeded')
            baseline = schedule_known(candidate, now, capacity, pending, running,
                                      max_tasks=self.max_tasks, max_slots=self.max_slots,
                                      blocked_until=commitments, _deadline=deadline)
            if not baseline.valid:
                return invalid('start_schedule:' + baseline.reason)
            jobs = {job.task_id: job for job in pending}
            jobs[candidate.task_id] = candidate
            deadlines = {task_id: utc_time(job.sla_deadline) for task_id, job in jobs.items()}
            for job in running:
                if getattr(job, 'deadline', None) is not None:
                    deadlines[job.task_id] = utc_time(job.deadline)

            def change(wait):
                own_delta, other_delta, own_tardy, other_tardy = 0, 0, 0., 0.
                for task_id, bound in deadlines.items():
                    f0, f1 = baseline.finishes[task_id], wait.finishes[task_id]
                    delta = int(f1 > bound) - int(f0 > bound)
                    tardy0 = max(0., (f0 - bound).total_seconds() / 60)
                    tardy1 = max(0., (f1 - bound).total_seconds() / 60)
                    extra = max(0., tardy1 - tardy0)
                    if task_id == candidate.task_id:
                        own_delta, own_tardy = delta, extra
                    else:
                        other_delta += delta
                        other_tardy += extra
                return own_delta, other_delta, own_tardy, other_tardy

            maximum = min(self.max_wait_slots, math.floor((latest - now) / STEP))
            safe, first_metrics = 0, None
            reason = 'known_queue_conditional'
            for delay in range(1, maximum + 1):
                if time.perf_counter() >= deadline:
                    return invalid('time_budget_exceeded')
                wait = schedule_known(candidate, now, capacity, pending, running, delay,
                                      max_tasks=self.max_tasks, max_slots=self.max_slots,
                                      blocked_until=commitments, _deadline=deadline)
                examined += 1
                if first_wait is None:
                    first_wait = wait
                if not wait.valid:
                    return invalid('wait_schedule:' + wait.reason)
                metrics = change(wait)
                if first_metrics is None:
                    first_metrics = metrics
                if metrics[0] + metrics[1] > 0:
                    reason = 'additional_known_queue_deadline_misses'
                    break  # Every supported positive prefix must satisfy the constraint.
                safe = delay * 15
            if time.perf_counter() >= deadline:
                return invalid('time_budget_exceeded')
            own_delta, other_delta, own_tardy, other_tardy = first_metrics
            estimate = RiskEstimate(
                safe_wait_budget_minutes=float(safe), delta_deadline_misses=own_delta + other_delta,
                extra_tardiness_minutes=own_tardy + other_tardy, risk_estimation_valid=True,
                risk_confidence=1., reason=reason, start_schedule=baseline, wait_schedule=first_wait,
                runtime_seconds=time.perf_counter() - started, examined_wait_slots=examined,
                candidate_delta_deadline_misses=own_delta, other_delta_deadline_misses=other_delta,
                candidate_extra_tardiness_minutes=own_tardy, other_extra_tardiness_minutes=other_tardy)
            self._cache[key] = estimate
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return estimate
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
            return invalid(f'invalid_snapshot:{exc}')
