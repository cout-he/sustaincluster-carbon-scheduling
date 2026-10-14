"""Common local executor: FIFO one-pass backfill, no global/network queue."""
import math
from collections import deque
from dataclasses import dataclass, field

from envs.sustaindc.sustaindc_env import SustainDC

from .contracts import STEP, finite, utc_time
from .metrics import interval_record, summarize, summarize_policy
from .policies import DecisionContext, ImmediateFIFO, PolicyDecision, latest_start


class LocalResourcePool:
    # Reuse upstream start/finish assignment, failed requeue and release.
    try_to_schedule_task = SustainDC.try_to_schedule_task
    release_resources = SustainDC.release_resources

    def __init__(self, capacity):
        self.capacity = tuple(finite(v, 'capacity', True) for v in capacity)
        if len(self.capacity) != 3:
            raise ValueError('Capacity requires CPU/GPU/MEM')
        self.available_cores, self.available_gpus, self.available_mem = self.capacity
        self.pending_tasks, self.running_tasks = deque(), []
        self.dc_id = 1

    def can_schedule(self, task):
        # Compare projected aggregate demands, avoiding subtraction roundoff
        # (e.g. 1 - fsum([.2,.6]) < .2). No oversubscription epsilon is used.
        return all(math.fsum([getattr(t, name) for t in self.running_tasks] + [getattr(task, name)]) <= limit
                   for name, limit in zip(('cores_req', 'gpu_req', 'mem_req'), self.capacity))

    def inventory(self):
        used = tuple(math.fsum(getattr(t, name) for t in self.running_tasks)
                     for name in ('cores_req', 'gpu_req', 'mem_req'))
        available = (self.available_cores, self.available_gpus, self.available_mem)
        for allocated, free, total in zip(used, available, self.capacity):
            if allocated > total or free < -1e-9 or not math.isclose(allocated + free, total, abs_tol=1e-9):
                raise RuntimeError('Resource conservation or capacity violation')
        # Recompute rather than accumulating roundoff through many releases.
        self.available_cores, self.available_gpus, self.available_mem = tuple(
            total - allocated for total, allocated in zip(self.capacity, used))
        return used


@dataclass(frozen=True)
class ExecutionResult:
    tasks: tuple
    events: tuple
    steps: tuple
    summary: dict
    policy_tasks: tuple = ()
    policy_events: tuple = ()
    policy_steps: tuple = ()
    policy_summary: dict = field(default_factory=dict)


class LocalExecutor:
    def __init__(self, workload, environment, capacity, power_model, policy=None, drain=False):
        if workload.start != environment.start or (environment.end < workload.end if drain
                                                   else environment.end != workload.end):
            raise ValueError('Workload and environment window mismatch')
        if any(t.dc_id != 1 for t in workload.tasks):
            raise ValueError('Local execution requires every task to originate at DC1')
        self.workload, self.environment, self.power_model = workload, environment, power_model
        self.pool = LocalResourcePool(capacity)
        self.policy = policy if policy is not None else ImmediateFIFO()
        self.specs = tuple(sorted(workload.tasks, key=lambda t: (t.arrival_time, t.source_row, t.source_task_index)))
        self.ledger, self.events, self.steps = {}, [], []
        self.policy_events, self.policy_steps = [], []
        self.has_run = False
        self.finished, self.sample, self.current_arrivals = False, None, ()
        self.drain, self.drained_at, self.observed_end = drain, None, environment.end

    def event(self, time, task, kind):
        self.events.append(dict(time=time.isoformat(), task_id=task.job_name, event=kind))

    def release(self, time):
        for task in self.pool.release_resources(time, None):
            entry = self.ledger[task.job_name]
            entry['state'] = 'completed'
            entry['completion_observed_at'] = time.isoformat()
            self.event(time, task, 'completed_resources_released')
        self.pool.inventory()

    def check_ownership(self):
        pending = [t.job_name for t in self.pool.pending_tasks]
        running = [t.job_name for t in self.pool.running_tasks]
        completed = [key for key, value in self.ledger.items() if value['state'] == 'completed']
        all_ids = pending + running + completed
        if len(set(all_ids)) != len(all_ids) or set(all_ids) != set(self.ledger):
            raise RuntimeError('Task ledger ownership violation')
        if any(self.ledger[key]['state'] != 'pending' for key in pending) or any(
                self.ledger[key]['state'] != 'running' for key in running):
            raise RuntimeError('Task queue/state mismatch')

    def start(self, stop_at_drain=False, until=None):
        """Prepare arrivals/releases/enqueue, suspend BEFORE policy or physics.

        `until` is an exclusive interaction cutoff. Only already-arrived tasks
        enter a partial result; the cutoff performs the usual completion flush.
        """
        if self.has_run:
            raise RuntimeError('Executor instances are single-use; construct fresh state for replay')
        self.has_run = True
        if stop_at_drain and not self.drain:
            raise ValueError('Stopping at drain requires explicit finite-cohort drain mode')
        self._until = self.environment.end if until is None else utc_time(until, True)
        if not self.environment.start < self._until <= self.environment.end:
            raise ValueError('Interaction cutoff outside environmental coverage')
        self.observed_end = self.environment.start
        self._iterator = self._execute(stop_at_drain)
        self.sample = next(self._iterator)
        return self.sample

    def advance(self, policy=None):
        """Execute one prepared interval, then prepare the next observed boundary.

        A replacement policy changes only the eligibility mode for that interval.
        The returned record is its actual energy/resource integral. The same
        pool, Task objects and power model remain alive throughout the iterator.
        """
        if not self.has_run or self.finished:
            raise RuntimeError('Advance requires a live prepared executor')
        previous = len(self.steps)
        try:
            self.sample = self._iterator.send(policy)
        except StopIteration:
            self.sample, self.current_arrivals, self.finished = None, (), True
        if len(self.steps) != previous + 1:
            raise RuntimeError('Each advance must account for exactly one interval')
        return self.steps[-1]

    def run(self, stop_at_drain=False):
        self.start(stop_at_drain)
        while not self.finished:
            self.advance()
        return self.result()

    def _execute(self, stop_at_drain):
        # Single execution path shared by batch run() and interactive advance().
        arrivals = {}
        for spec in self.specs:
            arrivals.setdefault(spec.arrival_time, []).append(spec)
        for sample in self.environment.samples:
            time = sample.time
            if time >= self._until:
                self.observed_end = self._until
                break
            self.observed_end = time
            incoming = []
            # Arrival is recorded first, but every completion releases capacity
            # before either old or new pending tasks can start at this boundary.
            for spec in arrivals.get(time, ()):
                task = spec.make_task()
                self.ledger[spec.task_id] = dict(spec=spec, task=task, state='pending',
                                               completion_observed_at=None,
                                               policy_defer_intervals=0, capacity_wait_intervals=0)
                incoming.append(task)
                self.event(time, task, 'arrived')
            self.release(time)
            if self.drain and time >= self.workload.end and not (
                    self.pool.pending_tasks or self.pool.running_tasks or incoming):
                if self.drained_at is None:
                    self.drained_at = time
                if stop_at_drain:
                    self.observed_end = time
                    break
            for task in incoming:
                self.pool.pending_tasks.append(task)
                self.event(time, task, 'enqueued')
            self.current_arrivals = tuple(self.ledger[t.job_name]['spec'] for t in incoming)
            selected_policy = yield sample
            if selected_policy is not None:
                self.policy = selected_policy
            # Every strategy receives ALL unstarted tasks, including old backlog.
            waiting = tuple(self.ledger[t.job_name]['spec'] for t in self.pool.pending_tasks)
            ordered = tuple(self.policy.order(waiting))
            originals = {s.task_id: s for s in waiting}
            if len(ordered) != len(waiting) or {s.task_id for s in ordered} != set(originals) or any(
                    s is not originals[s.task_id] for s in ordered):
                raise ValueError('Policy order must preserve every immutable pending task exactly once')
            self.pool.pending_tasks = deque(self.ledger[s.task_id]['task'] for s in ordered)
            context = DecisionContext(time, sample.ci_g_per_kwh)
            # Optional causal sensor: current scalars only, never the calendar.
            if hasattr(self.policy, 'on_step_environment'):
                self.policy.on_step_environment(sample)
            # Optional observed-state notifications. Legacy policies take exactly
            # their original path; no Task, pool, future arrivals or CI are exposed.
            if hasattr(self.policy, 'on_step_start'):
                from .cars_state import ResourceSnapshot, RunningTask, demand
                self.policy.on_step_start(ResourceSnapshot(time, self.pool.capacity,
                    (self.pool.available_cores,self.pool.available_gpus,self.pool.available_mem),
                    ordered, tuple(RunningTask(t.job_name,demand(self.ledger[t.job_name]['spec']),
                                               t.finish_time,self.ledger[t.job_name]['spec'].sla_deadline)
                                   for t in self.pool.running_tasks),
                    tuple(self.ledger[t.job_name]['spec'] for t in incoming)))
            detail = dict(time=time.isoformat(), decision_pending_ids=[s.task_id for s in waiting],
                          ordered_waiting_ids=[s.task_id for s in ordered],
                          policy_deferred_ids=[], capacity_waiting_ids=[], started_ids=[])
            # Shared one-pass allocation/backfill. Capacity has priority in wait
            # attribution: policy delay means this task COULD run at its turn.
            for rank, spec in enumerate(ordered):
                task = self.pool.pending_tasks.popleft()
                can_fit = self.pool.can_schedule(task)
                available = dict(cores=self.pool.available_cores, gpus=self.pool.available_gpus,
                                 mem=self.pool.available_mem)
                if not can_fit:
                    self.pool.try_to_schedule_task(task, time, None)
                    self.ledger[task.job_name]['capacity_wait_intervals'] += 1
                    kind, reason = 'capacity_waited', 'capacity_unavailable'
                    detail['capacity_waiting_ids'].append(task.job_name)
                else:
                    if hasattr(self.policy, 'on_candidate_resources'):
                        self.policy.on_candidate_resources(time,tuple(available.values()))
                    decision = self.policy.decide(spec, context)
                    if not isinstance(decision, PolicyDecision) or not isinstance(decision.defer, bool):
                        raise ValueError('Policy must return a validated PolicyDecision')
                    reason = decision.reason
                    if decision.defer:
                        self.pool.pending_tasks.append(task)
                        self.ledger[task.job_name]['policy_defer_intervals'] += 1
                        kind = 'policy_deferred'
                        detail['policy_deferred_ids'].append(task.job_name)
                        if hasattr(self.policy, 'on_task_deferred'):
                            self.policy.on_task_deferred(task.job_name,time + STEP)
                    else:
                        if not self.pool.try_to_schedule_task(task, time, None):
                            raise RuntimeError('Capacity changed between check and allocation')
                        kind = 'started'
                if kind == 'started':
                    task.dest_dc_id, task.dest_dc = 1, self.pool
                    self.ledger[task.job_name]['state'] = 'running'
                    self.event(time, task, 'started')
                    detail['started_ids'].append(task.job_name)
                    if hasattr(self.policy, 'on_task_started'):
                        from .cars_state import RunningTask, demand
                        self.policy.on_task_started(RunningTask(task.job_name,demand(spec),task.finish_time,spec.sla_deadline))
                self.policy_events.append(dict(time=time.isoformat(), task_id=task.job_name, event=kind,
                    reason=reason, policy=self.policy.name, decision_rank=rank,
                    ci_g_per_kwh=sample.ci_g_per_kwh,
                    threshold_g_per_kwh=getattr(self.policy, 'threshold_g_per_kwh', None),
                    latest_start=latest_start(spec).isoformat(), can_fit_at_decision=can_fit,
                    available_before=available))
                self.pool.inventory()
            self.check_ownership()
            used = self.pool.inventory()
            fractions = tuple(u / c for u, c in zip(used, self.pool.capacity))
            power = self.power_model.measure(*fractions, sample)
            self.steps.append(interval_record(time, sample, power, self.pool.capacity, used,
                                              len(self.pool.pending_tasks), len(self.pool.running_tasks)))
            if hasattr(self.policy, 'on_interval_completed'):
                from types import MappingProxyType
                self.policy.on_interval_completed(MappingProxyType(dict(self.steps[-1])))
            detail.update(pending_end_ids=[t.job_name for t in self.pool.pending_tasks],
                          running_ids=[t.job_name for t in self.pool.running_tasks])
            self.policy_steps.append(detail)
        else:
            self.observed_end = self._until
        # T is exclusively a completion flush: no arrivals, starts, power or drain.
        self.release(self.observed_end)
        self.check_ownership()
        if self.observed_end >= self.workload.end and len(self.ledger) != len(self.specs):
            raise RuntimeError('An arrived task is missing from the ledger')
        if self.drain and self.observed_end >= self.workload.end and self.drained_at is None and not (
                self.pool.pending_tasks or self.pool.running_tasks):
            self.drained_at = self.observed_end
        # result() is serialized by callers; finalization never adds another step.

    def extend_idle(self, endpoint):
        """Continue the same physical state to a common endpoint, with no scheduling."""
        endpoint = utc_time(endpoint, True)
        if not self.finished or not self.drain or self.drained_at is None or (
                self.pool.pending_tasks or self.pool.running_tasks):
            raise ValueError('Idle extension requires a completed drain run')
        if endpoint < self.observed_end or endpoint > self.environment.end:
            raise ValueError('Idle endpoint outside observed/frozen coverage')
        for sample in self.environment.samples:
            if self.observed_end <= sample.time < endpoint:
                power = self.power_model.measure(0,0,0,sample)
                self.steps.append(interval_record(sample.time,sample,power,self.pool.capacity,(0,0,0),0,0))
                self.policy_steps.append(dict(time=sample.time.isoformat(), decision_pending_ids=[],
                    ordered_waiting_ids=[], policy_deferred_ids=[], capacity_waiting_ids=[],
                    started_ids=[], pending_end_ids=[], running_ids=[]))
        self.observed_end = endpoint
        return self.result()

    def result(self):
        if not self.steps:
            raise RuntimeError('Result requires at least one accounted interval')
        rows, policy_rows = [], []
        for spec in self.specs:
            if spec.task_id not in self.ledger:
                continue  # Interactive prefixes do not reveal/count future tasks.
            entry, task = self.ledger[spec.task_id], self.ledger[spec.task_id]['task']
            state = entry['state']
            if state == 'completed':
                outcome = 'met' if task.sla_met else 'missed'
            else:
                outcome = 'missed_unfinished' if spec.sla_deadline <= self.observed_end else 'right_censored'
            row = spec.to_dict()
            row.update(state=state, start_time=task.start_time.isoformat() if task.start_time is not None else None,
                       finish_time=task.finish_time.isoformat() if task.finish_time is not None else None,
                       completion_observed_at=entry['completion_observed_at'],
                       sla_deadline=spec.sla_deadline.isoformat(), deadline_outcome=outcome,
                       sla_met=task.sla_met if state == 'completed' else None,
                       wait_intervals=task.wait_intervals, dest_dc_id=task.dest_dc_id,
                       transmission_delay_seconds=0.0, transmission_energy_kwh=0.0,
                       transmission_carbon_kg=0.0,
                       structurally_infeasible=any(v > c for v, c in zip(
                           (spec.cores_req, spec.gpu_req, spec.mem_req), self.pool.capacity)))
            rows.append(row)
            endpoint = task.start_time if task.start_time is not None else self.observed_end
            policy_rows.append(dict(task_id=spec.task_id, latest_start=latest_start(spec).isoformat(),
                policy_defer_intervals=entry['policy_defer_intervals'],
                capacity_wait_intervals=entry['capacity_wait_intervals'],
                observed_pending_minutes=(endpoint - spec.arrival_time).total_seconds() / 60))
        summary = summarize(rows, self.steps)
        if self.drain:
            summary.update(horizon='finite_cohort_drain', arrival_end=self.workload.end.isoformat(),
                           observation_end=self.observed_end.isoformat(),
                           drained_at=self.drained_at.isoformat() if self.drained_at is not None else None)
        return ExecutionResult(tuple(rows), tuple(self.events), tuple(self.steps), summary,
            tuple(policy_rows), tuple(self.policy_events), tuple(self.policy_steps),
            summarize_policy(self.policy, policy_rows, self.policy_steps, self.steps))
