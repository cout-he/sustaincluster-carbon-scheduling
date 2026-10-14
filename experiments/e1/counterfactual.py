"""One predeclared eligible Execute/Defer intervention, then unchanged EDF.

No executor, capacity, duration, task or upstream physics rules are duplicated.
The extra history and risk calculations are observations, never policy inputs.
"""
from collections import Counter
from dataclasses import asdict, dataclass, fields
import math

import pandas as pd

from .cars_state import ArrivalHistory
from .contracts import STEP, utc_time
from .counterfactual_audit import executor_prefix
from .counterfactual_cases import known_release_screen
from .engine import LocalExecutor
from .policies import EDFBackfill, PolicyDecision
from .task_risk_state import BudgetParameters, calculate_budget, latest_grid_start

VERSION = 'e1-7a-single-action-paired-edf-v1'
BRANCHES = ('force_execute', 'force_defer_once')


@dataclass(frozen=True)
class InterventionTarget:
    case_id: str
    time: object
    task_id: str
    candidate_occurrence: int = 1

    def __post_init__(self):
        object.__setattr__(self, 'time', utc_time(self.time, True))
        if not self.case_id or not self.task_id:
            raise ValueError('Target requires case and task identity')
        if isinstance(self.candidate_occurrence, bool) or not isinstance(self.candidate_occurrence, int) or self.candidate_occurrence < 1:
            raise ValueError('Target occurrence must be a positive integer')

    def to_dict(self):
        return dict(case_id=self.case_id, time=self.time.isoformat(), task_id=self.task_id,
                    candidate_occurrence=self.candidate_occurrence)


class ObservedEDF(EDFBackfill):
    """Keep EDF rules; notifications track only arrived/actually started tasks."""
    name = 'edf_backfill'

    def __init__(self, start, capacity, parameters=None, targets=(), action=None, prefix_callback=None):
        if action not in (None,) + BRANCHES:
            raise ValueError('Unsupported intervention action')
        keys = {field.name for field in fields(BudgetParameters)}
        self.parameters = (parameters if isinstance(parameters, BudgetParameters)
                           else BudgetParameters(**{key: value for key, value in (parameters or {}).items() if key in keys}))
        self.history = ArrivalHistory(start, capacity, self.parameters.history_minutes,
                                      cold_start_buffer_steps=self.parameters.cold_buffer_steps)
        self.capacity = self.history.capacity
        self.targets = tuple(targets)
        if len({target.case_id for target in self.targets}) != len(self.targets):
            raise ValueError('Duplicate target case ID')
        if len({(target.time, target.task_id, target.candidate_occurrence) for target in self.targets}) != len(self.targets):
            raise ValueError('Duplicate target decision')
        if action is not None and len(self.targets) != 1:
            raise ValueError('Intervention branch requires exactly one target')
        self.action, self.prefix_callback = action, prefix_callback
        self.records, self.interventions, self.occurrences = {}, [], Counter()

    def on_step_start(self, snapshot):
        if snapshot.capacity != self.capacity:
            raise ValueError('Observed capacity changed')
        self.snapshot, self.ordered = snapshot, snapshot.pending
        self.rank = {spec.task_id: index for index, spec in enumerate(self.ordered)}
        self.pending = {spec.task_id: spec for spec in self.ordered}
        self.running, self.available = list(snapshot.running), snapshot.available
        self.history.observe(snapshot.now, snapshot.arrivals)
        self.pressure = self.history.pressure(snapshot.now)

    def on_candidate_resources(self, now, available):
        if now != self.snapshot.now:
            raise ValueError('Stale observed candidate state')
        self.available = tuple(available)

    def on_task_started(self, running):
        if running.task_id not in self.pending:
            raise ValueError('Unknown observed start')
        del self.pending[running.task_id]
        self.running.append(running)

    def on_task_deferred(self, task_id, until):
        if task_id not in self.pending or until != self.snapshot.now + STEP:
            raise ValueError('Invalid observed defer')

    def decide(self, spec, context):
        if context.now != self.snapshot.now or self.pending.get(spec.task_id) is not spec:
            raise ValueError('Candidate missing from current observed state')
        key = (context.now, spec.task_id)
        self.occurrences[key] += 1
        matching = [target for target in self.targets if (target.time, target.task_id) == key
                    and target.candidate_occurrence == self.occurrences[key]]
        if not matching:
            return super().decide(spec, context)
        target = matching[0]
        if target.case_id in self.records:
            raise RuntimeError('Target was applied more than once')
        slack = (latest_grid_start(spec) - context.now).total_seconds() / 60
        if slack < 15 or spec.arrival_time > context.now:
            raise ValueError('Target must be arrived with at least one grid step of solo slack')
        predecessors = tuple(item for item in self.ordered[:self.rank[spec.task_id]] if item.task_id in self.pending)
        budget = calculate_budget(spec, context.now, self.capacity, self.running,
                                  predecessors, self.pressure, self.parameters)
        features = dict(time=context.now.isoformat(), task_id=spec.task_id,
            candidate_occurrence=self.occurrences[key], ci_g_per_kwh=context.ci_g_per_kwh,
            demand=[spec.cores_req, spec.gpu_req, spec.mem_req], duration_minutes=spec.duration,
            deadline=spec.sla_deadline.isoformat(), available=list(self.available),
            pending_count=len(self.pending), running_count=len(self.running), decision_rank=self.rank[spec.task_id],
            known_task_ids=sorted(self.history.seen), history_observed_total=len(self.history.seen),
            history_effective_minutes=self.pressure.effective_minutes,
            history_resource_work_minutes=list(self.pressure.resource_work_minutes),
            riskonly_pressure=self.pressure.rho, riskonly_buffer_steps=self.pressure.buffer_steps,
            latest_grid_start=latest_grid_start(spec).isoformat(), **asdict(budget))
        # Canonical JSON-friendly lists make schemas explicit across releases.
        features = {key: list(value) if isinstance(value, tuple) else value for key, value in features.items()}
        features.update(known_release_screen(spec, context.now, self.capacity,
                                             self.running, predecessors, 120))
        prefix = self.prefix_callback(self, spec, context) if self.prefix_callback else None
        self.records[target.case_id] = dict(target=target.to_dict(), features=features, prefix=prefix)
        if self.action is None:
            return super().decide(spec, context)
        changed = self.action == 'force_defer_once'
        self.interventions.append(dict(**target.to_dict(), original_defer=False, original_reason='eligible',
            changed_defer=changed, changed_reason=self.action, eligible_can_fit=True))
        return PolicyDecision(changed, self.action)

    def assert_targets_observed(self):
        if set(self.records) != {target.case_id for target in self.targets}:
            raise RuntimeError('Every target must be observed exactly once at an eligible candidate')
        if self.action is not None and len(self.interventions) != 1:
            raise RuntimeError('Intervention target must be applied exactly once')


@dataclass(frozen=True)
class ReferenceCapture:
    result: object
    records: dict


@dataclass(frozen=True)
class PairedRun:
    target: InterventionTarget
    results: dict
    own_results: dict
    prefixes: dict
    prefix_equal: bool
    features: dict
    interventions: dict
    common_end: object
    drained_at: dict
    task_impacts: tuple


def capture_reference(workload, environment, capacity, power_factory, targets, parameters=None):
    observer = ObservedEDF(workload.start, capacity, parameters, targets)
    executor = LocalExecutor(workload, environment, capacity, power_factory(), observer, drain=True)
    observer.prefix_callback = lambda policy, spec, context: executor_prefix(executor, policy, spec, context)
    result = executor.run(stop_at_drain=True)
    observer.assert_targets_observed()
    if executor.drained_at is None or result.summary['completed'] != len(workload.tasks):
        raise RuntimeError('Reference must complete the full drain')
    return ReferenceCapture(result, observer.records)


def _task_impacts(target, results, known):
    execute = {row['task_id']: row for row in results['force_execute'].tasks}
    deferred = {row['task_id']: row for row in results['force_defer_once'].tasks}
    policies = {name: {row['task_id']: row for row in result.policy_tasks} for name, result in results.items()}
    rows = []
    for task_id, first in execute.items():
        second = deferred[task_id]
        partition = ('candidate' if task_id == target.task_id else
                     'known_other' if task_id in known else 'post_intervention_arrival')
        start_difference = (utc_time(second['start_time']) - utc_time(first['start_time'])).total_seconds() / 60
        row = dict(task_id=task_id, partition=partition, arrival=first['arrival_time'], deadline=first['sla_deadline'],
            execute_start=first['start_time'], defer_start=second['start_time'],
            execute_finish=first['finish_time'], defer_finish=second['finish_time'],
            execute_deadline_outcome=first['deadline_outcome'], defer_deadline_outcome=second['deadline_outcome'],
            delta_start_minutes=start_difference,
            execute_wait_minutes=(utc_time(first['start_time']) - utc_time(first['arrival_time'])).total_seconds()/60,
            defer_wait_minutes=(utc_time(second['start_time']) - utc_time(second['arrival_time'])).total_seconds()/60,
            execute_capacity_wait_intervals=policies['force_execute'][task_id]['capacity_wait_intervals'],
            defer_capacity_wait_intervals=policies['force_defer_once'][task_id]['capacity_wait_intervals'])
        row['changed'] = any(first[key] != second[key] for key in ('start_time', 'finish_time', 'deadline_outcome'))
        rows.append(row)
    return tuple(rows)


def run_paired_case(workload, environment, capacity, power_factory, target, parameters=None, minimum_common_end=None):
    """Two complete replays; change one can-fit decision and preserve physics.

    `minimum_common_end` is a predeclared physical accounting endpoint, not an
    online input. Each branch retains its own power instance through idle tail.
    """
    if not isinstance(target, InterventionTarget):
        raise ValueError('Target must use the frozen InterventionTarget schema')
    specs = {spec.task_id: spec for spec in workload.tasks}
    if target.task_id not in specs or not specs[target.task_id].arrival_time <= target.time < environment.end:
        raise ValueError('Target absent, before arrival, or outside coverage')
    minimum = utc_time(minimum_common_end, True) if minimum_common_end is not None else workload.end
    if not workload.end <= minimum <= environment.end:
        raise ValueError('Common endpoint outside environmental coverage')
    input_identity = workload.sha256, environment.sha256
    executors, observers, own = {}, {}, {}
    for name in BRANCHES:
        observer = ObservedEDF(workload.start, capacity, parameters, (target,), name)
        executor = LocalExecutor(workload, environment, capacity, power_factory(), observer, drain=True)
        observer.prefix_callback = lambda policy, spec, context, ex=executor: executor_prefix(ex, policy, spec, context)
        own[name] = executor.run(stop_at_drain=True)
        observer.assert_targets_observed()
        if executor.drained_at is None or own[name].summary['completed'] != len(workload.tasks):
            raise RuntimeError('Both branches must complete the full drain; coverage insufficient')
        executors[name], observers[name] = executor, observer
    records = {name: observer.records[target.case_id] for name, observer in observers.items()}
    prefixes = {name: record['prefix'] for name, record in records.items()}
    if prefixes[BRANCHES[0]] != prefixes[BRANCHES[1]]:
        raise RuntimeError('Counterfactual pre-intervention prefix mismatch; pair invalid')
    if records[BRANCHES[0]]['features'] != records[BRANCHES[1]]['features']:
        raise RuntimeError('Counterfactual current online feature mismatch')
    common = max(minimum, *(executor.observed_end for executor in executors.values()))
    results = {name: executor.extend_idle(common) for name, executor in executors.items()}
    if input_identity != (workload.sha256, environment.sha256):
        raise RuntimeError('Frozen inputs changed during counterfactual run')
    features = records[BRANCHES[0]]['features']
    return PairedRun(target, results, own, prefixes, True, features,
        {name: observer.interventions for name, observer in observers.items()}, common,
        {name: executor.drained_at for name, executor in executors.items()},
        _task_impacts(target, results, set(features['known_task_ids'])))
