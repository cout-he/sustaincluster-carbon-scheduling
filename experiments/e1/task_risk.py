"""Independent task-matched EDF carbon policy; shared executor owns execution."""
from collections import Counter
from dataclasses import asdict, fields, replace
import math

from .cars import make_cars_policy
from .cars_state import ArrivalHistory, demand
from .contracts import STEP, finite
from .policies import EDFBackfill, PolicyDecision
from .task_risk_state import (BudgetParameters, VERSION, _budget_from_pressures, calculate_budget,
                              latest_grid_start, occupied_minutes)

POLICY_NAMES = ('task_risk_full', 'task_risk_no_history', 'task_risk_no_urgent',
                'task_risk_no_utilization', 'task_risk_max_required')
ALIAS = 'task_risk_budget_v1'


class TaskRiskPolicy(EDFBackfill):
    """Current state only; static task facts cached, exact fsum queries retained."""
    version = VERSION

    def __init__(self, name, parameters, experiment_start, capacity, *, query_backend='cached_work'):
        name = 'task_risk_full' if name == ALIAS else name
        if name not in POLICY_NAMES:
            raise ValueError('Unknown task-risk policy')
        self.name = name
        if query_backend not in ('reference', 'cached_work'):
            raise ValueError('Unknown task-risk query backend')
        self.query_backend = query_backend
        self.threshold_g_per_kwh = finite(parameters['threshold_g_per_kwh'], 'carbon threshold')
        keys = {f.name for f in fields(BudgetParameters)}
        self.parameters = BudgetParameters(**{k: v for k, v in parameters.items() if k in keys})
        switches = {'task_risk_no_history': 'a_h', 'task_risk_no_urgent': 'a_q',
                    'task_risk_no_utilization': 'a_u'}
        if name in switches:
            self.parameters = replace(self.parameters, **{switches[name]: 0})
        self.aggregation = 'max_required' if name == 'task_risk_max_required' else 'weighted'
        # Preserve legacy unknown and open-left observation window semantics.
        self.history = ArrivalHistory(experiment_start, capacity, self.parameters.history_minutes,
                                      cold_start_buffer_steps=self.parameters.cold_buffer_steps)
        self.capacity = self.history.capacity
        self.diagnostics = []
        self.reason_counts = Counter()
        self.counters = dict(candidate_checks=0, budget_checks=0, budget_guards=0,
                             policy_defer_checks=0, predecessor_scans=0)

    def on_step_start(self, snapshot):
        if snapshot.capacity != self.capacity:
            raise ValueError('Observed capacity differs from fixed policy capacity')
        self.snapshot = snapshot
        self.ordered = snapshot.pending
        self.rank = {s.task_id: i for i, s in enumerate(self.ordered)}
        self.pending = {s.task_id: s for s in self.ordered}
        self.running = list(snapshot.running)
        self.available = snapshot.available
        self.history.observe(snapshot.now, snapshot.arrivals)
        self.pressure = self.history.pressure(snapshot.now)
        self.grid_starts = {s.task_id: latest_grid_start(s) for s in self.ordered}
        self.urgent_work = {s.task_id: tuple(v*occupied_minutes(s) for v in demand(s))
                            for s in self.ordered if
                            (self.grid_starts[s.task_id]-snapshot.now).total_seconds()/60 <= self.parameters.history_minutes}

    def on_candidate_resources(self, now, available):
        if now != self.snapshot.now:
            raise ValueError('Stale task-risk candidate state')
        self.available = tuple(available)

    def on_task_started(self, running):
        if running.task_id not in self.pending:
            raise ValueError('Unknown task-risk start notification')
        del self.pending[running.task_id]
        self.running.append(running)

    def on_task_deferred(self, task_id, until):
        if task_id not in self.pending or until != self.snapshot.now + STEP:
            raise ValueError('Invalid task-risk defer notification')
        # Defer does not remove a competitor or reserve any running resource.

    def decide(self, spec, context):
        if context.now != self.snapshot.now or self.pending.get(spec.task_id) is not spec:
            raise ValueError('Candidate missing from current observed state')
        self.counters['candidate_checks'] += 1
        slack = (self.grid_starts[spec.task_id] - context.now).total_seconds()/60
        budget = None
        if context.ci_g_per_kwh <= self.threshold_g_per_kwh:
            decision = PolicyDecision(False, 'low_ci_execute')
        elif slack < 15:
            decision = PolicyDecision(False, 'no_grid_slack')
        else:
            predecessors = tuple(s for s in self.ordered[:self.rank[spec.task_id]]
                                 if s.task_id in self.pending)
            self.counters['predecessor_scans'] += len(predecessors)
            if self.query_backend == 'reference':
                budget = calculate_budget(spec, context.now, self.capacity, self.running,
                                          predecessors, self.pressure, self.parameters, self.aggregation)
            else:
                # Cache only immutable grid/work facts, never aggregate by float
                # subtraction. Same member order/products/fsum as reference.
                work = tuple(self.urgent_work[s.task_id] for s in predecessors if s.task_id in self.urgent_work)
                utilization = tuple(math.fsum(r.demand[k] for r in self.running)/self.capacity[k] for k in range(3))
                queue = tuple(math.fsum(row[k] for row in work)/(self.capacity[k]*self.parameters.history_minutes)
                              for k in range(3))
                budget = _budget_from_pressures(spec, context.now, self.capacity, utilization, queue,
                                                self.pressure, self.parameters, self.aggregation)
            self.counters['budget_checks'] += 1
            if budget.safe_slack_minutes < 15:
                self.counters['budget_guards'] += 1
                decision = PolicyDecision(False, 'task_risk_guard')
            else:
                self.counters['policy_defer_checks'] += 1
                decision = PolicyDecision(True, 'high_ci_defer')
        self.reason_counts[decision.reason] += 1
        self.diagnostics.append(dict(time=context.now.isoformat(), task_id=spec.task_id, policy=self.name,
            ci_g_per_kwh=context.ci_g_per_kwh, threshold_g_per_kwh=self.threshold_g_per_kwh,
            eligible=budget is not None, budget=asdict(budget) if budget is not None else None,
            latest_grid_start=self.grid_starts[spec.task_id].isoformat(), solo_slack_minutes=slack,
            available=list(self.available), pending_count=len(self.pending), running_count=len(self.running),
            effective_history_minutes=self.pressure.effective_minutes,
            history_observed_tasks=self.pressure.observed_tasks, defer=decision.defer, reason=decision.reason))
        return decision


def make_task_risk_policy(name, parameters, experiment_start, capacity):
    """Only this new registry knows the new names. Old constructors unchanged."""
    if name in POLICY_NAMES or name == ALIAS:
        return TaskRiskPolicy(name, parameters, experiment_start, capacity)
    return make_cars_policy(name, parameters, experiment_start, capacity)
