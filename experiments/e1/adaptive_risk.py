"""Versioned four-action eligibility; frozen v1 risk is a heuristic veto only."""
from dataclasses import asdict
import math
from numbers import Integral

from .policies import PolicyDecision
from .task_risk import TaskRiskPolicy
from .task_risk_state import BudgetParameters, _budget_from_pressures

SCHEMA = 'e1-6b-adaptive-delay-v1'
ACTION_SLACK_MINUTES = (None, 60, 30, 15)


class AdaptiveDelayPolicy(TaskRiskPolicy):
    """Same EDF hooks and exact frozen arithmetic; never owns task allocation."""
    version = SCHEMA

    def __init__(self, parameters, experiment_start, capacity, *, risk_filter=False):
        if not isinstance(risk_filter, bool):
            raise ValueError('risk_filter must be bool')
        if parameters.get('future_ci',False) or parameters.get('threshold_g_per_kwh') != 161.075:
            raise ValueError('Frozen threshold 161.075 and current CI only required')
        super().__init__('task_risk_full', parameters, experiment_start, capacity)
        if self.parameters != BudgetParameters():
            raise ValueError('E1-6A v1 risk coefficients are frozen')
        self.name = 'adaptive_task_delay'
        self.risk_filter = risk_filter
        self.set_action(0)

    def set_action(self, action):
        if isinstance(action,bool) or not isinstance(action,Integral) or not 0 <= action < 4:
            raise ValueError('Action must be an integer 0/1/2/3')
        self.action = int(action)

    def decide(self, spec, context):
        if context.now != self.snapshot.now or self.pending.get(spec.task_id) is not spec:
            raise ValueError('Candidate missing from current observed state')
        self.counters['candidate_checks'] += 1
        slack=(self.grid_starts[spec.task_id]-context.now).total_seconds()/60
        threshold=ACTION_SLACK_MINUTES[self.action]
        eligible=(self.action != 0 and context.ci_g_per_kwh > self.threshold_g_per_kwh
                  and slack >= threshold)
        budget=None
        veto=False
        if self.action == 0:
            decision=PolicyDecision(False,'adaptive_edf_execute')
        elif context.ci_g_per_kwh <= self.threshold_g_per_kwh:
            decision=PolicyDecision(False,'low_ci_execute')
        elif not eligible:
            decision=PolicyDecision(False,'below_action_slack')
        else:
            if self.risk_filter:
                predecessors=tuple(s for s in self.ordered[:self.rank[spec.task_id]]
                                   if s.task_id in self.pending)
                self.counters['predecessor_scans'] += len(predecessors)
                work=tuple(self.urgent_work[s.task_id] for s in predecessors
                           if s.task_id in self.urgent_work)
                utilization=tuple(math.fsum(r.demand[k] for r in self.running)/self.capacity[k]
                                  for k in range(3))
                queue=tuple(math.fsum(row[k] for row in work)/(self.capacity[k]*self.parameters.history_minutes)
                            for k in range(3))
                budget=_budget_from_pressures(spec,context.now,self.capacity,utilization,queue,
                                             self.pressure,self.parameters,self.aggregation)
                self.counters['budget_checks'] += 1
                veto=budget.safe_slack_minutes < threshold
            decision=PolicyDecision(not veto,'adaptive_risk_veto' if veto else 'adaptive_high_ci_defer')
            if veto:self.counters['budget_guards'] += 1
            else:self.counters['policy_defer_checks'] += 1
        self.reason_counts[decision.reason] += 1
        self.diagnostics.append(dict(time=context.now.isoformat(),task_id=spec.task_id,
            action=self.action,slack_threshold_minutes=threshold,ci_g_per_kwh=context.ci_g_per_kwh,
            solo_slack_minutes=slack,solo_eligible=eligible,risk_filter=self.risk_filter,
            filter_veto=veto,budget=asdict(budget) if budget is not None else None,
            available=list(self.available),pending_count=len(self.pending),running_count=len(self.running),
            defer=decision.defer,reason=decision.reason))
        return decision
