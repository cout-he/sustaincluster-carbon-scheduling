"""E1-4A isolates cap fallback from actual bounded shadow risk checks.

The historical CarsPolicy and its registry are unchanged. Every new decision
first evaluates the real RiskOnly rule, then applies exactly one extra module.
"""
from dataclasses import asdict

from .cars import CarsPolicy, make_cars_policy
from .cars_feasibility import PlannerResult, ShadowPlanner
from .cars_state import ResourceSnapshot
from .contracts import STEP
from .policies import PolicyDecision

ABLATION_SCHEMA = 'e1-4a-feasibility-ablation-v1'
ABLATION_POLICY_NAMES = ('edf_carbon_threshold', 'cars_risk_only', 'cars_full',
                         'risk_cap_only', 'risk_shadow_only')


class RiskAblationPolicy(CarsPolicy):
    """Same observed history and EDF order; no capacity or execution changes."""
    def __init__(self, name, parameters, experiment_start, capacity):
        if name not in ABLATION_POLICY_NAMES[3:]:
            raise ValueError('Unknown feasibility ablation')
        super().__init__('cars_risk_only', parameters, experiment_start, capacity)
        self.name = name
        self.max_pending = parameters.get('max_pending', 256)
        if type(self.max_pending) is not int or self.max_pending < 1:
            raise ValueError('Positive integer pending cap required')
        self.horizon_steps = parameters.get('horizon_steps', 8)
        self.shadow = name == 'risk_shadow_only'
        if self.shadow:
            self.planner = ShadowPlanner(self.horizon_steps, self.max_pending)
        self.counters.update(queue_limit_encounters=0, queue_limit_eligible=0,
            cap_execute_guards=0, uncertain_risk_fallbacks=0,
            shadow_safe=0, shadow_risk=0, shadow_uncertain=0)

    def _record_replacement(self, diag, original, decision):
        if decision.reason != original.reason:
            self.reason_counts[original.reason] -= 1
            if not self.reason_counts[original.reason]:
                del self.reason_counts[original.reason]
            self.reason_counts[decision.reason] += 1
        diag.update(reason=decision.reason, defer=decision.defer)
        return decision

    def decide(self, spec, context):
        # Use original RiskOnly decisions, pressure and side effects directly.
        original = super().decide(spec, context)
        diag = self.diagnostics[-1]
        over_cap = len(self.pending) > self.max_pending
        diag.update(ablation_schema=ABLATION_SCHEMA, shadow_enabled=self.shadow,
            pending_limit_exceeded=over_cap, risk_only_defer=original.defer,
            risk_only_reason=original.reason, shadow_planner_executed=False)
        self.counters['queue_limit_encounters'] += int(over_cap)
        if not original.defer:
            return original
        self.counters['queue_limit_eligible'] += int(over_cap)
        if not self.shadow:
            if over_cap:
                self.counters['cap_execute_guards'] += 1
                self.counters['conservative_fallbacks'] += 1
                return self._record_replacement(diag, original,
                    PolicyDecision(False, 'cap_execute_guard'))
            return original

        if over_cap:
            # Do not invoke the planner outside its declared planning range.
            # All pending jobs remain unverified; this is a RiskOnly fallback.
            unknown = tuple(sorted(self.pending))
            common = tuple(sorted(r.task_id for r in self.running
                if r.deadline is not None and r.finish_time > r.deadline))
            result = PlannerResult('uncertain', common, common, unknown, unknown,
                None, None, 0, context.now+self.horizon_steps*STEP, 'pending_limit', 0)
        else:
            snapshot = ResourceSnapshot(context.now, self.snapshot.capacity,
                self.available, tuple(self.pending.values()), tuple(self.running),
                blocked_until=tuple(self.blocked.items()))
            result = self.planner.compare(spec, snapshot)
            diag['shadow_planner_executed'] = True
        self.counters['planner_calls'] += result.planner_calls
        self.counters['shadow_'+result.status] += 1
        planner = asdict(result)
        for key in ('candidate_start_now', 'candidate_start_defer', 'horizon_end'):
            if planner[key] is not None:
                planner[key] = planner[key].isoformat()
        diag['planner'] = planner
        if result.status == 'risk':
            self.counters['known_risk_guards'] += 1
            return self._record_replacement(diag, original,
                PolicyDecision(False, 'known_deadline_risk'))
        if result.status == 'uncertain':
            self.counters['uncertain_risk_fallbacks'] += 1
            return self._record_replacement(diag, original,
                PolicyDecision(True, 'shadow_uncertain_risk_fallback'))
        return original


def make_ablation_policy(name, parameters, experiment_start, capacity):
    """Independent registry; historical controls use original classes exactly."""
    if name not in ABLATION_POLICY_NAMES:
        raise ValueError('Unknown E1-4A policy')
    if name in ABLATION_POLICY_NAMES[:3]:
        return make_cars_policy(name, parameters, experiment_start, capacity)
    return RiskAblationPolicy(name, parameters, experiment_start, capacity)
