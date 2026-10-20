"""Matched observation-only E1 rules on the unchanged E0 execution contract.

A uses EDF without active waiting. B and C propose the identical CI-triggered,
power-ranked prefix. B checks completion responsibility only; C uses E0 DDF.
Deadline values recorded by B are diagnostics and never approval conditions.
"""
import statistics
from time import perf_counter_ns

from experiments.platform_v1.environment import Action
from .adapter import Decision as E0Decision, ResearchAdapter
from .completion_guard import CompletionGuard
from .deadline_filter import CandidateCheck, FilterResult as E0FilterResult
from .deadline_filter import filter_requests, ranked_request
from .edf_calendar import edf_key, preview_now, simulate


def filter_completion_only(snapshot, reference, proposed):
    """Apply shared CG cumulatively to the fixed request without DDF rejection."""
    begin = perf_counter_ns()
    proposed = tuple(proposed)
    if reference.snapshot != snapshot or reference.forced_defer:
        raise ValueError('Reference must be the no-defer plan of the same immutable snapshot')
    if len(set(proposed)) != len(proposed) or not set(proposed) <= set(reference.first_slot.started):
        raise ValueError('Requests must be unique members of reference current-slot start set')
    if not CompletionGuard.accepts(reference):
        return E0FilterResult(proposed, (), (), reference, preview_now(snapshot), True,
                              perf_counter_ns() - begin)

    approved, checks = [], []
    on_time, late = reference.on_time_ids, reference.late_ids
    baseline_late = reference.total_lateness(late)
    for identity in proposed:
        start = perf_counter_ns()
        # A rejected position is consumed. Never refill or rebuild the prefix.
        trial_ids = tuple(approved + [identity])
        trial = simulate(snapshot, frozenset(trial_ids))
        accepted = CompletionGuard.accepts(trial)
        reasons = () if accepted else ('COMPLETION_GUARD_REJECTED',)
        lost = tuple(sorted(on_time - trial.on_time_ids))
        trial_late = trial.total_lateness(late)
        if accepted:
            approved.append(identity)
        checks.append(CandidateCheck(identity, trial_ids, accepted, reasons, lost,
            baseline_late, trial_late, trial.max_release, perf_counter_ns() - start,
            trial.elapsed_ns, trial.dispatch_slots, trial.task_checks))

    final = simulate(snapshot, frozenset(approved))
    preview = preview_now(snapshot, frozenset(approved))
    assert final.first_slot == preview, 'Final joint calendar and slot replay diverged'
    assert CompletionGuard.accepts(final)
    return E0FilterResult(proposed, tuple(approved), tuple(checks), final, preview, False,
                          perf_counter_ns() - begin)


class MatchedRuleAdapter(ResearchAdapter):
    """E1 arm with fixed 4h/24h trigger, power prefix <=16 and fixed20 action.

    Explicit action_id/proposed arguments are synthetic engineering overrides.
    Production callers omit both so the observation alone determines requests.
    The shared E0 observe/execute/summary methods retain their original checks.
    """

    def __init__(self, arm, cfg):
        if arm not in ('A', 'B', 'C'):
            raise ValueError('E1 arm must be A, B or C')
        evaluation = cfg['evaluation']
        if (evaluation['threshold_warmup_hours'], evaluation['threshold_history_hours'],
                evaluation['fixed_supply_c']) != (4, 24, 20):
            raise ValueError('E1 requires frozen 4h warmup, 24h history and fixed20 supply')
        self.arm = arm
        super().__init__()

    def reset(self):
        super().reset()
        self.last_decision_ns = 0
        self.last_trigger = None

    def decide(self, obs, action_id=None, proposed=None):
        begin = perf_counter_ns()
        try:
            snapshot = self.observe(obs)
            reference = simulate(snapshot)
            cg_pass = self.guard.inspect_reference(reference)
            now = snapshot.minute
            history = tuple((minute, ci) for minute, ci in obs['ci_history']
                            if now - 24 * 60 <= minute < now)
            threshold = statistics.median(ci for _, ci in history) if history else None
            warmup_pass = now >= 4 * 60
            triggered = warmup_pass and bool(history) and obs['ci_g_per_kwh'] > threshold
            self.last_trigger = dict(threshold=threshold, history=history,
                                     history_count=len(history), warmup_pass=warmup_pass,
                                     triggered=triggered)
            if self.arm == 'A':
                request = ()
            elif proposed is not None:
                request = tuple(proposed)
            elif action_id is not None:
                request = ranked_request(snapshot, reference, action_id)
            else:
                request = ranked_request(snapshot, reference, 3) if triggered else ()
            if len(set(request)) != len(request) or not set(request) <= set(reference.first_slot.started):
                raise ValueError('Request must be a fixed unique prefix of reference starters')

            # Preserve the intended request in Decision after any CG latch, as E0.
            effective = () if self.guard.failure_latched else request
            apply_filter = filter_requests if self.arm == 'C' else filter_completion_only
            result = apply_filter(snapshot, reference, effective)
            order = tuple(t.task_id for t in sorted(snapshot.pending,
                                                    key=lambda t: edf_key(t, snapshot.sla)))
            action = Action(order, frozenset(result.approved), 20.0)
            return E0Decision(snapshot, reference, result, request, action, cg_pass,
                              self.guard.failure_latched)
        finally:
            self.last_decision_ns = perf_counter_ns() - begin
