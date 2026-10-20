"""Strict EDF-reference DDF, with the shared CG on every counterfactual."""
from dataclasses import dataclass
from time import perf_counter_ns

from .edf_calendar import simulate, preview_now, edf_key
from .completion_guard import CompletionGuard

LATENESS_EPS_MINUTES = 1e-9  # fixed arithmetic tolerance, never tuned to outcomes


@dataclass(frozen=True)
class CandidateCheck:
    task_id: str
    trial_defer: tuple
    approved: bool
    reasons: tuple
    lost_on_time_ids: tuple
    reference_late_total: float
    trial_late_total: float
    max_release: float
    elapsed_ns: int
    calendar_elapsed_ns: int
    dispatch_slots: int
    task_checks: int


@dataclass(frozen=True)
class FilterResult:
    proposed: tuple
    approved: tuple
    checks: tuple
    plan: object
    preview: object
    reference_guard_failed: bool
    elapsed_ns: int


def ranked_request(snapshot, reference, action_id):
    if action_id not in range(7): raise ValueError('Action ID must be in 0..6')
    if action_id == 0: return ()
    candidates = {t.task_id: t for t in snapshot.pending
                  if t.task_id in reference.first_slot.started}
    def key(identity):
        t = candidates[identity]
        score = ((0.005*t.demand[0] + 0.225*t.demand[1]) / 0.94 if action_id <= 3
                 else edf_key(t, snapshot.sla)[0]-snapshot.minute-t.duration_minutes)
        return (-score, *edf_key(t, snapshot.sla))
    budget = (1, 4, 16)[(action_id-1) % 3]
    return tuple(sorted(candidates, key=key)[:budget])


def filter_requests(snapshot, reference, proposed):
    begin = perf_counter_ns()
    proposed = tuple(proposed)
    if reference.snapshot != snapshot or reference.forced_defer:
        raise ValueError('Reference must be the no-defer plan of the same immutable snapshot')
    if len(set(proposed)) != len(proposed) or not set(proposed) <= set(reference.first_slot.started):
        raise ValueError('Requests must be unique members of reference current-slot start set')
    if not CompletionGuard.accepts(reference):
        return FilterResult(proposed, (), (), reference, preview_now(snapshot), True, perf_counter_ns()-begin)
    approved = []; checks = []
    on_time = reference.on_time_ids; late = reference.late_ids
    baseline_late = reference.total_lateness(late)
    for identity in proposed:
        start = perf_counter_ns()
        # Always restart from the identical real snapshot. A rejection consumes
        # this prefix position; never replace it with a different candidate.
        trial_ids = tuple(approved + [identity])
        trial = simulate(snapshot, frozenset(trial_ids))
        lost = tuple(sorted(on_time - trial.on_time_ids))
        trial_late = trial.total_lateness(late)
        reasons = []
        if lost: reasons.append('REFERENCE_ON_TIME_SET_LOST')
        if trial_late > baseline_late + LATENESS_EPS_MINUTES:
            reasons.append('ORIGINAL_LATE_TOTAL_INCREASED')
        if not CompletionGuard.accepts(trial): reasons.append('COMPLETION_GUARD_REJECTED')
        if not reasons:
            assert len(trial.late_ids) <= len(reference.late_ids)
            approved.append(identity)
        checks.append(CandidateCheck(identity, trial_ids, not reasons, tuple(reasons), lost,
                                     baseline_late, trial_late, trial.max_release, perf_counter_ns()-start,
                                     trial.elapsed_ns, trial.dispatch_slots, trial.task_checks))
    # Repeat final joint counterfactual and separately preview the whole slot.
    final = simulate(snapshot, frozenset(approved))
    preview = preview_now(snapshot, frozenset(approved))
    assert final.first_slot == preview, 'Final joint calendar and slot replay diverged'
    assert on_time <= final.on_time_ids
    assert final.total_lateness(late) <= baseline_late + LATENESS_EPS_MINUTES
    assert CompletionGuard.accepts(final)
    return FilterResult(proposed, tuple(approved), tuple(checks), final, preview, False, perf_counter_ns()-begin)
