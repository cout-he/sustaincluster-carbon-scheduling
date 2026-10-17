"""Hand-checked causal queue continuations, bounded risk and physical masks."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.e1.cars_state import RunningTask
from tests.e1.helpers import START, task


def at(minutes):
    return START + pd.Timedelta(minutes=minutes)


def test_start_is_forced_once_and_physical_finish_differs_from_grid_release():
    from experiments.joint_v3.reservation import schedule_known
    candidate = task('candidate', duration=16, sla=5)
    other = task('other', duration=10, sla=2)
    plan = schedule_known(candidate, START, (1, 1, 1), (other, candidate), ())
    assert plan.valid and plan.settled
    assert plan.starts == {'candidate': START, 'other': at(30)}
    assert plan.finishes == {'candidate': at(16), 'other': at(40)}
    np.testing.assert_allclose(plan.loads, [[1, 0, 1]] * 3)


def test_wait_continuation_uses_same_running_state_and_edf_backfill():
    from experiments.joint_v3.reservation import schedule_known
    candidate = task('candidate', duration=20, cpu=.4, mem=.4, sla=5)
    urgent = task('urgent', duration=15, cpu=.8, mem=.8, sla=2)
    running = RunningTask('running', (.6, 0, .6), at(7), at(10))
    start = schedule_known(candidate, START, (1, 1, 1), (urgent,), (running,))
    wait = schedule_known(candidate, START, (1, 1, 1), (urgent,), (running,), delay_slots=1)
    assert start.valid and wait.valid
    assert start.finishes['running'] == wait.finishes['running'] == at(7)
    assert start.starts['urgent'] == at(30)
    assert wait.starts['urgent'] == at(15)
    assert wait.starts['candidate'] == at(30)
    np.testing.assert_allclose(start.loads[0], [1, 0, 1])
    np.testing.assert_allclose(wait.loads[0], [.6, 0, .6])


def test_already_committed_waits_are_preserved_in_both_counterfactuals():
    from experiments.joint_v3.reservation import schedule_known
    candidate = task('candidate', duration=15, cpu=.4, mem=.4, sla=6)
    deferred = task('deferred', duration=15, cpu=.6, mem=.6, sla=1)
    for delay in (0, 1):
        plan = schedule_known(candidate, START, (1, 1, 1), (deferred,), (),
                              delay_slots=delay, blocked_until={'deferred': at(30)})
        assert plan.valid
        assert plan.starts['deferred'] == at(30)


@pytest.mark.parametrize('resource', [0, 1, 2])
def test_all_three_resources_limit_nonpreemptive_execution(resource):
    from experiments.joint_v3.reservation import schedule_known
    demands = [0., 0., 0.]
    demands[resource] = .6
    candidate = task('candidate', duration=16, cpu=demands[0], gpu=demands[1],
                     mem=demands[2], sla=5)
    other = replace(candidate, task_id='other')
    plan = schedule_known(candidate, START, (1, 1, 1), (other,), ())
    assert plan.valid and plan.starts['other'] == at(30)
    assert np.max(plan.loads[:, resource]) == pytest.approx(.6)


def test_future_arrival_is_rejected_without_scheduling_it():
    from experiments.joint_v3.reservation import schedule_known
    plan = schedule_known(task('candidate'), START, (1, 1, 1),
                          (task('future', arrival=15),), ())
    assert not plan.valid and not plan.settled
    assert 'future' in plan.reason
    assert 'future' not in plan.starts


@pytest.mark.parametrize('kwargs,reason', [({'max_tasks': 1}, 'queue_limit'),
                                         ({'max_slots': 1}, 'horizon')])
def test_bounds_invalidate_instead_of_silently_truncating_known_work(kwargs, reason):
    from experiments.joint_v3.reservation import schedule_known
    plan = schedule_known(task('candidate'), START, (1, 1, 1),
                          (task('other'),), (), **kwargs)
    assert not plan.valid and not plan.settled
    assert reason in plan.reason
    assert plan.unresolved_task_ids


def test_running_job_beyond_horizon_is_not_reported_settled():
    from experiments.joint_v3.reservation import schedule_known
    plan = schedule_known(task('candidate', cpu=.1, mem=.1), START, (1, 1, 1), (),
                          (RunningTask('long', (.5, 0, .5), at(61)),), max_slots=4)
    assert not plan.valid and not plan.settled
    assert plan.finishes['long'] == at(61)
    assert 'long' in plan.unresolved_task_ids
    assert plan.unresolved_duration_minutes == 15


def test_unstarted_known_tail_has_serial_occupancy_bound_and_early_rejection_has_none():
    from experiments.joint_v3.reservation import schedule_known
    truncated = schedule_known(task('candidate'), START, (1, 1, 1),
                               (task('other', duration=16),), (), max_slots=1)
    assert truncated.unresolved_duration_minutes == 30
    rejected = schedule_known(task('candidate'), START, (1, 1, 1),
                              (task('other'),), (), max_tasks=1)
    assert rejected.unresolved_duration_minutes is None


def test_infeasible_immediate_start_is_invalid_and_does_not_overallocate():
    from experiments.joint_v3.reservation import schedule_known
    plan = schedule_known(task('candidate'), START, (1, 1, 1), (),
                          (RunningTask('running', (1, 0, 1), at(15)),))
    assert not plan.valid
    assert 'candidate' not in plan.starts
    assert np.all(plan.loads <= np.asarray((1, 1, 1)))


def test_duplicate_ids_and_oversized_tasks_fail_conservatively():
    from experiments.joint_v3.reservation import schedule_known
    candidate = task('candidate')
    duplicate = schedule_known(candidate, START, (1, 1, 1),
                               (task('other'), task('other')), ())
    impossible = schedule_known(task('huge', cpu=2), START, (1, 1, 1), (), ())
    assert not duplicate.valid and 'duplicate' in duplicate.reason
    assert not impossible.valid and 'capacity' in impossible.reason


def test_wait_risk_counts_endogenous_competition_and_minutes_not_just_solo_slack():
    from experiments.joint_v3.risk_estimator import RiskEstimator
    candidate = task('candidate', duration=15, sla=2)
    other = task('other', duration=30, sla=2)
    result = RiskEstimator({'time_budget_ms': 1000}).estimate(
        candidate, START, (1, 1, 1), (other,), ())
    assert result.risk_estimation_valid
    assert result.delta_deadline_misses == 1
    assert result.extra_tardiness_minutes == 15
    assert result.candidate_extra_tardiness_minutes == 15
    assert result.other_extra_tardiness_minutes == 0
    assert result.safe_wait_budget_minutes == 0
    assert result.examined_wait_slots == 1


def test_safe_wait_budget_is_positive_grid_bounded_and_deadline_uses_physical_finish():
    from experiments.joint_v3.risk_estimator import RiskEstimator
    candidate = task('candidate', duration=20, sla=3)
    result = RiskEstimator({'max_wait_slots': 16, 'time_budget_ms': 1000}).estimate(
        candidate, START, (1, 1, 1), (), ())
    assert result.risk_estimation_valid and result.risk_confidence > 0
    assert result.safe_wait_budget_minutes == 30
    assert result.delta_deadline_misses == 0
    assert result.extra_tardiness_minutes == 0
    bounded = RiskEstimator({'max_wait_slots': 1, 'time_budget_ms': 1000}).estimate(
        candidate, START, (1, 1, 1), (), ())
    assert bounded.safe_wait_budget_minutes == 15


def test_wait_can_improve_other_tasks_without_negative_extra_tardiness():
    from experiments.joint_v3.risk_estimator import RiskEstimator
    candidate = task('candidate', duration=20, cpu=.4, mem=.4, sla=5)
    urgent = task('urgent', duration=15, cpu=.8, mem=.8, sla=2)
    result = RiskEstimator({'time_budget_ms': 1000}).estimate(
        candidate, START, (1, 1, 1), (urgent,),
        (RunningTask('running', (.6, 0, .6), at(7)),))
    assert result.risk_estimation_valid
    assert result.delta_deadline_misses == -1
    assert result.extra_tardiness_minutes == 0
    assert result.other_delta_deadline_misses == -1


def test_unresolved_risk_and_expired_time_budget_forbid_discretionary_delay():
    from experiments.joint_v3.risk_estimator import RiskEstimator
    candidate = task('candidate', duration=15, sla=5)
    result = RiskEstimator({'max_tasks': 1}).estimate(
        candidate, START, (1, 1, 1), (task('other'),), ())
    assert not result.risk_estimation_valid and result.safe_wait_budget_minutes == 0
    timed = RiskEstimator({'time_budget_ms': 1e-9}).estimate(
        candidate, START, (1, 1, 1), (), ())
    assert not timed.risk_estimation_valid and timed.safe_wait_budget_minutes == 0
    assert 'time_budget' in timed.reason


def test_own_latest_start_fastpath_never_claims_uncomputed_queue_risk():
    from experiments.joint_v3.risk_estimator import RiskEstimator
    result = RiskEstimator().estimate(task('urgent', duration=15, sla=1),
                                      START, (1, 1, 1), (), ())
    assert not result.risk_estimation_valid and result.safe_wait_budget_minutes == 0
    assert result.start_schedule is result.wait_schedule is None
    assert result.reason == 'own_latest_start_no_wait'


def test_cache_keys_include_committed_waits_and_preserve_snapshot_inputs():
    from experiments.joint_v3.risk_estimator import RiskEstimator
    estimator = RiskEstimator({'time_budget_ms': 1000})
    candidate = task('candidate', duration=15, sla=5)
    other = task('other', duration=15, sla=2)
    pending = [other, candidate]
    first = estimator.estimate(candidate, START, (1, 1, 1), pending, ())
    cached = estimator.estimate(candidate, START, (1, 1, 1), pending, ())
    changed = estimator.estimate(candidate, START, (1, 1, 1), pending, (),
                                 blocked_until=(('other', at(30)),))
    assert cached.cache_hit and not first.cache_hit and not changed.cache_hit
    assert changed.wait_schedule.starts['other'] == at(30)
    assert pending == [other, candidate]


def guidance(**changes):
    values = dict(risk_estimation_valid=True, risk_confidence=1.,
                  safe_wait_budget_minutes=15., delta_deadline_misses=0)
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize('changes', [dict(risk_estimation_valid=False),
                                      dict(risk_confidence=0),
                                      dict(safe_wait_budget_minutes=0),
                                      dict(delta_deadline_misses=1)])
def test_uncertain_or_harmful_risk_masks_only_discretionary_wait(changes):
    from experiments.joint_v3.action_shield import shield
    result = shield(task('candidate', duration=15, sla=4), START,
                    (1, 1, 1), (1, 1, 1), (), (), guidance(**changes))
    assert result.mask == (True, False) and result.wait_masked


def test_capacity_wait_remains_legal_even_when_candidate_is_overdue():
    from experiments.joint_v3.action_shield import shield
    result = shield(task('candidate', duration=15, sla=1), at(30),
                    (0, 1, 0), (1, 1, 1), (), (), guidance())
    assert result.mask == (False, True) and result.capacity_wait


def test_causal_reservation_protects_known_urgent_task_even_when_risk_unreliable():
    from experiments.joint_v3.action_shield import shield
    candidate = task('candidate', duration=20, cpu=.4, mem=.4, sla=5)
    urgent = task('urgent', duration=15, cpu=.8, mem=.8, sla=2)
    result = shield(candidate, START, (.4, 1, .4), (1, 1, 1), (urgent,),
                    (RunningTask('running', (.6, 0, .6), at(7)),),
                    guidance(risk_estimation_valid=False))
    assert result.mask == (False, True)
    assert result.protected_task_ids == ('urgent',)
    assert result.resource_protection_triggered and result.risk_unreliable


def test_latest_start_and_ablation_preserve_physical_action_legality():
    from experiments.joint_v3.action_shield import shield
    urgent = task('urgent', duration=15, sla=1)
    assert shield(urgent, START, (1, 1, 1), (1, 1, 1), (), (), guidance()).mask == (True, False)
    flexible = task('flexible', duration=15, sla=4)
    assert shield(flexible, START, (1, 1, 1), (1, 1, 1), (), (), guidance(),
                  enabled=False).mask == (True, True)


def test_shield_rejects_unarrived_reservations():
    from experiments.joint_v3.action_shield import shield
    with pytest.raises(ValueError, match='future'):
        shield(task('candidate', duration=15, sla=5), START, (1, 1, 1), (1, 1, 1),
               (task('future', arrival=15),), (), guidance())


@pytest.mark.parametrize('changes', [dict(risk_confidence=None),
                                      dict(safe_wait_budget_minutes=None),
                                      dict(delta_deadline_misses=float('nan'))])
def test_malformed_risk_guidance_falls_back_to_a_physical_start(changes):
    from experiments.joint_v3.action_shield import shield
    result = shield(task('candidate', duration=15, sla=5), START,
                    (1, 1, 1), (1, 1, 1), (), (), guidance(**changes))
    assert result.mask == (True, False)


def test_edf_ties_are_reproducible_across_input_order():
    from experiments.joint_v3.reservation import schedule_known
    candidate = task('candidate', duration=15, sla=5)
    jobs = (task('b', duration=15, sla=3), task('a', duration=15, sla=3))
    first = schedule_known(candidate, START, (1, 1, 1), jobs, (), delay_slots=1)
    second = schedule_known(candidate, START, (1, 1, 1), jobs[::-1], (), delay_slots=1)
    assert first.starts == second.starts == {'a': START, 'b': at(15), 'candidate': at(30)}


def test_config_rejects_unbounded_or_nonpositive_planning_limits():
    from experiments.joint_v3.risk_estimator import RiskEstimator
    for config in ({'max_tasks': 0}, {'max_slots': 1000000}, {'max_wait_slots': 0},
                   {'time_budget_ms': float('inf')}):
        with pytest.raises(ValueError):
            RiskEstimator(config)
