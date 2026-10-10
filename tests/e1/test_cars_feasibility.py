"""Independent bounded counterfactual schedules, without physical simulation."""
from dataclasses import FrozenInstanceError, replace

import pandas as pd
import pytest

from .helpers import START, task


def snapshot(pending, running=(), capacity=(1, 1, 1), now=START, blocked=()):
    from experiments.e1.cars_state import ResourceSnapshot
    used = tuple(sum(r.demand[d] for r in running) for d in range(3))
    return ResourceSnapshot(now=now, capacity=capacity,
        available=tuple(c-u for c, u in zip(capacity, used)), pending=tuple(pending),
        running=tuple(running), blocked_until=tuple(blocked))


def planner(**kwargs):
    from experiments.e1.cars_feasibility import ShadowPlanner
    return ShadowPlanner(**kwargs)


def test_single_job_compares_forced_now_and_next_grid_start():
    p = planner()
    candidate = task('candidate', duration=30, sla=2)
    result = p.compare(candidate, snapshot([candidate]))
    assert result.status == 'safe'
    assert result.candidate_start_now == START
    assert result.candidate_start_defer == START + pd.Timedelta(minutes=15)
    assert result.missed_now == result.missed_defer == ()
    assert result.unknown_now == result.unknown_defer == ()
    assert result.checked_tasks == 1 and result.planner_calls == 2
    assert result.horizon_end == START + pd.Timedelta(minutes=120)
    with pytest.raises(FrozenInstanceError):
        result.status = 'risk'


def test_candidate_added_miss_is_risk():
    p = planner()
    candidate = task('candidate', duration=15, sla=1)
    result = p.compare(candidate, snapshot([candidate]))
    assert result.status == 'risk'
    assert result.missed_now == () and result.missed_defer == ('candidate',)


@pytest.mark.parametrize('resource', ['cpu', 'gpu', 'mem'])
def test_equal_miss_counts_with_new_other_job_miss_is_risk(resource):
    p = planner()
    # The candidate consumes one dimension; blocker consumes that plus another.
    second = 'gpu' if resource != 'gpu' else 'cpu'
    c = dict(cpu=0, gpu=0, mem=0); c[resource] = 1
    b = dict(c); b[second] = 1
    u = dict(cpu=0, gpu=0, mem=0); u[second] = 1
    candidate = task('candidate', duration=30, sla=3, **c)
    blocker = task('blocker', duration=60, sla=65/60, **b)
    affected = task('affected', duration=15, sla=70/15, **u)
    result = p.compare(candidate, snapshot([affected, candidate, blocker]))
    assert result.status == 'risk'
    assert result.missed_now == ('blocker',)
    assert result.missed_defer == ('affected',)
    assert result.candidate_start_now == START
    assert result.candidate_start_defer == START + pd.Timedelta(minutes=60)


@pytest.mark.parametrize('resource,dimension', [('cpu', 0), ('gpu', 1), ('mem', 2)])
def test_non_grid_running_finish_holds_capacity_until_next_grid(resource, dimension):
    p = planner()
    from experiments.e1.cars_state import RunningTask
    running_demand = [0, 0, 0]; running_demand[dimension] = 0.6
    running = RunningTask('running', tuple(running_demand), START + pd.Timedelta(minutes=5))
    c = dict(cpu=0, gpu=0, mem=0); c[resource] = 0.4
    q = dict(cpu=0, gpu=0, mem=0); q[resource] = 0.6
    candidate = task('candidate', duration=15, sla=6, **c)
    urgent = task('urgent', duration=15, sla=1, **q)
    result = p.compare(candidate, snapshot([candidate, urgent], [running]))
    assert result.status == 'safe'
    assert result.missed_now == result.missed_defer == ('urgent',)
    assert result.candidate_start_defer == START + pd.Timedelta(minutes=15)


def test_exact_finish_is_used_for_deadline_not_release_boundary():
    p = planner()
    candidate = task('candidate', duration=5, sla=4)
    result = p.compare(candidate, snapshot([candidate]))
    assert result.status == 'safe'  # deferred finish 05:20 equals deadline
    assert result.missed_defer == ()  # release would be 05:30


def test_known_start_beyond_horizon_finish_is_still_verifiable():
    p = planner()
    candidate = task('candidate', duration=500, sla=2)
    result = p.compare(candidate, snapshot([candidate]))
    assert result.status == 'safe'
    assert result.unknown_now == result.unknown_defer == ()


def test_backfills_behind_oversized_edf_head_without_cancellation():
    p = planner()
    candidate = task('candidate', cpu=1, mem=0, duration=15, sla=3)
    head = task('head', cpu=3, mem=0, duration=15, sla=1)
    small = task('small', cpu=1, mem=0, duration=15, sla=4)
    result = p.compare(candidate, snapshot([head, candidate, small], capacity=(2, 1, 1)))
    assert result.status == 'safe'
    assert result.missed_now == result.missed_defer == ('head',)
    assert result.unknown_now == result.unknown_defer == ()


def test_spanning_running_hold_creates_unknown_and_conservative_result():
    p = planner()
    from experiments.e1.cars_state import RunningTask
    running = RunningTask('running', (1, 0, 0), START + pd.Timedelta(minutes=300))
    candidate = task('candidate', cpu=0, mem=0, duration=15, sla=20)
    other = task('other', cpu=1, mem=0, duration=300, sla=4)
    result = p.compare(candidate, snapshot([candidate, other], [running]))
    assert result.status == 'uncertain'
    assert result.unknown_now == result.unknown_defer == ('other',)


def test_previous_deferred_job_is_blocked_in_both_counterfactuals():
    p = planner()
    candidate = task('candidate', cpu=0, mem=0, duration=15, sla=20)
    blocked = task('blocked', cpu=1, mem=0, duration=15, sla=2)
    state = snapshot([candidate, blocked], blocked=[('blocked', START + pd.Timedelta(minutes=30))])
    result = p.compare(candidate, state)
    assert result.missed_now == result.missed_defer == ('blocked',)
    assert result.status == 'safe'


def test_endpoint_has_eight_starts_and_equal_latest_start_is_unknown():
    p = planner()
    candidate = task('candidate', cpu=0, mem=0, duration=15, sla=20)
    boundary = task('boundary', cpu=1, mem=0, duration=15, sla=9)
    state = snapshot([candidate, boundary], blocked=[('boundary', START + pd.Timedelta(minutes=120))])
    result = p.compare(candidate, state)
    assert result.status == 'uncertain'
    assert result.unknown_now == result.unknown_defer == ('boundary',)
    earlier = replace(boundary, sla_multiplier=134/15)
    proven = p.compare(candidate, snapshot([candidate, earlier], blocked=state.blocked_until))
    assert proven.status == 'safe'
    assert proven.missed_now == proven.missed_defer == ('boundary',)


def test_queue_limit_has_no_planner_calls_and_keeps_unknown_scope():
    p = planner(max_pending=1)
    candidate, other = task('candidate', sla=10), task('other', sla=10)
    result = p.compare(candidate, snapshot([candidate, other]))
    assert result.status == 'uncertain' and result.planner_calls == 0
    assert result.checked_tasks == 0
    assert result.unknown_now == result.unknown_defer == ('candidate', 'other')
    assert 'limit' in result.reason


@pytest.mark.parametrize('resource', ['cpu', 'gpu', 'mem'])
def test_fractional_exact_fit_has_no_subtraction_roundoff_delay(resource):
    p = planner()
    specs = []
    for name, value in [('candidate', 0.2), ('b', 0.6), ('c', 0.2)]:
        demand = dict(cpu=0, gpu=0, mem=0); demand[resource] = value
        specs.append(task(name, duration=15, sla=2, **demand))
    result = p.compare(specs[0], snapshot(specs))
    assert result.status == 'safe'
    assert result.missed_now == result.missed_defer == ()
    assert result.candidate_start_defer == START + pd.Timedelta(minutes=15)


def test_stable_edf_ties_and_shared_preparation_do_not_reuse_stale_resources():
    p = planner()
    from experiments.e1.cars_state import RunningTask
    candidate = task('candidate', cpu=1, mem=0, duration=15, sla=3)
    a10 = task('a10', cpu=1, mem=0, duration=15, sla=3)
    a2 = task('a2', cpu=1, mem=0, duration=15, sla=3)
    state = snapshot([a2, candidate, a10])
    first = p.compare(candidate, state)
    assert first.candidate_start_defer == START + pd.Timedelta(minutes=30)
    assert p.compare(candidate, state) == first
    assert p.compare(candidate, snapshot([a10, a2, candidate])) == first
    running = RunningTask('prior_start', (1, 0, 0), START + pd.Timedelta(minutes=15))
    changed = snapshot([candidate, a10, a2], [running], capacity=(2, 1, 1))
    updated = p.compare(candidate, changed)
    assert updated.candidate_start_defer == START + pd.Timedelta(minutes=15)
    assert state.pending == (a2, candidate, a10)
    assert state.running == ()  # planner never writes to snapshots


def test_candidate_missing_is_uncertain_without_counterfactuals():
    p = planner()
    result = p.compare(task('missing'), snapshot([task('other')]))
    assert result.status == 'uncertain' and result.planner_calls == 0
    assert 'candidate' in result.reason


def test_running_deadline_miss_is_retained_as_common_known_pressure():
    p = planner()
    from experiments.e1.cars_state import RunningTask
    running = RunningTask('already_late', (0.2, 0, 0),
        START + pd.Timedelta(minutes=45), deadline=START + pd.Timedelta(minutes=30))
    candidate = task('candidate', cpu=0.2, mem=0, duration=15, sla=3)
    result = p.compare(candidate, snapshot([candidate], [running]))
    assert result.status == 'safe'
    assert result.missed_now == result.missed_defer == ('already_late',)


def test_queue_limit_retains_already_known_running_misses():
    p = planner(max_pending=1)
    from experiments.e1.cars_state import RunningTask
    running = RunningTask('already_late', (0.2, 0, 0),
        START + pd.Timedelta(minutes=45), deadline=START + pd.Timedelta(minutes=30))
    candidate, other = task('candidate', cpu=0.2), task('other', cpu=0.2)
    result = p.compare(candidate, snapshot([candidate, other], [running]))
    assert result.status == 'uncertain' and result.planner_calls == 0
    assert result.missed_now == result.missed_defer == ('already_late',)


@pytest.mark.parametrize('kwargs', [{'horizon_steps': 0}, {'horizon_steps': True},
    {'horizon_steps': 1.5}, {'max_pending': 0}, {'max_pending': True}])
def test_planner_configuration_rejects_invalid_integer_bounds(kwargs):
    with pytest.raises(ValueError):
        planner(**kwargs)
