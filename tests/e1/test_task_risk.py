"""E1-6A behavioral contracts, written before the task-risk implementation.

The fixtures exercise the existing real executor and upstream Task/resource
mechanism; KnownPower isolates physics while retaining interval integration.
"""
from dataclasses import asdict, replace

import pandas as pd
import pytest

from .helpers import START, KnownPower, inputs, task
from .test_policies import execute_policy, rows


THRESHOLD = 161.075
NEW_NAMES = (
    'task_risk_full', 'task_risk_no_history', 'task_risk_no_urgent',
    'task_risk_no_utilization', 'task_risk_max_required',
)


def policy(name='task_risk_full', capacity=(1, 1, 1), **parameters):
    from experiments.e1.task_risk import make_task_risk_policy
    return make_task_risk_policy(
        name, dict(threshold_g_per_kwh=THRESHOLD, **parameters), START, capacity)


def pressure(resources=(0, 0, 0), effective=120):
    from experiments.e1.cars_state import Pressure
    if effective == 0:
        return Pressure(0, (0, 0, 0), (None, None, None), None, 2, 0)
    return Pressure(effective, tuple(x * effective for x in resources),
                    tuple(resources), max(resources), 0, 0)


def budget(spec, now=START, capacity=(1, 1, 1), running=(),
           predecessors=(), history=None, parameters=None, aggregation='weighted'):
    from experiments.e1.task_risk_state import BudgetParameters, calculate_budget
    return calculate_budget(
        spec, now, capacity, tuple(running), tuple(predecessors),
        pressure() if history is None else history,
        parameters or BudgetParameters(), aggregation=aggregation)


def observe(p, pending, now=START, running=(), arrivals=None, capacity=(1, 1, 1)):
    from experiments.e1.cars_state import ResourceSnapshot
    ordered = p.order(tuple(pending))
    used = tuple(sum(r.demand[k] for r in running) for k in range(3))
    available = tuple(c - u for c, u in zip(capacity, used))
    p.on_step_start(ResourceSnapshot(
        now, capacity, available, ordered, tuple(running),
        tuple(pending if arrivals is None else arrivals)))
    return available


def test_fractional_duration_uses_raw_finish_for_slack_and_grid_for_occupation():
    from experiments.e1.task_risk_state import latest_grid_start, occupied_minutes
    candidate = task('b', duration=16, sla=35 / 16, mem=0)
    assert candidate.sla_deadline == START + pd.Timedelta(minutes=35)
    assert latest_grid_start(candidate) == START + pd.Timedelta(minutes=15)
    assert occupied_minutes(candidate) == 30
    assert budget(candidate).solo_slack_minutes == 15
    p = policy()
    result = execute_policy(
        [task('a', duration=15, sla=1, mem=0), candidate], p,
        minutes=60, ci=[200] * 4)
    row = rows(result)['b']
    assert row['start_time'] == (START + pd.Timedelta(minutes=15)).isoformat()
    assert row['finish_time'] == (START + pd.Timedelta(minutes=31)).isoformat()
    assert row['completion_observed_at'] == (START + pd.Timedelta(minutes=45)).isoformat()
    assert row['deadline_outcome'] == 'met'
    assert [s['used_cores'] for s in result.steps] == [1, 1, 1, 0]


def test_declared_weighted_formula_units_and_raw_history_are_exact():
    from experiments.e1.cars_state import RunningTask
    candidate = task('c', duration=60, cpu=5, gpu=5, mem=10, sla=3)
    predecessor = task('a', duration=16, cpu=4, gpu=10, mem=20, sla=2)
    running = RunningTask('r', (2, 8, 12), START + pd.Timedelta(minutes=30))
    got = budget(candidate, capacity=(10, 20, 40), running=(running,),
                 predecessors=(predecessor,), history=pressure((.9, .4, 1.5)))
    assert got.weights == pytest.approx((.5, .25, .25))
    assert got.utilization == pytest.approx((.2, .4, .3))
    assert got.urgent_pressure == pytest.approx((.1, .125, .125))
    assert got.history_pressure == (.9, .4, 1.5)
    assert got.resource_pressure == pytest.approx((.7, .525, 1.425))
    assert got.score == pytest.approx(.8375)
    assert got.buffer_minutes == 15
    assert got.solo_slack_minutes == 120 and got.safe_slack_minutes == 105
    assert got.cold_steps == 0


def test_urgent_filter_includes_overdue_and_horizon_equality_not_self_or_far_tasks():
    candidate = task('c', duration=60, cpu=1, mem=0, sla=5)
    overdue = task('a', duration=15, cpu=.1, mem=0, sla=.5)
    horizon = task('b', duration=15, cpu=.2, mem=0, sla=9)  # L=120
    far = task('z', duration=15, cpu=.4, mem=0, sla=10)     # L=135
    got = budget(candidate, predecessors=(overdue, horizon, far, candidate))
    assert got.urgent_pressure == pytest.approx(((.1 * 15 + .2 * 15) / 120, 0, 0))


def test_cold_unknown_and_cap_add_to_total_150_minutes_without_fake_zero_history():
    candidate = task('b', duration=60, cpu=1, mem=0, sla=5)
    predecessor = task('a', duration=120, cpu=20, mem=0, sla=1)
    got = budget(candidate, predecessors=(predecessor,), history=pressure(effective=0))
    assert got.history_pressure == (None, None, None)
    assert got.cold_steps == 2 and got.score == 20
    assert got.buffer_minutes == 150
    assert got.safe_slack_minutes == 90


def test_buffer_ceil_has_no_optimistic_epsilon_at_integer_boundary():
    candidate = task('b', duration=60, cpu=1, mem=0, sla=4)
    exact = budget(candidate, history=pressure((1.5, 0, 0)))
    just_over = budget(candidate, history=pressure((1.5000000000000002, 0, 0)))
    assert exact.buffer_minutes == 15
    assert just_over.buffer_minutes == 30


@pytest.mark.parametrize('name', NEW_NAMES)
def test_safe_slack_equal_one_step_defers_and_threshold_equal_executes(name):
    from experiments.e1.policies import DecisionContext
    candidate = task('a', duration=15, cpu=.1, mem=0, sla=4)  # S=45, cold B=30
    p = policy(name)
    available = observe(p, (candidate,))
    p.on_candidate_resources(START, available)
    decision = p.decide(candidate, DecisionContext(START, THRESHOLD + .001))
    assert decision.defer and decision.reason == 'high_ci_defer'
    assert p.diagnostics[-1]['eligible'] is True
    assert p.diagnostics[-1]['budget']['safe_slack_minutes'] == 15
    assert p.diagnostics[-1]['budget']['buffer_minutes'] == 30
    p = policy(name)
    p.on_candidate_resources(START, observe(p, (candidate,)))
    decision = p.decide(candidate, DecisionContext(START, THRESHOLD))
    assert not decision.defer and decision.reason == 'low_ci_execute'
    assert p.diagnostics[-1]['budget'] is None and not p.diagnostics[-1]['eligible']


def test_guard_executes_below_one_step_and_expired_task_has_no_budget():
    from experiments.e1.policies import DecisionContext
    candidate = task('a', duration=15, cpu=.1, mem=0, sla=3)  # S=30, B=30
    p = policy()
    p.on_candidate_resources(START, observe(p, (candidate,)))
    decision = p.decide(candidate, DecisionContext(START, 200))
    assert not decision.defer and decision.reason == 'task_risk_guard'
    assert p.diagnostics[-1]['budget']['safe_slack_minutes'] == 0
    expired = task('b', duration=15, sla=.5)
    p = policy()
    p.on_candidate_resources(START, observe(p, (expired,)))
    decision = p.decide(expired, DecisionContext(START, 200))
    assert not decision.defer and decision.reason == 'no_grid_slack'
    assert p.diagnostics[-1]['budget'] is None


@pytest.mark.parametrize('aggregation', ['weighted', 'max_required'])
def test_zero_resource_task_has_zero_buffer_even_cold_and_extreme_pressure(aggregation):
    candidate = task('a', duration=15, cpu=0, gpu=0, mem=0, sla=2)
    got = budget(candidate, history=pressure(effective=0), aggregation=aggregation)
    assert got.weights == (0, 0, 0) and got.score == 0
    assert got.buffer_minutes == 0 and got.safe_slack_minutes == 15
    p = policy('task_risk_full' if aggregation == 'weighted' else 'task_risk_max_required')
    result = execute_policy([candidate], p, ci=[200, 150])
    assert rows(result)['a']['start_time'] == (START + pd.Timedelta(minutes=15)).isoformat()
    assert result.policy_tasks[0]['policy_defer_intervals'] == 1


@pytest.mark.parametrize('resource,index', [('cpu', 0), ('gpu', 1), ('mem', 2)])
def test_task_weights_ignore_unrelated_resource_pressure(resource, index):
    q = dict(cpu=0, gpu=0, mem=0)
    q[resource] = .25
    candidate = task('a', duration=60, sla=2, **q)
    unrelated = tuple(0 if k == index else 10 for k in range(3))
    got = budget(candidate, history=pressure(unrelated))
    assert got.weights == tuple(1 if k == index else 0 for k in range(3))
    assert got.buffer_minutes == 0 and got.score == 0


def test_weighted_average_can_underestimate_small_demand_on_scarce_gpu():
    from experiments.e1.cars_state import RunningTask
    candidate = task('b', duration=60, cpu=.5, gpu=.001, mem=0, sla=2)
    running = RunningTask('r', (0, .999, 0), START + pd.Timedelta(minutes=60))
    weighted = budget(candidate, running=(running,), history=pressure((0, 8.5, 0)))
    maximum = budget(candidate, running=(running,), history=pressure((0, 8.5, 0)),
                     aggregation='max_required')
    assert weighted.weights == pytest.approx((.5 / .501, .001 / .501, 0))
    assert weighted.buffer_minutes == 15
    assert maximum.buffer_minutes == 120
    # Both methods examine the same currently feasible resource allocation.
    assert running.demand[1] + candidate.gpu_req == pytest.approx(1)


@pytest.mark.parametrize('high_resource,high_index,low_resource,low_index', [
    ('gpu', 1, 'cpu', 0), ('cpu', 0, 'gpu', 1),
])
def test_new_policy_expresses_task_subset_decision_unavailable_to_global_risk(
        high_resource, high_index, low_resource, low_index):
    from experiments.e1.cars import make_cars_policy
    from experiments.e1.cars_state import RunningTask
    from experiments.e1.policies import DecisionContext
    now = START + pd.Timedelta(minutes=15)
    def demand_one(resource, value):
        q = dict(cpu=0, gpu=0, mem=0)
        q[resource] = value
        return q
    old_finished = task('h', duration=15, sla=1, **demand_one(high_resource, .9))
    old_running = task('r', duration=30, sla=1, **demand_one(high_resource, .5))
    high = task('a', arrival=15, duration=60, sla=1.5, **demand_one(high_resource, .1))
    low = task('b', arrival=15, duration=60, sla=1.5, **demand_one(low_resource, .1))
    running_demand = tuple(.5 if k == high_index else 0 for k in range(3))
    running = RunningTask('r', running_demand, START + pd.Timedelta(minutes=30))
    p = policy()
    p.history.observe(START, (old_finished, old_running))
    available = observe(p, (high, low), now, (running,))
    p.on_candidate_resources(now, available)
    high_decision = p.decide(high, DecisionContext(now, 200))
    assert not high_decision.defer
    high_budget = p.diagnostics[-1]['budget']
    p.on_task_started(RunningTask(high.task_id, tuple(demand_one(high_resource, .1)[r]
        for r in ('cpu', 'gpu', 'mem')), now + pd.Timedelta(minutes=60), high.sla_deadline))
    remaining = tuple(a - (.1 if k == high_index else 0) for k, a in enumerate(available))
    p.on_candidate_resources(now, remaining)
    low_decision = p.decide(low, DecisionContext(now, 200))
    assert low_decision.defer
    assert high_budget['buffer_minutes'] > p.diagnostics[-1]['budget']['buffer_minutes']
    assert p.diagnostics[-1]['budget']['utilization'][high_index] == pytest.approx(.6)
    assert p.diagnostics[-1]['budget']['urgent_pressure'] == (0, 0, 0)
    old = make_cars_policy('cars_risk_only', {'threshold_g_per_kwh': THRESHOLD}, START, (1, 1, 1))
    old.history.observe(START, (old_finished, old_running))
    observe(old, (high, low), now, (running,))
    assert not old.decide(high, DecisionContext(now, 200)).defer
    assert not old.decide(low, DecisionContext(now, 200)).defer


def test_history_observed_once_and_open_left_expiry_does_not_restore_cold():
    from experiments.e1.cars_state import ResourceSnapshot
    p = policy()
    candidate = task('a', duration=16, cpu=.5, gpu=0, mem=0, sla=10)
    p.on_step_start(ResourceSnapshot(START, (1, 1, 1), (1, 1, 1), (candidate,), (), (candidate,)))
    assert p.history.pressure(START).observed_tasks == 1
    for _ in range(3):
        p.on_candidate_resources(START, (1, 1, 1))
    assert p.history.pressure(START).observed_tasks == 1
    assert p.history.pressure(START).resource_work_minutes == (15, 0, 0)
    p.on_step_start(ResourceSnapshot(START + pd.Timedelta(minutes=15),
        (1, 1, 1), (1, 1, 1), (candidate,), (), ()))
    assert p.history.pressure(START + pd.Timedelta(minutes=15)).rho_resources == (1, 0, 0)
    p.on_step_start(ResourceSnapshot(START + pd.Timedelta(minutes=120),
        (1, 1, 1), (1, 1, 1), (candidate,), (), ()))
    quiet = p.history.pressure(START + pd.Timedelta(minutes=120))
    assert quiet.observed_tasks == 0 and quiet.rho_resources == (0, 0, 0)
    assert budget(candidate, now=START + pd.Timedelta(minutes=120), history=quiet).cold_steps == 0


def test_no_history_removes_only_numeric_history_term_and_retains_common_cold():
    from experiments.e1.policies import DecisionContext
    candidate = task('a', duration=15, cpu=.1, mem=0, sla=4)
    full, no_history = policy(), policy('task_risk_no_history')
    for p in (full, no_history):
        p.on_candidate_resources(START, observe(p, (candidate,)))
        assert p.decide(candidate, DecisionContext(START, 200)).defer
        assert p.diagnostics[-1]['budget']['cold_steps'] == 2
        assert p.diagnostics[-1]['budget']['buffer_minutes'] == 30
    now = START + pd.Timedelta(minutes=15)
    high = task('b', duration=60, cpu=1, mem=0, sla=4)
    values = []
    for p in (full, no_history):
        p.on_candidate_resources(now, observe(p, (high,), now, arrivals=(high,)))
        p.decide(high, DecisionContext(now, 200))
        values.append(p.diagnostics[-1]['budget']['buffer_minutes'])
    assert values[0] > 0 and values[1] == 0


@pytest.mark.parametrize('coefficient,removed', [('a_u', 'utilization'), ('a_q', 'urgent_pressure'),
                                               ('a_h', 'history_pressure')])
def test_single_coefficient_ablations_leave_other_pressure_terms_intact(coefficient, removed):
    from experiments.e1.cars_state import RunningTask
    from experiments.e1.task_risk_state import BudgetParameters
    candidate = task('b', duration=60, cpu=1, mem=0, sla=5)
    predecessor = task('a', duration=60, cpu=1, mem=0, sla=2)
    running = RunningTask('r', (.5, 0, 0), START + pd.Timedelta(minutes=60))
    kwargs = dict(running=(running,), predecessors=(predecessor,), history=pressure((1.5, 0, 0)))
    full = budget(candidate, **kwargs)
    ablated = budget(candidate, parameters=BudgetParameters(**{coefficient: 0}), **kwargs)
    assert full.resource_pressure == (2, 0, 0)
    contribution = {'a_u': .5, 'a_q': .5, 'a_h': 1}[coefficient]
    assert ablated.resource_pressure == (2 - contribution, 0, 0)
    assert ablated.utilization == full.utilization
    assert ablated.urgent_pressure == full.urgent_pressure
    assert ablated.history_pressure == full.history_pressure


def test_same_step_start_removed_defer_and_capacity_failure_retained_in_prefix():
    p = policy()
    result = execute_policy([
        task('a', duration=60, cpu=.6, mem=0, sla=1),
        task('b', duration=30, cpu=.6, mem=0, sla=3),
        task('c', duration=60, cpu=.2, mem=0, sla=3),
        task('d', duration=60, cpu=.2, mem=0, sla=3.5),
    ], p, minutes=120, ci=[200] * 8)
    first = {d['task_id']: d for d in p.diagnostics if d['time'] == START.isoformat()}
    assert 'b' not in first  # capacity failed: no policy decision or fake budget
    assert first['c']['running_count'] == 1 and first['c']['pending_count'] == 3
    assert first['c']['budget']['utilization'] == pytest.approx((.6, 0, 0))
    assert first['c']['budget']['urgent_pressure'] == pytest.approx((.15, 0, 0))
    assert first['d']['budget']['urgent_pressure'] == pytest.approx((.25, 0, 0))
    assert first['c']['defer'] and first['d']['defer']
    assert result.policy_steps[0]['capacity_waiting_ids'] == ['b']
    assert result.policy_steps[0]['policy_deferred_ids'] == ['c', 'd']
    assert result.policy_steps[0]['running_ids'] == ['a']


def test_full_queue_edf_resorts_previous_waiters_with_new_arrivals():
    p = policy()
    result = execute_policy([
        task('z', duration=15, sla=1), task('a', duration=30, sla=5),
        task('b', arrival=15, duration=15, sla=1),
    ], p, minutes=60, ci=[100] * 4)
    assert result.policy_steps[1]['ordered_waiting_ids'] == ['b', 'a']
    assert rows(result)['b']['start_time'] == (START + pd.Timedelta(minutes=15)).isoformat()
    assert rows(result)['a']['start_time'] == (START + pd.Timedelta(minutes=30)).isoformat()


def test_stable_edf_tie_break_and_oversized_head_backfill():
    p = policy(capacity=(2, 1, 2))
    assert [s.task_id for s in p.order((task('a2', sla=2), task('a10', sla=2),
                                      task('b', arrival=15, sla=1)))] == ['a10', 'a2', 'b']
    result = execute_policy([task('a', cpu=3, sla=1), task('b', duration=30, sla=1),
                             task('c', sla=3)], p, ci=[100, 100], capacity=(2, 1, 2))
    assert result.policy_steps[0]['started_ids'] == ['b', 'c']
    assert rows(result)['a']['state'] == 'pending'
    assert rows(result)['a']['deadline_outcome'] == 'missed_unfinished'
    assert not any(d['task_id'] == 'a' for d in p.diagnostics)


@pytest.mark.parametrize('name', NEW_NAMES)
def test_all_new_policies_keep_task_identity_and_three_resource_conservation(name):
    specs = [task('a', duration=30, cpu=.6, gpu=.6, mem=.6, sla=2),
             task('b', cpu=.6, gpu=.6, mem=.6, sla=4),
             task('c', cpu=.4, gpu=.4, mem=.4, sla=4)]
    original = tuple(s.to_dict() for s in specs)
    result = execute_policy(specs, policy(name), minutes=90, ci=[200, 150, 200, 150, 150, 150])
    assert tuple(s.to_dict() for s in specs) == original
    assert len({r['task_id'] for r in result.tasks}) == len(specs)
    assert result.summary['completed'] + result.summary['pending'] + result.summary['running'] == len(specs)
    for step in result.steps:
        for resource in ('cores', 'gpus', 'mem'):
            assert 0 <= step[f'used_{resource}'] <= 1
            assert step[f'used_{resource}'] + step[f'available_{resource}'] == pytest.approx(1)


@pytest.mark.parametrize('name', NEW_NAMES)
def test_same_current_history_different_future_cannot_change_current_decision_or_energy(name):
    from experiments.e1.engine import LocalExecutor
    first = task('a', duration=15, sla=4)
    prefixes = []
    for future, ci, temperature in [(task('x', arrival=15, cpu=.1), 10, 5),
                                     (task('y', arrival=15, cpu=50), 999, 40)]:
        w, e = inputs([first, future], 60)
        e = replace(e, samples=tuple(replace(s, ci_g_per_kwh=200 if j == 0 else ci,
                     temperature_c=20 if j == 0 else temperature) for j, s in enumerate(e.samples)))
        p = policy(name)
        executor = LocalExecutor(w, e, (1, 1, 1), KnownPower(), p)
        executor.start(until=START + pd.Timedelta(minutes=15))
        executor.advance()
        prefixes.append((p.diagnostics, executor.result()))
    assert prefixes[0] == prefixes[1]
    assert prefixes[0][1].summary['arrived'] == 1


def test_expired_waiter_starts_without_cancellation_when_nonpreemptive_resource_releases():
    result = execute_policy([task('a', duration=45, sla=1),
                             task('b', arrival=15, duration=15, sla=1)],
                            policy(), minutes=60, ci=[200] * 4)
    assert rows(result)['b']['start_time'] == (START + pd.Timedelta(minutes=45)).isoformat()
    assert rows(result)['b']['deadline_outcome'] == 'missed'
    assert result.policy_tasks[1]['capacity_wait_intervals'] == 2
    assert result.policy_tasks[1]['policy_defer_intervals'] == 0


def test_terminal_cutoff_censors_arrived_work_and_never_creates_extra_energy_step():
    result = execute_policy([task('a', duration=60, sla=2), task('b', duration=45, sla=.5),
                             task('c', duration=15, cpu=0, mem=0, sla=5)],
                            policy(capacity=(2, 1, 2)), capacity=(2, 1, 2), ci=[200, 200])
    assert len(result.steps) == 2
    assert result.summary['arrived'] == 3
    assert rows(result)['a']['deadline_outcome'] == 'right_censored'
    assert rows(result)['b']['deadline_outcome'] == 'missed_unfinished'
    assert all(e['time'] < (START + pd.Timedelta(minutes=30)).isoformat() for e in result.policy_events)


def test_new_registry_alias_is_identical_and_old_registries_rl_actions_unchanged():
    from experiments.e1.task_risk import POLICY_NAMES
    from experiments.e1.policies import POLICY_NAMES as OLD_NAMES, POLICY_SCHEMA
    from experiments.e1.rl_env import ACTION_POLICIES, CarbonSchedulingEnv
    from experiments.e1.cars import make_cars_policy
    from experiments.e1.engine import LocalExecutor
    assert set(NEW_NAMES) <= set(POLICY_NAMES)
    assert OLD_NAMES == ('immediate_fifo', 'edf_backfill', 'carbon_threshold_fifo')
    assert POLICY_SCHEMA == 'e1-1b-policy-v1'
    assert ACTION_POLICIES == ('edf_backfill', 'edf_carbon_threshold', 'cars_risk_only')
    specs = [task('a', duration=15, sla=4), task('b', arrival=15)]
    full = execute_policy(specs, policy(), minutes=60, ci=[200, 150, 150, 150])
    alias = execute_policy(specs, policy('task_risk_budget_v1'), minutes=60, ci=[200, 150, 150, 150])
    assert asdict(full) == asdict(alias)
    w, e = inputs(specs, 60)
    for action, name in enumerate(ACTION_POLICIES):
        expected = LocalExecutor(w, e, (1, 1, 1), KnownPower(),
            make_cars_policy(name, {'threshold_g_per_kwh': THRESHOLD}, START, (1, 1, 1)), drain=True).run()
        env = CarbonSchedulingEnv(w, e, (1, 1, 1), KnownPower,
                                  {'threshold_g_per_kwh': THRESHOLD})
        env.reset(seed=6)
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        assert asdict(env.executor.result()) == asdict(expected)
        env.close()


def test_repeated_frozen_micro_replay_is_exact_including_policy_diagnostics():
    specs = [task('a', duration=16, cpu=.5, gpu=.25, mem=.5, sla=4),
             task('b', arrival=15, cpu=.5, gpu=.5, mem=.5, sla=3)]
    runs = []
    for _ in range(2):
        p = policy()
        result = execute_policy(specs, p, minutes=90, ci=[200, 200, 150, 150, 150, 150])
        runs.append((asdict(result), p.diagnostics, dict(p.reason_counts), p.counters))
    assert runs[0] == runs[1]

