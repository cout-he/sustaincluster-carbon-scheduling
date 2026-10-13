"""Behavior contracts for the independent four-action / 83-feature Gym interface."""
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from experiments.e1.engine import LocalExecutor
from .helpers import START, KnownPower, inputs, task


PARAMETERS = {'threshold_g_per_kwh': 161.075}


def env(workload=None, environment=None, capacity=(1, 1, 1), **kwargs):
    from experiments.e1.adaptive_rl_env import AdaptiveCarbonSchedulingEnv
    if workload is None:
        workload, environment = inputs([task('a')], 30)
    return AdaptiveCarbonSchedulingEnv(workload, environment, capacity, KnownPower,
                                       PARAMETERS, **kwargs)


def fixed_ci(environment, ci=200):
    return replace(environment, samples=tuple(replace(s, ci_g_per_kwh=ci)
                                              for s in environment.samples))


def rollout(live, action):
    observation, info = live.reset(seed=7)
    transitions = []
    done = False
    while not done:
        observation, reward, terminated, truncated, info = live.step(action)
        assert live.observation_space.contains(observation)
        assert np.isfinite(reward)
        transitions.append((reward, terminated, truncated, info))
        done = terminated or truncated
    return live.executor.result(), transitions


def core(result):
    return (result.tasks, result.events, result.steps, result.summary, result.policy_tasks)


def test_new_interface_preserves_base_forty_and_masks_entire_risk_block():
    from experiments.e1.rl_env import CarbonSchedulingEnv, RL_SCHEMA
    from experiments.e1.adaptive_rl_env import RL_SCHEMA as NEW_SCHEMA
    w, e = inputs([task('a', duration=30)], 60)
    old = CarbonSchedulingEnv(w, e, (1, 1, 1), KnownPower, PARAMETERS)
    enabled, disabled = env(w, e), env(w, e, risk_state=False)
    old_observation, _ = old.reset(seed=7)
    on_observation, on_info = enabled.reset(seed=7)
    off_observation, off_info = disabled.reset(seed=7)
    assert enabled.action_space.n == disabled.action_space.n == 4
    assert enabled.observation_space.shape == disabled.observation_space.shape == (83,)
    np.testing.assert_array_equal(on_observation[:40], old_observation)
    np.testing.assert_array_equal(off_observation[:40], old_observation)
    assert np.count_nonzero(on_observation[40:]) > 0
    np.testing.assert_array_equal(off_observation[40:], np.zeros(43, dtype=np.float32))
    assert NEW_SCHEMA != RL_SCHEMA and old.action_space.n == 3
    assert on_info['observation_schema'] == off_info['observation_schema'] == NEW_SCHEMA
    assert old.observation_space.shape == (40,)


@pytest.mark.parametrize('action,slack,defer', [
    (0, 120, False), (1, 45, False), (1, 60, True),
    (2, 15, False), (2, 30, True), (3, 0, False), (3, 15, True)])
def test_four_actions_use_current_slack_threshold_and_only_one_step(action, slack, defer):
    w, e = inputs([task('a', duration=15, sla=1 + slack / 15)], 150)
    live = env(w, fixed_ci(e), risk_filter=False)
    live.reset(seed=1)
    _, _, _, _, info = live.step(action)
    events = live.executor.policy_events
    assert len(events) == 1
    assert (events[0]['event'] == 'policy_deferred') == defer
    assert info['action'] == action
    assert info['interval']['interval_start'] == START.isoformat()
    assert live.executor.observed_end == START + pd.Timedelta(minutes=15)
    assert len(live.executor.steps) == 1


def test_raw_slack_is_floored_before_the_action_boundary():
    w, e = inputs([task('a', duration=16, sla=75.99 / 16)], 120)
    live = env(w, fixed_ci(e), risk_filter=False)
    live.reset(seed=7)
    live.step(1)
    assert live.executor.policy_events[0]['event'] == 'started'


@pytest.mark.parametrize('action', [1, 2, 3])
def test_threshold_equality_executes_for_every_defer_mode(action):
    w, e = inputs([task('a', duration=15, sla=10)], 180)
    live = env(w, fixed_ci(e, 161.075), risk_filter=True)
    live.reset(seed=7)
    live.step(action)
    assert live.executor.policy_events[0]['event'] == 'started'


@pytest.mark.parametrize('action,name,filter_enabled', [
    (0, 'edf_backfill', False),
    (3, 'edf_carbon_threshold', False),
    (3, 'task_risk_full', True)])
def test_constant_actions_match_existing_real_executor_ledger_energy_and_carbon(action, name, filter_enabled):
    from experiments.e1.task_risk import make_task_risk_policy
    w, e = inputs([task('a', duration=20, gpu=.6, sla=3),
                   task('b', duration=10, cpu=.5, gpu=.4, sla=5),
                   task('c', arrival=15, duration=31, gpu=.9, sla=3)], 150)
    e = fixed_ci(e)
    capacity = (2, 1, 3)
    baseline = LocalExecutor(w, e, capacity, KnownPower(),
        make_task_risk_policy(name, PARAMETERS, START, capacity), drain=True).run()
    live = env(w, e, capacity, risk_filter=filter_enabled)
    actual, _ = rollout(live, action)
    assert core(actual) == core(baseline)
    assert [row['started_ids'] for row in actual.policy_steps] == [row['started_ids'] for row in baseline.policy_steps]
    assert [row['policy_deferred_ids'] for row in actual.policy_steps] == [row['policy_deferred_ids'] for row in baseline.policy_steps]


def test_history_is_observed_even_when_all_previous_actions_were_edf():
    w, e = inputs([task('a', duration=15), task('b', arrival=15, duration=15)], 60)
    live = env(w, e)
    live.reset(seed=3)
    live.step(0)
    assert live.policy.history.pressure(START + pd.Timedelta(minutes=15)).observed_tasks == 2
    assert live.observation_raw['history_observed_tasks'] == 2
    assert live.observation_raw['history_resource_work_minutes'] == (30, 0, 30)


def test_same_current_history_ignores_future_workload_ci_and_weather():
    common = task('a', duration=20, sla=4)
    wa, ea = inputs([common, task('b', arrival=30)], 90)
    wb, eb = inputs([common, task('c', arrival=30, cpu=2, duration=60)], 90)
    eb = replace(eb, samples=tuple(replace(s, ci_g_per_kwh=900, temperature_c=35)
                                  if s.time >= START + pd.Timedelta(minutes=30) else s
                                  for s in eb.samples))
    a, b = env(wa, ea), env(wb, eb)
    oa, ia = a.reset(seed=11)
    ob, ib = b.reset(seed=11)
    np.testing.assert_array_equal(oa, ob)
    assert ia == ib
    oa, ra, ta, xa, ia = a.step(3)
    ob, rb, tb, xb, ib = b.step(3)
    np.testing.assert_array_equal(oa, ob)
    assert (ra, ta, xa, ia) == (rb, tb, xb, ib)
    _, ra, _, _, ia = a.step(2)
    _, rb, _, _, ib = b.step(2)
    assert ra == rb and ia['interval'] == ib['interval']
    assert a.executor.steps == b.executor.steps
    assert a.executor.policy_events == b.executor.policy_events


def test_exclusive_episode_cutoff_does_not_observe_or_load_next_arrivals():
    wa, ea = inputs([task('a'), task('b', arrival=15)], 45)
    wb, eb = inputs([task('a'), task('c', arrival=15, duration=60)], 45)
    eb = replace(eb, samples=tuple(replace(s, ci_g_per_kwh=999) if s.time > START else s
                                  for s in eb.samples))
    a, b = env(wa, ea, max_episode_steps=1), env(wb, eb, max_episode_steps=1)
    a.reset(seed=1)
    b.reset(seed=1)
    oa, ra, ta, xa, ia = a.step(0)
    ob, rb, tb, xb, ib = b.step(0)
    np.testing.assert_array_equal(oa, ob)
    assert (ra, ta, xa, ia) == (rb, tb, xb, ib)
    assert xa and not ta and ia['observed_counts']['arrived'] == 1


def test_deadline_reward_counts_first_miss_once_including_late_completion():
    w, e = inputs([task('a', duration=30),
                   task('b', arrival=15, duration=15, sla=1)], 60)
    live = env(w, e)
    _, transitions = rollout(live, 0)
    assert [t[3]['reward']['new_deadline_miss_ids'] for t in transitions] == [[], ['b'], [], []]
    assert live.executor.result().summary['completed'] == 2


def test_filter_queries_exclude_capacity_failed_candidates():
    w, e = inputs([task('a', duration=30, sla=1),
                   task('b', duration=15, sla=6)], 120)
    live = env(w, fixed_ci(e))
    live.reset(seed=1)
    info = live.step(3)[4]
    assert [row['event'] for row in live.executor.policy_events] == ['started', 'capacity_waited']
    assert info['risk_counts']['candidate_checks'] == 1
    assert len(info['policy_diagnostics']) == 1


def test_nonpreemptive_running_task_and_resource_release_boundary_survive_actions():
    w, e = inputs([task('a', duration=20, sla=1),
                   task('b', arrival=15, duration=15, sla=1)], 60)
    live = env(w, fixed_ci(e))
    live.reset(seed=1)
    live.step(0)
    running = live.executor.pool.running_tasks[0]
    live.step(3)
    assert running.start_time == START
    assert running.finish_time == START + pd.Timedelta(minutes=20)
    assert live.executor.ledger['a']['state'] == 'completed'
    assert [s['used_cores'] for s in live.executor.steps] == [1, 1]
    live.step(2)
    assert live.executor.ledger['b']['task'].start_time == START + pd.Timedelta(minutes=30)


def test_same_model_survives_steps_idle_tail_and_closes_once_per_reset():
    created = []
    class StatefulPower(KnownPower):
        def __init__(self):
            self.calls = self.closes = 0
            created.append(self)
        def measure(self, *args):
            self.calls += 1
            return super().measure(*args)
        def close(self):
            self.closes += 1
            assert self.closes == 1
    from experiments.e1.adaptive_rl_env import AdaptiveCarbonSchedulingEnv
    w, _ = inputs([task('a', duration=5)], 15)
    _, e = inputs([], 45)
    live = AdaptiveCarbonSchedulingEnv(w, e, (1, 1, 1), StatefulPower, PARAMETERS)
    result, transitions = rollout(live, 0)
    assert len(created) == 1 and created[0].calls == 3
    assert [s['total_energy_kwh'] for s in result.steps] == [1.5, 1, 1]
    assert transitions[-1][1:3] == (True, False)
    live.close()
    live.close()
    live.reset(seed=7)
    live.close()
    assert [p.closes for p in created] == [1, 1]


def test_stop_at_drain_terminates_without_one_extra_power_interval():
    w, _ = inputs([task('a', duration=5)], 15)
    _, e = inputs([], 45)
    live = env(w, e, stop_at_drain=True)
    live.reset(seed=7)
    _, _, terminated, truncated, _ = live.step(0)
    assert terminated and not truncated and len(live.executor.steps) == 1


def test_gym_checker_and_seeded_random_complete_rollout_preserve_conservation():
    from gymnasium.utils.env_checker import check_env
    w, e = inputs([task('a'), task('b', arrival=15, cpu=.2, gpu=.3)], 90)
    live = env(w, e)
    check_env(live, skip_render_check=True)
    traces = []
    for _ in range(2):
        live.reset(seed=123)
        trace = []
        done = False
        while not done:
            action = live.action_space.sample()
            observation, reward, terminated, truncated, info = live.step(action)
            trace.append((action, observation.tolist(), reward, terminated, truncated, info))
            live.executor.check_ownership()
            live.executor.pool.inventory()
            done = terminated or truncated
        assert live.executor.result().summary['completed'] == 2
        traces.append(trace)
    assert traces[0] == traces[1]
    with pytest.raises(RuntimeError):
        live.step(0)


@pytest.mark.parametrize('action', [-1, 4, 1.5, True])
def test_invalid_action_does_not_advance_physics(action):
    live = env()
    live.reset(seed=7)
    with pytest.raises(ValueError):
        live.step(action)
    assert live.executor.steps == []


@pytest.mark.parametrize('key,value', [('risk_state', 1), ('risk_filter', 'yes')])
def test_ablation_switches_reject_non_boolean_values(key, value):
    with pytest.raises(ValueError):
        env(**{key: value})


def test_empty_queue_risk_state_is_finite_and_future_ci_configuration_rejected():
    w, e = inputs([], 30)
    live = env(w, e)
    observation, _ = live.reset(seed=1)
    assert observation.shape == (83,) and np.isfinite(observation).all()
    from experiments.e1.adaptive_rl_env import AdaptiveCarbonSchedulingEnv
    with pytest.raises(ValueError):
        AdaptiveCarbonSchedulingEnv(w, e, (1, 1, 1), KnownPower,
                                    dict(PARAMETERS, future_ci=True))
