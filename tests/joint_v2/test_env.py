"""Real-plant V2 episode, fixed-window accounting and causal reservation tests."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.e1.cars_state import RunningTask
from experiments.e1.config import ROOT
from experiments.e1.contracts import STEP
from experiments.joint.config import load_config
from experiments.joint.env import JointEnv
from experiments.joint.power import COMPONENTS, JointPowerModel
from experiments.joint_v2.deadline_guard import legal_task_mask
from experiments.joint_v2.env import V2Env
from tests.e1.helpers import START, inputs, task


def fixture_scenario(tasks=(), arrival_minutes=30, total_minutes=180):
    workload, environment = inputs(tasks, total_minutes)
    workload = replace(workload, end=START + pd.Timedelta(minutes=arrival_minutes))
    datacenter = dict(location='US-CAL-CISO', total_cores=50000,
                      total_gpus=1000, total_mem=80000, hvac_setpoint_c=22)
    return SimpleNamespace(workload=workload, environment=environment, capacity=(2, 1, 3),
                           datacenter=datacenter, model_path=ROOT / 'configs/dcs/dc_config.json',
                           parameters=dict(threshold_g_per_kwh=150), role='train',
                           scenario_id='v2-fixture', identity={}, statistics={})


def fixture_config():
    v1 = load_config()
    return dict(v1_protocol=v1, environment=dict(idle_reference_setpoint_c=18.,
                settle_after_drain=True, deadline_guard_enabled=True),
                reward=dict(v1['reward'], carbon_mode='incremental_idle18', deadline_multiplier=10.),
                constraint=dict(lambda_max=5000.))


def start_if_legal(obs, mask):
    return 0 if mask[0] else 1


def lower_if_legal(obs, mask):
    return 0 if mask[0] else 1


def run_episode(env, cooling=lambda *_: 1):
    env.reset()
    outcomes = []
    while not env.done:
        outcomes.append(env.step(start_if_legal, cooling))
    return outcomes, env.summary()


def test_empty_arrival_window_cannot_end_early_and_settling_has_no_actor_actions():
    env = V2Env(fixture_scenario(), fixture_config())
    env.reset()
    first = env.step(start_if_legal, lambda *_: 1)
    assert not first.done
    second = env.step(start_if_legal, lambda *_: 1)
    assert not second.done and env.executor.observed_end == START + 2 * STEP
    cooling_calls = []
    settled = []
    while not env.done:
        outcome = env.step(lambda *_: pytest.fail('No task after drain'),
                           lambda *_: cooling_calls.append(1) or 1)
        settled.append(outcome)
        assert outcome.task_actions == outcome.cooling_actions == ()
        assert outcome.task_decisions == outcome.cooling_decisions == ()
    assert not cooling_calls
    assert len(settled) == 4
    assert [o.interval['setpoint_c'] for o in settled] == [21, 20, 19, 18]
    assert all(o.interval['total_energy_kwh'] > 0 for o in settled)
    summary = env.summary()
    assert summary['physical_steps'] == 6
    assert summary['deterministic_settling_steps'] == 4
    assert summary['fixed_window_complete'] and summary['complete_episode_for_constraint']
    assert summary['analytic_idle_intervals'] == 6
    env.close()


def test_drain_completion_releases_resources_and_avoids_long_idle_training():
    scenario = fixture_scenario([task('long', duration=75, cpu=1, sla=2)])
    env = V2Env(scenario, fixture_config())
    outcomes, summary = run_episode(env, lower_if_legal)
    assert len(outcomes) == 5
    assert summary['arrived'] == summary['completed'] == 1
    assert summary['running'] == summary['pending'] == summary['unfinished'] == 0
    assert summary['termination_reason'] == 'drained_settled'
    assert summary['deterministic_settling_steps'] == 0
    assert summary['physical_prefix_end'] == (START + 5 * STEP).isoformat()
    assert summary['fixed_window_intervals'] == 12
    env.executor.check_ownership()
    env.close()


def test_analytic_idle_tail_exactly_matches_full_physical_replay_all_components():
    scenario, config = fixture_scenario([task('a', duration=45, cpu=.5, sla=2)]), fixture_config()
    env = V2Env(scenario, config, evaluation=True)
    outcomes, summary = run_episode(env)
    physical_factory = lambda: JointPowerModel(scenario.datacenter, scenario.model_path)
    replay = JointEnv(scenario.workload, scenario.environment, scenario.capacity,
                      physical_factory, scenario.parameters, config['v1_protocol'], mode='B3')
    replay.reset()
    def same_operator(obs, mask):
        drained = (replay.executor.sample.time >= scenario.workload.end
                   and not (replay.executor.pool.pending_tasks or replay.executor.pool.running_tasks))
        return (0 if mask[0] else 1) if drained else 1
    while not replay.done:
        replay.step(start_if_legal, same_operator)
    full = replay.summary()
    assert len(outcomes) < full['intervals']
    assert summary['carbon_kg'] == pytest.approx(full['carbon_kg'], rel=1e-12)
    assert summary['total_energy_kwh'] == pytest.approx(full['total_energy_kwh'], rel=1e-12)
    for name in (*COMPONENTS, 'hvac'):
        assert summary[name + '_energy_kwh'] == pytest.approx(full[name + '_energy_kwh'], rel=1e-12)
        assert summary[name + '_carbon_kg'] == pytest.approx(full[name + '_carbon_kg'], rel=1e-12)
    assert summary['carbon_kg'] == pytest.approx(
        summary['background_carbon_kg'] + summary['incremental_carbon_kg'])
    assert summary['total_energy_kwh'] == pytest.approx(
        summary['background_energy_kwh'] + summary['incremental_energy_kwh'])
    assert sum(o.terms['incremental_carbon_kg'] for o in outcomes) == pytest.approx(summary['incremental_carbon_kg'])
    assert summary['physical_prefix_carbon_kg'] + summary['analytic_idle_tail_carbon_kg'] == pytest.approx(full['carbon_kg'])
    assert sum(summary[n + '_carbon_kg'] for n in COMPONENTS) == pytest.approx(summary['carbon_kg'])
    replay.close()
    env.close()


def test_reference_background_is_same_for_different_physical_episode_lengths():
    scenario = fixture_scenario([task('a', duration=45, cpu=.5, sla=2)])
    lower, hold = V2Env(scenario, fixture_config()), V2Env(scenario, fixture_config())
    _, lower_summary = run_episode(lower, lower_if_legal)
    _, hold_summary = run_episode(hold)
    assert lower_summary['physical_steps'] != hold_summary['physical_steps']
    assert lower_summary['background_carbon_kg'] == hold_summary['background_carbon_kg']
    for summary in (lower_summary, hold_summary):
        assert summary['carbon_kg'] == pytest.approx(summary['background_carbon_kg'] + summary['incremental_carbon_kg'])
    lower.close()
    hold.close()


def test_capped_unfinished_tasks_keep_liability_and_deadline_is_charged_once():
    scenario = fixture_scenario([task('long', duration=120, cpu=.5, sla=.1)], total_minutes=60)
    env = V2Env(scenario, fixture_config())
    env.set_deadline_multiplier(100.)
    outcomes, summary = run_episode(env, lower_if_legal)
    assert len(outcomes) == 4
    assert sum(o.terms['new_deadline_misses'] for o in outcomes) == 1
    assert sum(o.terms['deadline_cost'] for o in outcomes) == pytest.approx(.1)
    assert outcomes[-1].terms['terminal_cost'] > 0
    assert summary['unfinished'] == summary['deadline_misses'] == 1
    assert summary['completed'] == 0 and summary['dmr'] == 1
    assert not summary['feasible'] and not summary['complete_episode_for_constraint']
    env.close()
    env = V2Env(scenario, fixture_config(), max_episode_steps=1)
    _, prefix = run_episode(env)
    assert not prefix['fixed_window_complete']
    assert prefix['fixed_window_carbon_kg'] is None
    assert prefix['analytic_idle_intervals'] == 0
    assert prefix['carbon_scope'] == 'censored_physical_prefix'
    assert not prefix['complete_episode_for_constraint']
    with pytest.raises(ValueError):
        env.set_deadline_multiplier(5001)
    env.close()


def test_guard_reserves_known_running_release_for_blocked_urgent_task():
    scenario = fixture_scenario([task('blocker', duration=30, cpu=1, mem=1, sla=1),
                                task('urgent', arrival=15, duration=15, cpu=2, mem=1, sla=2),
                                task('flexible', arrival=15, duration=60, cpu=1, mem=1, sla=4)],
                                total_minutes=150)
    env = V2Env(scenario, fixture_config())
    outcomes, summary = run_episode(env, lower_if_legal)
    guarded = [a for o in outcomes for a in o.task_actions if a['protected_task_ids']]
    assert guarded[0]['task_id'] == 'flexible'
    assert guarded[0]['mask'] == [False, True] and guarded[0]['action'] == 1
    assert guarded[0]['protected_task_ids'] == ['urgent']
    assert summary['deadline_guard_triggers'] >= 1
    assert summary['deadline_guard_conflicts'] == 0
    assert summary['arrived'] == summary['completed'] == 3
    assert summary['deadline_misses'] == 0
    assert 0 <= summary['task_free_action_fraction'] <= 1
    env.executor.check_ownership()
    env.close()


def test_guard_never_empties_urgent_mask_or_reads_future_pending():
    urgent = task('urgent', duration=15, cpu=2, sla=2)
    candidate = task('candidate', duration=60, cpu=1, sla=1)
    running = RunningTask('running', (1, 0, 1), START + STEP)
    result = legal_task_mask(candidate, START, (1, 1, 2), (2, 1, 3),
                             (urgent, candidate), (running,))
    assert result.mask == (True, False) and result.conflict
    with pytest.raises(ValueError, match='future'):
        legal_task_mask(candidate, START, (1, 1, 2), (2, 1, 3),
                        (replace(urgent, arrival_time=START + STEP),), (running,))


def test_future_weather_tasks_and_report_tail_do_not_enter_current_reward_or_actor():
    scenario = fixture_scenario([task('a', duration=45, cpu=.5), task('future', arrival=15)], total_minutes=120)
    other = copy.copy(scenario)
    other.workload = replace(scenario.workload, tasks=(scenario.workload.tasks[0],
                             replace(scenario.workload.tasks[1], cores_req=100)))
    other.environment = replace(scenario.environment, samples=tuple(
        sample if i == 0 else replace(sample, ci_g_per_kwh=999, temperature_c=38, wet_bulb_c=30)
        for i, sample in enumerate(scenario.environment.samples)))
    first, second = V2Env(scenario, fixture_config()), V2Env(other, fixture_config())
    observations1, observations2 = first.reset(), second.reset()
    assert all(np.array_equal(a, b) for a, b in zip(observations1, observations2))
    seen1, seen2 = [], []
    result1 = first.step(start_if_legal, lambda obs, mask: seen1.append(obs.copy()) or 1)
    result2 = second.step(start_if_legal, lambda obs, mask: seen2.append(obs.copy()) or 1)
    assert np.array_equal(seen1[0], seen2[0])
    assert result1.reward == result2.reward
    assert result1.terms == result2.terms
    with pytest.raises(RuntimeError, match='report-only'):
        first._fixed_window_report()
    first.close()
    second.close()


def test_invalid_idle_reference_lower_bound_fails_instead_of_training_false_savings(monkeypatch):
    env = V2Env(fixture_scenario(), fixture_config())
    env.reset()
    reference = env._reference(env.executor.sample)
    invalid = replace(reference, it_kw=reference.it_kw * 100,
                       hvac_kw=reference.hvac_kw * 100, total_kw=reference.total_kw * 100,
                       components_kw={k: v * 100 for k, v in reference.components_kw.items()},
                       legacy_hvac_kw=reference.legacy_hvac_kw * 100,
                       legacy_total_kw=reference.legacy_total_kw * 100)
    monkeypatch.setattr(env, '_reference', lambda _: invalid)
    with pytest.raises(RuntimeError, match='lower bound'):
        env.step(start_if_legal, lambda *_: 1)
    env.close()
