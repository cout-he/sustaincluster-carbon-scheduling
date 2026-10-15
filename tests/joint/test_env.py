"""Behavioral contracts over the reused executor and the actual physical model."""
from dataclasses import replace

import numpy as np
import pytest

from experiments.e1.contracts import STEP
from experiments.joint.config import load_config
from experiments.joint.env import JointEnv, TASK_FIELDS, COOLING_FIELDS
from experiments.joint.power import JointPowerModel
from tests.e1.helpers import inputs, task


def make_env(tasks, minutes=60, mode='B3', environment=None):
    workload, original = inputs(tasks, minutes)
    cfg = load_config()
    # Small logical pool for allocation assertions, production-size plant for
    # the physical equations (upstream rack sizing cannot represent 2 cores).
    dc = dict(location='US-CAL-CISO', total_cores=50000, total_gpus=1000, total_mem=80000,
              hvac_setpoint_c=22)
    from experiments.e1.config import ROOT
    return JointEnv(workload, environment or original, (2, 1, 3),
                    lambda: JointPowerModel(dc, ROOT / 'configs/dcs/dc_config.json',
                                            cooling_config=cfg['cooling']),
                    dict(threshold_g_per_kwh=150), cfg, mode=mode)


def test_sequential_tasks_resources_masks_and_single_physical_clock():
    env = make_env([task('a', cpu=1), task('b', cpu=1), task('c', cpu=1)])
    env.reset()
    calls = []
    def actor(obs, mask):
        calls.append((obs.copy(), mask.copy()))
        return 0
    result = env.step(actor, lambda obs, mask: 1)
    assert len(calls) == 2  # third task cannot fit; no actor call
    assert result.interval['used_cores'] == 2
    assert result.interval['interval_hours'] == .25
    assert env.executor.observed_end == env.executor.workload.start + STEP
    assert result.interval['capacity_wait_count'] == 1
    assert len(set(d['task_id'] for d in result.task_actions)) == 2
    assert all(np.isfinite(obs).all() and np.max(np.abs(obs)) <= 1 for obs, _ in calls)
    env.close()


def test_task_specific_actions_slack_guard_and_no_silent_queue_limit():
    env = make_env([task('short', duration=15, sla=1),
                    task('wait', cpu=.5, duration=15, sla=8),
                    task('start', cpu=.5, duration=15, sla=8)])
    env.reset()
    seen = []
    def actor(obs, mask):
        seen.append(mask.copy())
        return 1 if mask[1] and len(seen) == 2 else 0
    result = env.step(actor, lambda obs, mask: 1)
    assert not seen[0][1]
    assert [x['action'] for x in result.task_actions] == [0, 1, 0]
    assert result.interval['pending'] == 1
    env.executor.check_ownership()
    env.close()


def test_empty_slot_cooling_still_advances_and_invalid_actions_rejected():
    env = make_env([])
    task_obs, cooling_obs = env.reset()
    assert len(task_obs) == len(TASK_FIELDS) and len(cooling_obs) == len(COOLING_FIELDS)
    outcome = env.step(lambda *_: pytest.fail('No pending task'), lambda *_: 0)
    assert len(outcome.task_actions) == 0
    assert outcome.interval['setpoint_c'] == 21
    assert len(outcome.cooling_actions) == 1
    env.close()
    env = make_env([task('urgent', sla=1)])
    env.reset()
    with pytest.raises(ValueError, match='mask|legal'):
        env.step(lambda *_: 1, lambda *_: 1)
    env.close()


def test_future_inputs_do_not_change_current_observations_or_first_transition():
    tasks = [task('a', duration=45, cpu=.5), task('future', arrival=30)]
    env1 = make_env(tasks)
    original = env1._environment
    changed = replace(original, samples=tuple(
        sample if i == 0 else replace(sample, ci_g_per_kwh=999, temperature_c=38, wet_bulb_c=30)
        for i, sample in enumerate(original.samples)))
    env2 = make_env([tasks[0], replace(tasks[1], cores_req=100)], environment=changed)
    obs1, obs2 = env1.reset(), env2.reset()
    assert all(np.array_equal(a, b) for a, b in zip(obs1, obs2))
    seen1, seen2 = [], []
    first = env1.step(lambda *_: 0, lambda obs, mask: seen1.append(obs.copy()) or 1)
    second = env2.step(lambda *_: 0, lambda obs, mask: seen2.append(obs.copy()) or 1)
    assert np.array_equal(seen1[0], seen2[0])
    timing = {'task_decision_seconds', 'cooling_decision_seconds'}
    assert {k:v for k,v in first.interval.items() if k not in timing} == {
        k:v for k,v in second.interval.items() if k not in timing}
    env1.close(); env2.close()


def test_endpoint_liability_deadline_once_and_per_mode_entrypoints():
    for mode in ('B0', 'B1', 'B2', 'B3', 'RiskOnly'):
        env = make_env([task('long', duration=75, cpu=.5, sla=.2)], 60, mode)
        env.reset()
        outcomes = [env.step(lambda *_: 0, lambda *_: 1) for _ in range(4)]
        assert sum(x.terms['new_deadline_misses'] for x in outcomes) == 1
        assert outcomes[-1].done and outcomes[-1].terms['terminal_cost'] > 0
        assert all(x.terms['terminal_cost'] == 0 for x in outcomes[:-1])
        summary = env.summary()
        assert summary['accounting'] == 'joint_full_hvac_v1'
        assert not summary['feasible'] and summary['dmr'] == 1
        env.close()


def test_large_queue_has_no_fixed_task_prefix_or_duplicate_decisions():
    env = make_env([task('job-' + str(i), cpu=.001, mem=0) for i in range(300)], 15)
    env.reset()
    outcome = env.step(lambda *_: 0, lambda *_: 1)
    assert len(outcome.task_actions) == 300
    assert len({r['task_id'] for r in outcome.task_actions}) == 300
    assert env.summary()['omitted_pending_tasks'] == 0
    assert env.summary()['completed'] == 300
    env.close()


def test_thermal_excess_is_a_separate_system_reward_cost():
    env = make_env([], 15)
    env.reset()
    outcome = env.step(lambda *_: 0, lambda *_: 1)
    interval = dict(outcome.interval, rack_inlet_max_c=30.)
    reward, terms = env.reward_model.calculate(env.executor, interval, terminal=True, inlet_limit_c=28.)
    assert terms['thermal_excess_c'] == 2.
    assert terms['thermal_safety_cost'] == 20.
    assert reward == pytest.approx(outcome.reward - 20.)
    env.close()
