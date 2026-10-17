"""Real-plant causal boundary and safe fallback contracts for RAC-MAPPO."""
import copy
import importlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tests.e1.helpers import task
from tests.joint_v2.test_env import fixture_scenario, fixture_config


def api():
    try:
        return importlib.import_module('experiments.joint_v3.env')
    except ModuleNotFoundError as error:
        pytest.fail('V3 causal environment is not implemented: ' + str(error))


class CausalForecast:
    def __init__(self):
        self.histories = []

    def predict(self, history, now):
        values = [sample.ci_g_per_kwh for sample in history]
        self.histories.append((now, values))
        return SimpleNamespace(mean=np.full(16, values[-1]),
            scenarios=np.full((8, 16), values[-1]), confidence=0., valid=False,
            reason='test_cold_start')


def config():
    value = fixture_config()
    value.update(risk={'max_tasks': 256, 'max_slots': 32, 'max_wait_slots': 2,
                      'time_budget_ms': 100, 'cache_size': 32},
                 carbon={'horizon': 16, 'max_fine_candidates': 1},
                 teacher={'threshold_kg': 0., 'initial_weight': .1, 'decay_fraction': .5})
    return value


def test_task40_keeps_v2_prefix_and_actor_executes_under_safe_fallback():
    module = api()
    from experiments.joint_v2.env import V2Env
    scenario = fixture_scenario([task('a', cpu=.1, duration=15, sla=4)])
    forecast = CausalForecast()
    env = module.V3Env(scenario, config(), forecast)
    old = V2Env(scenario, config())
    initial, baseline = env.reset(), old.reset()
    assert initial[0].shape == (40,)
    np.testing.assert_array_equal(initial[0][:35], baseline[0])
    calls = []
    outcome = env.step(lambda obs, mask: calls.append((obs, mask)) or 0, lambda *_: 1)
    assert calls and calls[0][0].shape == (40,)
    assert outcome.task_actions[0]['action'] == 0
    assert outcome.task_actions[0]['teacher_weight'] == 0
    assert not outcome.task_actions[0]['teacher_reliable']
    env.close(); old.close()


def test_global_task_boundary_has_arrivals_and_cooling_boundary_has_actual_starts():
    env = api().V3Env(fixture_scenario([task('a', cpu=1, duration=30, sla=4)]),
                      config(), CausalForecast())
    env.reset()
    env.step(lambda *_: 0, lambda *_: 1)
    before, after = env.task_global_state, env.cooling_global_state
    assert before['context'].shape == after['context'].shape == (26,)
    assert before['queue'].shape == (1, 5)
    assert after['queue'].shape == (0, 5)
    assert before['context'][0] == 0
    assert after['context'][0] == .5
    assert before['context'][3] > after['context'][3]
    assert np.max(np.abs(before['context'])) <= 1
    env.close()


def test_history_is_observed_once_per_slot_and_future_environment_does_not_leak():
    scenario = fixture_scenario([task('a', cpu=.1, duration=45, sla=4)], total_minutes=150)
    changed = copy.copy(scenario)
    changed.environment = replace(scenario.environment, samples=tuple(
        sample if i == 0 else replace(sample, ci_g_per_kwh=999., temperature_c=38.)
        for i, sample in enumerate(scenario.environment.samples)))
    forecasts = [CausalForecast(), CausalForecast()]
    envs = [api().V3Env(s, config(), f) for s, f in zip((scenario, changed), forecasts)]
    for env in envs:
        env.reset(); env.step(lambda *_: 0, lambda *_: 1)
    np.testing.assert_array_equal(envs[0].task_global_state['context'], envs[1].task_global_state['context'])
    assert len(forecasts[0].histories) == 1
    assert len(forecasts[0].histories[0][1]) == 1
    envs[0].step(lambda *_: 0, lambda *_: 1)
    assert len(forecasts[0].histories) == 2
    assert len(forecasts[0].histories[1][1]) == 2
    for env in envs: env.close()


def test_candidate_budget_falls_back_without_dropping_real_actions():
    jobs = [task(str(i), cpu=.1, mem=.1, duration=15, sla=4) for i in range(4)]
    env = api().V3Env(fixture_scenario(jobs), config(), CausalForecast())
    env.reset()
    result = env.step(lambda *_: 0, lambda *_: 1)
    assert len(result.task_actions) == 4
    assert sum(r['fine_estimation'] for r in result.task_actions) <= 1
    assert all(r['action'] == 0 for r in result.task_actions)
    assert len(env.executor.ledger) == 4
    env.close()


def test_reliable_teacher_wait_does_not_override_executing_actor():
    class FallingForecast:
        def predict(self, history, now):
            from experiments.joint_v3.ci_forecaster import CIForecast
            current = history[-1].ci_g_per_kwh
            return CIForecast(np.linspace(current, current*.1, 16), np.zeros((8, 16)), 1., True, 'controlled')
    env = api().V3Env(fixture_scenario([task('a', cpu=.5, duration=15, sla=4)]),
                      config(), FallingForecast())
    env.reset()
    row = env.step(lambda *_: 0, lambda *_: 1).task_actions[0]
    assert row['teacher_action'] == 1 and row['teacher_weight'] > 0
    assert row['action'] == 0 and not row['actor_teacher_agreement']
    assert row['robust_carbon_gain_kg'] > 0
    env.close()
