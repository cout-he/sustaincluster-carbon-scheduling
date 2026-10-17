"""Ex-post labels use observed signals, never forecasts as their own truth."""
import copy
import importlib
from dataclasses import replace

import numpy as np
import pytest

from tests.e1.helpers import task
from tests.joint_v2.test_env import fixture_scenario
from tests.joint_v3.test_env import config


class Persistence:
    def predict(self, history, now):
        from experiments.joint_v3.ci_forecaster import CIForecast
        return CIForecast(np.full(16, history[-1].ci_g_per_kwh), np.zeros((8, 16)), 1., True, 'controlled')


def scenario():
    return fixture_scenario([task('a', cpu=.5, duration=15, sla=4)],
                            arrival_minutes=240, total_minutes=360)


def make_env(current=None, **kwargs):
    from experiments.joint_v3.env import V3Env
    return V3Env(current or scenario(), config(), Persistence(), **kwargs)


def finish(env):
    while not env.done:
        env.step(lambda *_: 0, lambda *_: 1)


def mechanism_api():
    try:
        return importlib.import_module('experiments.joint_v3.mechanism_audit')
    except ModuleNotFoundError as error:
        pytest.fail('Independent observed-signal audit is not implemented: ' + str(error))


def test_ci_labels_are_observed_after_episode_with_hand_checked_persistence_error():
    env = make_env()
    env.reset(); finish(env)
    result = env.summary()
    assert 'ci_forecast_error' in result, 'No independent CI forecast labels have been computed'
    errors = result['ci_forecast_error']['by_horizon']
    assert errors[0]['mae_g_per_kwh'] == 0
    assert errors[1]['mae_g_per_kwh'] == pytest.approx(100.)
    assert errors[1]['rmse_g_per_kwh'] == pytest.approx(100.)
    assert errors[1]['count'] == len(env.records)-1
    assert errors[15]['mae_g_per_kwh'] == pytest.approx(1500.)
    assert sum(result['risk_reason_distribution'].values()) == result['task_action_count']
    env.close()


def test_real_future_changes_audit_label_but_cannot_change_first_actor_or_forecast():
    original = scenario()
    changed = copy.copy(original)
    changed.environment = replace(original.environment, samples=tuple(
        sample if index == 0 else replace(sample, ci_g_per_kwh=100.)
        for index, sample in enumerate(original.environment.samples)))
    first, second = make_env(original, evaluation=True), make_env(changed, evaluation=True)
    for env in (first, second):
        env.reset(); env.step(lambda *_: 0, lambda *_: 1)
    assert 'ci_forecast_mean' in first.records[0], 'Causal forecasts are not retained for independent scoring'
    np.testing.assert_array_equal(first.records[0]['ci_forecast_mean'], second.records[0]['ci_forecast_mean'])
    np.testing.assert_array_equal(first.action_records[0]['task_actions'][0]['observation'],
                                  second.action_records[0]['task_actions'][0]['observation'])
    audit = mechanism_api()
    with pytest.raises(RuntimeError, match='after|completed|done'):
        audit.audit_probes(first)
    for env in (first, second): finish(env)
    observed = [audit.audit_probes(env) for env in (first, second)]
    assert observed[0]['scored'] == observed[1]['scored'] == 1
    assert observed[0]['records'][0]['observed_signal_gain_kg'] != pytest.approx(
        observed[1]['records'][0]['observed_signal_gain_kg'])
    assert observed[1]['mean_gain_mae_kg'] == pytest.approx(0., abs=1e-10)
    for env in (first, second): env.close()


def test_unobserved_counterfactual_horizon_is_unscored_and_training_has_no_probe_buffer():
    audit = mechanism_api()
    evaluation = make_env(evaluation=True, max_episode_steps=1)
    training = make_env(max_episode_steps=1)
    for env in (evaluation, training): env.reset(); finish(env)
    assert len(evaluation.mechanism_probes) == 1
    assert not training.mechanism_probes
    result = audit.audit_probes(evaluation)
    assert result['scored'] == 0 and result['unscored'] == 1
    assert result['unscored_reasons']['unobserved_physical_prefix'] == 1
    assert result['sign_accuracy'] is None
    assert evaluation.summary()['ci_forecast_error']['by_horizon'][1]['count'] == 0
    for env in (evaluation, training): env.close()


def test_unestimated_candidates_are_not_counted_as_fast_power_proxies():
    from experiments.joint_v3.env import V3Env
    from tests.joint_v3.test_env import CausalForecast
    cfg = config()
    cfg['carbon']['max_fast_candidates'] = 1
    jobs = [task(str(index), cpu=.1, mem=.1, duration=15, sla=4) for index in range(4)]
    env = V3Env(fixture_scenario(jobs), cfg, CausalForecast())
    env.reset()
    outcome = env.step(lambda *_: 0, lambda *_: 1)
    assert sum(row['fine_estimation'] for row in outcome.task_actions) == 1
    assert sum(row['fast_approximation'] for row in outcome.task_actions) == 1
    assert sum(row['unestimated'] for row in outcome.task_actions) == 2
    assert outcome.task_actions[0]['carbon_reason'] == 'forecast_invalid:test_cold_start'
    assert all(row['action'] == 0 for row in outcome.task_actions)
    env.close()
