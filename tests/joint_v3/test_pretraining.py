"""Trustworthy teacher labels and physical initialization use Train only."""
import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from tests.e1.helpers import task
from tests.joint_v2.test_env import fixture_scenario
from tests.joint_v3.test_env import CausalForecast, config


def api(name):
    try:
        return importlib.import_module('experiments.joint_v3.' + name)
    except ModuleNotFoundError as error:
        pytest.fail('V3 teacher/pretraining is not implemented: ' + str(error))


def test_teacher_wait_requires_reliable_positive_gain_safe_budget_and_same_mask():
    teacher = api('teacher')
    risk = SimpleNamespace(risk_estimation_valid=True, risk_confidence=1.,
        safe_wait_budget_minutes=30., delta_deadline_misses=0)
    carbon = SimpleNamespace(counterfactual_valid=True, forecast_confidence=1.,
        robust_carbon_gain_kg=2.)
    label = teacher.teacher_label([True, True], risk, carbon, config())
    assert label['action'] == 1 and label['weight'] > 0
    risk.risk_estimation_valid = False
    assert teacher.teacher_label([True, True], risk, carbon, config())['weight'] == 0
    risk.risk_estimation_valid = True
    forced = teacher.teacher_label([True, False], risk, carbon, config())
    assert forced['action'] == 0 and forced['weight'] == 0
    assert teacher.teacher_weight(config(), 50, 100) == 0


def test_four_step_dp_uses_unchanged_isolated_model_and_reports_structural_18():
    pretraining = api('pretraining')
    from experiments.joint.power import JointPowerModel
    scenario = fixture_scenario()
    probe = JointPowerModel(scenario.datacenter, scenario.model_path)
    result = pretraining.cooling_dp_label(probe, (.5, 0., .2), scenario.environment.samples[0],
                                          22., config()['v1_protocol']['cooling'], horizon=4)
    assert result['action'] == 0
    assert result['minimum_power_setpoint_c'] == 18.
    assert result['horizon'] == 4
    assert set(result['setpoint_power_kw']) == {'18', '19', '20', '21', '22'}
    assert result['best_path'] == [21., 20., 19., 18.]
    probe.close()


class RecordingActor:
    def __init__(self):
        self.samples = []

    def component_hashes(self):
        return {'actor': 'recording', 'critic': 'unchanged'}

    def behavior_clone(self, samples, *, epochs, max_updates, deadline=None):
        self.samples = samples
        return {'updates': 0, 'samples': len(samples), 'budget_exhausted': False}


def test_collection_uses_actual_unique_states_stops_at_cap_and_rejects_heldout(tmp_path):
    pretraining = api('pretraining')
    scenario = fixture_scenario([task('a', cpu=.1, duration=60, sla=4)], total_minutes=180)
    system = SimpleNamespace(actors={'task': RecordingActor(), 'cooling': RecordingActor()})
    cfg = config()
    cfg['pretrain'] = {'cooling': {'max_states': 2, 'horizon': 4, 'demo_scenarios': 1,
                                 'bc_epochs': 1, 'max_updates': 1},
                       'task': {'max_samples': 2, 'demo_scenarios': 1, 'bc_epochs': 1, 'max_updates': 1}}
    report = pretraining.run_pretraining(system, [scenario], cfg, CausalForecast(), tmp_path)
    assert report['cooling']['distinct_observed_states'] == 2
    assert report['cooling']['samples'] == 2
    assert report['cooling']['collection_complete_episodes'] == 0
    assert len(system.actors['cooling'].samples) == 2
    assert all(len(row['observation']) == 17 for row in system.actors['cooling'].samples)
    scenario.role = 'development'
    with pytest.raises(ValueError, match='Train|train'):
        pretraining.run_pretraining(system, [scenario], cfg, CausalForecast(), tmp_path / 'bad')
