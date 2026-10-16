"""Real executor/plant pretraining on explicitly synthetic tiny fixtures."""
from dataclasses import replace
import time
import pytest

from experiments.e1.config import ROOT
from experiments.joint.config import load_config
from experiments.joint.env import TASK_FIELDS, COOLING_FIELDS
from experiments.joint_v2.ppo import IPPOAgent, PPOConfig
from experiments.joint_v2.pretrain_task import pretrain_task
from experiments.joint_v2.pretrain_cooling import pretrain_cooling
from experiments.joint_v2.scenarios import TrainingScenario
from tests.e1.helpers import inputs, task


def fixture_config():
    return dict(v1_protocol=load_config(), pretrain=dict(
        task=dict(enabled=True, demo_scenarios=1, max_samples=1, bc_epochs=1,
                  max_updates=2, min_wait_fraction=.05),
        cooling=dict(enabled=True, demo_scenarios=1, max_states=3, bc_epochs=1,
                     max_updates=2, ppo_steps=4, ppo_epochs=1)))


def fixture_scenario(role='train'):
    workload, environment = inputs([task('a', duration=30, cpu=.5),
                                    task('b', duration=30, cpu=.5),
                                    task('c', duration=30, cpu=.5)], 120)
    return TrainingScenario('fixture-' + role, role, workload, environment, (2., 1., 3.),
        dict(threshold_g_per_kwh=150),
        dict(location='US-CAL-CISO', total_cores=50000, total_gpus=1000,
             total_mem=80000, hvac_setpoint_c=22), ROOT / 'configs/dcs/dc_config.json',
        dict(role=role, fixture=True, provenance='synthetic unit fixture'), {})


def test_task_single_class_demo_is_reported_and_does_not_collapse_wait_exploration():
    cfg = fixture_config()
    agent = IPPOAgent(len(TASK_FIELDS), 2, PPOConfig(net_arch=(8, 8), n_epochs=1), 41)
    before = agent.parameter_hash()
    result = pretrain_task(agent, [fixture_scenario()], cfg)
    assert result['bc_skipped']
    assert result['reason'] == 'insufficient_free_choice_wait_demonstrations'
    assert result['free_choice_distribution'].get('1', 0) == 0
    assert result['observed_samples'] > result['selected_samples'] == 1
    assert result['omitted_samples'] == result['observed_samples'] - 1
    assert result['executor_tasks_omitted'] == 0
    assert agent.parameter_hash() == before


@pytest.mark.parametrize('pretrain', [pretrain_task, pretrain_cooling])
def test_pretraining_rejects_nontrain_role_before_environment_creation(pretrain):
    agent = IPPOAgent(len(TASK_FIELDS) if pretrain is pretrain_task else len(COOLING_FIELDS),
                      2 if pretrain is pretrain_task else 3, PPOConfig(net_arch=(8, 8)), 41)
    with pytest.raises(ValueError, match='[Tt]rain|role'):
        pretrain(agent, [fixture_scenario('validation')], fixture_config())


def test_cooling_pretraining_uses_real_full_plant_labels_and_updates_actor_critic(tmp_path):
    agent = IPPOAgent(len(COOLING_FIELDS), 3,
                      PPOConfig(net_arch=(8, 8), batch_size=4, n_epochs=1, target_kl=None), 42)
    before = agent.component_hashes()
    result = pretrain_cooling(agent, [fixture_scenario()], fixture_config(), tmp_path)
    assert result['accounting'] == 'joint_full_hvac_v1'
    assert result['sample_origin'] == 'real_trajectory_driven'
    assert result['synthetic_augmentation_samples'] == 0
    assert 0 < result['selected_states'] <= 3
    assert result['enumerated_actions'] >= result['selected_states']
    assert result['bc']['updates'] <= 2
    assert result['ppo']['optimizer_steps'] > 0
    assert result['ppo']['critic_optimizer_steps'] > 0
    assert result['ppo']['actor_optimizer_steps'] > 0
    assert result['physics']['power_spread_max_kw'] > 0
    assert agent.component_hashes()['actor'] != before['actor']
    assert agent.component_hashes()['critic'] != before['critic']
    assert (tmp_path / 'cooling_samples.json').exists()


def test_behavior_cloning_only_changes_actor_and_obeys_masks():
    agent = IPPOAgent(3, 2, PPOConfig(net_arch=(8, 8), batch_size=2), 42)
    before = agent.component_hashes()
    samples = [dict(observation=[.1, .2, .3], mask=[True, True], action=0),
               dict(observation=[.3, .2, .1], mask=[True, True], action=1)]
    result = agent.behavior_clone(samples, epochs=1, max_updates=1)
    assert result['updates'] == 1
    assert agent.component_hashes()['actor'] != before['actor']
    assert agent.component_hashes()['critic'] == before['critic']
    with pytest.raises(ValueError, match='mask'):
        agent.behavior_clone([dict(observation=[0., 0., 0.], mask=[True, False], action=1)])


@pytest.mark.parametrize('pretrain', [pretrain_task, pretrain_cooling])
def test_expired_pretraining_budget_returns_without_environment_steps_or_model_changes(pretrain):
    task_stage = pretrain is pretrain_task
    agent = IPPOAgent(len(TASK_FIELDS) if task_stage else len(COOLING_FIELDS),
                      2 if task_stage else 3, PPOConfig(net_arch=(8, 8)), 41)
    before = agent.parameter_hash()
    result = pretrain(agent, [fixture_scenario()], fixture_config(), deadline=time.monotonic()-1.)
    assert result['budget_exhausted']
    assert not result['all_selected_scenarios_complete']
    assert result['physical_slots' if task_stage else 'demonstration_physical_slots'] == 0
    assert result['bc']['updates'] == 0
    assert agent.parameter_hash() == before


def test_demonstration_selection_uses_distinct_original_fragments_before_augmentations():
    from experiments.joint_v2.pretrain_task import _train_scenarios

    first = fixture_scenario()
    first = replace(first, identity=dict(first.identity, source_fragment_id='original-a'))
    augmented = replace(first, scenario_id='augmentation-a')
    other = replace(first, scenario_id='original-b', identity=dict(first.identity, source_fragment_id='original-b'))
    assert [row.scenario_id for row in _train_scenarios([first, augmented, other], 2)] == [
        first.scenario_id, other.scenario_id]
