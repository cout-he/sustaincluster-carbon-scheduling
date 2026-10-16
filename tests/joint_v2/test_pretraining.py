"""Stage commits must resume exactly after a bounded prefix is rolled back."""
import random

import numpy as np

from experiments.joint_v2.checkpoint import CheckpointStore, global_rng_state, restore_global_rng
from experiments.joint_v2.ppo import IPPOAgent, PPOConfig
from experiments.joint_v2.scenario_sampler import ScenarioSampler
from experiments.joint_v2 import pretraining
from .test_pretrain import fixture_scenario, fixture_config


def setup_transaction(tmp_path):
    agents = {role: IPPOAgent(3, 2, PPOConfig(net_arch=(8, 8), batch_size=2), seed)
              for role, seed in (('task', 41), ('cooling', 42))}
    for agent in agents.values():
        agent.behavior_clone([dict(observation=[.1, .2, .3], mask=[True, True], action=0)],
                             epochs=1, max_updates=1)
    scenarios = [fixture_scenario()]
    sampler = ScenarioSampler(scenarios, 43)
    state = dict(completed_episodes=0, pretrain={}, pretraining_skipped=False,
                 agents={role: agent.export_state() for role, agent in agents.items()},
                 sampler=sampler.state_dict(), global_rng=global_rng_state())
    store = CheckpointStore(tmp_path/'checkpoints', config_identity='config', data_identity='data')
    store.save(state)
    return state, agents, sampler, store, scenarios


def mutate_training_state(agent):
    import torch
    agent.act(np.asarray([.1, .2, .3], dtype=np.float32), [True, True])
    agent.behavior_clone([dict(observation=[.1, .2, .3], mask=[True, True], action=0)],
                         epochs=1, max_updates=1)
    random.random(); np.random.random(); torch.rand(1)


def test_prefix_stage_rolls_back_parameters_optimizer_rng_sampler_and_latest(monkeypatch, tmp_path):
    import torch
    state, agents, sampler, store, scenarios = setup_transaction(tmp_path)
    committed = store.load()
    before = {role: agent.parameter_hash() for role, agent in agents.items()}

    def incomplete(agent, scenarios, config, output, *, deadline=None):
        mutate_training_state(agent)
        sampler.sample()
        return dict(budget_exhausted=True, all_selected_scenarios_complete=False,
                    observed_samples=1, selected_samples=1)

    monkeypatch.setattr(pretraining, 'pretrain_task', incomplete)
    assert not pretraining.run_pretraining(state, agents, sampler, store, scenarios,
                                           fixture_config(), tmp_path, None)
    assert {role: agent.parameter_hash() for role, agent in agents.items()} == before
    assert all(agent.optimizer_steps == committed['agents'][role]['counters']['optimizer_steps']
               for role, agent in agents.items())
    for role, agent in agents.items():
        actual_optimizer = agent.export_state()['optimizer']['state']
        for key, expected_moments in committed['agents'][role]['optimizer']['state'].items():
            assert torch.equal(actual_optimizer[key]['exp_avg'], expected_moments['exp_avg'])
            assert torch.equal(actual_optimizer[key]['exp_avg_sq'], expected_moments['exp_avg_sq'])
    assert sampler.state_dict() == committed['sampler']
    assert state['pretrain'] == store.load()['pretrain'] == {}
    assert not state.get('pretraining_stages', {}).get('task')
    assert not (tmp_path/'pretrain/task.pt').exists()
    actual_rng = (random.random(), np.random.random(), torch.rand(3))
    restore_global_rng(committed['global_rng'])
    expected_rng = (random.random(), np.random.random(), torch.rand(3))
    assert actual_rng[:2] == expected_rng[:2]
    assert torch.equal(actual_rng[2], expected_rng[2])
    for role in agents:
        reference = IPPOAgent.from_state(committed['agents'][role])
        obs = np.asarray([.1, .2, .3], dtype=np.float32)
        assert [agents[role].act(obs, [True, True]).action for _ in range(10)] == [
            reference.act(obs, [True, True]).action for _ in range(10)]


def test_completed_task_stage_is_not_repeated_when_cooling_resumes(monkeypatch, tmp_path):
    state, agents, sampler, store, scenarios = setup_transaction(tmp_path)

    def completed_task(agent, scenarios, config, output, *, deadline=None):
        mutate_training_state(agent)
        return dict(budget_exhausted=False, all_selected_scenarios_complete=True,
                    bc_skipped=True, reason='insufficient_free_choice_wait_demonstrations')

    def interrupted_cooling(agent, scenarios, config, output, *, deadline=None):
        mutate_training_state(agent)
        return dict(budget_exhausted=False, complete=False)

    monkeypatch.setattr(pretraining, 'pretrain_task', completed_task)
    monkeypatch.setattr(pretraining, 'pretrain_cooling', interrupted_cooling)
    assert not pretraining.run_pretraining(state, agents, sampler, store, scenarios,
                                           fixture_config(), tmp_path, None)
    assert store.load()['pretraining_stages'] == {'task': True}
    task_hash = agents['task'].parameter_hash()
    assert (tmp_path/'pretrain/task.pt').exists()
    assert not (tmp_path/'pretrain/cooling.pt').exists()
    state = store.load()
    agents = {role: IPPOAgent.from_state(payload) for role, payload in state['agents'].items()}

    def must_not_repeat(*args, **kwargs):
        raise AssertionError('Committed task stage was repeated')

    def completed_cooling(agent, scenarios, config, output, *, deadline=None):
        mutate_training_state(agent)
        return dict(budget_exhausted=False, all_selected_scenarios_complete=True)

    monkeypatch.setattr(pretraining, 'pretrain_task', must_not_repeat)
    monkeypatch.setattr(pretraining, 'pretrain_cooling', completed_cooling)
    assert pretraining.run_pretraining(state, agents, sampler, store, scenarios,
                                      fixture_config(), tmp_path, None)
    assert agents['task'].parameter_hash() == task_hash
    assert store.load()['pretraining_stages'] == {'task': True, 'cooling': True}
    assert IPPOAgent.load(tmp_path/'pretrain/task.pt').parameter_hash() == task_hash
    assert (tmp_path/'pretrain/cooling.pt').exists()
