"""V2 observability must preserve V1's physical-slot PPO objective."""
from dataclasses import replace
import io
import math
import time

import numpy as np
import pytest

from experiments.joint.ppo import IPPOAgent as V1Agent
from experiments.joint_v2.ppo import IPPOAgent, PPOConfig, SlotTransition


def config(**overrides):
    options = dict(n_steps=8, batch_size=4, n_epochs=2, net_arch=(8, 8), target_kl=None)
    options.update(overrides)
    return PPOConfig(**options)


def test_instrumentation_preserves_v1_joint_objective_and_parameter_updates():
    cfg = config()
    old, new = V1Agent(3, 2, cfg, 43), IPPOAgent(3, 2, cfg, 43)
    slots = []
    for index, count in enumerate([0, 1, 3, 2]):
        obs = np.asarray([index / 4, .2, .4], dtype=np.float32)
        decisions = [old.act(obs, [True, True]) for _ in range(count)]
        slots.append(SlotTransition(obs, decisions, -index - 1., index == 3))
    old_metrics, new_metrics = old.update(slots), new.update(slots)
    assert new.parameter_hash() == old.parameter_hash()
    for name in ('policy_loss', 'value_loss', 'entropy', 'mean_return'):
        assert new_metrics[name] == pytest.approx(old_metrics[name], abs=1e-10)
    assert new_metrics['candidates_per_slot'] == [0, 1, 3, 2]
    assert new_metrics['free_choices_per_slot'] == [0, 1, 3, 2]
    assert new_metrics['actor_optimizer_steps'] == 2
    assert new_metrics['critic_optimizer_steps'] == 2
    assert new_metrics['effective_actor_steps'] > 0
    assert new_metrics['gradient_norm_max'] > 0
    assert new_metrics['joint_log_ratio']['count'] == 6  # 3 active slots x 2 epochs


def test_joint_log_ratio_is_a_sum_of_conditional_ratios():
    import torch

    agent = IPPOAgent(3, 2, config(n_epochs=1, gamma=0., normalize_advantage=False,
                                vf_coef=0., ent_coef=0.), 5)
    with torch.no_grad():
        for parameter in agent.policy.parameters():
            parameter.zero_()
    obs = np.zeros(3, dtype=np.float32)
    decision = agent.act(obs, [True, True])
    old = replace(decision, log_prob=decision.log_prob - math.log(1.1))
    metrics = agent.update([SlotTransition(obs, [old] * 3, 1., True)])
    assert metrics['joint_log_ratio']['mean'] == pytest.approx(3 * math.log(1.1), abs=1e-6)
    assert metrics['policy_loss'] == pytest.approx(-1.2, abs=1e-6)


def test_forced_only_update_has_no_actor_gradient_or_momentum_change():
    agent = IPPOAgent(3, 2, config(n_epochs=1, normalize_advantage=False), 5)
    obs = np.asarray([.1, .2, .3], dtype=np.float32)
    agent.update([SlotTransition(obs, [agent.act(obs, [True, True])], -2., True)])
    before = agent.component_hashes()
    forced = agent.act(obs, [True, False])
    metrics = agent.update([SlotTransition(obs, [forced, forced], -3., True)])
    after = agent.component_hashes()
    assert before['actor'] == after['actor']
    assert before['critic'] != after['critic']
    assert metrics['actor_optimizer_steps'] == 0
    assert metrics['critic_optimizer_steps'] == 1
    assert metrics['zero_effective_actor_update']
    assert metrics['free_choices_per_slot'] == [0]
    assert metrics['candidates_per_slot'] == [2]


def test_kl_rejection_reports_lost_budget_and_zero_effective_actor_update():
    agent = IPPOAgent(3, 2, config(n_epochs=2, target_kl=.001), 5)
    obs = np.zeros(3, dtype=np.float32)
    bad = replace(agent.act(obs, [True, True]), log_prob=-100.)
    metrics = agent.update([SlotTransition(obs, [bad], -1., True)])
    assert metrics['target_kl_stopped'] or metrics['numerical_guard_triggered']
    assert metrics['optimizer_steps'] == 0
    assert metrics['zero_effective_actor_update']
    assert metrics['planned_minibatches'] == 2


def test_export_state_preserves_optimizer_rng_counters_and_v2_class():
    import torch

    agent = IPPOAgent(3, 2, config(), 11)
    obs = np.asarray([.1, .3, .5], dtype=np.float32)
    slots = [SlotTransition(obs, [agent.act(obs, [True, True])], -1., True)]
    agent.update(slots)
    state = agent.export_state()
    stream = io.BytesIO()
    torch.save(state, stream)
    stream.seek(0)
    restored = IPPOAgent.from_state(torch.load(stream, weights_only=True))
    assert type(restored) is IPPOAgent
    assert restored.component_hashes() == agent.component_hashes()
    assert restored.actor_optimizer_steps == agent.actor_optimizer_steps
    assert [restored.act(obs, [True, True]).action for _ in range(10)] == [
        agent.act(obs, [True, True]).action for _ in range(10)]
    agent.update(slots)
    restored.update(slots)
    assert restored.parameter_hash() == agent.parameter_hash()


def test_export_state_is_a_snapshot_and_not_a_live_parameter_view():
    agent = IPPOAgent(3, 2, config(), 11)
    before = agent.parameter_hash()
    state = agent.export_state()
    obs = np.asarray([.1, .3, .5], dtype=np.float32)
    agent.update([SlotTransition(obs, [agent.act(obs, [True, True])], -1., True)])
    assert agent.parameter_hash() != before
    assert IPPOAgent.from_state(state).parameter_hash() == before


def test_expired_minibatch_deadline_prevents_bc_and_ppo_optimizer_updates():
    agent = IPPOAgent(3, 2, config(), 11)
    obs = np.asarray([.1, .3, .5], dtype=np.float32)
    before = agent.parameter_hash()
    expired = time.monotonic()-1.
    bc = agent.behavior_clone([dict(observation=obs, mask=[True, True], action=0)], deadline=expired)
    assert bc['budget_exhausted'] and bc['updates'] == 0
    ppo = agent.update([SlotTransition(obs, [agent.act(obs, [True, True])], -1., True)], deadline=expired)
    assert ppo['budget_exhausted'] and ppo['optimizer_steps'] == 0
    assert ppo['attempted_minibatches'] == 0
    assert agent.parameter_hash() == before


def test_restored_training_does_not_mutate_source_snapshot_adam_moments():
    import torch

    agent = IPPOAgent(3, 2, config(), 11)
    obs = np.asarray([.1, .3, .5], dtype=np.float32)
    slots = [SlotTransition(obs, [agent.act(obs, [True, True])], -1., True)]
    agent.update(slots)
    snapshot = agent.export_state()
    moments = {key: value['exp_avg'].clone() for key, value in snapshot['optimizer']['state'].items()}
    restored = IPPOAgent.from_state(snapshot)
    restored.update(slots)
    for key, before in moments.items():
        assert torch.equal(snapshot['optimizer']['state'][key]['exp_avg'], before)
