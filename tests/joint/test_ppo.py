"""Clock, joint-action likelihood, masking, and checkpoint acceptance tests.

These fixtures never open a workload bundle or a held-out split.  They exercise
the real SB3 actor/critic and optimizer with hand-computed physical-slot targets.
"""
from dataclasses import replace
import math

import numpy as np
import pytest

from experiments.joint.ppo import (
    Decision, IPPOAgent, PPOConfig, SlotTransition, compute_gae,
)


def small_agent(**overrides):
    options = dict(n_steps=8, batch_size=8, n_epochs=1, gamma=0.9,
                   gae_lambda=0.8, net_arch=(8, 8), target_kl=None)
    options.update(overrides)
    return IPPOAgent(3, 2, PPOConfig(**options), seed=41)


def test_gae_discounts_once_per_physical_slot_including_empty_slot():
    # There are three physical slots regardless of their candidate counts.
    advantages, returns = compute_gae(
        [1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [False, False, True],
        bootstrap_value=999.0, gamma=0.5, gae_lambda=1.0,
    )
    np.testing.assert_allclose(returns, [2.75, 3.5, 3.0])
    np.testing.assert_allclose(advantages, [2.75, 3.5, 3.0])


def test_gae_stops_at_each_terminal_and_bootstraps_only_rollout_cut():
    advantages, returns = compute_gae(
        [1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [False, True, False],
        bootstrap_value=4.0, gamma=0.5, gae_lambda=1.0,
    )
    np.testing.assert_allclose(returns, [2.0, 2.0, 5.0])
    np.testing.assert_allclose(advantages, [2.0, 2.0, 5.0])


def test_gae_subtracts_boundary_values_and_terminal_liability_is_not_bootstrapped():
    advantages, returns = compute_gae(
        [1.0, 2.0], [10.0, 20.0], [False, True],
        bootstrap_value=100.0, gamma=0.9, gae_lambda=0.8,
    )
    np.testing.assert_allclose(advantages, [-3.96, -18.0], atol=1e-6)
    np.testing.assert_allclose(returns, [6.04, 2.0], atol=1e-6)


def test_joint_ratio_multiplies_task_likelihoods_but_reward_is_once_per_slot():
    import torch

    agent = small_agent(gamma=0.0, normalize_advantage=False, ent_coef=0.0, vf_coef=0.0)
    with torch.no_grad():
        for parameter in agent.policy.parameters():
            parameter.zero_()
    observation = np.zeros(3, dtype=np.float32)
    decision = agent.act(observation, [True, True], deterministic=True)
    # pi_new / pi_old = 1.1 for EACH conditional task action.  Thus the
    # three-task slot has ratio 1.331 and is clipped ONCE to 1.2.
    old = replace(decision, log_prob=decision.log_prob - math.log(1.1))
    slots = [
        SlotTransition(observation, [old], 1.0, True, value=0.0),
        SlotTransition(observation, [old, old, old], 1.0, True, value=0.0),
        SlotTransition(observation, [], 1.0, True, value=0.0),
    ]
    metrics = agent.update(slots)
    assert metrics['policy_loss'] == pytest.approx(-(1.1 + 1.2) / 3, abs=1e-6)
    assert metrics['mean_reward'] == pytest.approx(1.0)
    assert metrics['mean_return'] == pytest.approx(1.0)
    assert metrics['slots'] == 3
    assert metrics['decisions'] == 4
    assert metrics['actor_slots'] == 2


def test_empty_task_slots_train_critic_without_actor_or_entropy_weight():
    import torch

    agent = small_agent(normalize_advantage=False)
    before_actor = {name: tensor.detach().clone() for name, tensor in agent.policy.named_parameters()
                    if 'policy_net' in name or 'action_net' in name}
    before_critic = agent.policy.value_net.weight.detach().clone()
    observation = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    metrics = agent.update([
        SlotTransition(observation, [], -2.0, False),
        SlotTransition(observation, [], -3.0, True),
    ])
    assert metrics['policy_loss'] == 0.0
    assert metrics['entropy'] == 0.0
    assert metrics['actor_slots'] == 0
    assert not torch.equal(before_critic, agent.policy.value_net.weight)
    for name, tensor in agent.policy.named_parameters():
        if name in before_actor:
            assert torch.equal(before_actor[name], tensor)


def test_forced_only_slots_do_not_move_actor_via_existing_adam_momentum():
    import torch

    agent = small_agent(normalize_advantage=False, ent_coef=0.01)
    observation = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    # Populate Adam's actor moments first; zero gradients on the next update
    # would still move these parameters unless the actor graph is absent.
    first = agent.act(observation, [True, True])
    agent.update([SlotTransition(observation, [first], -2.0, True)])
    actor_before = {name: tensor.detach().clone() for name, tensor in agent.policy.named_parameters()
                    if 'policy_net' in name or 'action_net' in name}
    assert any(agent.policy.optimizer.state[parameter].get('exp_avg', torch.zeros_like(parameter)).abs().sum() > 0
               for name, parameter in agent.policy.named_parameters() if name in actor_before)
    critic_before = agent.policy.value_net.weight.detach().clone()
    forced = agent.act(observation, [True, False])
    metrics = agent.update([SlotTransition(observation, [forced, forced], -3.0, True)])
    assert metrics['policy_loss'] == 0.0
    assert metrics['entropy'] == 0.0
    assert metrics['actor_slots'] == 0
    assert metrics['decisions'] == 2
    for name, tensor in agent.policy.named_parameters():
        if name in actor_before:
            assert torch.equal(actor_before[name], tensor)
            assert tensor.grad is None
    assert not torch.equal(critic_before, agent.policy.value_net.weight)


def test_action_mask_is_applied_and_stored_for_recomputation():
    agent = small_agent(vf_coef=0.0, ent_coef=0.0, normalize_advantage=False)
    observation = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    mask = np.asarray([False, True])
    decision = agent.act(observation, mask)
    assert decision.action == 1
    assert decision.log_prob == pytest.approx(0.0, abs=1e-6)
    # Caller mutations must not alter the exact rollout input or legal set.
    observation[:] = 99.0
    mask[:] = True
    np.testing.assert_array_equal(decision.observation, np.asarray([0.1, 0.2, 0.3], dtype=np.float32))
    np.testing.assert_array_equal(decision.action_mask, [False, True])
    metrics = agent.update([
        SlotTransition(decision.observation, [decision], -1.0, True),
    ])
    assert metrics['approx_kl'] == pytest.approx(0.0, abs=1e-6)
    assert metrics['entropy'] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize('mask', [[False, False], [True], [1, 0]])
def test_invalid_or_non_boolean_action_masks_are_rejected(mask):
    with pytest.raises(ValueError, match='mask'):
        small_agent().act(np.zeros(3, dtype=np.float32), mask)


def test_saved_rollout_illegal_action_cannot_enter_update():
    agent = small_agent()
    observation = np.zeros(3, dtype=np.float32)
    bad = Decision(observation, np.asarray([True, False]), 1, -0.5)
    with pytest.raises(ValueError, match='legal|mask'):
        agent.update([SlotTransition(observation, [bad], -1.0, True)])


def test_nonterminal_update_requires_an_explicit_boundary_bootstrap():
    agent = small_agent()
    observation = np.zeros(3, dtype=np.float32)
    with pytest.raises(ValueError, match='bootstrap'):
        agent.update([SlotTransition(observation, [], -1.0, False)])


def test_huge_joint_log_ratio_stops_before_an_unsafe_optimizer_update():
    agent = small_agent(target_kl=0.01, normalize_advantage=False)
    observation = np.zeros(3, dtype=np.float32)
    decision = agent.act(observation, [True, True])
    corrupted = replace(decision, log_prob=-1000.0)
    before = agent.parameter_hash()
    metrics = agent.update([
        SlotTransition(observation, [corrupted] * 32, -1.0, True),
    ])
    assert metrics['stopped_early']
    assert metrics['optimizer_steps'] == 0
    assert agent.parameter_hash() == before
    assert all(math.isfinite(value) for value in metrics.values()
               if isinstance(value, (int, float)))


def test_extreme_finite_joint_log_ratio_cannot_make_kl_negative_or_bypass_guard():
    agent = small_agent(target_kl=None, normalize_advantage=False)
    observation = np.zeros(3, dtype=np.float32)
    decision = agent.act(observation, [True, True])
    # exp is capped at exp(60). Subtracting this *uncapped* log ratio would
    # produce a negative KL and let the numerically invalid batch optimize.
    corrupted = replace(decision, log_prob=-1e100)
    before = agent.parameter_hash()
    metrics = agent.update([SlotTransition(observation, [corrupted], -1.0, True)])
    assert metrics['stopped_early']
    assert metrics['numerical_guard_triggered']
    assert math.isfinite(metrics['approx_kl']) and metrics['approx_kl'] >= 0
    assert metrics['optimizer_steps'] == 0
    assert agent.parameter_hash() == before


def test_both_independent_actor_critics_update_and_checkpoint_roundtrip(tmp_path):
    import torch

    task = small_agent(normalize_advantage=False)
    cooling = IPPOAgent(3, 3, task.config, seed=42)
    for name, agent in [('task', task), ('cooling', cooling)]:
        before = agent.parameter_hash()
        slots = []
        for index in range(4):
            observation = np.asarray([index / 4, 0.25, 0.5], dtype=np.float32)
            decisions = [agent.act(observation, np.ones(agent.action_dim, dtype=bool))]
            slots.append(SlotTransition(observation, decisions, -1.0-index, index == 3))
        metrics = agent.update(slots)
        assert metrics['optimizer_steps'] > 0
        assert agent.parameter_hash() != before
        assert len(agent.policy.optimizer.state_dict()['state']) > 0
        checkpoint = tmp_path / f'{name}.pt'
        probe = np.asarray([0.2, 0.25, 0.5], dtype=np.float32)
        mask = np.ones(agent.action_dim, dtype=bool)
        deterministic = agent.act(probe, mask, deterministic=True).action
        agent.save(checkpoint)
        expected_samples = [agent.act(probe, mask).action for _ in range(10)]
        restored = IPPOAgent.load(checkpoint)
        assert restored.parameter_hash() == agent.parameter_hash()
        assert restored.num_slots == agent.num_slots == 4
        assert restored.optimizer_steps == agent.optimizer_steps
        assert restored.act(probe, mask, deterministic=True).action == deterministic
        assert [restored.act(probe, mask).action for _ in range(10)] == expected_samples
        # Identical optimizer + minibatch RNG state must continue identically.
        agent.update(slots)
        restored.update(slots)
        assert restored.parameter_hash() == agent.parameter_hash()
        for left, right in zip(agent.policy.parameters(), restored.policy.parameters()):
            assert torch.equal(left, right)


def test_agents_have_independent_sampling_rng_streams():
    observation = np.zeros(3, dtype=np.float32)
    task = small_agent()
    reference = small_agent()
    unrelated = IPPOAgent(3, 2, task.config, seed=7)
    actual = []
    for _ in range(12):
        unrelated.act(observation, [True, True])
        actual.append(task.act(observation, [True, True]).action)
    assert actual == [reference.act(observation, [True, True]).action for _ in range(12)]
