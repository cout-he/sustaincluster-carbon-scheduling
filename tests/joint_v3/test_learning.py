"""Physical-clock costs, masked learning, and full-state reproducibility."""
import copy
import numpy as np
import pytest
import torch


def state(queue=()):
    return dict(context=np.zeros(26, dtype=np.float32),
                queue=np.asarray(queue, dtype=np.float32).reshape(-1, 5))


def test_deadline_cost_uses_final_cohort_not_candidate_count():
    from experiments.joint_v3.reward import episode_costs, GroupConstraints
    costs = episode_costs([dict(incremental_carbon_kg=100, new_deadline_misses=2,
        wait_cost=.2, thermal_safety_cost=0., terminal_cost=.3)], 10)
    assert costs.tolist() == [[.1, .2, .5]]
    groups = GroupConstraints()
    key = groups.key(.75, 1.5)
    for _ in range(20):
        groups.update(key, 0., role='train', complete=True)
    assert groups.values[key] == 10.
    heavy = groups.key(1.25, 1.5)
    groups.update(heavy, .2, role='train', complete=True)
    assert groups.values[heavy] > 10.
    with pytest.raises(ValueError):
        groups.update(key, .2, role='development', complete=True)


def test_global_queue_pool_is_permutation_invariant_and_empty_finite():
    from experiments.joint_v3.critics import DualCostCritic
    critic = DualCostCritic()
    q = [[.1, .2, .3, .4, .5], [.6, .5, .4, .3, .2]]
    a = critic([state(q), state(q[::-1]), state()])
    assert a.shape == (3, 2)
    assert torch.isfinite(a).all()
    assert torch.allclose(a[0], a[1], atol=1e-7)


def test_forced_actions_have_zero_actor_update_and_resume_sampling_exact():
    from experiments.joint_v3.mappo import RACSystem
    system = RACSystem(seed=19)
    obs = np.zeros(40, dtype=np.float32)
    decision = system.actors['task'].act(obs, np.array([True, False]))
    before = system.actors['task'].parameter_hash()
    row = dict(task_global=state(), cooling_global=state(), task_boundary=obs,
        cooling_boundary=np.zeros(17, dtype=np.float32), task_decisions=[decision],
        cooling_decisions=[], task_actions=[], done=True,
        terms=dict(incremental_carbon_kg=100, new_deadline_misses=0,
                   wait_cost=0., thermal_safety_cost=0., terminal_cost=0.))
    report = system.update([row], total_tasks=1, multiplier=10., auxiliary_weight=0.)
    assert system.actors['task'].parameter_hash() == before
    assert report['task']['actor_optimizer_steps'] == 0
    assert report['critic_optimizer_steps'] > 0
    saved = copy.deepcopy(system.export_state())
    restored = RACSystem.from_state(saved)
    mask = np.array([True, True])
    assert [system.actors['task'].act(obs, mask).action for _ in range(30)] == [
        restored.actors['task'].act(obs, mask).action for _ in range(30)]


def test_joint_likelihood_sums_free_decisions_once_per_physical_slot():
    from experiments.joint_v3.mappo import MaskedActor, joint_statistics
    actor = MaskedActor(40, 2, seed=11)
    obs = np.zeros(40, dtype=np.float32)
    a = actor.act(obs, np.array([True, True]))
    b = actor.act(obs, np.array([True, True]))
    forced = actor.act(obs, np.array([True, False]))
    new, old, entropy, active = joint_statistics(actor, [[a, b, forced], []])
    assert old.tolist() == pytest.approx([a.log_prob + b.log_prob, 0.])
    assert new.detach().numpy().tolist() == pytest.approx(old.tolist(), abs=1e-6)
    assert active.tolist() == [True, False]
    assert entropy[1] == 0.
