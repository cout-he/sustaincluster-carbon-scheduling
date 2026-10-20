"""Artificial numeric PPO tests; never reads workload or evaluation data."""
import importlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch


def ppo_module():
    # The first RED run fails as an assertion until the new module exists.
    assert importlib.util.find_spec('experiments.sca_ppo.ppo') is not None, 'PPO core is not implemented'
    return importlib.import_module('experiments.sca_ppo.ppo')


def artificial_batch(trainer, size=12, sampled=None):
    states = torch.linspace(-1.0, 1.0, size * 27).reshape(size, 27)
    with torch.no_grad():
        actions, log_probs, values = trainer.model.act(states)
    terminated = torch.zeros(size, dtype=torch.bool)
    terminated[-1] = True
    return dict(states=states, actions=actions, log_probs=log_probs, values=values,
                rewards=torch.linspace(-.4, .7, size), terminated=terminated,
                policy_sampled=torch.ones(size, dtype=torch.bool) if sampled is None else sampled)


def model_parameters(model):
    return {name: value.detach().clone() for name, value in model.named_parameters()}


class ActorCriticTests(unittest.TestCase):
    def test_actor_values_and_saved_action_log_probs_agree(self):
        module = ppo_module()
        module.seed_everything(17)
        model = module.ActorCritic()
        states = torch.arange(81, dtype=torch.float32).reshape(3, 27) / 81
        logits, values = model(states)
        self.assertEqual(tuple(logits.shape), (3, 7))
        self.assertEqual(tuple(values.shape), (3,))
        actions, old_log_probs, sampled_values = model.act(states)
        log_probs, entropy, evaluated_values = model.evaluate_actions(states, actions)
        torch.testing.assert_close(log_probs, old_log_probs)
        torch.testing.assert_close(values, sampled_values)
        torch.testing.assert_close(values, evaluated_values)
        self.assertTrue(torch.all(entropy > 0))
        deterministic, _, _ = model.act(states, deterministic=True)
        torch.testing.assert_close(deterministic, logits.argmax(dim=-1))
        (log_probs.mean() + evaluated_values.mean()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_seed_reproduces_initialization_and_sampling(self):
        module = ppo_module()
        module.seed_everything(31)
        first = module.ActorCritic()
        output = first.act(torch.ones(16, 27))
        expected = (random.random(), np.random.random(), torch.rand(2))
        module.seed_everything(31)
        second = module.ActorCritic()
        actual = second.act(torch.ones(16, 27))
        for left, right in zip(output, actual):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual(expected[0], random.random())
        self.assertEqual(expected[1], np.random.random())
        torch.testing.assert_close(expected[2], torch.rand(2), rtol=0, atol=0)


class GAETests(unittest.TestCase):
    def test_gamma_one_preserves_every_physical_slot_with_separate_lambda(self):
        advantages, returns = ppo_module().compute_gae(
            [1., 2., 3.], [.5, 1., 1.5], [False, False, True],
            last_value=900., gamma=1., lambda_gae=.5)
        # deltas=[1.5,2.5,1.5]; no compression of the middle forced slot.
        torch.testing.assert_close(advantages, torch.tensor([3.125, 3.25, 1.5]))
        torch.testing.assert_close(returns, torch.tensor([3.625, 4.25, 3.]))

    def test_true_termination_cuts_advantage_and_rollout_cut_bootstraps(self):
        module = ppo_module()
        terminal, _ = module.compute_gae([1., 2.], [.5, 1.], [False, True], 7., 1., 1.)
        truncated, returns = module.compute_gae([1., 2.], [.5, 1.], [False, False], 7., 1., 1.)
        torch.testing.assert_close(terminal, torch.tensor([2.5, 1.]))
        torch.testing.assert_close(truncated, torch.tensor([9.5, 8.]))
        torch.testing.assert_close(returns, torch.tensor([10., 9.]))
        cut, _ = module.compute_gae([1., 2., 3.], [0., 0., 0.], [False, True, True], 0., 1., 1.)
        torch.testing.assert_close(cut, torch.tensor([3., 2., 3.]))

    def test_gamma_and_lambda_are_independent(self):
        advantages, _ = ppo_module().compute_gae([1., 2.], [0., 0.], [False, True], 0., .5, 1.)
        torch.testing.assert_close(advantages, torch.tensor([2., 2.]))
        advantages, _ = ppo_module().compute_gae([1., 2.], [0., 0.], [False, True], 0., 1., .25)
        torch.testing.assert_close(advantages, torch.tensor([1.5, 2.]))


class PPOUpdateTests(unittest.TestCase):
    def test_backward_updates_actual_parameters_and_reports_finite_metrics(self):
        trainer = ppo_module().PPOTrainer(dict(seed=19, update_epochs=2, minibatch_size=5))
        batch = artificial_batch(trainer)
        before = model_parameters(trainer.model)
        metrics = trainer.update(batch)
        json.dumps(metrics, allow_nan=False)
        self.assertEqual(metrics['optimizer_steps'], 6)
        self.assertEqual(metrics['critic_samples'], 24)
        self.assertEqual(metrics['policy_samples'], 24)
        self.assertGreater(metrics['parameter_delta_l2'], 0)
        self.assertGreater(metrics['actor_parameter_delta_l2'], 0)
        self.assertGreater(metrics['grad_norm'], 0)
        self.assertTrue(any(not torch.equal(value, before[name]) for name, value in trainer.model.named_parameters()))
        self.assertTrue(trainer.optimizer.state)
        for key in ('policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_fraction'):
            self.assertTrue(np.isfinite(metrics[key]), key)

    def test_clipped_surrogate_uses_conservative_bound_for_both_signs(self):
        trainer = ppo_module().PPOTrainer(dict(seed=3, update_epochs=1, minibatch_size=2,
                                             value_coef=0., entropy_coef=0.))
        with torch.no_grad():
            for parameter in trainer.model.parameters():
                parameter.zero_()
        batch = dict(states=torch.zeros(2, 27), actions=torch.tensor([0, 1]),
                     log_probs=torch.tensor([-np.log(7.) - np.log(2.), -np.log(7.) - np.log(.5)]),
                     values=torch.zeros(2), rewards=torch.tensor([1., -1.]),
                     terminated=torch.ones(2, dtype=torch.bool), policy_sampled=torch.ones(2, dtype=torch.bool))
        metrics = trainer.update(batch)
        # Positive advantage ratio 2 clips to 1.2; negative ratio .5 clips to .8.
        self.assertAlmostEqual(metrics['policy_loss'], -.2, places=6)
        self.assertAlmostEqual(metrics['clip_fraction'], 1., places=6)
        self.assertAlmostEqual(metrics['approx_kl'], .25, places=6)
        self.assertAlmostEqual(metrics['entropy'], np.log(7.), places=6)

    def test_forced_slots_train_critic_without_policy_head_gradients(self):
        trainer = ppo_module().PPOTrainer(dict(seed=3, update_epochs=2, minibatch_size=64))
        batch = artificial_batch(trainer, size=288, sampled=torch.zeros(288, dtype=torch.bool))
        # No proposal was sampled, so these fields are absent rather than fabricated.
        batch['actions'] = torch.full((288,), -1)
        batch['log_probs'] = [None] * 288
        before = model_parameters(trainer.model)
        metrics = trainer.update(batch)
        self.assertEqual(metrics['critic_samples'], 576)
        self.assertEqual(metrics['policy_samples'], 0)
        for key in ('policy_loss', 'entropy', 'approx_kl', 'clip_fraction', 'actor_parameter_delta_l2'):
            self.assertEqual(metrics[key], 0., key)
        self.assertGreater(metrics['parameter_delta_l2'], 0)
        self.assertTrue(all(torch.equal(value, before[name]) for name, value in trainer.model.named_parameters()
                            if name.startswith('policy_head.')))
        self.assertTrue(all(value.grad is None for value in trainer.model.policy_head.parameters()))

    def test_forced_rows_do_not_affect_actor_entropy_or_advantage_normalization(self):
        module = ppo_module()
        config = dict(seed=22, update_epochs=1, minibatch_size=4, value_coef=0.)
        first = module.PPOTrainer(config)
        batch = artificial_batch(first, size=4, sampled=torch.tensor([True, False, True, False]))
        batch['terminated'][:] = True
        batch['values'][:] = 0
        batch['rewards'] = torch.tensor([1., 200., 3., -300.])
        alternate = {key: value.clone() for key, value in batch.items()}
        alternate['states'][~batch['policy_sampled']] = 400.
        alternate['log_probs'][~batch['policy_sampled']] = -10000.
        alternate['actions'][~batch['policy_sampled']] = -1
        alternate['rewards'][~batch['policy_sampled']] *= 7
        second = module.PPOTrainer(config)
        left = first.update(batch)
        right = second.update(alternate)
        for key in ('policy_loss', 'entropy', 'approx_kl', 'clip_fraction'):
            self.assertAlmostEqual(left[key], right[key], places=7)
        for left_p, right_p in zip(first.model.parameters(), second.model.parameters()):
            torch.testing.assert_close(left_p, right_p, rtol=0, atol=0)

    def test_critic_uses_rewards_of_forced_slots_in_gae(self):
        module = ppo_module()
        first = module.PPOTrainer(dict(seed=7, update_epochs=1))
        batch = artificial_batch(first, size=3, sampled=torch.tensor([True, False, False]))
        batch['rewards'] = torch.tensor([0., 2., 3.])
        batch['values'] = torch.zeros(3)
        metrics = first.update(batch)
        # gamma=1, lambda=.95; returns [4.6075,4.85,3.0], mean=4.1525.
        self.assertAlmostEqual(metrics['return_mean'], 4.1525, places=5)

    def test_nonterminal_update_requires_explicit_bootstrap(self):
        trainer = ppo_module().PPOTrainer(dict(update_epochs=1))
        batch = artificial_batch(trainer, size=3)
        batch['terminated'][-1] = False
        with self.assertRaisesRegex(ValueError, 'last_value|bootstrap'):
            trainer.update(batch)
        batch['last_value'] = 1.25
        self.assertGreater(trainer.update(batch)['optimizer_steps'], 0)

    def test_nonfinite_rollout_rejected_before_any_parameter_change(self):
        module = ppo_module()
        for field in ('states', 'values', 'rewards', 'log_probs'):
            trainer = module.PPOTrainer(dict(seed=9, update_epochs=1))
            batch = artificial_batch(trainer)
            batch[field].reshape(-1)[0] = float('nan')
            before = model_parameters(trainer.model)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'finite|NaN'):
                trainer.update(batch)
            self.assertTrue(all(torch.equal(p, before[n]) for n, p in trainer.model.named_parameters()))
        with self.assertRaisesRegex(ValueError, 'finite|NaN'):
            module.compute_gae([float('nan')], [0.], [True])

    def test_bad_shapes_and_sampled_actions_fail_before_update(self):
        module = ppo_module()
        for field, replacement in (('rewards', torch.zeros(2)), ('policy_sampled', torch.ones(12)),
                                   ('actions', torch.full((12,), 7))):
            trainer = module.PPOTrainer(dict(update_epochs=1))
            batch = artificial_batch(trainer)
            batch[field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                trainer.update(batch)


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_restores_forward_optimizer_counters_and_all_rngs(self):
        module = ppo_module()
        trainer = module.PPOTrainer(dict(seed=83, update_epochs=2, minibatch_size=5))
        trainer.update(artificial_batch(trainer))
        root = Path('research_notes/sca-ppo/unit-tmp')
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path = Path(directory) / 'ppo.pt'
            trainer.save(path, metadata={'artificial': True, 'schema': 'unit-test'})
            states = torch.linspace(-1., 1., 27 * 12).reshape(12, 27)
            expected_forward = trainer.model(states)
            expected_rng = (random.random(), np.random.random(), torch.rand(3))
            next_batch = artificial_batch(trainer)
            expected_metrics = trainer.update(next_batch)
            expected_model = model_parameters(trainer.model)
            restored = module.PPOTrainer.load(path)
            self.assertEqual(restored.metadata, {'artificial': True, 'schema': 'unit-test'})
            for left, right in zip(expected_forward, restored.model(states)):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            self.assertEqual(expected_rng[0], random.random())
            self.assertEqual(expected_rng[1], np.random.random())
            torch.testing.assert_close(expected_rng[2], torch.rand(3), rtol=0, atol=0)
            actual_batch = artificial_batch(restored)
            actual_metrics = restored.update(actual_batch)
            self.assertEqual(expected_metrics, actual_metrics)
            self.assertEqual(trainer.update_count, restored.update_count)
            self.assertEqual(trainer.optimizer_steps, restored.optimizer_steps)
            for name, value in restored.model.named_parameters():
                torch.testing.assert_close(expected_model[name], value, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
