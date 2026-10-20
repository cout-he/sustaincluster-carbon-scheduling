"""Numeric-only PyTorch PPO for the frozen SCA-PPO v0.2 observation/action API.

The critic and GAE retain every physical slot. Only actually sampled proposals
enter the clipped policy objective and entropy term. Rewards/returns are never
normalized here; only the sampled policy advantages are standardized.
"""
from copy import deepcopy
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


DEFAULT_CONFIG = dict(
    input_dim=27, hidden_sizes=[64, 64], action_dim=7,
    learning_rate=3e-4, adam_eps=1e-5, clip_range=.2, update_epochs=4,
    minibatch_size=64, entropy_coef=.01, value_coef=.5, max_grad_norm=.5,
    gamma=1., lambda_gae=.95, seed=0,
)


def seed_everything(seed):
    """Seed collection/update RNGs; the caller controls CPU thread count."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _finite_tensor(value, name, *, dtype=torch.float32):
    try:
        result = torch.as_tensor(value, dtype=dtype, device='cpu').detach().clone()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f'{name} must be a numeric tensor') from exc
    if not torch.isfinite(result).all():
        raise ValueError(f'{name} must be finite (no NaN or infinity)')
    return result


def _boolean_vector(value, name, length):
    try:
        result = torch.as_tensor(value, device='cpu').detach().clone()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f'{name} must be a boolean vector') from exc
    if result.dtype != torch.bool or result.shape != (length,):
        raise ValueError(f'{name} must be a boolean vector of length {length}')
    return result


def compute_gae(rewards, values, terminated, last_value=0., gamma=1., lambda_gae=.95):
    """Compute GAE on full physical time, cutting only at true terminations.

    A rollout cut uses ``last_value``; a true terminal boundary always uses zero.
    ``gamma`` and ``lambda_gae`` have separate roles and are never conflated.
    """
    rewards = _finite_tensor(rewards, 'rewards')
    values = _finite_tensor(values, 'values')
    if rewards.ndim != 1 or rewards.numel() == 0 or values.shape != rewards.shape:
        raise ValueError('rewards and values must be equal-length nonempty vectors')
    terminated = _boolean_vector(terminated, 'terminated', len(rewards))
    bootstrap = _finite_tensor(last_value, 'last_value')
    if bootstrap.numel() != 1:
        raise ValueError('last_value must be scalar')
    for name, number in (('gamma', gamma), ('lambda_gae', lambda_gae)):
        if not math.isfinite(number) or not 0 <= number <= 1:
            raise ValueError(f'{name} must be finite and in [0, 1]')
    advantages = torch.empty_like(rewards)
    following_advantage = torch.zeros((), dtype=rewards.dtype)
    following_value = bootstrap.reshape(())
    for index in range(len(rewards) - 1, -1, -1):
        continuing = (~terminated[index]).to(rewards.dtype)
        delta = rewards[index] + gamma * continuing * following_value - values[index]
        following_advantage = delta + gamma * lambda_gae * continuing * following_advantage
        advantages[index] = following_advantage
        following_value = values[index]
    returns = advantages + values
    if not torch.isfinite(advantages).all() or not torch.isfinite(returns).all():
        raise ValueError('GAE advantages and returns must remain finite')
    return advantages, returns


class ActorCritic(nn.Module):
    """One shared two-layer Tanh backbone with categorical and scalar heads."""

    def __init__(self, input_dim=27, hidden_sizes=(64, 64), action_dim=7):
        super().__init__()
        dimensions = (input_dim, *hidden_sizes, action_dim)
        if len(hidden_sizes) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
                                         for v in dimensions):
            raise ValueError('positive input/action dimensions and two positive hidden sizes required')
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]), nn.Tanh(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.Tanh(),
        )
        # Standard PyTorch Linear initialization, reproducible under seed_everything.
        self.policy_head = nn.Linear(hidden_sizes[1], action_dim)
        self.value_head = nn.Linear(hidden_sizes[1], 1)

    def forward(self, states):
        states = torch.as_tensor(states, dtype=torch.float32, device='cpu')
        if states.ndim not in (1, 2) or states.shape[-1] != self.input_dim:
            raise ValueError(f'states must have final dimension {self.input_dim}')
        if not torch.isfinite(states).all():
            raise ValueError('states must be finite (no NaN or infinity)')
        hidden = self.backbone(states)
        logits, values = self.policy_head(hidden), self.value_head(hidden).squeeze(-1)
        if not torch.isfinite(logits).all() or not torch.isfinite(values).all():
            raise ValueError('network outputs must be finite')
        return logits, values

    @torch.no_grad()
    def act(self, states, deterministic=False):
        logits, values = self(states)
        distribution = Categorical(logits=logits)
        actions = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return actions, distribution.log_prob(actions), values

    def evaluate_actions(self, states, actions):
        logits, values = self(states)
        actions = torch.as_tensor(actions, device='cpu')
        if actions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError('actions must contain integer action IDs')
        if actions.shape != logits.shape[:-1] or (actions < 0).any() or (actions >= self.action_dim).any():
            raise ValueError('actions must be valid IDs with the states batch shape')
        distribution = Categorical(logits=logits)
        return distribution.log_prob(actions), distribution.entropy(), values


def _config(config):
    supplied = {} if config is None else dict(config)
    if set(supplied) - set(DEFAULT_CONFIG):
        raise ValueError(f'unknown PPO config keys: {sorted(set(supplied) - set(DEFAULT_CONFIG))}')
    result = deepcopy(DEFAULT_CONFIG)
    result.update(supplied)
    result['hidden_sizes'] = list(result['hidden_sizes'])
    for name in ('input_dim', 'action_dim', 'update_epochs', 'minibatch_size'):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    for name in ('learning_rate', 'adam_eps', 'max_grad_norm'):
        if not math.isfinite(result[name]) or result[name] <= 0:
            raise ValueError(f'{name} must be positive and finite')
    for name in ('clip_range', 'entropy_coef', 'value_coef'):
        if not math.isfinite(result[name]) or result[name] < 0:
            raise ValueError(f'{name} must be nonnegative and finite')
    for name in ('gamma', 'lambda_gae'):
        if not math.isfinite(result[name]) or not 0 <= result[name] <= 1:
            raise ValueError(f'{name} must be finite and in [0, 1]')
    return result


class PPOTrainer:
    """Own the real network, Adam optimizer, update counters, and local checkpoint."""

    checkpoint_schema = 'sca-ppo-v02-pytorch-checkpoint'
    checkpoint_version = 1

    def __init__(self, config=None):
        self.config = _config(config)
        seed_everything(self.config['seed'])
        self.model = ActorCritic(self.config['input_dim'], self.config['hidden_sizes'], self.config['action_dim'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'],
                                          eps=self.config['adam_eps'])
        self.update_count = 0
        self.optimizer_steps = 0
        self.metadata = {}

    def _prepare(self, batch):
        required = {'states', 'actions', 'log_probs', 'values', 'rewards', 'terminated', 'policy_sampled'}
        missing = required - batch.keys()
        if missing:
            raise ValueError(f'missing rollout fields: {sorted(missing)}')
        states = _finite_tensor(batch['states'], 'states')
        if states.ndim != 2 or states.shape[1] != self.config['input_dim'] or len(states) == 0:
            raise ValueError('states must be a nonempty [T,input_dim] tensor')
        size = len(states)
        sampled = _boolean_vector(batch['policy_sampled'], 'policy_sampled', size)
        terminated = _boolean_vector(batch['terminated'], 'terminated', size)
        values = _finite_tensor(batch['values'], 'values')
        rewards = _finite_tensor(batch['rewards'], 'rewards')
        if values.shape != (size,) or rewards.shape != (size,):
            raise ValueError('values and rewards must have shape [T]')
        try:
            actions = torch.as_tensor(batch['actions'], device='cpu').detach().clone()
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError('actions must be an integer vector') from exc
        if actions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8) or actions.shape != (size,):
            raise ValueError('actions must be an integer vector of shape [T]')
        if (actions[sampled] < 0).any() or (actions[sampled] >= self.config['action_dim']).any():
            raise ValueError('sampled actions must be valid action IDs')
        raw_logs = batch['log_probs']
        if not isinstance(raw_logs, torch.Tensor):
            try:
                if len(raw_logs) != size:
                    raise ValueError('log_probs must have shape [T]')
                # None denotes no sampling event. The placeholder is never indexed
                # by any policy calculation, and is not a forced-action logprob.
                raw_logs = [0. if value is None and not sampled[index] else value
                            for index, value in enumerate(raw_logs)]
            except TypeError as exc:
                raise ValueError('log_probs must have shape [T]') from exc
        log_probs = _finite_tensor(raw_logs, 'log_probs')
        if log_probs.shape != (size,) or (log_probs[sampled] > 1e-5).any():
            raise ValueError('log_probs must have shape [T] and sampled log_probs must be <= 0')
        if not terminated[-1] and 'last_value' not in batch:
            raise ValueError('a nonterminal rollout requires explicit last_value bootstrap')
        last_value = batch.get('last_value', 0.)
        advantages, returns = compute_gae(rewards, values, terminated, last_value,
                                          self.config['gamma'], self.config['lambda_gae'])
        policy_advantages = advantages.clone()
        if int(sampled.sum()) > 1:
            chosen = advantages[sampled]
            policy_advantages[sampled] = (chosen - chosen.mean()) / (chosen.std(unbiased=False) + 1e-8)
        return states, actions.long(), log_probs, sampled, policy_advantages, returns

    def update(self, batch):
        states, actions, old_log_probs, sampled, advantages, returns = self._prepare(batch)
        self.model.train()
        before = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}
        history = []
        policy_count = critic_count = steps = 0
        for _ in range(self.config['update_epochs']):
            order = torch.randperm(len(states))
            for start in range(0, len(states), self.config['minibatch_size']):
                indices = order[start:start + self.config['minibatch_size']]
                chosen_indices = indices[sampled[indices]]
                # Evaluate the critic at ALL physical slots. An absent sampling
                # event never enters Categorical.log_prob or the entropy mean.
                _, predicted_values = self.model(states[indices])
                value_loss = .5 * (predicted_values - returns[indices]).square().mean()
                zero = torch.zeros((), dtype=torch.float32)
                policy_loss, entropy, approx_kl, clip_fraction = zero, zero, zero, zero
                if len(chosen_indices):
                    logs, entropies, _ = self.model.evaluate_actions(states[chosen_indices], actions[chosen_indices])
                    log_ratio = logs - old_log_probs[chosen_indices]
                    ratio = log_ratio.exp()
                    unclipped = ratio * advantages[chosen_indices]
                    clipped = ratio.clamp(1 - self.config['clip_range'], 1 + self.config['clip_range']) * advantages[chosen_indices]
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    entropy = entropies.mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fraction = ((ratio - 1).abs() > self.config['clip_range']).float().mean()
                loss = policy_loss + self.config['value_coef'] * value_loss - self.config['entropy_coef'] * entropy
                if not all(torch.isfinite(item).all() for item in (value_loss, policy_loss, entropy, approx_kl, clip_fraction, loss)):
                    raise ValueError('PPO loss and diagnostics must remain finite')
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'],
                                                           error_if_nonfinite=True)
                self.optimizer.step()
                if not all(torch.isfinite(parameter).all() for parameter in self.model.parameters()):
                    raise ValueError('PPO parameters must remain finite after optimizer step')
                count = len(chosen_indices)
                history.append((len(indices), count, float(policy_loss.detach()), float(value_loss.detach()),
                                float(entropy.detach()), float(approx_kl.detach()), float(clip_fraction.detach()),
                                float(grad_norm)))
                policy_count += count
                critic_count += len(indices)
                steps += 1
        self.update_count += 1
        self.optimizer_steps += steps
        deltas = {name: float((parameter.detach() - before[name]).square().sum())
                  for name, parameter in self.model.named_parameters()}
        def delta(prefix):
            return math.sqrt(sum(value for name, value in deltas.items() if name.startswith(prefix)))
        def average(column, actor=False):
            denominator = policy_count if actor else critic_count
            return sum(row[column] * row[1 if actor else 0] for row in history) / denominator if denominator else 0.
        # actor_parameter_delta_l2 means the exclusive policy head: the shared
        # backbone can change under critic-only training and is reported separately.
        metrics = dict(policy_loss=average(2, True), value_loss=average(3), entropy=average(4, True),
                       approx_kl=average(5, True), clip_fraction=average(6, True),
                       grad_norm=max(row[7] for row in history), optimizer_steps=steps,
                       policy_samples=policy_count, critic_samples=critic_count,
                       sampled_slots=int(sampled.sum()), physical_slots=len(states),
                       actor_parameter_delta_l2=delta('policy_head.'),
                       backbone_parameter_delta_l2=delta('backbone.'),
                       value_parameter_delta_l2=delta('value_head.'),
                       parameter_delta_l2=math.sqrt(sum(deltas.values())),
                       return_mean=float(returns.mean()), update_count=self.update_count,
                       total_optimizer_steps=self.optimizer_steps)
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError('PPO metrics must remain finite')
        return metrics

    def save(self, path, metadata=None):
        """Save a trusted local checkpoint, including all RNG and Adam state."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = deepcopy(self.metadata if metadata is None else metadata)
        checkpoint = dict(schema=self.checkpoint_schema, version=self.checkpoint_version,
                          config=deepcopy(self.config), model=self.model.state_dict(),
                          optimizer=self.optimizer.state_dict(), update_count=self.update_count,
                          optimizer_steps=self.optimizer_steps, metadata=deepcopy(self.metadata),
                          rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                                   torch=torch.get_rng_state(),
                                   cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None),
                          deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        temporary = destination.with_name(destination.name + '.tmp')
        try:
            with temporary.open('wb') as stream:
                torch.save(checkpoint, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path):
        """Restore only a checkpoint the caller created/trusts locally.

        Python/NumPy RNG tuples require pickle; never load an untrusted download.
        The global RNGs are restored AFTER model/optimizer construction.
        """
        checkpoint = torch.load(Path(path), map_location='cpu', weights_only=False)
        if checkpoint.get('schema') != cls.checkpoint_schema or checkpoint.get('version') != cls.checkpoint_version:
            raise ValueError('unsupported SCA-PPO checkpoint schema/version')
        trainer = cls(checkpoint['config'])
        trainer.model.load_state_dict(checkpoint['model'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        trainer.update_count = checkpoint['update_count']
        trainer.optimizer_steps = checkpoint['optimizer_steps']
        for counter in (trainer.update_count, trainer.optimizer_steps):
            if isinstance(counter, bool) or not isinstance(counter, int) or counter < 0:
                raise ValueError('invalid checkpoint update counters')
        if not all(torch.isfinite(parameter).all() for parameter in trainer.model.parameters()):
            raise ValueError('checkpoint parameters must be finite')
        trainer.metadata = deepcopy(checkpoint['metadata'])
        random.setstate(checkpoint['rng']['python'])
        np.random.set_state(checkpoint['rng']['numpy'])
        torch.set_rng_state(checkpoint['rng']['torch'])
        if checkpoint['rng']['cuda'] is not None:
            if not torch.cuda.is_available():
                raise ValueError('checkpoint CUDA RNG requires an available CUDA device')
            torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])
        torch.use_deterministic_algorithms(checkpoint['deterministic_algorithms'])
        return trainer
