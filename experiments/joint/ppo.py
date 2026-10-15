"""Independent SB3 actor/critics with one PPO transition per physical slot.

The task policy is autoregressive inside a 15-minute slot: resources and the
candidate observation change after each decision.  Its *macro action* is that
whole sequence.  Consequently ``log pi(sequence) = sum(log pi(action|state))``
and PPO clips ONE joint likelihood ratio against ONE physical-slot advantage.
The shared reward is recorded once, including slots with no task decisions.
Gamma and GAE lambda therefore advance once per physical slot, never per task.
The task critic observes the boundary state (queue aggregates, no candidate).
The cooling critic observes its actual post-scheduling decision boundary.

An empty task sequence, or one containing only single-legal-action decisions,
has probability one and no actor/entropy term; its boundary value and reward
remain in GAE and value training.  All losses average over physical slots,
avoiding an accidental weighting by candidate count.
Masks and conditional observations are saved and reused exactly during PPO.

Products of many conditional probabilities can give high-variance joint
ratios.  The target-KL guard stops BEFORE optimizing an unsafe minibatch.
Log ratios are accumulated in float64; ONLY exponentiation is bounded to
[-60,60] for numerical safety, independently of PPO's ordinary ratio clipping.
Any raw log ratio outside that numerical range rejects the minibatch before
optimization, even when the optional target-KL guard is disabled. KL logging
uses the same bounded log ratio as exponentiation so it cannot turn negative
through a mismatch of capped and uncapped terms.
This is a research baseline, not a claim that long action sequences are easy
to optimize.  The actor/critic network and Adam optimizer are reused from SB3;
only the physical-clock rollout objective differs from ordinary SB3 PPO.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    n_steps: int = 128
    batch_size: int = 64
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    target_kl: float | None = 0.03
    net_arch: tuple[int, ...] = (64, 64)

    def __post_init__(self):
        for name in ('n_steps', 'batch_size', 'n_epochs'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('learning_rate', 'gamma', 'gae_lambda', 'clip_range',
                     'ent_coef', 'vf_coef', 'max_grad_norm'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{name} must be finite numeric')
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError('learning_rate and max_grad_norm must be positive')
        if not (0 <= self.gamma <= 1 and 0 <= self.gae_lambda <= 1):
            raise ValueError('gamma and gae_lambda must lie in [0,1]')
        if not 0 < self.clip_range < 1 or self.ent_coef < 0 or self.vf_coef < 0:
            raise ValueError('clip_range must lie in (0,1); loss weights must be nonnegative')
        if not isinstance(self.normalize_advantage, bool):
            raise ValueError('normalize_advantage must be bool')
        if self.target_kl is not None and (
            isinstance(self.target_kl, bool) or not isinstance(self.target_kl, (int, float))
            or not math.isfinite(self.target_kl) or self.target_kl <= 0
        ):
            raise ValueError('target_kl must be None or finite positive numeric')
        if not isinstance(self.net_arch, (tuple, list)) or not self.net_arch or any(
            isinstance(width, bool) or not isinstance(width, int) or width < 1
            for width in self.net_arch
        ):
            raise ValueError('net_arch must contain positive integer widths')
        object.__setattr__(self, 'net_arch', tuple(self.net_arch))


@dataclass(frozen=True)
class Decision:
    observation: np.ndarray
    action_mask: np.ndarray
    action: int
    log_prob: float


@dataclass(frozen=True)
class SlotTransition:
    boundary_observation: np.ndarray
    decisions: Sequence[Decision]
    reward: float
    done: bool
    value: float | None = None


def compute_gae(rewards, values, dones, *, bootstrap_value=0.0, gamma=0.99, gae_lambda=0.95):
    """Return (advantages, targets), one entry per 15-minute physical slot.

    True terminals include any endpoint liability in their reward and ignore
    bootstrap.  A collection cutoff with an unfinished episode requires the
    actual next boundary value; it is not an environment terminal.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones)
    if rewards.ndim != 1 or values.shape != rewards.shape or dones.shape != rewards.shape:
        raise ValueError('rewards, values and dones must have the same one-dimensional shape')
    if dones.dtype.kind != 'b':
        raise ValueError('dones must contain booleans')
    if not (np.isfinite(rewards).all() and np.isfinite(values).all()
            and math.isfinite(bootstrap_value)):
        raise ValueError('GAE inputs must be finite')
    if not (math.isfinite(gamma) and math.isfinite(gae_lambda)
            and 0 <= gamma <= 1 and 0 <= gae_lambda <= 1):
        raise ValueError('gamma and gae_lambda must lie in [0,1]')
    advantages = np.empty_like(rewards)
    next_value, next_advantage = float(bootstrap_value), 0.0
    for index in range(len(rewards) - 1, -1, -1):
        continuing = 0.0 if dones[index] else 1.0
        delta = rewards[index] + gamma * continuing * next_value - values[index]
        next_advantage = delta + gamma * gae_lambda * continuing * next_advantage
        advantages[index] = next_advantage
        next_value = values[index]
    return advantages, advantages + values


class IPPOAgent:
    """A discrete independent policy; instantiate separately for task/cooling.

    Sampling and minibatch RNGs belong to this instance, so calling one agent
    cannot consume the other agent's random stream.  Checkpoints save both RNGs,
    Adam moments, configuration, policy parameters, and training counters.
    CPU/single-thread execution matches the existing reproducibility baseline.
    """
    checkpoint_version = 1

    def __init__(self, obs_dim, action_dim, config=None, seed=0):
        import torch
        from stable_baselines3.common.policies import ActorCriticPolicy

        for name, value, minimum in (('obs_dim', obs_dim, 1), ('action_dim', action_dim, 2)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError('seed must be a nonnegative integer below 2**63')
        self.config = config if isinstance(config, PPOConfig) else PPOConfig(**(config or {}))
        self.obs_dim, self.action_dim, self.seed = obs_dim, action_dim, seed
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        # Isolate parameter initialization from other agents and external RNGs.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.policy = ActorCriticPolicy(
                gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32),
                gym.spaces.Discrete(action_dim),
                lr_schedule=lambda _: self.config.learning_rate,
                net_arch=list(self.config.net_arch),
                activation_fn=torch.nn.Tanh,
            ).to('cpu')
        self._sampling_rng = torch.Generator(device='cpu').manual_seed(seed)
        self._minibatch_rng = np.random.default_rng(seed)
        self.num_slots = 0
        self.num_decisions = 0
        self.optimizer_steps = 0
        self.rollout_updates = 0

    def _observation(self, observation):
        result = np.asarray(observation, dtype=np.float32)
        if result.shape != (self.obs_dim,) or not np.isfinite(result).all():
            raise ValueError(f'observation must be finite with shape ({self.obs_dim},)')
        return result.copy()

    def _mask(self, mask):
        result = np.asarray(mask)
        if result.dtype.kind != 'b' or result.shape != (self.action_dim,) or not result.any():
            raise ValueError('action mask must be boolean, match action dimension, and contain a legal action')
        return result.copy()

    def _distribution(self, observations, masks):
        import torch

        distribution = self.policy.get_distribution(observations).distribution
        return torch.distributions.Categorical(logits=distribution.logits.masked_fill(~masks, -torch.inf))

    def act(self, observation, mask, deterministic=False):
        import torch

        observation, mask = self._observation(observation), self._mask(mask)
        self.policy.set_training_mode(False)
        with torch.no_grad():
            distribution = self._distribution(
                torch.from_numpy(observation[None]), torch.from_numpy(mask[None]),
            )
            if deterministic:
                action = distribution.probs.argmax(dim=-1)
            else:
                action = torch.multinomial(distribution.probs, 1, generator=self._sampling_rng).squeeze(-1)
            log_prob = float(distribution.log_prob(action).item())
        return Decision(observation, mask, int(action.item()), log_prob)

    def value(self, boundary_observation):
        return float(self.values(np.asarray([self._observation(boundary_observation)]))[0])

    def values(self, boundary_observations):
        import torch

        observations = np.asarray(boundary_observations, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[1] != self.obs_dim or not np.isfinite(observations).all():
            raise ValueError('boundary observations must be a finite [slots,obs_dim] array')
        self.policy.set_training_mode(False)
        with torch.no_grad():
            return self.policy.predict_values(torch.from_numpy(observations)).flatten().numpy().copy()

    def update(self, trajectory: Sequence[SlotTransition], bootstrap_value=None):
        """Optimize physical-slot PPO; ``n_steps`` is a collector budget hint.

        Whole episodes may exceed ``n_steps`` and the last minibatch may be
        smaller than ``batch_size``.  This avoids inventing a premature cooling
        boundary observation when its next task scheduling has not occurred.
        A nonterminal rollout cut must pass an explicit, correct next boundary
        value.  Terminal trajectories use zero bootstrap by definition.
        """
        import torch

        slots = list(trajectory)
        if not slots:
            raise ValueError('PPO trajectory must contain at least one physical slot')
        boundary = np.stack([self._observation(slot.boundary_observation) for slot in slots])
        for slot in slots:
            if not isinstance(slot.done, (bool, np.bool_)) or not math.isfinite(slot.reward):
                raise ValueError('each slot requires a finite reward and boolean done')
            if slot.value is not None and not math.isfinite(slot.value):
                raise ValueError('stored boundary value must be finite')
            for decision in slot.decisions:
                self._observation(decision.observation)
                mask = self._mask(decision.action_mask)
                if (isinstance(decision.action, bool)
                        or not isinstance(decision.action, (int, np.integer))
                        or not 0 <= decision.action < self.action_dim or not mask[decision.action]):
                    raise ValueError('saved decision action must be legal under its saved mask')
                if not math.isfinite(decision.log_prob) or decision.log_prob > 1e-5:
                    raise ValueError('saved conditional log probability must be finite and <= 0')
                if mask.sum() == 1 and abs(decision.log_prob) > 1e-5:
                    raise ValueError('a forced decision must have log probability zero under its saved mask')
        if bootstrap_value is None:
            if not slots[-1].done:
                raise ValueError('nonterminal rollout cut requires explicit next-boundary bootstrap_value')
            bootstrap_value = 0.0
        if not math.isfinite(bootstrap_value):
            raise ValueError('bootstrap_value must be finite')
        old_values = self.values(boundary).astype(np.float64)
        for index, slot in enumerate(slots):
            if slot.value is not None:
                old_values[index] = slot.value
        rewards = np.asarray([slot.reward for slot in slots], dtype=np.float64)
        advantages, returns = compute_gae(
            rewards, old_values, [bool(slot.done) for slot in slots],
            bootstrap_value=bootstrap_value, gamma=self.config.gamma, gae_lambda=self.config.gae_lambda,
        )
        if self.config.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        boundary_tensor = torch.from_numpy(boundary)
        advantage_tensor = torch.as_tensor(advantages, dtype=torch.float64)
        return_tensor = torch.as_tensor(returns, dtype=torch.float32)
        history = []
        optimizer_steps, epochs_started, stopped_early = 0, 0, False
        numerical_guard_triggered = False
        self.policy.set_training_mode(True)
        for _ in range(self.config.n_epochs):
            epochs_started += 1
            permutation = self._minibatch_rng.permutation(len(slots))
            for start in range(0, len(slots), self.config.batch_size):
                indices = permutation[start:start + self.config.batch_size]
                batch = [slots[int(index)] for index in indices]
                count = len(batch)
                active = torch.as_tensor([
                    any(np.count_nonzero(decision.action_mask) > 1 for decision in slot.decisions)
                    for slot in batch
                ], dtype=torch.bool)
                new_log_probs = torch.zeros(count, dtype=torch.float64)
                old_log_probs = torch.zeros(count, dtype=torch.float64)
                entropies = torch.zeros(count, dtype=torch.float64)
                # Forced factors are exactly one. Excluding their graph also
                # leaves actor .grad=None in all-forced minibatches, preventing
                # old Adam momentum from moving an actor with no choice here.
                flattened = [(position, decision) for position, slot in enumerate(batch)
                             for decision in slot.decisions if np.count_nonzero(decision.action_mask) > 1]
                if flattened:
                    positions = torch.as_tensor([position for position, _ in flattened], dtype=torch.long)
                    observations = torch.as_tensor(np.stack([d.observation for _, d in flattened]), dtype=torch.float32)
                    masks = torch.as_tensor(np.stack([d.action_mask for _, d in flattened]), dtype=torch.bool)
                    actions = torch.as_tensor([d.action for _, d in flattened], dtype=torch.long)
                    distribution = self._distribution(observations, masks)
                    new_log_probs = new_log_probs.scatter_add(0, positions, distribution.log_prob(actions).double())
                    old_log_probs = old_log_probs.scatter_add(
                        0, positions, torch.as_tensor([d.log_prob for _, d in flattened], dtype=torch.float64),
                    )
                    entropies = entropies.scatter_add(0, positions, distribution.entropy().double())
                log_ratio = new_log_probs - old_log_probs
                bounded_log_ratio = log_ratio.clamp(-60.0, 60.0)
                ratio = torch.exp(bounded_log_ratio)
                selected_advantages = advantage_tensor[indices]
                surrogate = torch.minimum(
                    ratio * selected_advantages,
                    ratio.clamp(1 - self.config.clip_range, 1 + self.config.clip_range) * selected_advantages,
                )
                policy_loss = -(surrogate * active).mean()
                predictions = self.policy.predict_values(boundary_tensor[indices]).flatten()
                value_loss = torch.nn.functional.mse_loss(predictions, return_tensor[indices])
                entropy = entropies.mean()
                with torch.no_grad():
                    approximate_kl = float(((ratio - 1) - bounded_log_ratio)[active].mean()) if active.any() else 0.0
                    clip_fraction = float(((ratio - 1).abs() > self.config.clip_range)[active].double().mean()) if active.any() else 0.0
                    oversized_log_ratio = bool((log_ratio.abs() > 60.0).any())
                metric = dict(policy_loss=float(policy_loss.detach()), value_loss=float(value_loss.detach()),
                              entropy=float(entropy.detach()), approx_kl=approximate_kl,
                              clip_fraction=clip_fraction, batch_slots=count)
                if not all(math.isfinite(value) for value in metric.values()):
                    raise RuntimeError('nonfinite physical-slot PPO update metric')
                history.append(metric)
                if oversized_log_ratio or (self.config.target_kl is not None
                                           and approximate_kl > 1.5 * self.config.target_kl):
                    numerical_guard_triggered = oversized_log_ratio
                    stopped_early = True
                    break
                loss = policy_loss + self.config.vf_coef * value_loss - self.config.ent_coef * entropy
                self.policy.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm,
                                               error_if_nonfinite=True)
                self.policy.optimizer.step()
                self.optimizer_steps += 1
                optimizer_steps += 1
            if stopped_early:
                break
        self.rollout_updates += 1
        self.num_slots += len(slots)
        decisions = sum(len(slot.decisions) for slot in slots)
        self.num_decisions += decisions
        total_weight = sum(row['batch_slots'] for row in history)
        metrics = {name: math.fsum(row[name] * row['batch_slots'] for row in history) / total_weight
                   for name in ('policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_fraction')}
        metrics.update(slots=len(slots), decisions=decisions,
                       actor_slots=sum(any(np.count_nonzero(d.action_mask) > 1 for d in slot.decisions)
                                       for slot in slots),
                       mean_reward=float(rewards.mean()), mean_return=float(returns.mean()),
                       optimizer_steps=optimizer_steps, updates=self.rollout_updates,
                       total_optimizer_steps=self.optimizer_steps, epochs=epochs_started,
                       stopped_early=stopped_early,
                       numerical_guard_triggered=numerical_guard_triggered)
        return metrics

    def parameter_hash(self):
        digest = hashlib.sha256()
        for name, tensor in sorted(self.policy.state_dict().items()):
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(json.dumps(dict(name=name, dtype=str(array.dtype), shape=list(array.shape)),
                                     sort_keys=True, separators=(',', ':')).encode('utf-8'))
            digest.update(array.tobytes(order='C'))
        return digest.hexdigest()

    def save(self, path):
        import torch

        torch.save(dict(
            version=self.checkpoint_version, obs_dim=self.obs_dim, action_dim=self.action_dim,
            config=asdict(self.config), seed=self.seed, policy=self.policy.state_dict(),
            optimizer=self.policy.optimizer.state_dict(), sampling_rng=self._sampling_rng.get_state(),
            minibatch_rng=self._minibatch_rng.bit_generator.state,
            counters=dict(num_slots=self.num_slots, num_decisions=self.num_decisions,
                          optimizer_steps=self.optimizer_steps, rollout_updates=self.rollout_updates),
        ), Path(path))

    @classmethod
    def load(cls, path):
        import torch

        checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
        if checkpoint.get('version') != cls.checkpoint_version:
            raise ValueError('unsupported joint PPO checkpoint version')
        agent = cls(checkpoint['obs_dim'], checkpoint['action_dim'], checkpoint['config'], checkpoint['seed'])
        agent.policy.load_state_dict(checkpoint['policy'])
        agent.policy.optimizer.load_state_dict(checkpoint['optimizer'])
        agent._sampling_rng.set_state(checkpoint['sampling_rng'])
        agent._minibatch_rng.bit_generator.state = checkpoint['minibatch_rng']
        for name in ('num_slots', 'num_decisions', 'optimizer_steps', 'rollout_updates'):
            value = checkpoint['counters'][name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError('invalid saved PPO counter')
            setattr(agent, name, value)
        return agent
