"""V2 observability and resumable state over V1's exact macro-slot PPO.

ActorCriticPolicy, masks, GAE, seeds and action sampling are inherited from
experiments.joint.ppo. The update implementation below is an attributed copy
of V1 with diagnostics added: it still SUMS conditional log probabilities,
clips one sequence ratio, and uses one reward and advantage per physical slot.
V1 files and objective are unchanged. Forced actions have probability one.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import math
import os
from pathlib import Path
import time
from typing import Sequence

import numpy as np

from experiments.joint.ppo import (
    Decision, PPOConfig, SlotTransition, compute_gae, IPPOAgent as _V1Agent,
)


def _distribution_summary(values):
    array = np.asarray(values, dtype=np.float64)
    nonfinite_count = int(np.count_nonzero(~np.isfinite(array)))
    array = array[np.isfinite(array)]
    if not array.size:
        return dict(count=0, nonfinite_count=nonfinite_count, mean=0., std=0., min=0., max=0., q05=0., q50=0., q95=0.)
    scale = max(1., float(np.max(np.abs(array))))
    scaled = array / scale
    return dict(count=int(array.size), nonfinite_count=nonfinite_count,
                mean=float(scaled.mean() * scale), std=float(scaled.std() * scale),
                min=float(array.min()), max=float(array.max()),
                **{name: float(np.quantile(scaled, quantile) * scale)
                   for name, quantile in (('q05', .05), ('q50', .5), ('q95', .95))})


class IPPOAgent(_V1Agent):
    checkpoint_version = 2
    _counter_names = ('num_slots', 'num_decisions', 'optimizer_steps', 'rollout_updates',
                      'actor_optimizer_steps', 'critic_optimizer_steps', 'effective_actor_steps',
                      'effective_critic_steps', 'zero_actor_updates', 'bc_optimizer_steps')

    def __init__(self, obs_dim, action_dim, config=None, seed=0):
        super().__init__(obs_dim, action_dim, config, seed)
        self.actor_optimizer_steps = 0
        self.critic_optimizer_steps = 0
        self.effective_actor_steps = 0
        self.effective_critic_steps = 0
        self.zero_actor_updates = 0
        self.bc_optimizer_steps = 0
        self.provenance = {}

    def _component_parameters(self, component):
        names = ('policy_net', 'action_net') if component == 'actor' else ('value_net',)
        return [(name, parameter) for name, parameter in self.policy.named_parameters()
                if any(part in name for part in names)]

    def component_hashes(self):
        result = {}
        for component in ('actor', 'critic'):
            digest = hashlib.sha256()
            for name, parameter in self._component_parameters(component):
                digest.update(name.encode('utf8'))
                digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
            result[component] = digest.hexdigest()
        return result

    def export_state(self):
        # Deep copy makes this a point-in-time snapshot even if a caller keeps
        # the dictionary in memory while training the next episode.
        return deepcopy(dict(
            schema='jointrl-v2-ippo-state', version=self.checkpoint_version,
            agent_class=type(self).__module__ + '.' + type(self).__name__,
            objective='v1_macro_slot_joint_ratio', obs_dim=self.obs_dim, action_dim=self.action_dim,
            config=asdict(self.config), seed=self.seed, policy=self.policy.state_dict(),
            optimizer=self.policy.optimizer.state_dict(), sampling_rng=self._sampling_rng.get_state(),
            minibatch_rng=self._minibatch_rng.bit_generator.state, provenance=self.provenance,
            counters={name: getattr(self, name) for name in self._counter_names},
        ))

    @classmethod
    def from_state(cls, state):
        if state.get('schema') != 'jointrl-v2-ippo-state' or state.get('version') != cls.checkpoint_version:
            raise ValueError('Unsupported V2 IPPO state')
        if state.get('objective') != 'v1_macro_slot_joint_ratio':
            raise ValueError('Saved PPO objective differs from the declared macro-slot joint ratio')
        if set(state['counters']) != set(cls._counter_names):
            raise ValueError('Incomplete or unexpected V2 IPPO state counters')
        agent = cls(state['obs_dim'], state['action_dim'], state['config'], state['seed'])
        agent.policy.load_state_dict(state['policy'])
        # Optimizer loading may retain CPU moment tensors from its input.
        # Keep reusable transaction/checkpoint snapshots immutable on resume.
        agent.policy.optimizer.load_state_dict(deepcopy(state['optimizer']))
        agent._sampling_rng.set_state(state['sampling_rng'])
        agent._minibatch_rng.bit_generator.state = deepcopy(state['minibatch_rng'])
        agent.provenance = deepcopy(state.get('provenance', {}))
        for name, value in state['counters'].items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError('Invalid V2 IPPO state counter')
            setattr(agent, name, value)
        return agent

    def save(self, path):
        import torch

        path = Path(path)
        temporary = path.with_name(path.name + '.tmp')
        with temporary.open('wb') as stream:
            torch.save(self.export_state(), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    @classmethod
    def load(cls, path):
        import torch

        return cls.from_state(torch.load(Path(path), map_location='cpu', weights_only=True))

    def behavior_clone(self, samples, *, epochs=1, max_updates=4, deadline=None):
        """Bounded masked cross-entropy initialization; no critic gradients."""
        import torch

        if epochs < 0 or max_updates < 0:
            raise ValueError('BC epochs and max_updates must be nonnegative')
        prepared = []
        for sample in samples:
            observation, mask = self._observation(sample['observation']), self._mask(sample['mask'])
            action = int(sample['action'])
            if not 0 <= action < self.action_dim or not mask[action]:
                raise ValueError('BC label violates its saved action mask')
            if mask.sum() > 1:
                prepared.append((observation, mask, action))
        losses, gradients, steps = [], [], 0
        budget_exhausted = deadline is not None and time.monotonic() >= deadline
        before = self.component_hashes()
        self.policy.set_training_mode(True)
        for _ in range(epochs):
            if deadline is not None and time.monotonic() >= deadline:
                budget_exhausted = True
                break
            order = self._minibatch_rng.permutation(len(prepared))
            for start in range(0, len(order), self.config.batch_size):
                if deadline is not None and time.monotonic() >= deadline:
                    budget_exhausted = True
                    break
                if steps >= max_updates:
                    break
                batch = [prepared[int(index)] for index in order[start:start + self.config.batch_size]]
                observations = torch.as_tensor(np.stack([row[0] for row in batch]), dtype=torch.float32)
                masks = torch.as_tensor(np.stack([row[1] for row in batch]), dtype=torch.bool)
                actions = torch.as_tensor([row[2] for row in batch], dtype=torch.long)
                loss = -self._distribution(observations, masks).log_prob(actions).mean()
                self.policy.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm,
                                                      error_if_nonfinite=True)
                self.policy.optimizer.step()
                losses.append(float(loss.detach()))
                gradients.append(float(norm))
                steps += 1
            if steps >= max_updates or budget_exhausted:
                break
        self.bc_optimizer_steps += steps
        self.actor_optimizer_steps += steps
        self.optimizer_steps += steps
        after = self.component_hashes()
        return dict(updates=steps, samples=len(prepared), mean_loss=float(np.mean(losses)) if losses else 0.,
                    gradient_norm_max=max(gradients, default=0.),
                    budget_exhausted=budget_exhausted,
                    actor_changed=before['actor'] != after['actor'],
                    critic_changed=before['critic'] != after['critic'])

    def update(self, trajectory: Sequence[SlotTransition], bootstrap_value=None, *, deadline=None):
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
        candidates = [len(slot.decisions) for slot in slots]
        free_choices = [sum(np.count_nonzero(d.action_mask) > 1 for d in slot.decisions) for slot in slots]
        initial_components = self.component_hashes()
        joint_log_ratios, gradient_norms = [], []
        actor_steps = critic_steps = effective_actor = effective_critic = 0
        target_kl_stopped = False
        budget_exhausted = False
        planned_minibatches = self.config.n_epochs * math.ceil(len(slots) / self.config.batch_size)
        self.policy.set_training_mode(True)
        for _ in range(self.config.n_epochs):
            if deadline is not None and time.monotonic() >= deadline:
                budget_exhausted = True
                stopped_early = True
                break
            epochs_started += 1
            permutation = self._minibatch_rng.permutation(len(slots))
            for start in range(0, len(slots), self.config.batch_size):
                if deadline is not None and time.monotonic() >= deadline:
                    budget_exhausted = True
                    stopped_early = True
                    break
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
                    joint_log_ratios.extend(log_ratio[active].detach().cpu().tolist())
                metric = dict(policy_loss=float(policy_loss.detach()), value_loss=float(value_loss.detach()),
                              entropy=float(entropy.detach()), approx_kl=approximate_kl,
                              clip_fraction=clip_fraction, batch_slots=count)
                if not all(math.isfinite(value) for value in metric.values()):
                    raise RuntimeError('nonfinite physical-slot PPO update metric')
                history.append(metric)
                if oversized_log_ratio or (self.config.target_kl is not None
                                           and approximate_kl > 1.5 * self.config.target_kl):
                    numerical_guard_triggered = oversized_log_ratio
                    target_kl_stopped = (self.config.target_kl is not None
                                         and approximate_kl > 1.5 * self.config.target_kl)
                    stopped_early = True
                    break
                loss = policy_loss + self.config.vf_coef * value_loss - self.config.ent_coef * entropy
                self.policy.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm,
                                                      error_if_nonfinite=True)
                gradient_norms.append(float(norm))
                before_step = self.component_hashes()
                actor_steps += any(p.grad is not None for _, p in self._component_parameters('actor'))
                critic_steps += any(p.grad is not None for _, p in self._component_parameters('critic'))
                self.policy.optimizer.step()
                after_step = self.component_hashes()
                effective_actor += before_step['actor'] != after_step['actor']
                effective_critic += before_step['critic'] != after_step['critic']
                self.optimizer_steps += 1
                optimizer_steps += 1
            if stopped_early:
                break
        self.rollout_updates += 1
        self.num_slots += len(slots)
        decisions = sum(len(slot.decisions) for slot in slots)
        self.num_decisions += decisions
        self.actor_optimizer_steps += actor_steps
        self.critic_optimizer_steps += critic_steps
        self.effective_actor_steps += effective_actor
        self.effective_critic_steps += effective_critic
        self.zero_actor_updates += effective_actor == 0
        total_weight = sum(row['batch_slots'] for row in history)
        metrics = {name: math.fsum(row[name] * row['batch_slots'] for row in history) / total_weight
                   if total_weight else 0.
                   for name in ('policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_fraction')}
        metrics.update(slots=len(slots), decisions=decisions,
                       actor_slots=sum(any(np.count_nonzero(d.action_mask) > 1 for d in slot.decisions)
                                       for slot in slots),
                       mean_reward=float(rewards.mean()), mean_return=float(returns.mean()),
                       optimizer_steps=optimizer_steps, updates=self.rollout_updates,
                       total_optimizer_steps=self.optimizer_steps, epochs=epochs_started,
                       stopped_early=stopped_early,
                       numerical_guard_triggered=numerical_guard_triggered)
        final_components = self.component_hashes()
        metrics.update(
            candidates_per_slot=candidates, free_choices_per_slot=free_choices,
            candidate_count_summary=_distribution_summary(candidates),
            free_choice_count_summary=_distribution_summary(free_choices),
            joint_log_ratio=_distribution_summary(joint_log_ratios),
            joint_log_ratio_scope='active slot evaluations across attempted PPO minibatches and epochs',
            gradient_norm_mean=float(np.mean(gradient_norms)) if gradient_norms else 0.,
            gradient_norm_max=max(gradient_norms, default=0.), gradient_norm_scope='global norm before clipping',
            actor_optimizer_steps=int(actor_steps), critic_optimizer_steps=int(critic_steps),
            effective_actor_steps=int(effective_actor), effective_critic_steps=int(effective_critic),
            actor_parameter_changed=initial_components['actor'] != final_components['actor'],
            critic_parameter_changed=initial_components['critic'] != final_components['critic'],
            planned_minibatches=planned_minibatches, attempted_minibatches=len(history),
            target_kl_stopped=bool(target_kl_stopped),
            budget_exhausted=budget_exhausted,
            target_kl_stop_fraction=(planned_minibatches - optimizer_steps) / planned_minibatches
                                   if target_kl_stopped else 0.,
            zero_effective_actor_update=effective_actor == 0,
            zero_effective_actor_update_fraction=self.zero_actor_updates / self.rollout_updates,
            component_hashes=final_components,
        )
        return metrics

