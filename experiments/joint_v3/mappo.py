"""Masked dual-cost MAPPO with exact autoregressive physical-slot likelihoods."""
from copy import deepcopy
import hashlib
import math
import time
import numpy as np
import torch
from torch import nn
from experiments.joint.ppo import Decision, compute_gae
from experiments.joint_v2.ppo import _distribution_summary
from .critics import DualCostCritic, LocalDualCritic
from .reward import episode_costs


class MaskedActor:
    def __init__(self, obs_dim, action_dim, seed=0, config=None):
        self.obs_dim, self.action_dim, self.seed = obs_dim, action_dim, int(seed)
        self.config = dict(learning_rate=1e-4, max_grad_norm=.5)
        self.config.update(config or {})
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.net = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 64),
                                     nn.Tanh(), nn.Linear(64, action_dim))
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config['learning_rate'])
        self.rng = torch.Generator().manual_seed(self.seed)
        self.bc_optimizer_steps = self.optimizer_steps = 0

    def distribution(self, observations, masks):
        return torch.distributions.Categorical(logits=self.net(observations).masked_fill(~masks, -torch.inf))

    def act(self, observation, mask, deterministic=False):
        obs, legal = np.asarray(observation, dtype=np.float32), np.asarray(mask, dtype=bool)
        if obs.shape != (self.obs_dim,) or not np.isfinite(obs).all():
            raise ValueError('Invalid actor observation')
        if legal.shape != (self.action_dim,) or not legal.any():
            raise ValueError('Empty or invalid actor mask')
        with torch.no_grad():
            dist = self.distribution(torch.from_numpy(obs[None]), torch.from_numpy(legal[None]))
            action = (dist.probs.argmax(-1) if deterministic else
                      torch.multinomial(dist.probs, 1, generator=self.rng).flatten())
            logp = float(dist.log_prob(action)[0])
        return Decision(obs.copy(), legal.copy(), int(action[0]), logp)

    def parameter_hash(self):
        h = hashlib.sha256()
        for name, tensor in self.net.state_dict().items():
            h.update(name.encode()); h.update(tensor.detach().numpy().tobytes())
        return h.hexdigest()

    def component_hashes(self):
        return dict(actor=self.parameter_hash())

    def behavior_clone(self, samples, *, epochs=1, max_updates=64, deadline=None):
        valid = [s for s in samples if s.get('weight', 1.) > 0 and sum(s['mask']) > 1]
        losses, norms = [], []
        updates = 0
        for _ in range(epochs):
            for offset in range(0, len(valid), 64):
                if updates >= max_updates or (deadline is not None and time.monotonic() >= deadline):
                    break
                batch = valid[offset:offset+64]
                obs = torch.tensor(np.asarray([s['observation'] for s in batch]), dtype=torch.float32)
                mask = torch.tensor(np.asarray([s['mask'] for s in batch]), dtype=torch.bool)
                actions = torch.tensor([s['action'] for s in batch], dtype=torch.long)
                weights = torch.tensor([s.get('weight', 1.) for s in batch], dtype=torch.float32)
                if not mask.gather(1, actions[:, None]).all():
                    raise ValueError('Teacher violates actor mask')
                loss = -(weights*self.distribution(obs, mask).log_prob(actions)).mean()
                self.optimizer.zero_grad(); loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.config['max_grad_norm'])
                if not torch.isfinite(loss) or not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite auxiliary update')
                self.optimizer.step(); updates += 1
                losses.append(float(loss.detach())); norms.append(float(norm))
        self.bc_optimizer_steps += updates
        return dict(samples=len(valid), optimizer_steps=updates, loss=_distribution_summary(losses),
                    gradient_norm=_distribution_summary(norms), budget_exhausted=bool(
                        deadline is not None and time.monotonic() >= deadline))

    def export_state(self):
        return deepcopy(dict(obs_dim=self.obs_dim, action_dim=self.action_dim, seed=self.seed,
            config=self.config, net=self.net.state_dict(), optimizer=self.optimizer.state_dict(),
            rng=self.rng.get_state(), optimizer_steps=self.optimizer_steps, bc_optimizer_steps=self.bc_optimizer_steps))

    @classmethod
    def from_state(cls, state):
        obj = cls(state['obs_dim'], state['action_dim'], state['seed'], state['config'])
        obj.net.load_state_dict(state['net']); obj.optimizer.load_state_dict(deepcopy(state['optimizer']))
        obj.rng.set_state(state['rng'])
        obj.optimizer_steps, obj.bc_optimizer_steps = state['optimizer_steps'], state['bc_optimizer_steps']
        return obj


def joint_statistics(actor, sequences):
    """Sum conditional logs; forced/empty sequences contribute constant zero."""
    observations, masks, actions, owners, old = [], [], [], [], np.zeros(len(sequences))
    for slot, decisions in enumerate(sequences):
        for d in decisions:
            if np.count_nonzero(d.action_mask) <= 1:
                continue
            observations.append(d.observation); masks.append(d.action_mask)
            actions.append(d.action); owners.append(slot); old[slot] += d.log_prob
    new = torch.zeros(len(sequences), dtype=torch.float64)
    entropy = torch.zeros(len(sequences), dtype=torch.float64)
    active = torch.zeros(len(sequences), dtype=torch.bool)
    if owners:
        dist = actor.distribution(torch.tensor(np.asarray(observations), dtype=torch.float32),
                                  torch.tensor(np.asarray(masks), dtype=torch.bool))
        index = torch.tensor(owners, dtype=torch.long)
        new = new.index_add(0, index, dist.log_prob(torch.tensor(actions)).double())
        entropy = entropy.index_add(0, index, dist.entropy().double())
        active[index] = True
    return new, torch.from_numpy(old), entropy, active


class RACSystem:
    def __init__(self, seed=20261017, config=None, centralized=True):
        self.seed, self.centralized = int(seed), bool(centralized)
        self.config = dict(learning_rate=1e-4, gamma=1., gae_lambda=.95, n_epochs=4,
            batch_size=32, clip_range=.2, ent_coef=.01, vf_coef=.5, max_grad_norm=.5, target_kl=.03)
        self.config.update(config or {})
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        self.actors = dict(task=MaskedActor(40, 2, seed, self.config),
                           cooling=MaskedActor(17, 3, seed+1, self.config))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+2)
            self.critics = nn.ModuleDict({role: (DualCostCritic() if centralized else LocalDualCritic(dim))
                                          for role, dim in (('task', 40), ('cooling', 17))})
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=self.config['learning_rate'])
        self.rng = np.random.default_rng(seed+3)
        self.updates = self.physical_slots = 0

    def _states(self, rows, role):
        return [r[role+('_global' if self.centralized else '_boundary')] for r in rows]

    def update(self, rows, *, total_tasks, multiplier, auxiliary_weight=0., deadline=None):
        if not rows or not rows[-1]['done'] or any(r['done'] for r in rows[:-1]):
            raise ValueError('Only one complete physical episode is a PPO transaction')
        costs = episode_costs([r['terms'] for r in rows], total_tasks)
        cfg, n = self.config, len(rows)
        dones = np.asarray([r['done'] for r in rows], dtype=bool)
        advantages, targets, report = {}, {}, {}
        for role in self.actors:
            with torch.no_grad():
                values = self.critics[role](self._states(rows, role)).numpy()
            ac, tc = compute_gae(costs[:, 0], values[:, 0], dones,
                gamma=cfg['gamma'], gae_lambda=cfg['gae_lambda'])
            ad, td = compute_gae(costs[:, 1], values[:, 1], dones,
                gamma=cfg['gamma'], gae_lambda=cfg['gae_lambda'])
            ao, _ = compute_gae(costs[:, 2], np.zeros(n), dones,
                gamma=cfg['gamma'], gae_lambda=cfg['gae_lambda'])
            # No independent channel standardization: lambda retains its stated units.
            advantages[role] = -(ac+float(multiplier)*ad+ao)
            targets[role] = torch.tensor(np.column_stack((tc, td)), dtype=torch.float32)
            report[role] = dict(actor_optimizer_steps=0, actor_loss=[], entropy=[], kl=[], clip_fraction=[],
                gradient_norm=[], teacher_loss=[], teacher_loss_relative=[], teacher_agreement=[],
                advantage=_distribution_summary(advantages[role]), carbon_advantage=_distribution_summary(ac),
                deadline_advantage=_distribution_summary(ad), auxiliary_advantage=_distribution_summary(ao),
                stopped_kl=0, stopped_log_ratio=0)
        report.update(critic_optimizer_steps=0, carbon_critic_loss=[], deadline_critic_loss=[], critic_gradient_norm=[])
        for _ in range(cfg['n_epochs']):
            if deadline is not None and time.monotonic() >= deadline:
                report['budget_exhausted'] = True; break
            for offset in range(0, n, cfg['batch_size']):
                if deadline is not None and time.monotonic() >= deadline:
                    report['budget_exhausted'] = True
                    break
                # Shuffled order is generated once per epoch, not once per minibatch.
                if offset == 0:
                    order = self.rng.permutation(n)
                indices = order[offset:offset+cfg['batch_size']]
                batch = [rows[i] for i in indices]
                loss_c, loss_d = [], []
                for role in self.actors:
                    predicted = self.critics[role](self._states(batch, role))
                    diff = (predicted-targets[role][indices]).square()
                    loss_c.append(diff[:, 0].mean()); loss_d.append(diff[:, 1].mean())
                carbon_loss, deadline_loss = torch.stack(loss_c).mean(), torch.stack(loss_d).mean()
                critic_loss = cfg['vf_coef']*(carbon_loss+deadline_loss)
                self.critic_optimizer.zero_grad(); critic_loss.backward()
                norm = nn.utils.clip_grad_norm_(self.critics.parameters(), cfg['max_grad_norm'])
                if not torch.isfinite(critic_loss) or not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite dual critic training')
                self.critic_optimizer.step(); report['critic_optimizer_steps'] += 1
                report['carbon_critic_loss'].append(float(carbon_loss.detach()))
                report['deadline_critic_loss'].append(float(deadline_loss.detach()))
                report['critic_gradient_norm'].append(float(norm))
                for role, actor in self.actors.items():
                    r = report[role]
                    if r['stopped_kl'] or r['stopped_log_ratio']:
                        continue
                    new, old, entropy, active = joint_statistics(actor, [b[role+'_decisions'] for b in batch])
                    if not active.any():
                        continue
                    logratio = new-old
                    if torch.any(torch.abs(logratio.detach()[active]) > 60):
                        r['stopped_log_ratio'] += 1; continue
                    ratio = logratio.exp()
                    kl = float(((ratio-1)-logratio).detach()[active].mean())
                    r['kl'].append(kl)
                    if cfg.get('target_kl') is not None and kl > cfg['target_kl']:
                        r['stopped_kl'] += 1; continue
                    adv = torch.from_numpy(advantages[role][indices])
                    clipped = ratio.clamp(1-cfg['clip_range'], 1+cfg['clip_range'])
                    objective = torch.minimum(ratio*adv, clipped*adv)
                    ppo_loss = -(objective*active).mean()
                    teacher_loss = torch.tensor(0.)
                    if role == 'task' and auxiliary_weight > 0:
                        demos = [a for b in batch for a in b.get('task_actions', ())
                                 if a.get('teacher_reliable') and a.get('teacher_weight', 0.) > 0 and all(a['mask'])]
                        if demos:
                            dist = actor.distribution(torch.tensor(np.asarray([d['observation'] for d in demos]), dtype=torch.float32),
                                torch.tensor(np.asarray([d['mask'] for d in demos]), dtype=torch.bool))
                            teacher = torch.tensor([d['teacher_action'] for d in demos])
                            weights = torch.tensor([d['teacher_weight'] for d in demos], dtype=torch.float32)
                            teacher_loss = -(weights*dist.log_prob(teacher)).mean()
                            r['teacher_agreement'].append(float((dist.probs.argmax(-1) == teacher).float().mean()))
                    ent = (entropy*active).mean()
                    loss = ppo_loss+auxiliary_weight*teacher_loss-cfg['ent_coef']*ent
                    actor.optimizer.zero_grad(); loss.backward()
                    grad = nn.utils.clip_grad_norm_(actor.net.parameters(), cfg['max_grad_norm'])
                    if not torch.isfinite(loss) or not torch.isfinite(grad):
                        raise FloatingPointError('Nonfinite actor training')
                    actor.optimizer.step(); actor.optimizer_steps += 1; r['actor_optimizer_steps'] += 1
                    r['actor_loss'].append(float(ppo_loss.detach())); r['entropy'].append(float(ent.detach()))
                    r['gradient_norm'].append(float(grad)); r['teacher_loss'].append(float(teacher_loss.detach()))
                    r['teacher_loss_relative'].append(float(abs(auxiliary_weight*teacher_loss.detach())/(abs(ppo_loss.detach())+1e-8)))
                    r['clip_fraction'].append(float(((ratio.detach()-1).abs() > cfg['clip_range'])[active].float().mean()))
        for role in self.actors:
            for name, value in list(report[role].items()):
                if isinstance(value, list): report[role][name] = _distribution_summary(value)
        for name in ('carbon_critic_loss', 'deadline_critic_loss', 'critic_gradient_norm'):
            report[name] = _distribution_summary(report[name])
        self.updates += 1; self.physical_slots += n
        report.update(physical_slots=n, cost_sums=costs.sum(0).tolist(),
                      advantage_normalization='none; channels retain units', auxiliary_weight=auxiliary_weight)
        return report

    def export_state(self):
        return deepcopy(dict(schema='rac-mappo-system-v1', seed=self.seed, centralized=self.centralized,
            config=self.config, actors={k: a.export_state() for k, a in self.actors.items()},
            critics=self.critics.state_dict(), critic_optimizer=self.critic_optimizer.state_dict(),
            rng=self.rng.bit_generator.state, updates=self.updates, physical_slots=self.physical_slots))

    @classmethod
    def from_state(cls, state):
        if state['schema'] != 'rac-mappo-system-v1': raise ValueError('Wrong algorithm checkpoint')
        obj = cls(state['seed'], state['config'], state['centralized'])
        obj.actors = {k: MaskedActor.from_state(v) for k, v in state['actors'].items()}
        obj.critics.load_state_dict(state['critics'])
        obj.critic_optimizer.load_state_dict(deepcopy(state['critic_optimizer']))
        obj.rng.bit_generator.state = deepcopy(state['rng'])
        obj.updates, obj.physical_slots = state['updates'], state['physical_slots']
        return obj
