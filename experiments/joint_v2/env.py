"""V2 finite-cohort IPPO environment over the unchanged V1 execution kernel.

Actors share one advancing plant. A separate, explicitly counterfactual probe
evaluates idle18 at current weather for the action-independent carbon baseline.
After drain, a deterministic operator lowers to18 before termination; all of
its real energy is charged. Omitted idle18 future intervals are integrated only
when producing a report, and never enter actors, returns or terminal rewards.
"""
import math
import time
from collections import Counter
from dataclasses import replace
from itertools import takewhile

import numpy as np

from experiments.e1.contracts import content_hash, file_hash
from experiments.e1.policies import PolicyDecision
from experiments.joint.env import (
    JointEnv, TASK_FIELDS, COOLING_FIELDS, _TaskController, select)
from experiments.joint.power import COMPONENTS, JointPowerModel

from .deadline_guard import legal_task_mask
from .reward import V2SystemReward


# Cache only algebraic reference outputs by physical model and current weather.
# No policy state, task identity, CI or future-dependent quantities are cached.
_IDLE_CACHE = {}


class _V2TaskController(_TaskController):
    def decide(self, spec, context):
        started = time.perf_counter()
        env = self.env
        # CarsPolicy preserves the snapshot's EDF insertion order and removes
        # starts immediately. Inspect only earlier tasks still waiting, rather
        # than rescanning the entire later queue for every feasible candidate.
        earlier_pending = tuple(takewhile(lambda s: s.sla_deadline < spec.sla_deadline,
                                           self.pending.values()))
        result = legal_task_mask(spec, context.now,
                                 (env.executor.pool.available_cores, env.executor.pool.available_gpus,
                                  env.executor.pool.available_mem), env.capacity,
                                 earlier_pending, tuple(self.running),
                                 enabled=env.guard_enabled)
        mask = np.asarray(result.mask, dtype=bool)
        obs = env.task_observation(spec, controller=self)
        action, decision = select(env._task_actor, obs, mask)
        if hasattr(decision, 'log_prob'):
            env._task_decisions.append(decision)
        guarded = bool(result.protected_task_ids) and not result.conflict
        env._guard_triggers += int(guarded)
        env._guard_conflicts += int(result.conflict)
        reason = 'deadline_reservation_wait' if guarded else ('task_ppo_start' if action == 0 else 'task_ppo_wait')
        env._task_actions.append(dict(task_id=spec.task_id, action=action, mask=mask.tolist(),
                                      observation=obs.tolist(), reason=reason,
                                      protected_task_ids=list(result.protected_task_ids),
                                      deadline_guard_conflict=result.conflict))
        env._task_seconds += time.perf_counter() - started
        return PolicyDecision(action == 1, reason)


class _V2CoolingBridge:
    def __init__(self, env, physical):
        self.env, self.physical, self.last = env, physical, None

    def measure(self, cpu, gpu, mem, sample):
        env = self.env
        pool = env.executor.pool
        drained = sample.time >= env._workload.end and not (pool.pending_tasks or pool.running_tasks)
        env._slot_stage = ('arrival' if sample.time < env._workload.end else
                           'postdrain_idle' if drained else 'drain')
        env._cooling_observation = env.cooling_observation((cpu, gpu, mem), sample)
        mask = self.physical.action_mask()
        before = time.perf_counter()
        if drained:
            action = 0 if self.physical.setpoint_c > 18. else 1
            env._operator_actions.append(dict(time=sample.time.isoformat(), action=action,
                                              setpoint_before_c=self.physical.setpoint_c,
                                              source='deterministic_idle_settling'))
        else:
            action, decision = select(env._cooling_actor, env._cooling_observation, mask)
            if hasattr(decision, 'log_prob'):
                env._cooling_decisions.append(decision)
            env._cooling_actions.append(dict(action=action, mask=list(mask),
                                             observation=env._cooling_observation.tolist(),
                                             setpoint_c=self.physical.setpoint_c + action - 1))
        self.physical.apply_action(action)
        env._cooling_seconds += time.perf_counter() - before
        self.last = self.physical.measure(cpu, gpu, mem, sample)
        # Only the sample just used by the advancing plant may affect reward.
        env._current_reference = env._reference(sample)
        return self.last

    def close(self):
        self.physical.close()


class V2Env(JointEnv):
    def __init__(self, scenario, config, *, evaluation=False, max_episode_steps=None):
        if scenario.role not in ('train', 'development', 'validation'):
            raise ValueError('V2 environment allows Train/development/Validation only; Test is sealed')
        if not 0 <= (scenario.environment.end - scenario.workload.end).total_seconds() <= 30 * 86400:
            raise ValueError('V2 physical safety tail cannot exceed 30 days after arrivals end')
        self.scenario, self.v2_config, self.evaluation = scenario, config, bool(evaluation)
        v1 = config['v1_protocol']
        options = config.get('environment', {})
        if (options.get('idle_reference_setpoint_c', 18.) != 18.
                or options.get('settle_after_drain', True) is not True):
            raise ValueError('V2 fixed-window identity requires idle18 and deterministic drain settling')
        self.guard_enabled = bool(options.get('deadline_guard_enabled', True))
        self._reward_config = dict(config.get('reward', v1['reward']))
        self._deadline_multiplier = float(self._reward_config.get('deadline_multiplier', 10.))
        self._reference_model = None
        signature = content_hash(dict(datacenter=scenario.datacenter,
                                      physical_sha256=file_hash(scenario.model_path), cooling=v1['cooling']))
        self._reference_cache = _IDLE_CACHE.setdefault(signature, {})
        self._fixed_report = None
        factory = lambda: JointPowerModel(scenario.datacenter, scenario.model_path,
                                          cooling_config=v1['cooling'])
        super().__init__(scenario.workload, scenario.environment, scenario.capacity, factory,
                         scenario.parameters, v1, mode='B3', max_episode_steps=max_episode_steps)
        self.set_deadline_multiplier(self._deadline_multiplier)

    def reset(self, seed=None):
        result = super().reset(seed=seed)
        self.controller = _V2TaskController(self)
        self.executor.policy = self.controller
        self.bridge = _V2CoolingBridge(self, self.physical)
        self.executor.power_model = self.bridge
        self._guard_triggers = self._guard_conflicts = 0
        self._operator_actions = []
        self._current_reference = None
        self._slot_stage = None
        self._fixed_report = None
        self.termination_reason = None
        self.reward_model = V2SystemReward(self._reward_config, lambda: self._current_reference,
                                           lambda: self._slot_stage)
        self.reward_model.set_deadline_multiplier(self._deadline_multiplier)
        return result

    def _reference(self, sample):
        key = (float(sample.temperature_c), float(sample.wet_bulb_c))
        reading = self._reference_cache.get(key)
        if reading is None:
            if self._reference_model is None:
                self._reference_model = self.power_factory()
                self._reference_model.set_setpoint(18.)
            reading = self._reference_model.measure(0., 0., 0., sample)
            self._reference_cache[key] = reading
        return reading

    def set_deadline_multiplier(self, value):
        value = float(value)
        maximum = float(self.v2_config.get('constraint', {}).get('lambda_max', 5000.))
        if not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError('Deadline multiplier must remain within the configured finite bound')
        self._deadline_multiplier = value
        if hasattr(self, 'reward_model'):
            self.reward_model.set_deadline_multiplier(value)

    def _drained_settled(self):
        return (self.executor.observed_end >= self._workload.end
                and not (self.executor.pool.pending_tasks or self.executor.pool.running_tasks)
                and self.physical.setpoint_c == 18.)

    def step(self, task_actor=None, cooling_actor=None):
        guard_before, conflict_before = self._guard_triggers, self._guard_conflicts
        operator_before = len(self._operator_actions)
        result = super().step(task_actor, cooling_actor)
        if self._drained_settled():
            self.done = True
            self.termination_reason = 'drained_settled'
        elif self.done:
            self.termination_reason = ('fixed_window_limit' if self.executor.observed_end == self._environment.end
                                       else 'training_step_limit')
        diagnostics = dict(task_candidate_count=len(result.task_actions),
                           task_free_action_count=sum(all(a['mask']) for a in result.task_actions),
                           deadline_guard_triggers=self._guard_triggers - guard_before,
                           deadline_guard_conflicts=self._guard_conflicts - conflict_before,
                           cooling_actor_action_count=len(result.cooling_actions),
                           deterministic_settling=len(self._operator_actions) > operator_before)
        diagnostics['operator_cooling_action'] = (self._operator_actions[-1]['action']
                                                    if diagnostics['deterministic_settling'] else None)
        self.records[-1].update(diagnostics)
        self.executor.steps[-1].update(diagnostics)
        return replace(result, done=self.done, interval=dict(result.interval, **diagnostics))

    def _fixed_window_report(self):
        if self._fixed_report is not None:
            return self._fixed_report
        if not self.done:
            raise RuntimeError('Future idle integration is report-only after an episode ends')
        fixed_complete = self._drained_settled() or self.executor.observed_end == self._environment.end
        background = {name: dict(energy_kwh=0., carbon_kg=0.) for name in COMPONENTS}
        tail = {name: dict(energy_kwh=0., carbon_kg=0., power_kw=[]) for name in COMPONENTS}
        tail_count = 0
        for sample in self._environment.samples:
            reference = self._reference(sample)
            is_tail = fixed_complete and sample.time >= self.executor.observed_end
            if is_tail:
                tail_count += 1
            for name, power in reference.components_kw.items():
                energy, carbon = power * .25, power * .25 * sample.ci_g_per_kwh / 1000
                background[name]['energy_kwh'] += energy
                background[name]['carbon_kg'] += carbon
                if is_tail:
                    tail[name]['energy_kwh'] += energy
                    tail[name]['carbon_kg'] += carbon
                    tail[name]['power_kw'].append(power)
        self._fixed_report = dict(complete=fixed_complete, background=background, tail=tail,
                                  tail_count=tail_count)
        return self._fixed_report

    def summary(self):
        summary = super().summary()
        report = self._fixed_window_report()
        summary.update(protocol='joint-task-hvac-ippo-v2', scenario_id=self.scenario.scenario_id,
                       scenario_role=self.scenario.role, evaluation=self.evaluation,
                       termination_reason=self.termination_reason,
                       physical_steps=len(self.records), simulated_intervals=len(self.records),
                       analytic_idle_intervals=report['tail_count'],
                       fixed_window_intervals=len(self._environment.samples),
                       fixed_window_complete=report['complete'],
                       physical_prefix_carbon_kg=summary['carbon_kg'],
                       physical_prefix_energy_kwh=summary['total_energy_kwh'],
                       operational_end=self._environment.end.isoformat(),
                       physical_prefix_end=self.executor.observed_end.isoformat(),
                       carbon_scope='fixed_operational_window' if report['complete'] else 'censored_physical_prefix')
        summary['background_carbon_kg'] = math.fsum(v['carbon_kg'] for v in report['background'].values())
        summary['background_energy_kwh'] = math.fsum(v['energy_kwh'] for v in report['background'].values())
        summary['training_background_carbon_kg'] = math.fsum(r['background_carbon_kg'] for r in self.records)
        summary['incremental_carbon_kg'] = math.fsum(r['incremental_carbon_kg'] for r in self.records)
        summary['incremental_energy_kwh'] = math.fsum(r['incremental_energy_kwh'] for r in self.records)
        summary['analytic_idle_tail_carbon_kg'] = math.fsum(v['carbon_kg'] for v in report['tail'].values())
        summary['analytic_idle_tail_energy_kwh'] = math.fsum(v['energy_kwh'] for v in report['tail'].values())
        summary['carbon_kg'] += summary['analytic_idle_tail_carbon_kg']
        summary['total_energy_kwh'] += summary['analytic_idle_tail_energy_kwh']
        summary['fixed_window_carbon_kg'] = summary['carbon_kg'] if report['complete'] else None
        for name in COMPONENTS:
            for metric in ('energy_kwh', 'carbon_kg'):
                summary[name + '_' + metric] += report['tail'][name][metric]
                summary['background_' + name + '_' + metric] = report['background'][name][metric]
            powers = [r[name + '_power_kw'] for r in self.records] + report['tail'][name]['power_kw']
            summary[name + '_power_mean_kw'] = math.fsum(powers) / len(powers)
            summary[name + '_power_max_kw'] = max(powers)
        for metric in ('energy_kwh', 'carbon_kg'):
            summary['hvac_' + metric] = math.fsum(summary[n + '_' + metric] for n in COMPONENTS[1:])
            summary['background_hvac_' + metric] = math.fsum(
                report['background'][n][metric] for n in COMPONENTS[1:])
        hvac_powers = [r['hvac_power_kw'] for r in self.records]
        hvac_powers += [math.fsum(report['tail'][n]['power_kw'][i] for n in COMPONENTS[1:])
                        for i in range(report['tail_count'])]
        summary['hvac_power_mean_kw'] = math.fsum(hvac_powers) / len(hvac_powers)
        summary['hvac_power_max_kw'] = max(hvac_powers)
        summary['active_execution_carbon_kg'] = math.fsum(r['carbon_kg'] for r in self.records if r['running'] > 0)
        for stage in ('arrival', 'drain', 'postdrain_idle'):
            summary[stage + '_phase_carbon_kg'] = math.fsum(r['carbon_kg'] for r in self.records if r['stage'] == stage)
        summary['physical_post_drain_idle_carbon_kg'] = summary['postdrain_idle_phase_carbon_kg']
        summary['postdrain_idle_phase_carbon_kg'] += summary['analytic_idle_tail_carbon_kg']
        summary['post_drain_idle_carbon_kg'] = summary['postdrain_idle_phase_carbon_kg']
        summary['deadline_guard_triggers'] = self._guard_triggers
        summary['deadline_guard_conflicts'] = self._guard_conflicts
        task_actions = [a for row in self.action_records for a in row['task_actions']]
        summary['task_free_action_count'] = sum(all(a['mask']) for a in task_actions)
        summary['task_action_count'] = len(task_actions)
        summary['task_free_action_fraction'] = (summary['task_free_action_count'] / len(task_actions)
                                                 if task_actions else 0.)
        summary['cooling_action_count'] = sum(len(row['cooling_actions']) for row in self.action_records)
        summary['deterministic_settling_steps'] = len(self._operator_actions)
        summary['deadline_multiplier'] = self._deadline_multiplier
        summary['scenario_task_count'] = len(self._workload.tasks)
        summary['all_arrivals_observed'] = (self.executor.observed_end >= self._workload.end
                                             and summary['arrived'] == len(self._workload.tasks))
        summary['unobserved_future_tasks'] = len(self._workload.tasks) - summary['arrived']
        summary['complete_episode_for_constraint'] = bool(
            report['complete'] and summary['all_arrivals_observed'] and summary['unfinished'] == 0
            and summary['deadline_right_censored'] == 0)
        summary['censored'] = not summary['complete_episode_for_constraint']
        summary['feasible'] = bool(summary['feasible'] and summary['complete_episode_for_constraint'])
        summary['policy_defer_intervals'] = sum(len(s['policy_deferred_ids']) for s in self.executor.policy_steps)
        summary['capacity_wait_intervals'] = sum(len(s['capacity_waiting_ids']) for s in self.executor.policy_steps)
        settings = Counter(str(r['setpoint_c']) for r in self.records)
        settings['18.0'] += report['tail_count']
        summary['setpoint_distribution'] = dict(sorted(settings.items()))
        summary['reward_term_means'] = {key: value / len(self.records) for key, value in summary['reward'].items()}
        if report['complete'] and not math.isclose(
                summary['carbon_kg'], summary['background_carbon_kg'] + summary['incremental_carbon_kg'],
                rel_tol=1e-9, abs_tol=1e-8):
            raise RuntimeError('V2 fixed-window carbon identity failed')
        return summary

    def close(self):
        super().close()
        if self._reference_model is not None:
            self._reference_model.close()
            self._reference_model = None
