"""Task-by-task actions and one cooling action through the unchanged E1 executor.

This is an explicit two-controller interface rather than a fixed-size Gym
Discrete environment. The executor alone owns arrivals, allocation, completion,
non-preemption and the sole advancing physical model.
"""
import math
import time
from collections import Counter, deque
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from experiments.e1.cars import CarsPolicy
from experiments.e1.contracts import STEP
from experiments.e1.engine import LocalExecutor
from experiments.e1.policies import PolicyDecision, latest_start

from . import ACCOUNTING
from .power import COMPONENTS
from .reward import SystemReward

MODES = ('B0', 'B1', 'B2', 'B3', 'RiskOnly')
TASK_FIELDS = (
    'candidate_present', 'task_cpu', 'task_gpu', 'task_mem',
    'task_cpu_fraction', 'task_gpu_fraction', 'task_mem_fraction',
    'duration', 'wait_minutes', 'slack_minutes',
    'available_cpu', 'available_gpu', 'available_mem',
    'cpu_utilization', 'gpu_utilization', 'mem_utilization',
    'running_count', 'pending_count', 'urgent_count',
    'pending_cpu_demand', 'pending_gpu_demand', 'pending_mem_demand',
    'ci_current', 'ci_lag1', 'ci_lag2', 'ci_lag3', 'ci_lag4', 'ci_history_fraction',
    'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',
    'setpoint', 'previous_it_power', 'previous_hvac_power')
COOLING_FIELDS = (
    'cpu_utilization', 'gpu_utilization', 'mem_utilization', 'current_it_kw',
    'previous_it_kw', 'previous_hvac_kw', 'setpoint_c',
    'previous_rack_inlet_max_c', 'previous_rack_outlet_max_c',
    'thermal_sensor_available', 'temperature_c', 'wet_bulb_c', 'ci_g_per_kwh',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos')


def bounded(value, scale=1.):
    return float(value) / (float(scale) + abs(float(value)))


def vector(values):
    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all() or np.max(np.abs(result)) > 1:
        raise RuntimeError('Nonfinite or unnormalized joint observation')
    return result


def select(actor, obs, mask):
    if not callable(actor):
        raise ValueError('An actor callback is required for the trained controller')
    mask = np.asarray(mask, dtype=bool)
    decision = actor(obs.copy(), mask.copy())
    action = getattr(decision, 'action', decision)
    if isinstance(action, (bool, np.bool_)) or not isinstance(action, Integral):
        raise ValueError('Action must be an integer legal under its mask')
    action = int(action)
    if not 0 <= action < len(mask) or not mask[action]:
        raise ValueError('Action violates the legal action mask')
    return action, decision


@dataclass(frozen=True)
class SlotOutcome:
    task_boundary_observation: np.ndarray
    task_decisions: tuple
    cooling_boundary_observation: np.ndarray
    cooling_decisions: tuple
    reward: float
    done: bool
    terms: dict
    interval: dict
    task_actions: tuple
    cooling_actions: tuple


class _TaskController(CarsPolicy):
    def __init__(self, env):
        super().__init__('cars_risk_only', env.parameters, env._workload.start, env.capacity)
        self.env = env
        self.name = 'joint_' + env.mode

    def on_step_start(self, snapshot):
        super().on_step_start(snapshot)
        self._demand = [math.fsum(getattr(s, name) for s in snapshot.pending)
                        for name in ('cores_req', 'gpu_req', 'mem_req')]
        self._urgent = sum(snapshot.now + STEP > latest_start(s) for s in snapshot.pending)

    def on_task_started(self, running):
        spec = self.pending[running.task_id]
        for i, name in enumerate(('cores_req', 'gpu_req', 'mem_req')):
            self._demand[i] -= getattr(spec, name)
        self._urgent -= self.snapshot.now + STEP > latest_start(spec)
        super().on_task_started(running)

    def decide(self, spec, context):
        started = time.perf_counter()
        env = self.env
        mask = np.asarray((True, context.now + STEP <= latest_start(spec)), dtype=bool)
        obs = env.task_observation(spec, controller=self)
        if env.mode in ('B1', 'B3'):
            action, decision = select(env._task_actor, obs, mask)
            if hasattr(decision, 'log_prob'):
                env._task_decisions.append(decision)
            reason = 'task_ppo_start' if action == 0 else 'task_ppo_wait'
        elif env.mode == 'RiskOnly':
            old = super().decide(spec, context)
            action, reason = int(old.defer), old.reason
            if not mask[action]:
                raise RuntimeError('RiskOnly returned an illegal task action')
        else:
            action, reason = 0, 'edf_start'
        env._task_actions.append(dict(task_id=spec.task_id, action=action, mask=mask.tolist(),
                                      observation=obs.tolist(), reason=reason))
        env._task_seconds += time.perf_counter() - started
        return PolicyDecision(action == 1, reason)


class _CoolingBridge:
    def __init__(self, env, physical):
        self.env, self.physical = env, physical
        self.last = None

    def measure(self, cpu, gpu, mem, sample):
        env = self.env
        env._cooling_observation = env.cooling_observation((cpu, gpu, mem), sample)
        mask = self.physical.action_mask()
        before = time.perf_counter()
        if env.mode in ('B2', 'B3'):
            action, decision = select(env._cooling_actor, env._cooling_observation, mask)
            if hasattr(decision, 'log_prob'):
                env._cooling_decisions.append(decision)
        else:
            action = 1
        self.physical.apply_action(action)
        env._cooling_seconds += time.perf_counter() - before
        env._cooling_actions.append(dict(action=action, mask=list(mask),
                                        observation=env._cooling_observation.tolist(),
                                        setpoint_c=self.physical.setpoint_c))
        self.last = self.physical.measure(cpu, gpu, mem, sample)
        return self.last

    def close(self):
        self.physical.close()


class JointEnv:
    def __init__(self, workload, environment, capacity, power_factory, parameters, config, *,
                 mode='B3', max_episode_steps=None):
        if mode not in MODES:
            raise ValueError('Unsupported control mode')
        if parameters.get('future_ci', False):
            raise ValueError('Future information is forbidden')
        if max_episode_steps is not None and (isinstance(max_episode_steps, bool) or
                not isinstance(max_episode_steps, int) or max_episode_steps < 1):
            raise ValueError('max_episode_steps must be a positive integer')
        self._workload, self._environment = workload, environment
        self.capacity, self.parameters = tuple(capacity), dict(parameters)
        self.config, self.mode, self.power_factory = config, mode, power_factory
        self.max_episode_steps = max_episode_steps
        self.executor, self.physical, self.done = None, None, True

    def reset(self, seed=None):
        self.close()
        self.physical = self.power_factory()
        self.bridge = _CoolingBridge(self, self.physical)
        self.controller = _TaskController(self)
        self.executor = LocalExecutor(self._workload, self._environment, self.capacity,
                                      self.bridge, self.controller, drain=True)
        until = self._environment.end
        if self.max_episode_steps is not None:
            until = min(until, self._workload.start + STEP * self.max_episode_steps)
        self.executor.start(until=until)
        self.reward_model = SystemReward(self.config['reward'])
        self.records, self.reward_records, self.action_records = [], [], []
        self.ci_history = deque(maxlen=5)
        self.ci_history.append(float(self.executor.sample.ci_g_per_kwh))
        self.done = False
        # The cooling preview is causal but is NOT the actor's later post-task
        # observation or an allowed bootstrap for a cut cooling trajectory.
        return self.task_observation(), self.cooling_observation()

    def task_observation(self, spec=None, controller=None):
        pool, now = self.executor.pool, self.executor.sample.time
        norm = self.config['normalization']
        used = pool.inventory()
        if controller is None:
            pending = tuple(self.executor.ledger[t.job_name]['spec'] for t in pool.pending_tasks)
            count = len(pending)
            urgent = sum(now + STEP > latest_start(s) for s in pending)
            demand = tuple(math.fsum(getattr(s, name) for s in pending)
                           for name in ('cores_req', 'gpu_req', 'mem_req'))
        else:
            count, urgent, demand = len(controller.pending), controller._urgent, controller._demand
        if spec is None:
            candidate = [0.] * 10
        else:
            needs = (spec.cores_req, spec.gpu_req, spec.mem_req)
            candidate = [1., *[bounded(v, c) for v, c in zip(needs, self.capacity)],
                         *[bounded(v / c) for v, c in zip(needs, self.capacity)],
                         bounded(spec.duration, norm['minutes']),
                         bounded((now - spec.arrival_time).total_seconds() / 60, norm['minutes']),
                         bounded((latest_start(spec) - now).total_seconds() / 60, norm['minutes'])]
        free = (pool.available_cores, pool.available_gpus, pool.available_mem)
        values = candidate + [v / c for v, c in zip(free, self.capacity)]
        values += [v / c for v, c in zip(used, self.capacity)]
        values += [bounded(v, norm['tasks']) for v in (len(pool.running_tasks), count, urgent)]
        values += [bounded(v / c) for v, c in zip(demand, self.capacity)]
        ci = self.ci_history[-1]
        values += [bounded(ci, norm['ci'])] + [bounded(
            self.ci_history[-1-i] if len(self.ci_history) > i else self.ci_history[0], norm['ci'])
            for i in range(1, 5)] + [(len(self.ci_history)-1)/4]
        angle = 2 * math.pi * (now.hour * 60 + now.minute) / 1440
        weekday = 2 * math.pi * now.weekday() / 7
        last = self.bridge.last
        values += [math.sin(angle), math.cos(angle), math.sin(weekday), math.cos(weekday),
                   bounded(self.physical.setpoint_c, norm['temperature_c']),
                   bounded(last.it_kw if last else 0., norm['power_kw']),
                   bounded(last.hvac_kw if last else 0., norm['power_kw'])]
        result = vector(values)
        if len(result) != len(TASK_FIELDS):
            raise RuntimeError('Task observation schema mismatch')
        return result

    def cooling_observation(self, loads=None, sample=None):
        if loads is None:
            loads = tuple(v / c for v, c in zip(self.executor.pool.inventory(), self.capacity))
        raw = self.physical.cooling_observation(*loads, sample or self.executor.sample)
        norm = self.config['normalization']
        values = []
        for key in COOLING_FIELDS:
            value = raw[key]
            if key.endswith('_kw'):
                value = bounded(value, norm['power_kw'])
            elif key.endswith('_c'):
                value = bounded(value, norm['temperature_c'])
            elif key == 'ci_g_per_kwh':
                value = bounded(value, norm['ci'])
            values.append(value)
        return vector(values)

    def step(self, task_actor=None, cooling_actor=None):
        if self.done:
            raise RuntimeError('reset() required before another joint step')
        self._task_actor, self._cooling_actor = task_actor, cooling_actor
        boundary = self.task_observation()
        self._task_decisions, self._cooling_decisions = [], []
        self._task_actions, self._cooling_actions = [], []
        self._task_seconds, self._cooling_seconds = 0., 0.
        interval = self.executor.advance()
        self.done = self.executor.finished
        reading = self.bridge.last
        interval.update(accounting=ACCOUNTING, setpoint_c=reading.setpoint_c,
                        rack_inlet_max_c=reading.rack_inlet_max_c,
                        rack_outlet_max_c=reading.rack_outlet_max_c,
                        crac_thermal_kw=reading.crac_thermal_kw,
                        legacy_total_kw=reading.legacy_total_kw,
                        legacy_hvac_kw=reading.legacy_hvac_kw,
                        task_decision_seconds=self._task_seconds,
                        cooling_decision_seconds=self._cooling_seconds,
                        task_decision_count=len(self._task_actions),
                        capacity_wait_count=len(self.executor.policy_steps[-1]['capacity_waiting_ids']))
        for name, power in reading.components_kw.items():
            interval[name + '_power_kw'] = power
            interval[name + '_energy_kwh'] = power * .25
            interval[name + '_carbon_kg'] = power * .25 * interval['ci_g_per_kwh'] / 1000
        interval['hvac_carbon_kg'] = reading.hvac_kw * .25 * interval['ci_g_per_kwh'] / 1000
        reward, terms = self.reward_model.calculate(
            self.executor, interval, terminal=self.done,
            inlet_limit_c=self.config['cooling']['rack_inlet_limit_c'])
        interval['thermal_excess_c'] = terms['thermal_excess_c']
        self.records.append(dict(interval))
        self.reward_records.append(dict(reward=reward, **terms))
        self.action_records.append(dict(task_actions=list(self._task_actions),
                                        cooling_actions=list(self._cooling_actions)))
        outcome = SlotOutcome(boundary, tuple(self._task_decisions), self._cooling_observation.copy(),
                              tuple(self._cooling_decisions), reward, self.done, terms, dict(interval),
                              tuple(self._task_actions), tuple(self._cooling_actions))
        if not self.done:
            self.ci_history.append(float(self.executor.sample.ci_g_per_kwh))
        return outcome

    def summary(self):
        if not self.done or not self.records:
            raise RuntimeError('Summary requires a completed finite episode')
        result = self.executor.result()
        summary = dict(result.summary, accounting=ACCOUNTING, mode=self.mode,
                       observation_end=self.executor.observed_end.isoformat(),
                       task_observation_fields=list(TASK_FIELDS), cooling_observation_fields=list(COOLING_FIELDS))
        missed = summary['deadline_missed_completed'] + summary['deadline_missed_unfinished']
        summary['deadline_misses'] = missed
        summary['dmr'] = missed / summary['arrived'] if summary['arrived'] else 0.
        summary['unfinished'] = summary['pending'] + summary['running']
        summary['unfinished_tasks'] = summary['unfinished']
        summary['censored_tasks'] = summary['deadline_right_censored']
        summary['temperature_violations'] = sum(r['thermal_excess_c'] > 0 for r in self.records)
        summary['max_temperature_excess_c'] = max(r['thermal_excess_c'] for r in self.records)
        summary['max_rack_inlet_c'] = max(r['rack_inlet_max_c'] for r in self.records)
        summary['feasible'] = bool(summary['dmr'] <= .05 and summary['temperature_violations'] == 0
                                   and summary['unfinished'] == 0 and summary['deadline_right_censored'] == 0)
        completed = [r for r in result.tasks if r['state'] == 'completed']
        from experiments.e1.contracts import utc_time
        jct = [(utc_time(r['finish_time']) - utc_time(r['arrival_time'])).total_seconds()/60 for r in completed]
        waits = [r['observed_pending_minutes'] for r in result.policy_tasks]
        summary['jct_completed_mean_minutes'] = math.fsum(jct)/len(jct) if jct else None
        summary['waiting_mean_minutes'] = math.fsum(waits)/len(waits) if waits else None
        summary['waiting_total_task_minutes'] = math.fsum(waits)
        for name in (*COMPONENTS, 'hvac'):
            for metric in ('energy_kwh', 'carbon_kg'):
                summary[name+'_'+metric] = math.fsum(r[name+'_'+metric] for r in self.records)
            summary[name+'_power_mean_kw'] = math.fsum(r[name+'_power_kw'] for r in self.records)/len(self.records)
            summary[name+'_power_max_kw'] = max(r[name+'_power_kw'] for r in self.records)
        for controller in ('task', 'cooling'):
            summary[controller+'_decision_seconds'] = math.fsum(r[controller+'_decision_seconds'] for r in self.records)
            counts = Counter(x['action'] for row in self.action_records for x in row[controller+'_actions'])
            summary[controller+'_action_distribution'] = {str(k): v for k, v in sorted(counts.items())}
        summary['capacity_wait_intervals'] = sum(r['capacity_wait_count'] for r in self.records)
        summary['task_decision_scope'] = 'all resource-feasible pending tasks, EDF, once per slot; no cap'
        summary['omitted_pending_tasks'] = 0
        summary['reward'] = {key: math.fsum(row[key] for row in self.reward_records) for key in (
            'reward', 'carbon_cost', 'wait_cost', 'deadline_cost', 'thermal_safety_cost', 'terminal_cost')}
        return summary

    def close(self):
        if self.physical is not None:
            self.physical.close()
            self.physical = None
        self.done = True
