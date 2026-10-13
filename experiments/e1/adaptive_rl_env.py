"""Independent four-action Gym interface over the unchanged E1 execution kernel.

The inner legacy environment retains its 40-dimensional observation and reward
implementation. This wrapper advances its one persistent LocalExecutor directly,
then appends current-known task risk statistics. It never calls legacy step()
with a new action, and never creates a second allocation or physics path.
"""
import gymnasium as gym
import numpy as np

from .adaptive_observation import RISK_FIELDS, risk_observation
from .adaptive_risk import AdaptiveDelayPolicy
from .cars_state import ResourceSnapshot, RunningTask, demand
from .rl_env import CarbonSchedulingEnv, OBSERVATION_FIELDS as BASE_FIELDS


RL_SCHEMA = 'e1-6b-adaptive-delay-env-v1'
OBSERVATION_FIELDS = BASE_FIELDS + RISK_FIELDS


class AdaptiveCarbonSchedulingEnv(gym.Env):
    """One action sets a 60/30/15-minute current-slack threshold for one step.

    ``risk_filter`` replaces solo slack with the frozen E1-6A safe slack only in
    eligibility decisions. It is a heuristic restriction, not an SLA guarantee.
    ``risk_state=False`` masks every feature of the added block, preserving the
    same shape and parameter count for all four RL ablation groups.
    """
    metadata = {'render_modes': []}
    render_mode = None

    def __init__(self, workload, environment, capacity, power_factory, parameters, *,
                 risk_state=True, risk_filter=True, reward_config=None,
                 normalization=None, max_episode_steps=None, stop_at_drain=False):
        super().__init__()
        if not isinstance(risk_state, bool) or not isinstance(risk_filter, bool):
            raise ValueError('risk_state and risk_filter must be bool')
        self.risk_state, self.risk_filter = risk_state, risk_filter
        self.inner = CarbonSchedulingEnv(
            workload, environment, capacity, power_factory, parameters,
            reward_config=reward_config, normalization=normalization,
            max_episode_steps=max_episode_steps, stop_at_drain=stop_at_drain)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            -1., 1., shape=(len(OBSERVATION_FIELDS),), dtype=np.float32)
        self.policy = None
        self._done = True

    @property
    def executor(self):
        return self.inner.executor

    @property
    def normalization(self):
        return self.inner.normalization

    @property
    def reward_config(self):
        return self.inner.reward_config

    @property
    def capacity(self):
        return self.inner.capacity

    def _observe_policy_boundary(self):
        # Observe on every step, including EDF action 0 and disabled filtering.
        # ArrivalHistory deduplicates the same arrivals subsequently supplied by
        # the executor's on_step_start notification.
        self.policy.history.observe(self.executor.observed_end,
                                    self.executor.current_arrivals)

    def _snapshot(self):
        executor = self.executor
        pending = tuple(executor.ledger[t.job_name]['spec']
                        for t in executor.pool.pending_tasks)
        ordered = self.policy.order(pending)
        running = tuple(RunningTask(
            t.job_name, demand(executor.ledger[t.job_name]['spec']),
            t.finish_time, executor.ledger[t.job_name]['spec'].sla_deadline)
            for t in executor.pool.running_tasks)
        available = (executor.pool.available_cores, executor.pool.available_gpus,
                     executor.pool.available_mem)
        return ResourceSnapshot(executor.observed_end, self.capacity, available,
                                ordered, running, executor.current_arrivals)

    def _observation(self, base=None):
        base = self.inner._observation() if base is None else base
        if self.risk_state:
            snapshot = self._snapshot()
            pressure = self.policy.history.pressure(snapshot.now)
            features, raw = risk_observation(snapshot, pressure, self.normalization)
            raw = dict(raw, enabled=True)
        else:
            features = np.zeros(len(RISK_FIELDS), dtype=np.float32)
            raw = {'enabled': False}
        self.observation_raw = dict(self.inner.observation_raw, risk=raw)
        observation = np.concatenate((base, features)).astype(np.float32, copy=False)
        if not self.observation_space.contains(observation):
            raise RuntimeError('Invalid adaptive online observation')
        return observation

    def _info(self):
        info = self.inner._info()
        info.update(observation_schema=RL_SCHEMA, risk_state=self.risk_state,
                    risk_filter=self.risk_filter)
        return info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.action_space.seed(seed)
        base, _ = self.inner.reset(seed=seed, options=options)
        self.policy = AdaptiveDelayPolicy(
            self.inner.parameters, self.inner._workload.start, self.capacity,
            risk_filter=self.risk_filter)
        self._observe_policy_boundary()
        self._done = False
        return self._observation(base), self._info()

    def step(self, action):
        if self._done:
            raise RuntimeError('reset() is required before another interaction')
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError('Action must be 0, 1, 2 or 3')
        action = int(action)
        self.policy.set_action(action)
        first_diagnostic = len(self.policy.diagnostics)
        interval = self.executor.advance(self.policy)
        diagnostics = self.policy.diagnostics[first_diagnostic:]
        completed = (self.executor.observed_end >= self.inner._workload.end and
                     not (self.executor.pool.pending_tasks or self.executor.pool.running_tasks))
        terminated = bool(self.executor.finished and completed)
        truncated = bool(self.executor.finished and not completed)
        self._done = self.inner._done = terminated or truncated
        self.inner._observe_boundary()
        self._observe_policy_boundary()
        reward, terms = self.inner._reward(interval, truncated)
        observation = self._observation()
        info = self._info()
        info.update(action=action, mode=self.policy.name, interval=dict(interval),
                    reward=terms, policy_diagnostics=list(diagnostics),
                    risk_counts=dict(
                        candidate_checks=len(diagnostics),
                        solo_eligible=sum(row['solo_eligible'] for row in diagnostics),
                        filter_veto=sum(row['filter_veto'] for row in diagnostics),
                        deferred=sum(row['defer'] for row in diagnostics)))
        if self._done:
            info['final_summary'] = dict(self.executor.result().summary)
            info['termination_reason'] = ('all_completed_at_accounting_end' if terminated
                                          else 'episode_or_coverage_limit')
        return observation, float(reward), terminated, truncated, info

    def close(self):
        self.inner.close()
        self._done = True
