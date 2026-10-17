"""Causal RAC-MAPPO sensors around the unchanged V2 physical execution path."""
from collections import Counter, deque
from dataclasses import replace
from types import SimpleNamespace
import math
import time

import numpy as np

from experiments.e1.contracts import STEP
from experiments.e1.policies import PolicyDecision, latest_start
from experiments.joint.env import TASK_FIELDS as V2_TASK_FIELDS, COOLING_FIELDS, bounded, select, vector
from experiments.joint_v2.env import V2Env, _V2TaskController, _V2CoolingBridge

from .teacher import teacher_label


TASK_FIELDS = V2_TASK_FIELDS + ('safe_wait_budget_minutes', 'extra_tardiness_minutes',
    'mean_carbon_gain_kg', 'std_carbon_gain_kg', 'robust_carbon_gain_kg')


def _invalid_risk(reason):
    return SimpleNamespace(safe_wait_budget_minutes=0., delta_deadline_misses=0,
        extra_tardiness_minutes=0., risk_estimation_valid=False, risk_confidence=0.,
        reason=reason, start_schedule=None, wait_schedule=None, runtime_seconds=0., cache_hit=False)


def _invalid_carbon(reason):
    return SimpleNamespace(mean_carbon_gain_kg=0., std_carbon_gain_kg=0., robust_carbon_gain_kg=0.,
        forecast_confidence=0., counterfactual_valid=False, tail_liability_kg=0.,
        estimator_runtime_ms=0., reason=reason, start_settled=False, wait_settled=False)


class _V3TaskController(_V2TaskController):
    def on_step_start(self, snapshot):
        super().on_step_start(snapshot)
        self.env._observe_current()
        self.env._fine_this_slot = 0
        self.env._fast_this_slot = 0
        self.env._estimation_seconds_this_slot = 0.
        self.env.task_global_state = self.env._global_state(tuple(self.pending.values()))

    def decide(self, spec, context):
        from .action_shield import shield
        env = self.env
        started = time.perf_counter()
        risk, carbon, fine, fast = env._estimate(spec, context.now, self)
        pool = env.executor.pool
        result = shield(spec, context.now,
            (pool.available_cores, pool.available_gpus, pool.available_mem), env.capacity,
            tuple(self.pending.values()), tuple(self.running), risk,
            enabled=env.v3_config.get('variant', 'full') != 'no_queue_risk')
        mask = np.asarray(result.mask, dtype=bool)
        env._active_features = env._features(risk, carbon)
        observation = env.task_observation(spec, controller=self)
        progress = env.teacher_progress
        label_risk = risk
        if env.v3_config.get('variant') == 'no_queue_risk':
            fields = vars(risk).copy()
            fields.update(safe_wait_budget_minutes=max(0., math.floor(
                (latest_start(spec)-context.now).total_seconds()/900.) * 15.), delta_deadline_misses=0)
            label_risk = SimpleNamespace(**fields)
        label = teacher_label(mask, label_risk, carbon, env.v3_config,
            physical_steps=progress[0], total_steps=progress[1])
        env._current_teacher = label
        action, decision = select(env._task_actor, observation, mask)
        if hasattr(decision, 'log_prob'):
            env._task_decisions.append(decision)
        env._guard_triggers += int(result.resource_protection_triggered)
        env._guard_conflicts += int(result.conflict)
        record = dict(task_id=spec.task_id, action=action, mask=mask.tolist(),
            observation=observation.tolist(), reason='rac_mappo_wait' if action else 'rac_mappo_start',
            protected_task_ids=list(result.protected_task_ids), deadline_guard_conflict=result.conflict,
            teacher_action=label['action'], teacher_weight=label['weight'], teacher_reliable=label['reliable'],
            teacher_reason=label['reason'], actor_teacher_agreement=action == label['action'],
            fine_estimation=fine, fast_approximation=fast, unestimated=not fine and not fast,
            fine_carbon_estimation=fine and carbon.estimator_runtime_ms > 0.,
            carbon_unestimated=not fast and not (fine and carbon.estimator_runtime_ms > 0.),
            fast_approximation_rule='current_local_system_marginal_power_equal_work_ci_shift' if fast else None,
            unestimated_fallback_rule='zero_gain_zero_confidence_conservative_start' if not fine and not fast else None,
            safe_wait_budget_minutes=float(risk.safe_wait_budget_minutes),
            extra_tardiness_minutes=float(risk.extra_tardiness_minutes),
            delta_deadline_misses=int(risk.delta_deadline_misses),
            risk_estimation_valid=bool(risk.risk_estimation_valid), risk_confidence=float(risk.risk_confidence),
            risk_reason=risk.reason, risk_runtime_ms=1000. * risk.runtime_seconds,
            risk_cache_hit=bool(risk.cache_hit),
            mean_carbon_gain_kg=float(carbon.mean_carbon_gain_kg),
            std_carbon_gain_kg=float(carbon.std_carbon_gain_kg),
            robust_carbon_gain_kg=float(carbon.robust_carbon_gain_kg),
            forecast_confidence=float(carbon.forecast_confidence),
            counterfactual_valid=bool(carbon.counterfactual_valid),
            tail_liability_kg=float(carbon.tail_liability_kg), carbon_reason=carbon.reason,
            tail_liability_known=bool(getattr(carbon, 'tail_liability_known', True)),
            estimator_runtime_ms=float(carbon.estimator_runtime_ms),
            start_settled=bool(carbon.start_settled), wait_settled=bool(carbon.wait_settled),
            wait_masked=bool(result.wait_masked), start_masked=bool(result.start_masked),
            risk_unreliable=bool(result.risk_unreliable), shield_reasons=list(result.reasons))
        if action == 1 and risk.risk_estimation_valid and risk.safe_wait_budget_minutes >= 15:
            env._predicted_safe_wait_ids.add(spec.task_id)
        env._task_actions.append(record)
        env._task_seconds += time.perf_counter() - started
        return PolicyDecision(action == 1, record['reason'])


class _V3CoolingBridge(_V2CoolingBridge):
    def measure(self, cpu, gpu, mem, sample):
        self.env.cooling_global_state = self.env._global_state()
        return super().measure(cpu, gpu, mem, sample)


class V3Env(V2Env):
    def __init__(self, scenario, config, forecaster, *, evaluation=False, max_episode_steps=None):
        from .risk_estimator import RiskEstimator
        from .carbon_counterfactual import CarbonCounterfactual
        self.v3_config, self.forecaster = config, forecaster
        self._active_features = np.zeros(5, dtype=np.float32)
        self._forecast_history = deque(maxlen=9)
        self._forecast_time = None
        self._forecast = None
        self._carbon = None
        self._risk = RiskEstimator(config.get('risk', {}))
        self.teacher_progress = (0, config.get('budget', {}).get('total_physical_steps', 65536))
        self._fine_this_slot = 0
        self._fast_this_slot = 0
        self._estimation_seconds_this_slot = 0.
        self.mechanism_probes = []
        self._mechanism_audit = None
        super().__init__(scenario, config, evaluation=evaluation, max_episode_steps=max_episode_steps)
        self._carbon = CarbonCounterfactual(self.power_factory, config.get('carbon', {}))

    def reset(self, seed=None):
        self._forecast_history.clear()
        self._forecast_time = None
        self._forecast = None
        self._active_features = np.zeros(5, dtype=np.float32)
        self._predicted_safe_wait_ids = set()
        self._fine_this_slot = 0
        self._fast_this_slot = 0
        self._estimation_seconds_this_slot = 0.
        self.mechanism_probes = []
        self._mechanism_audit = None
        result = super().reset(seed=seed)
        self.controller = _V3TaskController(self)
        self.executor.policy = self.controller
        self.bridge = _V3CoolingBridge(self, self.physical)
        self.executor.power_model = self.bridge
        self._forecast_history.append(self.executor.sample)
        self.task_global_state = self.cooling_global_state = self._global_state()
        return result

    @property
    def forecast(self):
        return self._forecast

    @property
    def forecast_history(self):
        return tuple(self._forecast_history)

    def _observe_current(self):
        sample = self.executor.sample
        if self._forecast_time == sample.time:
            return
        if not self._forecast_history or self._forecast_history[-1].time != sample.time:
            self._forecast_history.append(sample)
        self._forecast = self.forecaster.predict(tuple(self._forecast_history), sample.time)
        self._forecast_time = sample.time

    def _estimate(self, spec, now, controller):
        options = self.v3_config.get('carbon', {})
        limit = int(options.get('max_fine_candidates', 32))
        slot_ms = float(options.get('slot_time_budget_ms',
                        self.v3_config.get('risk', {}).get('time_budget_ms', 100.)))
        if now + STEP > latest_start(spec):
            return _invalid_risk('no_own_grid_slack'), _invalid_carbon('no_safe_wait'), False, False
        if self._fine_this_slot >= limit or self._estimation_seconds_this_slot * 1000. >= slot_ms:
            risk = _invalid_risk('physical_slot_estimator_budget')
            if (self._fast_this_slot < int(options.get('max_fast_candidates', 32))
                    and self.v3_config.get('variant') != 'no_carbon_features'):
                started = time.perf_counter()
                self._fast_this_slot += 1
                try:
                    carbon = self._carbon.fast_estimate(spec, self._forecast, self.executor.sample,
                        self.physical.setpoint_c, capacity=self.capacity,
                        current_used=self.executor.pool.inventory())
                    return risk, carbon, False, True
                finally:
                    self._estimation_seconds_this_slot += time.perf_counter()-started
            return risk, _invalid_carbon('unestimated_after_bounded_fine_and_fast_budgets'), False, False
        self._fine_this_slot += 1
        started = time.perf_counter()
        try:
            risk = self._risk.estimate(spec, now, self.capacity, tuple(controller.pending.values()),
                tuple(controller.running), blocked_until=tuple(controller.blocked.items()))
            if not risk.risk_estimation_valid:
                reason = 'risk_invalid:' + risk.reason
            elif risk.safe_wait_budget_minutes < 15 and self.v3_config.get('variant') != 'no_queue_risk':
                reason = 'no_supported_safe_wait'
            elif self.v3_config.get('variant') == 'no_carbon_features':
                reason = 'carbon_features_disabled'
            elif self._forecast is None:
                reason = 'forecast_unavailable'
            elif not self._forecast.valid:
                reason = 'forecast_invalid:' + self._forecast.reason
            else:
                reason = None
            carbon = _invalid_carbon(reason or 'not_yet_estimated')
            if reason is None:
                carbon = self._carbon.estimate(risk.start_schedule, risk.wait_schedule,
                    self._forecast, self.executor.sample, self.physical.setpoint_c, capacity=self.capacity)
                if self.evaluation and carbon.counterfactual_valid and len(self.mechanism_probes) < 32:
                    self.mechanism_probes.append(dict(task_id=spec.task_id, initial_timestamp=now.isoformat(),
                        initial_setpoint_c=float(self.physical.setpoint_c), capacity=list(self.capacity),
                        forecast_horizon=len(self._forecast.mean),
                        mean_carbon_gain_kg=float(carbon.mean_carbon_gain_kg),
                        std_carbon_gain_kg=float(carbon.std_carbon_gain_kg),
                        robust_carbon_gain_kg=float(carbon.robust_carbon_gain_kg),
                        forecast_confidence=float(carbon.forecast_confidence),
                        start_loads=np.asarray(risk.start_schedule.loads).tolist(),
                        wait_loads=np.asarray(risk.wait_schedule.loads).tolist(),
                        known_task_ids_start=sorted(risk.start_schedule.finishes),
                        known_task_ids_wait=sorted(risk.wait_schedule.finishes),
                        start_settled=bool(carbon.start_settled), wait_settled=bool(carbon.wait_settled)))
            return risk, carbon, True, False
        finally:
            self._estimation_seconds_this_slot += time.perf_counter() - started

    def _features(self, risk, carbon):
        values = [bounded(risk.safe_wait_budget_minutes, 240.),
                       bounded(risk.extra_tardiness_minutes, 240.),
                       bounded(carbon.mean_carbon_gain_kg, 100.),
                       bounded(carbon.std_carbon_gain_kg, 100.),
                       bounded(carbon.robust_carbon_gain_kg, 100.)]
        if self.v3_config.get('variant') == 'no_queue_risk':
            values[:2] = [0., 0.]
        if self.v3_config.get('variant') == 'no_carbon_features':
            values[2:] = [0., 0., 0.]
        return vector(values)

    def task_observation(self, spec=None, controller=None):
        original = super().task_observation(spec, controller)
        extra = self._active_features if spec is not None else np.zeros(5, dtype=np.float32)
        return np.concatenate((original, extra)).astype(np.float32)

    def _global_state(self, pending=None):
        pool, sample = self.executor.pool, self.executor.sample
        if pending is None:
            pending = tuple(self.executor.ledger[t.job_name]['spec'] for t in pool.pending_tasks)
        pending = sorted(pending, key=lambda s: (s.sla_deadline, s.arrival_time, s.task_id))
        used = pool.inventory()
        predicted = (np.asarray(self._forecast.mean) if self._forecast is not None
                     else np.full(16, float(sample.ci_g_per_kwh)))
        if predicted.shape != (16,):
            raise ValueError('V3 centralized state requires a frozen sixteen-slot forecast')
        last = self.bridge.last
        context = vector([*[v/c for v, c in zip(used, self.capacity)],
            bounded(len(pending), 1000.), bounded(sample.ci_g_per_kwh, 1000.),
            *[bounded(value, 1000.) for value in predicted],
            bounded(sample.temperature_c, 40.), bounded(sample.wet_bulb_c, 40.),
            bounded(self.physical.setpoint_c, 40.),
            bounded(last.it_kw if last else 0., 1000.), bounded(last.hvac_kw if last else 0., 1000.)])
        queue = np.asarray([[*[getattr(s, n)/c for n, c in zip(
            ('cores_req', 'gpu_req', 'mem_req'), self.capacity)], bounded(s.duration, 240.),
            bounded((latest_start(s)-sample.time).total_seconds()/60., 240.)]
            for s in pending[:256]], dtype=np.float32).reshape((-1, 5))
        return dict(context=context, queue=queue)

    def step(self, task_actor=None, cooling_actor=None):
        result = super().step(task_actor, cooling_actor)
        actions = result.task_actions
        diagnostics = dict(fine_estimation_count=sum(a['fine_estimation'] for a in actions),
            fast_approximation_count=sum(a['fast_approximation'] for a in actions),
            unestimated_count=sum(a['unestimated'] for a in actions),
            fine_carbon_estimation_count=sum(a['fine_carbon_estimation'] for a in actions),
            carbon_unestimated_count=sum(a['carbon_unestimated'] for a in actions),
            risk_unreliable_count=sum(a['risk_unreliable'] for a in actions),
            wait_masked_count=sum(a['wait_masked'] for a in actions),
            start_masked_count=sum(a['start_masked'] for a in actions),
            estimator_seconds=self._estimation_seconds_this_slot,
            ci_forecast_mean=np.asarray(self._forecast.mean, dtype=float).tolist(),
            ci_forecast_valid=bool(self._forecast.valid),
            ci_forecast_confidence=float(self._forecast.confidence), ci_forecast_reason=self._forecast.reason)
        self.records[-1].update(diagnostics)
        return replace(result, interval=dict(result.interval, **diagnostics))

    def summary(self):
        result = super().summary()
        actions = [a for row in self.action_records for a in row['task_actions']]
        for key in ('fine_estimation', 'fast_approximation', 'unestimated', 'fine_carbon_estimation',
                    'carbon_unestimated', 'risk_unreliable', 'wait_masked', 'start_masked'):
            result[key + '_count'] = sum(bool(a[key]) for a in actions)
        reliable = [a for a in actions if a['teacher_reliable'] and a['teacher_weight'] > 0]
        result.update(protocol='joint-rl-v3-rac-mappo-v1', task_observation_dim=40,
            teacher_action_distribution=dict(Counter(str(a['teacher_action']) for a in reliable)),
            teacher_reliable_count=len(reliable), teacher_estimate_failure_count=sum(not a['teacher_reliable'] for a in actions),
            actor_teacher_agreement=(sum(a['actor_teacher_agreement'] for a in reliable)/len(reliable) if reliable else None),
            unique_policy_deferred_tasks=len({a['task_id'] for a in actions if a['action'] == 1}),
            estimator_runtime_ms=math.fsum(a['risk_runtime_ms']+a['estimator_runtime_ms'] for a in actions),
            carbon_cache_hit_rate=self._carbon.cache_hit_rate,
            predicted_safe_wait_tasks=len(self._predicted_safe_wait_ids))
        free = [a for a in actions if all(a['mask'])]
        result.update(risk_reason_distribution=dict(Counter(a['risk_reason'] for a in actions)),
            carbon_reason_distribution=dict(Counter(a['carbon_reason'] for a in actions)),
            reliable_risk_count=sum(a['risk_estimation_valid'] for a in actions),
            reliable_carbon_count=sum(a['counterfactual_valid'] for a in actions),
            free_action_count=len(free), active_wait_count=sum(a['action'] == 1 for a in free),
            free_action_distribution=dict(Counter(str(a['action']) for a in free)),
            active_wait_distribution=dict(Counter(a['risk_reason'] for a in free if a['action'] == 1)),
            active_wait_fraction=sum(a['action'] == 1 for a in free)/len(free) if free else 0.)
        result['predicted_safe_but_missed_tasks'] = sum(
            (entry['task'].finish_time > entry['spec'].sla_deadline if entry['task'].finish_time is not None
             else entry['spec'].sla_deadline < self.executor.observed_end)
            for name, entry in self.executor.ledger.items() if name in self._predicted_safe_wait_ids)
        from .mechanism_audit import forecast_errors, audit_probes
        result['ci_forecast_error'] = forecast_errors(self.records)
        if self.evaluation:
            result['mechanism_audit'] = audit_probes(self)
        return result

    def close(self):
        super().close()
        if self._carbon is not None:
            self._carbon.close()
