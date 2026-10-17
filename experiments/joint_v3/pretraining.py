"""Real-trajectory Cooling DP initialization followed by reliable Task labels.

Each retained Cooling state was actually observed after scheduling Train jobs.
Enumerated counterfactual setpoints are labels, never extra independent states.
Demonstration collection may stop early at a sample cap and is never reported
as a successful training episode or used to update a constraint multiplier.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from experiments.joint.power import JointPowerModel
from experiments.joint_v2.checkpoint import atomic_json
from experiments.joint_v2.env import V2Env

from .env import V3Env


def _expired(deadline):
    return deadline is not None and time.monotonic() >= deadline


def _count(options, name, default, *, zero=False):
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise ValueError('Invalid bounded pretraining option: ' + name)
    return value


def cooling_dp_label(probe, loads, sample, setpoint_c, cooling, *, horizon=4):
    """Exact tiny DP, with frozen actual load/current weather/current CI.

    The original algebraic model has no thermal inertia. All five admissible
    setpoints are probed using identical physical inputs; the Bellman paths
    enforce the one-degree-per-slot actuator limit and rack inlet constraint.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 16:
        raise ValueError('Cooling DP requires a finite horizon between one and sixteen')
    if len(loads) != 3 or not np.isfinite(loads).all() or any(v < 0 or v > 1 for v in loads):
        raise ValueError('Cooling DP loads must be three physical utilization fractions')
    minimum, maximum = cooling['min_setpoint_c'], cooling['max_setpoint_c']
    if minimum != 18. or maximum != 22. or cooling['action_step_c'] != 1.:
        raise ValueError('Cooling DP preserves the frozen 18..22 C one-degree actuator')
    levels = tuple(float(z) for z in range(18, 23))
    if float(setpoint_c) not in levels:
        raise ValueError('Cooling DP starting setpoint is outside the physical domain')
    readings, costs, safe = {}, {}, {}
    original = probe.setpoint_c
    try:
        for level in levels:
            probe.set_setpoint(level)
            reading = probe.measure(*loads, sample)
            readings[level] = reading
            safe[level] = reading.rack_inlet_max_c <= cooling['rack_inlet_limit_c']
            costs[level] = reading.total_kw * .25 * float(sample.ci_g_per_kwh) / 1000.
    finally:
        probe.set_setpoint(original)
    memo = {}

    def solve(current, remaining):
        if remaining == 0:
            return 0., ()
        key = (current, remaining)
        if key not in memo:
            candidates = []
            for action in range(3):
                target = current + action - 1
                if target not in costs or not safe[target]:
                    continue
                later_cost, later_path = solve(target, remaining - 1)
                candidates.append((costs[target] + later_cost, action != 1, action,
                                   (target,) + later_path))
            if not candidates:
                raise ValueError('Cooling DP has no thermally safe continuation')
            best = min(candidates)
            memo[key] = best[0], best[3]
        return memo[key]

    total, path = solve(float(setpoint_c), horizon)
    action_costs = {}
    for action in range(3):
        target = float(setpoint_c) + action - 1
        if target in costs and safe[target]:
            action_costs[str(action)] = costs[target] + solve(target, horizon - 1)[0]
    best_level = min((z for z in levels if safe[z]), key=lambda z: (readings[z].total_kw, z))
    return dict(action=int(path[0] - float(setpoint_c) + 1), horizon=horizon,
        predicted_carbon_kg=total, best_path=list(path), action_costs_kg=action_costs,
        setpoint_power_kw={str(int(z)): readings[z].total_kw for z in levels},
        setpoint_safe={str(int(z)): bool(safe[z]) for z in levels},
        minimum_power_setpoint_c=best_level,
        continuation='frozen_actual_load_current_weather_current_ci',
        accounting='joint_full_hvac_v1')


def _ordered_train(scenarios):
    scenarios = list(scenarios)
    if not scenarios or any(s.role != 'train' for s in scenarios):
        raise ValueError('Physical/Task pretraining requires a nonempty Train-only pool')
    # Round-robin source fragments before adding further synthetic variants.
    groups = {}
    for scenario in sorted(scenarios, key=lambda s: s.scenario_id):
        fragment = scenario.identity.get('source_fragment_id',
            scenario.statistics.get('source_fragment_id', scenario.workload.start.isoformat()))
        groups.setdefault(str(fragment), []).append(scenario)
    result = []
    for index in range(max(map(len, groups.values()))):
        for fragment in sorted(groups):
            if index < len(groups[fragment]):
                result.append(groups[fragment][index])
    return result


def _identity(observation, scenario, sample, loads):
    # Date/source and the full actual physical input disambiguate observed
    # states. Changing only scenario ID cannot manufacture new state samples.
    payload = dict(time=sample.time.isoformat(), observation=list(map(float, observation)),
        loads=list(map(float, loads)), temperature=float(sample.temperature_c),
        wet_bulb=float(sample.wet_bulb_c), ci=float(sample.ci_g_per_kwh))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _actor_hashes(actors):
    return {name: actor.component_hashes() for name, actor in actors.items()}


def run_pretraining(system, train, cfg, forecaster, output, deadline=None):
    ordered = _ordered_train(train)  # Reject held-out roles before output or payload work.
    options = cfg.get('pretrain', {})
    cooling_options, task_options = options.get('cooling', {}), options.get('task', {})
    cooling_max = _count(cooling_options, 'max_states', 4096)
    task_max = _count(task_options, 'max_samples', 4096)
    cooling_scenes = ordered[:_count(cooling_options, 'demo_scenarios', 96)] if cooling_options.get('enabled', True) else []
    task_scenes = ordered[:_count(task_options, 'demo_scenarios', 4)] if task_options.get('enabled', True) else []
    horizon = _count(cooling_options, 'horizon', 4)
    actors = system.actors
    before = _actor_hashes(actors)
    samples = dict(cooling=[], task=[])
    seen = set()
    minimums, distributions = Counter(), Counter()
    cooling_observed = cooling_steps = cooling_completed = cooling_failed = 0
    cooling_collection, task_collection = [], []
    for scenario in cooling_scenes:
        if _expired(deadline) or len(samples['cooling']) >= cooling_max:
            break
        env = V2Env(scenario, cfg)
        probe = JointPowerModel(scenario.datacenter, scenario.model_path,
                               cooling_config=cfg['v1_protocol']['cooling'])
        try:
            env.reset(seed=cfg.get('seed', 20261017))

            def cooling_teacher(observation, mask):
                nonlocal cooling_observed, cooling_failed
                cooling_observed += 1
                loads = tuple(v/c for v, c in zip(env.executor.pool.inventory(), env.capacity))
                sample = env.executor.sample
                identity = _identity(observation, scenario, sample, loads)
                try:
                    label = cooling_dp_label(probe, loads, sample, env.physical.setpoint_c,
                        cfg['v1_protocol']['cooling'], horizon=horizon)
                except ValueError:
                    cooling_failed += 1
                    return 0 if mask[0] else 1
                action = label['action']
                if not bool(mask[action]):
                    raise RuntimeError('DP label violates the live physical action mask')
                if identity not in seen and len(samples['cooling']) < cooling_max:
                    seen.add(identity)
                    samples['cooling'].append(dict(observation=np.asarray(observation).tolist(),
                        mask=np.asarray(mask, dtype=bool).tolist(), weight=1.,
                        scenario_id=scenario.scenario_id, time=sample.time.isoformat(), state_sha256=identity,
                        loads=list(loads), current_setpoint_c=env.physical.setpoint_c, **label))
                    minimums[str(label['minimum_power_setpoint_c'])] += 1
                    distributions[str(action)] += 1
                return action

            while not env.done and len(samples['cooling']) < cooling_max and not _expired(deadline):
                env.step(lambda obs, mask: 0 if mask[0] else 1, cooling_teacher)
                cooling_steps += 1
            cooling_completed += int(env.done)
            cooling_collection.append(dict(scenario_id=scenario.scenario_id, complete=env.done,
                physical_steps=len(env.records), purpose='demonstration_collection_no_constraint_update'))
        finally:
            probe.close(); env.close()
    cooling_bc = actors['cooling'].behavior_clone(samples['cooling'],
        epochs=_count(cooling_options, 'bc_epochs', 4, zero=True),
        max_updates=_count(cooling_options, 'max_updates', 256, zero=True), deadline=deadline)

    task_observed = task_steps = task_completed = task_failures = 0
    task_distribution = Counter()
    for scenario in task_scenes:
        if _expired(deadline) or len(samples['task']) >= task_max:
            break
        env = V3Env(scenario, cfg, forecaster)
        try:
            env.reset(seed=cfg.get('seed', 20261017))

            def task_teacher(observation, mask):
                return env._current_teacher['action']

            def cooling_actor(observation, mask):
                actor = actors['cooling']
                return actor.act(observation, mask, deterministic=True) if hasattr(actor, 'act') else (0 if mask[0] else 1)

            while not env.done and len(samples['task']) < task_max and not _expired(deadline):
                outcome = env.step(task_teacher, cooling_actor)
                task_steps += 1
                for row in outcome.task_actions:
                    task_observed += 1
                    if row['teacher_weight'] <= 0 or not row['teacher_reliable']:
                        task_failures += 1
                        continue
                    if len(samples['task']) >= task_max:
                        break
                    samples['task'].append(dict(observation=row['observation'], mask=row['mask'],
                        action=row['teacher_action'], weight=row['teacher_weight'],
                        scenario_id=scenario.scenario_id, time=outcome.interval['interval_start'],
                        robust_carbon_gain_kg=row['robust_carbon_gain_kg'],
                        safe_wait_budget_minutes=row['safe_wait_budget_minutes']))
                    task_distribution[str(row['teacher_action'])] += 1
            task_completed += int(env.done)
            task_collection.append(dict(scenario_id=scenario.scenario_id, complete=env.done,
                physical_steps=len(env.records), purpose='demonstration_collection_no_constraint_update'))
        finally:
            env.close()
    task_bc = actors['task'].behavior_clone(samples['task'],
        epochs=_count(task_options, 'bc_epochs', 2, zero=True),
        max_updates=_count(task_options, 'max_updates', 64, zero=True), deadline=deadline)
    report = dict(schema='joint-v3-pretraining-v1', test_access=False,
        budget_exhausted=_expired(deadline), before_component_hashes=before,
        after_component_hashes=_actor_hashes(actors),
        cooling=dict(samples=len(samples['cooling']), target_states=cooling_max,
            distinct_observed_states=len(seen), observed_callback_states=cooling_observed,
            independent_state_shortfall=max(0, cooling_max-len(seen)),
            collection_physical_steps=cooling_steps, collection_complete_episodes=cooling_completed,
            collection_scenarios=cooling_collection, failures=cooling_failed,
            action_distribution=dict(distributions), minimum_power_setpoint_distribution=dict(minimums),
            structural_18_fraction=minimums.get('18.0', 0)/max(1, len(seen)),
            state_count_scope='unique_actual_observed_physical_inputs; augmented_traces_are_not_independent_realizations',
            continuation='four_slot_DP_frozen_actual_load_current_weather_current_CI', bc=cooling_bc),
        task=dict(samples=len(samples['task']), observed_candidates=task_observed,
            collection_physical_steps=task_steps, collection_complete_episodes=task_completed,
            collection_scenarios=task_collection, unreliable_or_forced_labels=task_failures,
            teacher_action_distribution=dict(task_distribution), bc=task_bc))
    if output is not None:
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out/'pretraining-samples.json', dict(schema='joint-v3-pretraining-samples-v1',
            source_role='train', test_access=False, **samples))
        atomic_json(out/'pretraining-report.json', report)
    return report
