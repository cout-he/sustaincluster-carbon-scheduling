"""Finite canonical-plant action labels and bounded cooling-only PPO.

The independent probe is an explicitly counterfactual labeling tool. It never
advances or mutates the live plant. The live trajectory remains one shared
physical environment, driven by real Train tasks and causal guarded EDF.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import replace

import numpy as np

from experiments.joint import ACCOUNTING
from experiments.joint.power import JointPowerModel

from .ppo import SlotTransition
from .pretrain_task import _deadline_expired, _edf_action, _positive, _train_scenarios, _write


def enumerate_cooling_actions(probe, loads, sample, setpoint, mask, cooling):
    """Evaluate legal 1-degree actions at identical CURRENT loads/weather."""
    records = []
    for action, legal in enumerate(mask):
        if not legal:
            continue
        target = float(setpoint) + (action - 1) * cooling['action_step_c']
        probe.set_setpoint(target)
        reading = probe.measure(*loads, sample)
        records.append(dict(
            action=action, setpoint_c=target, total_kw=reading.total_kw,
            it_kw=reading.it_kw, hvac_kw=reading.hvac_kw,
            components_kw=dict(reading.components_kw),
            carbon_kg=reading.total_kw * .25 * float(sample.ci_g_per_kwh) / 1000,
            rack_inlet_max_c=reading.rack_inlet_max_c,
            thermal_excess_c=max(0., reading.rack_inlet_max_c - cooling['rack_inlet_limit_c']),
        ))
    if not records:
        raise ValueError('Cooling enumeration requires a legal action')
    safe = [record for record in records if record['thermal_excess_c'] == 0.]
    pool = safe or records
    best = min(pool, key=lambda row: (row['thermal_excess_c'], row['total_kw'], row['action'] != 1))
    ranked_power = sorted(row['total_kw'] for row in pool)
    margin = ranked_power[1] - ranked_power[0] if len(ranked_power) > 1 else 0.
    return int(best['action']), records, float(margin)


def _physics_summary(samples):
    directions, distribution, setpoints = Counter(), Counter(), Counter()
    spreads, margins = [], []
    for sample in samples:
        ordered = sorted(sample['actions'], key=lambda row: row['setpoint_c'])
        powers = np.asarray([row['total_kw'] for row in ordered])
        delta = np.diff(powers)
        tolerance = 1e-9 * max(1., float(np.max(np.abs(powers))))
        if np.all(np.abs(delta) <= tolerance):
            direction = 'flat'
        elif np.all(delta >= -tolerance):
            direction = 'nondecreasing'
        elif np.all(delta <= tolerance):
            direction = 'nonincreasing'
        else:
            direction = 'nonmonotonic'
        directions[direction] += 1
        distribution[str(sample['action'])] += 1
        setpoints[str(sample['current_setpoint_c'])] += 1
        spreads.append(float(powers.max() - powers.min()))
        margins.append(sample['power_margin_kw'])
    dominant = max(distribution.values(), default=0) / len(samples) if samples else 0.
    return dict(monotonic_direction_counts=dict(directions), optimal_action_distribution=dict(distribution),
                dominant_optimal_action_fraction=dominant, current_setpoint_distribution=dict(setpoints),
                power_spread_max_kw=max(spreads, default=0.),
                power_spread_mean_kw=float(np.mean(spreads)) if spreads else 0.,
                power_margin_mean_kw=float(np.mean(margins)) if margins else 0.,
                interpretation_scope='legal local actions at each observed current setpoint; no global-optimality claim')


def pretrain_cooling(agent, scenarios, config, output=None, *, deadline=None):
    options = config.get('pretrain', {}).get('cooling', {})
    selected = _train_scenarios(scenarios, options.get('demo_scenarios', 2))
    if not options.get('enabled', True):
        return dict(stage='cooling_pretrain_v2', enabled=False, reason='disabled',
                    budget_exhausted=_deadline_expired(deadline))
    maximum = _positive(options.get('max_states', 128), 'max_states')
    epochs = _positive(options.get('bc_epochs', 1), 'bc_epochs', allow_zero=True)
    maximum_updates = _positive(options.get('max_updates', 4), 'max_updates', allow_zero=True)
    ppo_steps = _positive(options.get('ppo_steps', 32), 'ppo_steps', allow_zero=True)
    ppo_epochs = _positive(options.get('ppo_epochs', 1), 'ppo_epochs')
    protocol = config['v1_protocol']
    from .env import V2Env

    samples, observed_states, demo_slots = [], 0, 0
    before = agent.component_hashes()
    identities = []
    budget_exhausted = _deadline_expired(deadline)
    per_scenario = int(np.ceil(maximum / len(selected)))
    for scenario in selected:
        if _deadline_expired(deadline):
            budget_exhausted = True
            break
        env = V2Env(scenario, config, evaluation=False)
        probe = JointPowerModel(scenario.datacenter, scenario.model_path, cooling_config=protocol['cooling'])
        retained_here = 0
        try:
            env.reset(seed=agent.seed)

            def demonstrator(observation, mask):
                nonlocal observed_states, retained_here
                observed_states += 1
                if len(samples) >= maximum or retained_here >= per_scenario:
                    return 1
                # The callback runs after current task scheduling and BEFORE
                # the live physical step. The first three features are actual
                # utilization fractions, not transformed proxies.
                loads = tuple(float(used) / float(capacity) for used, capacity in
                              zip(env.executor.pool.inventory(), env.capacity))
                sample = env.executor.sample
                setpoint = env.physical.setpoint
                action, records, margin = enumerate_cooling_actions(
                    probe, loads, sample, setpoint, mask, protocol['cooling'])
                samples.append(dict(
                    scenario_id=scenario.scenario_id, time=sample.time.isoformat(),
                    observation=observation.tolist(), mask=np.asarray(mask, dtype=bool).tolist(),
                    action=action, current_setpoint_c=float(setpoint), loads=list(loads),
                    temperature_c=float(sample.temperature_c), wet_bulb_c=float(sample.wet_bulb_c),
                    ci_g_per_kwh=float(sample.ci_g_per_kwh), actions=records,
                    power_margin_kw=margin, origin='real_trajectory_driven', accounting=ACCOUNTING,
                    source_trace_augmented=bool(scenario.statistics.get('synthetic', False)),
                    scenario_augmentation=deepcopy(scenario.statistics.get('augmentation', {})),
                ))
                retained_here += 1
                return action

            while True:
                if _deadline_expired(deadline):
                    budget_exhausted = True
                    break
                outcome = env.step(_edf_action, demonstrator)
                demo_slots += 1
                if outcome.done:
                    break
            identities.append(dict(scenario_id=scenario.scenario_id, identity=deepcopy(scenario.identity),
                                   complete=bool(env.done),
                                   unobserved_future_tasks=len(scenario.workload.tasks)-len(env.executor.ledger)))
        finally:
            env.close()
            probe.close()
    bc = agent.behavior_clone(samples, epochs=epochs, max_updates=maximum_updates, deadline=deadline)
    budget_exhausted = budget_exhausted or bc.get('budget_exhausted', False) or _deadline_expired(deadline)
    ppo = dict(optimizer_steps=0, actor_optimizer_steps=0, critic_optimizer_steps=0, physical_slots=0)
    if budget_exhausted:
        ppo['skipped_reason'] = 'budget_exhausted'
    if ppo_steps and not budget_exhausted:
        # Loads are exogenous to Cooling: exactly the real tasks executed by
        # guarded EDF. Temperature actions still affect the real IT+HVAC model
        # and the controller's own reachable setpoint state at later slots.
        env = V2Env(selected[0], config, evaluation=False, max_episode_steps=ppo_steps)
        trajectory = []
        old_config = agent.config
        try:
            env.reset(seed=agent.seed)
            while True:
                if _deadline_expired(deadline):
                    budget_exhausted = True
                    break
                outcome = env.step(_edf_action, agent.act)
                interval = outcome.interval
                reward_options = protocol['reward']
                carbon_cost = (reward_options.get('alpha', 1.) * interval['carbon_kg']
                               / reward_options['carbon_scale_kg'])
                excess = max(0., interval['rack_inlet_max_c'] - protocol['cooling']['rack_inlet_limit_c'])
                thermal_cost = (reward_options['thermal_weight'] * excess
                                / reward_options['thermal_scale_degree_steps'])
                boundary = outcome.cooling_boundary_observation
                trajectory.append(SlotTransition(boundary, outcome.cooling_decisions,
                                                  -carbon_cost-thermal_cost, outcome.done,
                                                  agent.value(boundary)))
                if outcome.done:
                    break
            budget_exhausted = budget_exhausted or _deadline_expired(deadline)
            if trajectory and trajectory[-1].done and not budget_exhausted:
                agent.config = replace(old_config, n_epochs=ppo_epochs)
                ppo = agent.update(trajectory, bootstrap_value=0., deadline=deadline)
                budget_exhausted = ppo.get('budget_exhausted', False)
            else:
                ppo['skipped_reason'] = 'budget_exhausted_before_complete_cooling_prefix'
            ppo.update(physical_slots=len(trajectory),
                       collection_complete=bool(trajectory and trajectory[-1].done))
        finally:
            agent.config = old_config
            env.close()
    summary = dict(
        stage='cooling_pretrain_v2', enabled=True, accounting=ACCOUNTING,
        sample_origin='real_trajectory_driven',
        synthetic_augmentation_samples=sum(sample['source_trace_augmented'] for sample in samples),
        independent_synthetic_load_samples=0,
        observed_states=observed_states, selected_states=len(samples),
        omitted_label_states=observed_states-len(samples),
        enumerated_actions=sum(len(sample['actions']) for sample in samples),
        scenarios=identities, demonstration_physical_slots=demo_slots,
        selection_rule='distinct original fragments first; first quota per scenario, capped overall; no task drops within observed prefix',
        budget_exhausted=budget_exhausted,
        all_selected_scenarios_complete=(len(identities) == len(selected)
                                        and all(row['complete'] for row in identities)),
        unprocessed_scenarios=len(selected)-len(identities),
        physics=_physics_summary(samples), bc=bc, ppo=ppo,
        ppo_objective='finite cooling-only real guarded-EDF load prefix: actual total carbon plus thermal cost',
        ppo_horizon_steps=ppo_steps, ppo_terminal_bootstrap=0.,
        ppo_task_completion_claim=False,
        normalization=deepcopy(protocol['normalization']),
        component_hashes_before=before, component_hashes_after=agent.component_hashes(),
        limitations='Local one-step power labels initialize the actor; they do not establish long-term optimality',
    )
    agent.provenance['cooling_pretrain'] = deepcopy(summary)
    _write(output, 'cooling', summary, samples)
    return summary
