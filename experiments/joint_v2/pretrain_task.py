"""Guard-consistent EDF demonstrations without suppressing wait exploration."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import math
import time

import numpy as np

from experiments.e1.contracts import write_json


def _positive(value, name, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f'{name} must be a {"nonnegative" if allow_zero else "positive"} integer')
    return value


def _train_scenarios(scenarios, count):
    scenarios = list(scenarios)
    if not scenarios:
        raise ValueError('Pretraining requires at least one Train scenario')
    # Check every role BEFORE constructing any environment or model probe.
    if any(getattr(scenario, 'role', None) != 'train' for scenario in scenarios):
        raise ValueError('Pretraining accepts only Train role scenarios')
    count = _positive(count, 'demo_scenarios')
    # Pool order groups augmentations. Select distinct original task fragments
    # first rather than spending both demonstration slots on one source copy.
    selected, seen = [], set()
    for scenario in scenarios:
        fragment = scenario.identity.get('source_fragment_id', scenario.scenario_id)
        if fragment not in seen:
            selected.append(scenario)
            seen.add(fragment)
        if len(selected) == count:
            break
    return selected


def _deadline_expired(deadline):
    if deadline is not None and (isinstance(deadline, bool) or not math.isfinite(deadline)):
        raise ValueError('deadline must be None or a finite absolute monotonic time')
    return deadline is not None and time.monotonic() >= deadline


def _edf_action(observation, mask):
    # V2 reservation protection may force a feasible long task to wait.
    # Its forced label is not evidence of free-choice wait demonstrations.
    return 0 if mask[0] else 1


def _write(output, name, summary, samples):
    if output is None:
        return
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / (name + '_samples.json'), samples)
    write_json(output / (name + '_pretrain.json'), summary)


def pretrain_task(agent, scenarios, config, output=None, *, deadline=None):
    """Collect the same 35 actor features, masks, resources and current slack.

    The sample cap affects supervised storage only. Every task remains in the
    real executor. Scenarios run to their legitimate endpoint unless the shared
    wall-clock deadline stops collection; those records are explicit prefixes.
    Free-choice EDF labels ordinarily have only the start class; in that case
    BC is explicitly skipped, retaining the initialized stochastic actor.
    """
    options = config.get('pretrain', {}).get('task', {})
    selected = _train_scenarios(scenarios, options.get('demo_scenarios', 2))
    if not options.get('enabled', True):
        return dict(stage='task_pretrain_v2', enabled=False, bc_skipped=True, reason='disabled',
                    budget_exhausted=_deadline_expired(deadline))
    maximum = _positive(options.get('max_samples', 2048), 'max_samples')
    epochs = _positive(options.get('bc_epochs', 1), 'bc_epochs', allow_zero=True)
    maximum_updates = _positive(options.get('max_updates', 4), 'max_updates', allow_zero=True)
    threshold = float(options.get('min_wait_fraction', .05))
    if not np.isfinite(threshold) or not 0 <= threshold <= .5:
        raise ValueError('min_wait_fraction must lie in [0,.5]')
    from .env import V2Env

    samples, distribution, free_distribution = [], Counter(), Counter()
    observed, forced, physical_slots = 0, 0, 0
    before = agent.component_hashes()
    scenario_records = []
    budget_exhausted = _deadline_expired(deadline)
    for scenario in selected:
        if _deadline_expired(deadline):
            budget_exhausted = True
            break
        env = V2Env(scenario, config, evaluation=False)
        try:
            env.reset(seed=agent.seed)

            def demonstrator(observation, mask):
                nonlocal observed, forced
                action = _edf_action(observation, mask)
                observed += 1
                distribution[str(action)] += 1
                free = np.count_nonzero(mask) > 1
                if free:
                    free_distribution[str(action)] += 1
                else:
                    forced += 1
                if len(samples) < maximum:
                    normalized_slack = float(observation[9])
                    slack = normalized_slack * config['v1_protocol']['normalization']['minutes'] / (
                        1 - abs(normalized_slack)) if abs(normalized_slack) < 1 else None
                    samples.append(dict(
                        scenario_id=scenario.scenario_id, observation=observation.tolist(),
                        mask=np.asarray(mask, dtype=bool).tolist(), action=int(action),
                        available_resources=dict(cpu=float(env.executor.pool.available_cores),
                                                 gpu=float(env.executor.pool.available_gpus),
                                                 mem=float(env.executor.pool.available_mem)),
                        slack_minutes=slack, slack_normalized=normalized_slack,
                        free_choice=bool(free), origin='real_trajectory_driven',
                        source_trace_augmented=bool(scenario.statistics.get('synthetic', False)),
                        scenario_augmentation=deepcopy(scenario.statistics.get('augmentation', {})),
                    ))
                return action

            while True:
                if _deadline_expired(deadline):
                    budget_exhausted = True
                    break
                outcome = env.step(demonstrator, lambda observation, mask: 1)
                physical_slots += 1
                if outcome.done:
                    break
            scenario_records.append(dict(scenario_id=scenario.scenario_id,
                                         identity=deepcopy(scenario.identity),
                                         arrived=len(env.executor.ledger),
                                         endpoint=env.executor.observed_end.isoformat(),
                                         complete=bool(env.done),
                                         unobserved_future_tasks=len(scenario.workload.tasks)-len(env.executor.ledger)))
        finally:
            env.close()
    free_count = sum(free_distribution.values())
    wait_fraction = free_distribution.get('1', 0) / free_count if free_count else 0.
    diverse = (free_distribution.get('0', 0) > 0 and free_distribution.get('1', 0) > 0
               and wait_fraction >= threshold)
    # Apply the diversity check to retained labels too: an early storage cap
    # must not conceal that the actual supervised minibatches have one class.
    retained_actions = {sample['action'] for sample in samples if sample['free_choice']}
    budget_exhausted = budget_exhausted or _deadline_expired(deadline)
    skipped = budget_exhausted or not diverse or retained_actions != {0, 1}
    bc = dict(updates=0, samples=0, actor_changed=False, critic_changed=False)
    if not skipped:
        bc = agent.behavior_clone(samples, epochs=epochs, max_updates=maximum_updates, deadline=deadline)
        budget_exhausted = bc.get('budget_exhausted', False)
    summary = dict(
        stage='task_pretrain_v2', enabled=True, teacher='EDF with identical V2 deadline masks',
        bc_skipped=skipped,
        reason=('budget_exhausted' if budget_exhausted else 'insufficient_free_choice_wait_demonstrations'
                if skipped else 'bounded_masked_behavior_cloning'),
        observed_samples=observed, selected_samples=len(samples), omitted_samples=observed-len(samples),
        omitted_fraction=(observed-len(samples))/observed if observed else 0., executor_tasks_omitted=0,
        class_distribution=dict(distribution), free_choice_distribution=dict(free_distribution),
        forced_demonstrations=forced, free_choice_wait_fraction=wait_fraction, bc=bc,
        selection_rule='distinct original fragments first; first max_samples decisions; no task drops within observed prefix',
        budget_exhausted=budget_exhausted,
        all_selected_scenarios_complete=(len(scenario_records) == len(selected)
                                        and all(row['complete'] for row in scenario_records)),
        unprocessed_scenarios=len(selected)-len(scenario_records),
        physical_slots=physical_slots, scenarios=scenario_records, sample_origin='real_trajectory_driven',
        scenario_augmented_samples=sum(sample['source_trace_augmented'] for sample in samples),
        independent_synthetic_load_samples=0,
        normalization=deepcopy(config['v1_protocol']['normalization']),
        component_hashes_before=before, component_hashes_after=agent.component_hashes(),
        limitations='EDF initialization is neither carbon optimization nor a deadline guarantee',
    )
    agent.provenance['task_pretrain'] = deepcopy(summary)
    _write(output, 'task', summary, samples)
    return summary
