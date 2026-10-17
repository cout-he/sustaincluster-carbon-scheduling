"""Finite complete-episode MAPPO transactions; setup and selection share wall budget."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import time
import numpy as np
from experiments.e1.contracts import content_hash, file_hash
from experiments.joint_v2.scenarios import load_pool
from experiments.joint_v2.scenario_sampler import ScenarioSampler
from experiments.joint_v2.selection import aggregate
from .checkpoint import Store, atomic_json, code_identity, global_rng_state, restore_global_rng
from .config import load_config
from .mappo import RACSystem
from .reward import GroupConstraints


def _snapshot(state, system, forecaster, sampler, constraints):
    state.update(system=system.export_state(), forecaster=forecaster.export_state(),
                 sampler=sampler.state_dict(), constraints=constraints.export_state(), global_rng=global_rng_state())
    return state


def collect(system, scenario, cfg, forecaster, *, deadline=None, remaining=None, deterministic=False):
    from .env import V3Env
    env = V3Env(scenario, cfg, forecaster, evaluation=deterministic)
    rows = []
    try:
        env.reset(seed=cfg['seed'])
        while not env.done:
            if ((deadline is not None and time.monotonic() >= deadline)
                    or (remaining is not None and len(rows) >= remaining)):
                return None, None, dict(physical_steps=len(rows), reason='incomplete_episode_rolled_back')
            out = env.step(*(lambda obs, mask, a=system.actors[role]:
                a.act(obs, mask, deterministic=deterministic) for role in ('task', 'cooling')))
            rows.append(dict(task_global=deepcopy(env.task_global_state),
                cooling_global=deepcopy(env.cooling_global_state),
                task_boundary=out.task_boundary_observation.copy(),
                cooling_boundary=out.cooling_boundary_observation.copy(),
                task_decisions=out.task_decisions, cooling_decisions=out.cooling_decisions,
                task_actions=out.task_actions, done=out.done, terms=out.terms))
        summary = env.summary()
        return rows, summary, None
    finally:
        env.close()


def development(system, scenarios, cfg, forecaster, deadline=None):
    result = []
    for scenario in scenarios:
        if scenario.role != 'development': raise ValueError('Model selection only permits Development')
        _, summary, partial = collect(system, scenario, cfg, forecaster, deadline=deadline, deterministic=True)
        if partial: return None
        result.append(summary)
    return result


def run(pool, output, *, config=None, resume=False, scenarios=None):
    import torch
    from .ci_forecaster import CIForecaster
    started = time.monotonic()
    cfg = load_config(config)
    out = Path(output)
    if out.exists() and not resume: raise FileExistsError('Use explicit --resume for an existing run')
    out.mkdir(parents=True, exist_ok=True)
    # Roles are restricted before any scenario payload is opened by the V2 loader.
    train, dev = scenarios if scenarios is not None else (load_pool(pool, ('train',)), load_pool(pool, ('development',)))
    if not train or not dev or any(s.role != 'train' for s in train) or any(s.role != 'development' for s in dev):
        raise ValueError('Need nonempty Train/Development sets')
    dev = sorted(dev, key=lambda s: s.scenario_id)[:cfg['validation']['max_development_scenarios']]
    data = content_hash(dict(pool_sha256=file_hash(Path(pool)/'manifest.json') if pool else None,
        train=[s.identity for s in train], development=[s.identity for s in dev]))
    code = code_identity()
    store = Store(out/'checkpoints', content_hash(cfg), data, code['sha256'])
    sampler = ScenarioSampler(train, cfg['seed'])
    if resume:
        state = store.load(); store.restore_aliases(state)
        system = RACSystem.from_state(state['system'])
        forecaster = CIForecaster.from_state(state['forecaster'])
        constraints = GroupConstraints.from_state(state['constraints'])
        sampler.load_state_dict(state['sampler']); restore_global_rng(state['global_rng'])
    else:
        atomic_json(out/'config.json', cfg)
        atomic_json(out/'identity.json', dict(config_identity=content_hash(cfg), data_identity=data,
            code_identity=code['sha256'], code_manifest=code['files'], test_access=False))
        random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
        system = RACSystem(cfg['seed'], cfg['ppo'], cfg['variant'] != 'independent_critic')
        forecaster = CIForecaster(**cfg['forecast']).fit(train)
        constraints = GroupConstraints(cfg['constraint'])
        state = dict(schema=cfg['schema'], config=cfg, elapsed_seconds=0., completed_episodes=0,
            physical_steps=0, episodes=[], validations=[], best_feasible=None, best_deadline=None,
            curriculum=dict(stage=2, success_streak=0), pretraining=None, test_access=False)
        store.save(_snapshot(state, system, forecaster, sampler, constraints))
    previous_elapsed = state['elapsed_seconds']
    deadline = started+max(0., cfg['budget']['max_wall_seconds']-previous_elapsed)
    stop, partial = 'step_budget', None
    try:
        if state['pretraining'] is None:
            from .pretraining import run_pretraining
            state['pretraining'] = run_pretraining(system, train, cfg, forecaster, out/'pretraining', deadline=deadline)
            if time.monotonic() >= deadline:
                stop = 'wall_budget_during_pretraining_rolled_back'
            else:
                state['elapsed_seconds'] = previous_elapsed+time.monotonic()-started
                store.save(_snapshot(state, system, forecaster, sampler, constraints))
        while time.monotonic() < deadline:
            remaining = cfg['budget']['total_physical_steps']-state['physical_steps']
            if remaining <= 0: break
            if cfg['budget'].get('max_episodes') is not None and state['completed_episodes'] >= cfg['budget']['max_episodes']:
                stop = 'episode_budget'; break
            scenario = sampler.sample(stage=2)
            rows, summary, partial = collect(system, scenario, cfg, forecaster, deadline=deadline, remaining=remaining)
            if partial:
                stop = partial['reason']; break
            group = constraints.key(scenario.identity.get('resource_pressure_factor', 1.),
                                    scenario.identity.get('sla_multiplier', 1.5))
            teacher = cfg['teacher']
            weight = (teacher['initial_weight']*max(0., 1-state['physical_steps']/(
                cfg['budget']['total_physical_steps']*teacher['decay_fraction'])) if teacher['enabled'] else 0.)
            updates = system.update(rows, total_tasks=summary['arrived'], multiplier=constraints.values[group],
                                    auxiliary_weight=weight, deadline=deadline)
            if time.monotonic() >= deadline:
                stop = 'wall_budget_during_update_rolled_back'; break
            constraint = constraints.update(group, summary['dmr'], role='train',
                                            complete=summary['complete_episode_for_constraint'])
            state['physical_steps'] += len(rows); state['completed_episodes'] += 1
            record = dict(episode=state['completed_episodes'], physical_steps=state['physical_steps'],
                scenario_id=scenario.scenario_id, source_fragment_id=scenario.identity.get('source_fragment_id'),
                group=group, summary=summary, updates=updates, constraint=constraint,
                elapsed_seconds=previous_elapsed+time.monotonic()-started)
            state['episodes'].append(record)
            # Snapshot after all episode updates, before selection: no partial gradients can leak.
            state['elapsed_seconds'] = previous_elapsed+time.monotonic()-started
            store.save(_snapshot(state, system, forecaster, sampler, constraints))
            if state['completed_episodes'] == 1 or state['completed_episodes'] % cfg['validation']['every_episodes'] == 0:
                validation = development(system, dev, cfg, forecaster, deadline)
                if validation is not None:
                    score = aggregate(validation)
                    event = dict(episode=state['completed_episodes'], score=score, scenarios=validation)
                    state['validations'].append(event)
                    names = []
                    if state['best_deadline'] is None or score['deadline_key'] < state['best_deadline']['score']['deadline_key']:
                        state['best_deadline'] = event; names.append('best_deadline')
                    if score['feasible'] and (state['best_feasible'] is None or score['carbon_kg'] < state['best_feasible']['score']['carbon_kg']):
                        state['best_feasible'] = event; names.append('best_feasible')
                    if names:
                        pointer = store.save(_snapshot(state, system, forecaster, sampler, constraints), tuple(names))
                        for name in names: state[name+'_pointer'] = pointer
            state['elapsed_seconds'] = previous_elapsed+time.monotonic()-started
            store.save(_snapshot(state, system, forecaster, sampler, constraints))
            from .monitor import flush
            flush(out, state)
            print(json.dumps(dict(seed=cfg['seed'], variant=cfg['variant'], episode=state['completed_episodes'],
                physical_steps=state['physical_steps'], dmr=summary['dmr'], carbon_kg=summary['carbon_kg'],
                incremental_carbon_kg=summary['incremental_carbon_kg'], group=group,
                multiplier=constraints.values[group], seconds=state['elapsed_seconds'])), flush=True)
        else:
            stop = 'wall_budget'
    except KeyboardInterrupt:
        stop = 'interrupted_uncommitted_episode_rolled_back'
    # Rollback to immutable latest covers partial sampling, action RNGs and optimizers.
    committed = store.load(); store.restore_aliases(committed)
    committed['elapsed_seconds'] = previous_elapsed+time.monotonic()-started
    committed['last_run'] = dict(stop_reason=stop, partial_episode=partial, wall_seconds=time.monotonic()-started)
    store.save(committed, ('latest', 'final'))
    from .monitor import flush
    flush(out, committed)
    result = dict(schema=cfg['schema'], seed=cfg['seed'], variant=cfg['variant'],
        completed_episodes=committed['completed_episodes'], physical_steps=committed['physical_steps'],
        elapsed_seconds=committed['elapsed_seconds'], last_run=committed['last_run'],
        best_feasible=committed['best_feasible'], best_deadline=committed['best_deadline'],
        parameter_hashes={k: a.parameter_hash() for k, a in RACSystem.from_state(committed['system']).actors.items()},
        test_access=False)
    atomic_json(out/'run-result.json', result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool', required=True); p.add_argument('--output', required=True)
    p.add_argument('--config', required=True); p.add_argument('--resume', action='store_true')
    args = p.parse_args(argv)
    print(json.dumps(run(args.pool, args.output, config=args.config, resume=args.resume)), flush=True)


if __name__ == '__main__': main()
