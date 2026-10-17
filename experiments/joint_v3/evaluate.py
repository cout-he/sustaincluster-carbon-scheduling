"""Frozen Development comparison B0–B5; all runs use common physical window."""
import argparse
from copy import deepcopy
import csv
import json
from dataclasses import replace
from pathlib import Path
from experiments.e1.contracts import file_hash
from experiments.joint_v2.checkpoint import CheckpointStore, atomic_json
from experiments.joint_v2.env import V2Env
from experiments.joint_v2.ppo import IPPOAgent
from experiments.joint_v2.scenarios import load_pool
from experiments.joint.env import JointEnv
from experiments.joint.power import JointPowerModel
from .checkpoint import Store, code_identity
from .ci_forecaster import CIForecaster
from .config import load_config
from .env import V3Env
from .mappo import RACSystem


METHODS = ('B0', 'B1', 'B2', 'B3', 'B4', 'B5')


def scenario_matrix(scenarios):
    """Predeclared sensitivity augmentations; never independent measured cohorts."""
    result = []
    for scene in scenarios:
        for pressure in (.75, 1., 1.25):
            for sla in (1.5, 2.):
                tasks = tuple(replace(t, cores_req=t.cores_req*pressure, gpu_req=t.gpu_req*pressure,
                    mem_req=t.mem_req*pressure, sla_multiplier=sla) for t in scene.workload.tasks)
                workload = replace(scene.workload, tasks=tasks,
                    resource_multiplier=scene.workload.resource_multiplier*pressure, sla_multiplier=sla)
                identity = dict(scene.identity, resource_pressure_factor=pressure, sla_multiplier=sla,
                    sensitivity_parent=scene.scenario_id, augmentation_independence=False)
                stats = dict(scene.statistics, augmentation=dict(resource_pressure_factor=pressure, sla_multiplier=sla))
                result.append(replace(scene, scenario_id=scene.scenario_id+f'-eval-r{pressure:g}-s{sla:g}',
                                      workload=workload, identity=identity, statistics=stats))
    return result


def load_model(folder, alias='best_feasible', *, check_code=True):
    root = Path(folder)
    cfg = load_config(root/'config.json')
    meta = json.loads((root/'identity.json').read_text(encoding='utf8'))
    if check_code and meta['code_identity'] != code_identity()['sha256']:
        raise ValueError('Frozen trained source identity changed')
    store = Store(root/'checkpoints', meta['config_identity'], meta['data_identity'], meta['code_identity'])
    state = store.load(alias)
    return RACSystem.from_state(state['system']), CIForecaster.from_state(state['forecaster']), cfg, state


def load_v2(folder):
    root = Path(folder)
    pointer = json.loads((root/'best_feasible.json').read_text(encoding='utf8'))
    state = CheckpointStore(root, config_identity=pointer['config_identity'], data_identity=pointer['data_identity']).load('best_feasible')
    if state['completed_episodes'] != 52:
        raise ValueError('B3 must use frozen V2 episode52 best')
    return {k: IPPOAgent.from_state(v) for k, v in state['agents'].items()}


def immediate(obs, mask):
    return 0 if mask[0] else 1


def lower(obs, mask):
    return 0 if mask[0] else 1


def run_method(method, scenario, cfg, system=None, forecaster=None, v2=None):
    if method not in METHODS: raise ValueError('Unknown comparator')
    if method in ('B0', 'B1'):
        return run_fixed_immediate(method, scenario, cfg)
    execution_config = deepcopy(cfg)
    if method == 'B2':
        execution_config['environment']['deadline_guard_enabled'] = False
    env = V3Env(scenario, execution_config, forecaster, evaluation=True) if method in ('B4', 'B5') else V2Env(scenario, execution_config, evaluation=True)
    try:
        env.reset(seed=cfg['seed'])
        # Fixed18 baseline begins at18 and holds it; V1 historical initial22 remains unchanged.
        if method in ('B1', 'B4'): env.physical.set_setpoint(18.)
        task = ((lambda o, m: system.actors['task'].act(o, m, deterministic=True)) if method in ('B4', 'B5')
                else (lambda o, m: v2['task'].act(o, m, deterministic=True)) if method == 'B3' else immediate)
        cooling = ((lambda o, m: system.actors['cooling'].act(o, m, deterministic=True)) if method in ('B2', 'B5')
            else (lambda o, m: v2['cooling'].act(o, m, deterministic=True)) if method == 'B3' else (lambda *_: 1))
        while not env.done: env.step(task, cooling)
        summary = env.summary()
        summary.update(comparison_method=method, seed=cfg['seed'], variant=cfg['variant'],
                       baseline_cooling='fixed18_active_then_idle18' if method in ('B1', 'B4') else
                           'fixed22_active_then_shared_idle18' if method == 'B0' else 'trained_then_shared_idle18')
        # Preserve per-task distinction between voluntary deferral and executor capacity wait.
        deferred = [a['task_id'] for row in env.action_records for a in row['task_actions']
                    if a['action'] == 1 and all(a['mask'])]
        summary['actively_deferred_unique_tasks'] = len(set(deferred))
        summary['active_deferral_decisions'] = len(deferred)
        summary['actively_deferred_task_fraction'] = len(set(deferred))/max(1, summary['arrived'])
        return summary
    finally:
        env.close()


def run_fixed_immediate(method, scenario, cfg):
    """Fixed means fixed throughout the full window, including the idle tail."""
    factory = lambda: JointPowerModel(scenario.datacenter, scenario.model_path,
                                      cooling_config=cfg['v1_protocol']['cooling'])
    env = JointEnv(scenario.workload, scenario.environment, scenario.capacity, factory,
                    scenario.parameters, cfg['v1_protocol'], mode='B0')
    reference, cache = factory(), {}
    try:
        reference.set_setpoint(18.)
        env.reset(seed=cfg['seed'])
        env.physical.set_setpoint(22. if method == 'B0' else 18.)
        while not env.done: env.step()
        s = env.summary()
        background = 0.
        for sample in scenario.environment.samples:
            key = (float(sample.temperature_c), float(sample.wet_bulb_c))
            if key not in cache: cache[key] = reference.measure(0., 0., 0., sample).total_kw
            background += cache[key]*.25*sample.ci_g_per_kwh/1000
        s.update(comparison_method=method, scenario_id=scenario.scenario_id,
            scenario_role=scenario.role, seed=cfg['seed'], variant=cfg['variant'],
            fixed_window_complete=True, fixed_window_carbon_kg=s['carbon_kg'],
            background_carbon_kg=background, incremental_carbon_kg=s['carbon_kg']-background,
            complete_episode_for_constraint=bool(s['unfinished'] == 0 and s['deadline_right_censored'] == 0),
            actively_deferred_unique_tasks=0, active_deferral_decisions=0,
            actively_deferred_task_fraction=0., baseline_cooling='fixed_entire_operating_window',
            physical_steps=len(env.records), carbon_scope='fixed_operational_window')
        return s
    finally:
        env.close(); reference.close()


def evaluate(pool, checkpoint, output, *, role='development', methods=METHODS,
             alias='best_feasible', v2_checkpoint=None):
    if role != 'development':
        raise ValueError('Validation and Test are sealed in the Development entry')
    if any(m not in METHODS for m in methods): raise ValueError('Unknown comparison method')
    out = Path(output)
    if out.exists(): raise FileExistsError('Evaluation output is immutable')
    system, forecaster, cfg, state = load_model(checkpoint, alias)
    scenes = scenario_matrix(load_pool(pool, ('development',)))
    old = load_v2(v2_checkpoint) if 'B3' in methods else None
    out.mkdir(parents=True)
    atomic_json(out/'protocol.json', dict(role=role, methods=list(methods), config=cfg,
        pool_manifest_sha256=file_hash(Path(pool)/'manifest.json'), checkpoint_alias=alias,
        source_identity=code_identity()['sha256'], test_access=False,
        shared_idle_rule='all policies lower1C per slot to18 after same workload drains'))
    results = []
    for scene in scenes:
        for method in methods:
            summary = run_method(method, scene, cfg, system, forecaster, old)
            results.append(summary)
            atomic_json(out/'results.json', dict(results=results, complete=False, test_access=False))
            print(json.dumps(dict(method=method, scenario=scene.scenario_id, carbon_kg=summary['carbon_kg'], dmr=summary['dmr'])), flush=True)
    atomic_json(out/'results.json', dict(results=results, complete=True, test_access=False))
    keys = ('comparison_method', 'scenario_id', 'seed', 'variant', 'arrived', 'completed', 'dmr',
        'unfinished', 'deadline_right_censored', 'waiting_mean_minutes', 'jct_completed_mean_minutes',
        'actively_deferred_task_fraction', 'active_deferral_decisions', 'capacity_wait_intervals',
        'carbon_kg', 'incremental_carbon_kg', 'it_carbon_kg', 'hvac_carbon_kg',
        'total_energy_kwh', 'it_energy_kwh', 'hvac_energy_kwh', 'max_rack_inlet_c', 'temperature_violations')
    with (out/'comparison.csv').open('w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore'); writer.writeheader(); writer.writerows(results)
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True); p.add_argument('--role', default='development')
    p.add_argument('--methods', nargs='+', default=list(METHODS)); p.add_argument('--alias', default='best_feasible')
    p.add_argument('--v2-checkpoint')
    a = p.parse_args(argv)
    evaluate(a.pool, a.checkpoint, a.output, role=a.role, methods=a.methods, alias=a.alias, v2_checkpoint=a.v2_checkpoint)


if __name__ == '__main__': main()
