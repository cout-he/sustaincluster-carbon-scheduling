"""Deterministic, common-endpoint comparison on guarded frozen Validation."""
import argparse
import csv
import json
import math
from pathlib import Path

from experiments.e1.contracts import content_hash, file_hash, write_json

from . import ACCOUNTING
from .inputs import build_env, load_joint_bundle, resolve_config
from .train import _json_line, _positive, source_provenance


MODES = ('B0', 'B1', 'B2', 'B3', 'RiskOnly')
POLICIES = {'B1': ('task',), 'B2': ('cooling',), 'B3': ('task', 'cooling')}


def feasibility(summary):
    """Operational gate; incomplete/censored work never counts as feasible."""
    try:
        dmr = float(summary['dmr'])
        temperature = float(summary['temperature_violations'])
        unfinished = summary.get('unfinished_tasks', summary.get('unfinished'))
        if unfinished is None:
            unfinished = summary['pending'] + summary['running']
        censored = summary.get('censored_tasks', summary.get('deadline_right_censored'))
        return bool(math.isfinite(dmr) and 0 <= dmr <= .05 and temperature == 0
                    and unfinished == 0 and censored == 0)
    except (KeyError, TypeError, ValueError):
        return False


def validate_metadata(folder, mode, config, *, metadata=None):
    """Validate role and byte identities BEFORE torch checkpoint loading."""
    from .env import TASK_FIELDS, COOLING_FIELDS

    root = Path(folder)
    meta = metadata if metadata is not None else json.loads((root / 'metadata.json').read_text(encoding='utf8'))
    if meta.get('training_role') != 'train' or meta.get('input_identity', {}).get('split') != 'train':
        raise ValueError('Checkpoint provenance must identify Train-only fitting role')
    if mode not in POLICIES or meta.get('mode') != mode:
        raise ValueError('Checkpoint mode provenance differs from requested mode')
    if meta.get('accounting') != ACCOUNTING or meta.get('config_sha256') != content_hash(config):
        raise ValueError('Checkpoint accounting or config provenance differs')
    if meta.get('test_access') is not False or meta['input_identity'].get('test_evaluation') is not False:
        raise ValueError('Checkpoint must preserve the sealed Test rule')
    if (meta.get('task_observation_fields') != list(TASK_FIELDS)
            or meta.get('cooling_observation_fields') != list(COOLING_FIELDS)):
        raise ValueError('Checkpoint observation schema differs')
    expected = {name + '.pt' for name in POLICIES[mode]}
    if set(meta.get('checkpoint_sha256', {})) != expected:
        raise ValueError('Checkpoint inventory differs for mode ' + mode)
    for filename, sha in meta['checkpoint_sha256'].items():
        if file_hash(root / filename) != sha:
            raise ValueError('Checkpoint byte hash differs: ' + filename)
    for filename, key in (('training_config.json', 'training_config_sha256'),
                          ('provenance.json', 'provenance_sha256')):
        if file_hash(root / filename) != meta.get(key):
            raise ValueError('Checkpoint provenance file differs: ' + filename)
    training = json.loads((root / 'training_config.json').read_text(encoding='utf8'))
    provenance = json.loads((root / 'provenance.json').read_text(encoding='utf8'))
    if (content_hash(training['protocol']) != meta['config_sha256']
            or provenance['input_identity'] != meta['input_identity']
            or provenance['config_sha256'] != meta['config_sha256']
            or provenance.get('training_role') != 'train'
            or provenance.get('accounting') != ACCOUNTING):
        raise ValueError('Checkpoint provenance content differs')
    if training['execution']['mode'] != mode or training['execution']['smoke'] != meta.get('smoke'):
        raise ValueError('Checkpoint execution mode or smoke provenance differs')
    # Reuse the guarded original loader rather than trusting a label in JSON.
    original = load_joint_bundle(meta['train_bundle'], 'train')
    if original.identity != meta['input_identity']:
        raise ValueError('Checkpoint Train input identity differs from frozen bundle')
    return meta


def _accounting_audit(summary, intervals):
    from .power import COMPONENTS, HVAC_COMPONENTS

    for row in intervals:
        component_power = math.fsum(row[name + '_power_kw'] for name in COMPONENTS)
        component_energy = math.fsum(row[name + '_energy_kwh'] for name in COMPONENTS)
        hvac = math.fsum(row[name + '_energy_kwh'] for name in HVAC_COMPONENTS)
        carbon = math.fsum(row[name + '_carbon_kg'] for name in COMPONENTS)
        expected = (component_power, component_energy, hvac, carbon)
        actual = (row['total_power_kw'], row['total_energy_kwh'], row['hvac_energy_kwh'], row['carbon_kg'])
        if any(not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8) for a, b in zip(actual, expected)):
            raise RuntimeError('Joint electrical component accounting does not reconcile')
        if row.get('accounting') != ACCOUNTING or row['interval_hours'] != .25:
            raise RuntimeError('Incorrect accounting boundary or physical clock')
    if not math.isclose(summary['carbon_kg'], math.fsum(r['carbon_kg'] for r in intervals),
                        rel_tol=1e-9, abs_tol=1e-8):
        raise RuntimeError('Summary carbon does not reconcile to actual intervals')


def evaluate(validation_bundle, models_root, output, *, config=None, max_episode_steps=None):
    from .ppo import IPPOAgent

    cfg = resolve_config(config)
    validation = load_joint_bundle(validation_bundle, 'validation')
    if max_episode_steps is not None:
        _positive(max_episode_steps, 'max_episode_steps')
    model_root = Path(models_root)
    metadata, agents = {}, {}
    train_identity = None
    for mode in POLICIES:
        folder = model_root / mode
        meta = validate_metadata(folder, mode, cfg)
        if train_identity is None:
            train_identity = meta['input_identity']
        elif train_identity != meta['input_identity']:
            raise ValueError('All modes must use identical frozen Train input')
        if float(meta['input_identity']['factor']) != float(validation.factor):
            raise ValueError('Train and Validation pressure factors must match')
        if meta['input_identity']['raw_source_sha256'] != validation.identity['raw_source_sha256']:
            raise ValueError('Train and Validation original source identities differ')
        metadata[mode] = meta
        agents[mode] = {name: IPPOAgent.load(folder / (name + '.pt')) for name in POLICIES[mode]}
        for name, agent in agents[mode].items():
            if agent.parameter_hash() != meta['final_parameter_hashes'].get(name):
                raise ValueError('Loaded checkpoint parameter identity differs')
    out = Path(output)
    if out.exists():
        raise FileExistsError('Refusing to overwrite Validation output: ' + str(out))
    out.mkdir(parents=True)
    smoke = max_episode_steps is not None or any(meta['smoke'] for meta in metadata.values())
    write_json(out / 'provenance.json', dict(source_provenance(), accounting=ACCOUNTING,
        validation_bundle=str(Path(validation_bundle).resolve()), input_identity=validation.identity,
        train_input_identity=train_identity, config_sha256=content_hash(cfg),
        checkpoint_metadata_sha256={mode: file_hash(model_root / mode / 'metadata.json') for mode in POLICIES},
        deterministic=True, test_access=False, smoke=smoke))
    results, fairness = {}, {}
    common_end, common_tasks, common_resource_hours = None, None, None
    for mode in MODES:
        folder = out / mode
        folder.mkdir()
        env = build_env(validation, validation_bundle, cfg, mode, max_episode_steps)
        mode_agents = agents.get(mode, {})
        callbacks = {name: (lambda observation, mask, policy=agent:
                           policy.act(observation, mask, deterministic=True))
                     for name, agent in mode_agents.items()}
        try:
            env.reset(seed=int(cfg['seed']))
            with (folder / 'intervals.jsonl').open('w', encoding='utf8') as stream:
                while True:
                    outcome = env.step(callbacks.get('task'), callbacks.get('cooling'))
                    _json_line(stream, dict(outcome.interval, reward=outcome.reward,
                        reward_terms=outcome.terms, task_actions=outcome.task_actions,
                        cooling_actions=outcome.cooling_actions))
                    if outcome.done:
                        break
            summary = dict(env.summary(), input_identity=validation.identity,
                           config_sha256=content_hash(cfg), deterministic=True, smoke=smoke)
            summary['feasible'] = feasibility(summary)
            execution = env.executor.result()
            tasks = list(execution.tasks)
            _accounting_audit(summary, env.records)
            task_ids = sorted(row['task_id'] for row in tasks)
            hours = {name: math.fsum(row['used_' + name] * row['interval_hours'] for row in env.records)
                     for name in ('cores', 'gpus', 'mem')}
            if common_end is None:
                common_end, common_tasks, common_resource_hours = summary['observation_end'], task_ids, hours
            elif summary['observation_end'] != common_end or task_ids != common_tasks:
                raise RuntimeError('Compared methods do not share endpoint and arrived task cohort')
            resource_equal = all(math.isclose(hours[k], common_resource_hours[k], rel_tol=1e-9, abs_tol=1e-7)
                                 for k in hours)
            if mode != 'B0' and not smoke and summary['unfinished'] == 0 and results['B0']['unfinished'] == 0 and not resource_equal:
                raise RuntimeError('Completed finite workloads have unequal resource hours')
            fairness[mode] = dict(common_endpoint=True, identical_arrived_task_ids=True,
                                  resource_hours=hours, resource_hours_equal_to_B0=resource_equal,
                                  completed=summary['completed'], unfinished=summary['unfinished'],
                                  censored=summary['deadline_right_censored'], accounting_reconciled=True)
            write_json(folder / 'tasks.json', tasks)
            write_json(folder / 'policy_tasks.json', list(execution.policy_tasks))
            write_json(folder / 'summary.json', summary)
            results[mode] = summary
            print(json.dumps(dict(mode=mode, phase='validation', carbon_kg=summary['carbon_kg'],
                                  dmr=summary['dmr'], feasible=summary['feasible'], smoke=smoke)), flush=True)
        finally:
            env.close()
    candidates = {mode: dict(eligible=not smoke and results[mode]['feasible'],
                             operational_gate=results[mode]['feasible'], carbon_kg=results[mode]['carbon_kg'],
                             dmr=results[mode]['dmr'], smoke=smoke,
                             checkpoint_sha256=metadata[mode]['checkpoint_sha256']) for mode in POLICIES}
    eligible = [mode for mode, row in candidates.items() if row['eligible']]
    selected = min(eligible, key=lambda mode: (results[mode]['carbon_kg'], results[mode]['dmr'], mode)) if eligible else None
    selection = dict(status='smoke_only' if smoke else ('eligible_candidates' if eligible else 'no_feasible_candidate'),
                     selected_mode=selected, candidates=candidates, split='validation',
                     gate='DMR <= 0.05; zero temperature violations; zero unfinished/censored tasks',
                     interpretation='one final candidate per mode; operational gate, no safety guarantee', test_access=False)
    comparison = dict(schema='joint-five-mode-validation-v1', accounting=ACCOUNTING, split='validation',
                      smoke=smoke, results=results, selection=selection, fairness=fairness,
                      common_endpoint=common_end, test_access=False)
    write_json(out / 'selection.json', selection)
    write_json(out / 'fairness.json', fairness)
    write_json(out / 'comparison.json', comparison)
    columns = ['mode', 'carbon_kg', 'total_energy_kwh', 'it_energy_kwh', 'hvac_energy_kwh',
               'dmr', 'deadline_misses', 'temperature_violations', 'max_rack_inlet_c',
               'completed', 'unfinished', 'deadline_right_censored', 'waiting_mean_minutes',
               'jct_completed_mean_minutes', 'feasible', 'smoke']
    with (out / 'comparison.csv').open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: summary.get(key) for key in columns} for summary in results.values())
    return comparison


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument('--validation-bundle', required=True)
    command.add_argument('--models-root', required=True)
    command.add_argument('--output', required=True)
    command.add_argument('--config')
    command.add_argument('--max-episode-steps', type=int, help='Smoke evaluation only')
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    evaluate(args.validation_bundle, args.models_root, args.output,
             config=args.config, max_episode_steps=args.max_episode_steps)


if __name__ == '__main__':
    main()
