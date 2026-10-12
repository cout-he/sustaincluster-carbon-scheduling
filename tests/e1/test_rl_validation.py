"""Independent sealed evidence and numerical mode-selection validation."""
import importlib.util
import json
from dataclasses import replace

import pandas as pd
import pytest

from .helpers import START, KnownPower, task
from .test_temporal_run import fixture


def driver():
    assert importlib.util.find_spec('experiments.e1.rl_validation') is not None, (
        'Independent E1-5A validation driver is not implemented')
    from experiments.e1 import rl_validation
    return rl_validation


def sealed(tmp_path):
    from experiments.e1.contracts import content_hash, file_hash, write_json
    archive = tmp_path/'archive'
    sources = {'experiments/e1/engine.py': b'old kernel\n',
               'experiments/e1/cars.py': b'unchanged policy\n'}
    for name, payload in sources.items():
        target = archive/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    (archive/'science.json').write_text('{"arrived":1}\n', encoding='utf-8')
    write_json(archive/'runtime.json', {'excluded_from_scientific_hashes': True})
    write_json(archive/'run_status.json', {'state': 'complete'})
    hashes = {name: file_hash(archive/'source'/name) for name in sources}
    provenance = dict(sources=hashes, source_sha256=content_hash(hashes),
        output_hashes={'science.json': file_hash(archive/'science.json')},
        runtime_hashes={'runtime.json': file_hash(archive/'runtime.json')},
        excluded_outputs=['runtime.json', 'run_status.json', 'provenance.json', 'source/**'])
    write_json(archive/'provenance.json', provenance)
    current = {**hashes, 'experiments/e1/engine.py': 'intentionally-refactored-kernel',
               'experiments/e1/rl_validation.py': 'new-validation-source'}
    return archive, provenance, current


def test_sealed_archive_accepts_only_declared_kernel_change_and_new_sources(tmp_path):
    # A blanket current-source equality check would reject the authorized refactor.
    module = driver()
    archive, _, current = sealed(tmp_path)
    evidence = module.verify_sealed_archive(archive, {'science.json'}, current_sources=current)
    assert evidence['changed_original_sources'] == ['experiments/e1/engine.py']
    assert evidence['new_sources'] == ['experiments/e1/rl_validation.py']
    assert evidence['scientific_files_checked'] == 1
    appended = ['experiments/e1/feasibility_ablations.py', 'experiments/e1/feasibility_experiment.py',
        'experiments/e1/diagnose_feasibility.py', 'experiments/e1/feasibility_intervention.py',
        'tests/e1/test_feasibility_ablations.py', 'tests/e1/test_feasibility_experiment.py',
        'tests/e1/test_diagnose_feasibility.py', 'tests/e1/test_feasibility_intervention.py',
        'experiments/e1/rl_env.py', 'experiments/e1/rl_validation.py',
        'tests/e1/test_stepwise.py', 'tests/e1/test_rl_env.py', 'tests/e1/test_rl_validation.py']
    current.update({name: 'declared appended source' for name in appended})
    evidence = module.verify_sealed_archive(archive, {'science.json'}, current_sources=current)
    assert evidence['new_sources'] == sorted(appended)


@pytest.mark.parametrize('damage', ['old_policy', 'unlisted_new', 'snapshot', 'snapshot_extra',
                                 'source_manifest', 'science', 'output_manifest', 'runtime'])
def test_sealed_archive_rejects_source_and_science_damage(tmp_path, damage):
    # Catches accepting an altered old policy, damaged archive, or incomplete manifest.
    module = driver()
    archive, provenance, current = sealed(tmp_path)
    if damage == 'old_policy':
        current['experiments/e1/cars.py'] = 'changed'
    elif damage == 'unlisted_new':
        current['experiments/e1/unapproved.py'] = 'new'
    elif damage == 'snapshot':
        (archive/'source/experiments/e1/engine.py').write_text('tampered')
    elif damage == 'snapshot_extra':
        (archive/'source/extra.py').write_text('unlisted')
    elif damage == 'source_manifest':
        provenance['source_sha256'] = 'corrupt'
    elif damage == 'science':
        (archive/'science.json').write_text('{"arrived":2}')
    elif damage == 'runtime':
        (archive/'runtime.json').write_text('{}')
    else:
        provenance['output_hashes'] = {}
    (archive/'provenance.json').write_text(json.dumps(provenance), encoding='utf-8')
    with pytest.raises(ValueError):
        module.verify_sealed_archive(archive, {'science.json'}, current_sources=current)


def test_result_comparison_catches_interval_physics_and_ledger_changes(tmp_path):
    # Catches a comparator that checks only aggregate carbon or completion counts.
    module = driver()
    from experiments.e1.engine import LocalExecutor
    from experiments.e1.run import _write_result
    workload, environment = fixture([task('a')])
    result = LocalExecutor(workload, environment, (1, 1, 1), KnownPower(), drain=True).run()
    archive = tmp_path/'old'
    archive.mkdir()
    _write_result(archive, result)
    report = module.compare_archived_result(result, archive, tmp_path/'matched')
    assert report['all_files_byte_identical'] and report['files_checked'] == 8
    assert report['resource_hours'] == {'cores': 0.25, 'gpus': 0.0, 'mem': 0.25}
    damaged_step = {**result.steps[0], 'it_power_kw': result.steps[0]['it_power_kw'] + 1}
    damaged = replace(result, steps=(damaged_step, *result.steps[1:]))
    with pytest.raises(RuntimeError, match='intervals.csv'):
        module.compare_archived_result(damaged, archive, tmp_path/'physics-changed')
    row = {**result.tasks[0], 'start_time': None}
    with pytest.raises(RuntimeError, match='ledger.jsonl'):
        module.compare_archived_result(replace(result, tasks=(row,)), archive, tmp_path/'ledger-changed')


def test_interval_comparison_accepts_only_zero_formatting_and_rejects_tiny_numeric_change(tmp_path):
    # Catches hiding real CSV differences with floating tolerance or misreporting byte equality.
    module = driver()
    from experiments.e1.engine import LocalExecutor
    from experiments.e1.run import _write_result
    workload, environment = fixture([task('a')])
    result = LocalExecutor(workload, environment, (1, 1, 1), KnownPower(), drain=True).run()
    legacy_idle = {**result.steps[-1], 'used_cores': 0, 'used_gpus': 0, 'used_mem': 0,
                   'it_power_kw': 2, 'total_power_kw': 4}
    archive = tmp_path/'legacy-idle'
    archive.mkdir()
    _write_result(archive, replace(result, steps=(*result.steps[:-1], legacy_idle)))
    report = module.compare_archived_result(result, archive, tmp_path/'numerically-equal')
    assert report['all_numerical_fields_exact']
    assert not report['all_files_byte_identical']
    assert report['serialization_only_differences'] == ['intervals.csv']
    intervals = archive/'intervals.csv'
    intervals.write_text(intervals.read_text().replace(',100,20,', ',100.000000000000000001,20,', 1),
                         encoding='utf-8')
    with pytest.raises(RuntimeError, match='intervals.csv'):
        module.compare_archived_result(result, archive, tmp_path/'numeric-mismatch')


@pytest.mark.parametrize('action,name', [(0, 'edf_backfill'), (1, 'edf_carbon_threshold'),
                                       (2, 'cars_risk_only')])
def test_fixed_action_rollout_matches_original_policy_and_full_idle_endpoint(action, name):
    # Catches action mapping errors, premature drain stopping, and reset-per-step physics.
    module = driver()
    from experiments.e1.cars import make_cars_policy
    from experiments.e1.cars_experiment import FIXED_PARAMETERS
    from experiments.e1.engine import LocalExecutor
    workload, environment = fixture([task('a', duration=15, sla=3), task('b', arrival=15)],
                                    safety_minutes=120)
    parameters = {**FIXED_PARAMETERS, 'threshold_g_per_kwh': 137.64999999999998}
    endpoint = START + pd.Timedelta(minutes=90)
    clipped = replace(environment, end=endpoint,
                      samples=tuple(s for s in environment.samples if s.time < endpoint))
    old = LocalExecutor(workload, clipped, (1, 1, 1), KnownPower(),
                        policy=make_cars_policy(name, parameters, workload.start, (1, 1, 1)),
                        drain=True).run()
    result, trace = module.run_fixed_action(workload, environment, (1, 1, 1), KnownPower,
                                           parameters, action, endpoint)
    assert result == old
    assert len(trace) == result.summary['intervals'] == 6
    assert result.summary['observation_end'] == endpoint.isoformat()
    assert {row['action'] for row in trace} == {action}


def test_random_rollout_is_seeded_complete_finite_cohort_without_performance_claims():
    # Catches ignoring the declared seed or stopping while arrived work remains unfinished.
    module = driver()
    from experiments.e1.cars_experiment import FIXED_PARAMETERS
    workload, environment = fixture([task('a', duration=30, sla=3), task('b', arrival=15)],
                                    safety_minutes=180)
    first, trace = module.run_random_actions(workload, environment, (1, 1, 1), KnownPower,
                                            FIXED_PARAMETERS, seed=20261012)
    repeated, repeated_trace = module.run_random_actions(workload, environment, (1, 1, 1),
        KnownPower, FIXED_PARAMETERS, seed=20261012)
    assert first == repeated and trace == repeated_trace
    assert first.summary['arrived'] == first.summary['completed'] == 2
    assert first.summary['pending'] == first.summary['running'] == 0
    assert trace[-1]['terminated'] and not trace[-1]['truncated']
    assert all(row['action'] in (0, 1, 2) for row in trace)
    assert all(row['reward_terms']['carbon_kg'] == row['carbon_kg'] for row in trace)


def tiny_campaigns(tmp_path, monkeypatch):
    from experiments.e1 import cars_experiment, contention, feasibility_experiment, temporal
    from experiments.e1.config import ROOT
    from experiments.e1.contracts import file_hash
    from .test_cars_experiment import setup_reference
    # These fixtures test scheduling/import relationships; real historical source
    # coverage is checked separately. Avoid copying the full repository 11 times.
    sources = {name: file_hash(ROOT/name) for name in
               ('experiments/e1/cars.py', 'experiments/e1/power.py')}
    for module in (cars_experiment, contention, feasibility_experiment, temporal, driver()):
        monkeypatch.setattr(module, 'experiment_sources', lambda: dict(sources))
    _, _, _, reference, parameters, _ = setup_reference(tmp_path, monkeypatch, factors=[1, 2])
    monkeypatch.setattr(cars_experiment, 'UpstreamPowerModel', lambda *args: KnownPower())
    monkeypatch.setattr(feasibility_experiment, 'UpstreamPowerModel', lambda *args: KnownPower())
    historical = tmp_path/'e1-3'
    cars_experiment.run_reference_levels(reference, parameters, historical)
    ablations = tmp_path/'e1-4a'
    feasibility_experiment.run_reference_levels(historical, ablations)
    return historical, ablations


def test_tiny_campaign_runs_all_controls_and_modes_and_retains_original_provenance(tmp_path, monkeypatch):
    # Catches wrong historical endpoint selection, lost E1-4A links, or regenerated inputs.
    module = driver()
    old, later = tiny_campaigns(tmp_path, monkeypatch)
    old_provenance = (old/'provenance.json').read_bytes()
    later_provenance = (later/'provenance.json').read_bytes()
    monkeypatch.setattr(module, 'UpstreamPowerModel', lambda *args: KnownPower())
    output = tmp_path/'validation'
    rows = module.run_validation(old, later, output)
    assert len(rows) == 14
    assert {r['factor'] for r in rows} == {1.5, 2.0}
    assert sum(r['run_kind'] == 'direct_policy' for r in rows) == 8
    assert sum(r['run_kind'] == 'fixed_mode' for r in rows) == 6
    assert all(r['all_numerical_fields_exact'] and r['arrived'] == r['completed'] for r in rows)
    assert all(r['all_files_byte_identical'] for r in rows if r['run_kind'] == 'direct_policy')
    evidence = json.loads((output/'archive_integrity.json').read_text())
    assert evidence['e1_3']['scientific_files_checked'] == 6
    assert set(evidence['levels']) == {'load_1p50', 'load_2p00'}
    random = json.loads((output/'random_load_1p50/validation.json').read_text())
    assert random['all_arrived_tasks_completed'] and random['seed'] == 20261012
    assert not random['training_performed'] and not random['performance_claim']
    trajectory = pd.read_csv(output/'random_load_1p50/trajectory.csv')
    assert {'reward_wait_task_minutes', 'reward_new_deadline_miss_ids',
            'reward_remaining_resource_work_minutes', 'reward_terminal_cost'} <= set(trajectory)
    assert (old/'provenance.json').read_bytes() == old_provenance
    assert (later/'provenance.json').read_bytes() == later_provenance
    for archive, relative in ((old, 'load_1p00/cars_full/summary.json'),
                              (old, 'load_1p25/cars_known_only/diagnostics.jsonl'),
                              (later, 'load_2p00/risk_shadow_only/diagnostics.jsonl')):
        target = archive/relative
        original = target.read_bytes()
        try:
            target.write_bytes(original+b'\n')
            with pytest.raises(ValueError, match='scientific file hash mismatch'):
                module.verify_historical_campaign(archive)
        finally:
            target.write_bytes(original)


def test_historical_import_rejects_self_hashed_false_frozen_identity(tmp_path, monkeypatch):
    # Catches trusting a self-consistent reference hash without checking its declaration.
    module = driver()
    from experiments.e1.contracts import content_hash, file_hash, write_json
    _, later = tiny_campaigns(tmp_path, monkeypatch)
    child = later/'load_1p50'
    identity = json.loads((child/'reference.json').read_text())
    identity['frozen_import'] = False
    write_json(child/'reference.json', identity)
    child_provenance = json.loads((child/'provenance.json').read_text())
    child_provenance['input_hashes']['reference'] = content_hash(identity)
    child_provenance['output_hashes']['reference.json'] = file_hash(child/'reference.json')
    write_json(child/'provenance.json', child_provenance)
    identities = json.loads((later/'input_identities.json').read_text())
    next(item for item in identities if item['factor'] == 1.5)['reference'] = identity
    write_json(later/'input_identities.json', identities)
    parent = json.loads((later/'provenance.json').read_text())
    parent['child_output_manifests']['load_1p50'] = child_provenance
    parent['input_hashes']['identities'] = content_hash(identities)
    parent['output_hashes']['input_identities.json'] = file_hash(later/'input_identities.json')
    write_json(later/'provenance.json', parent)
    with pytest.raises(ValueError, match='frozen reference identity'):
        module.verify_historical_campaign(later)
