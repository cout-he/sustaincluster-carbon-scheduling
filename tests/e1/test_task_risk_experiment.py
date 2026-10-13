"""E1-6A independent fairness, provenance, Train gate and replay behavior."""
from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.e1.contracts import file_hash, write_json
from experiments.e1.rl_evaluate import EvaluationBundle
from .helpers import START, KnownPower, inputs, task


def bundle(split='train', factor=1., factory=KnownPower, oversized=False):
    workload, env = inputs([task('a', duration=30, sla=3, cpu=2 if oversized else 1)], 30)
    end = START + pd.Timedelta(minutes=180)
    samples = tuple(replace(env.samples[0], time=START + pd.Timedelta(minutes=i),
        ci_g_per_kwh=200 if i < 30 else 100) for i in range(0, 180, 15))
    env = replace(env, end=end, samples=samples)
    return EvaluationBundle(split, workload, env, (1, 1, 1), factory,
        {'threshold_g_per_kwh':161.075}, 510001 if split == 'train' else 520001, factor)


def test_reject_test_before_power_or_output_creation(tmp_path):
    from experiments.e1.task_risk_experiment import run_group
    def forbidden():
        raise AssertionError('Test must be rejected before touching physics')
    target = tmp_path/'test'
    with pytest.raises(ValueError, match='Test|test'):
        run_group(bundle('test', factory=forbidden), target)
    assert not target.exists()


def test_refuse_existing_output_without_touching_marker(tmp_path):
    from experiments.e1.task_risk_experiment import run_group
    out = tmp_path/'old'; out.mkdir(); (out/'keep').write_text('keep')
    with pytest.raises(FileExistsError):
        run_group(bundle(), out)
    assert list(out.iterdir()) == [out/'keep']
    assert (out/'keep').read_text() == 'keep'


def test_all_eight_policies_share_input_finish_and_common_idle_physics(tmp_path):
    from experiments.e1.task_risk_experiment import GROUP_POLICIES, run_group
    live = []
    class PersistentPower(KnownPower):
        def __init__(self):
            self.calls = 0; self.closed = False; live.append(self)
        def measure(self, *args):
            assert not self.closed
            self.calls += 1
            return super().measure(*args)
        def close(self):
            self.closed = True
    result = run_group(bundle(factory=PersistentPower), tmp_path/'group')
    assert len(GROUP_POLICIES) == len(live) == len(result['common_rows']) == 8
    assert all(model.closed for model in live)
    assert len({model.calls for model in live}) == 1
    assert result['fairness']['all_tasks_completed']
    assert result['fairness']['equal_resource_hours']
    assert result['fairness']['idle_included']
    assert len({r['observation_end'] for r in result['common_rows']}) == 1
    assert all(r['completed'] == r['arrived'] == 1 for r in result['common_rows'])
    assert len({r['cores_hours'] for r in result['common_rows']}) == 1
    assert set(result['results']) == set(GROUP_POLICIES)
    for row in result['results']['edf_backfill'].steps:
        assert row['used_cores'] <= 1 and row['used_gpus'] <= 1 and row['used_mem'] <= 1
    assert result['results']['edf_backfill'].summary['carbon_kg'] > 0
    assert result['results']['edf_backfill'].summary['total_energy_kwh'] == 5


def test_unsafe_incomplete_cohort_remains_in_ledger_and_not_carbon_claimable(tmp_path):
    from experiments.e1.task_risk_experiment import run_group
    result = run_group(bundle(oversized=True), tmp_path/'incomplete')
    assert not result['fairness']['all_tasks_completed']
    assert not result['fairness']['equal_workload_carbon_comparison_valid']
    for row in result['common_rows']:
        assert row['completed'] == 0 and row['pending'] == row['arrived'] == 1
        assert row['deadline_missed_unfinished'] == 1
        assert row['termination'] == 'safety_cap_incomplete'


def test_historical_prefix_regression_keeps_old_archive_unchanged(tmp_path):
    from experiments.e1.rl_evaluate import evaluate_group
    from experiments.e1.task_risk_experiment import run_group
    original = bundle(); history = tmp_path/'history'
    evaluate_group(original, history)
    before = {p.relative_to(history):file_hash(p) for p in history.rglob('*') if p.is_file()}
    result = run_group(original, tmp_path/'new', reference_dir=history)
    assert result['baseline_regression']['matched']
    assert result['baseline_regression']['policies_checked'] == 3
    assert before == {p.relative_to(history):file_hash(p) for p in history.rglob('*') if p.is_file()}


def test_replay_scientific_outputs_match_and_refuses_tampering(tmp_path):
    from experiments.e1.task_risk_experiment import run_group, replay_group
    original = bundle(); old = tmp_path/'first'; new = tmp_path/'replay'
    run_group(original, old)
    replayed = replay_group(old, new, bundle=original)
    assert replayed['replay']['scientific_outputs_exact']
    a = json.loads((old/'scientific-hashes.json').read_text())
    b = json.loads((new/'scientific-hashes.json').read_text())
    assert a == b
    with (old/'edf_backfill'/'ledger.jsonl').open('a') as stream:
        stream.write('{}\n')
    with pytest.raises(ValueError, match='hash|artifact'):
        replay_group(old, tmp_path/'tampered', bundle=original)


def test_train_gate_requires_three_correct_train_groups_and_same_frozen_source(tmp_path):
    from experiments.e1.task_risk_experiment import run_group, write_train_gate, validate_train_gate, PARAMETERS
    groups = []
    for factor in (1., 1.5, 2.):
        path = tmp_path/f'train-{factor:g}'
        run_group(bundle(factor=factor), path)
        groups.append(path)
    gate = tmp_path/'gate.json'
    write_train_gate(groups, gate)
    saved = json.loads(gate.read_text())
    assert saved['functional_passed'] and saved['not_performance_selected']
    assert validate_train_gate(gate, PARAMETERS)['functional_passed']
    changed = dict(PARAMETERS, k=2)
    with pytest.raises(ValueError, match='parameter'):
        validate_train_gate(gate, changed)
    saved['source_sha256'] = '0'*64
    write_json(gate, saved)
    with pytest.raises(ValueError, match='source'):
        validate_train_gate(gate, PARAMETERS)


def test_train_gate_rejects_incomplete_or_validation_results(tmp_path):
    from experiments.e1.task_risk_experiment import run_group, write_train_gate
    incomplete = tmp_path/'incomplete'
    run_group(bundle(oversized=True), incomplete)
    with pytest.raises(ValueError, match='three|Train|train|complete'):
        write_train_gate([incomplete], tmp_path/'bad-gate.json')
    validation = tmp_path/'validation'
    run_group(bundle('validation'), validation)
    with pytest.raises(ValueError, match='three|Train|train'):
        write_train_gate([validation]*3, tmp_path/'bad-gate2.json')
