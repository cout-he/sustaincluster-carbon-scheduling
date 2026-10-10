"""One entry point, independently pinned inputs, replay and historical regression."""
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from .test_run import configuration, SCIENTIFIC_FILES


def cfg_for(cfg, policy):
    result = json.loads(json.dumps(cfg))
    result['policy'] = policy
    if policy == 'carbon_threshold_fifo':
        result['policy_params'] = {'threshold_g_per_kwh': 80}
    return result


def test_factory_resolves_all_configured_policies_without_access_to_trace():
    from experiments.e1.policies import make_policy
    assert make_policy({'policy': 'immediate_fifo'}).name == 'immediate_fifo'
    assert make_policy({'policy': 'edf_backfill'}).name == 'edf_backfill'
    assert make_policy({'policy': 'carbon_threshold_fifo',
                        'policy_params': {'threshold_g_per_kwh': 200}}).threshold_g_per_kwh == 200


def test_policy_configs_require_fixed_literal_threshold_and_reject_irrelevant_parameters(tmp_path):
    from experiments.e1.config import validate_config
    cfg, _ = configuration(tmp_path)
    assert validate_config(cfg_for(cfg, 'edf_backfill'))['policy'] == 'edf_backfill'
    assert validate_config(cfg_for(cfg, 'carbon_threshold_fifo'))['policy_params']['threshold_g_per_kwh'] == 80
    bad = cfg_for(cfg, 'carbon_threshold_fifo'); del bad['policy_params']
    with pytest.raises(ValueError): validate_config(bad)
    for policy in ('immediate_fifo', 'edf_backfill'):
        bad = cfg_for(cfg, policy); bad['policy_params'] = {'threshold_g_per_kwh': 80}
        with pytest.raises(ValueError): validate_config(bad)
    for value in (-1, True, float('nan'), 'test_window_quantile'):
        bad = cfg_for(cfg, 'carbon_threshold_fifo'); bad['policy_params']['threshold_g_per_kwh'] = value
        with pytest.raises(ValueError): validate_config(bad)


def test_three_strategy_frozen_entry_identical_inputs_and_strict_replay_without_raw_files(tmp_path):
    from experiments.e1.run import run_experiment, run_frozen_experiment, replay_experiment
    cfg, raw_inputs = configuration(tmp_path)
    cfg['workload']['sla_multiplier'] = 3
    base = tmp_path / 'base'; run_experiment(cfg, base)
    for raw in raw_inputs: raw.unlink()
    original = json.loads((base / 'provenance.json').read_text())
    for policy in ('immediate_fifo', 'edf_backfill', 'carbon_threshold_fifo'):
        output, replay = tmp_path / policy, tmp_path / (policy + '-replay')
        run_frozen_experiment(cfg_for(cfg, policy), base, output)
        replay_experiment(output, replay)
        for name in ('workload.json', 'environment.json', 'dc_config.json'):
            assert (base / name).read_bytes() == (output / name).read_bytes()
        new = json.loads((output / 'provenance.json').read_text())
        assert new['input_hashes'] == original['input_hashes']
        assert new['frozen_from'] == str(base.resolve())
        for name in (*SCIENTIFIC_FILES, 'policy_ledger.jsonl', 'policy_events.jsonl',
                     'policy_intervals.jsonl', 'policy_summary.json'):
            assert (output / name).read_bytes() == (replay / name).read_bytes(), name
        diagnostics = json.loads((output / 'policy_summary.json').read_text())
        assert diagnostics['policy'] == policy
        assert diagnostics['policy_defer_intervals'] == (1 if policy == 'carbon_threshold_fifo' else 0)


def test_cross_version_frozen_import_is_explicit_and_strict_replay_stays_strict(tmp_path):
    from experiments.e1.run import run_experiment, run_frozen_experiment, replay_experiment
    from experiments.e1.contracts import content_hash, file_hash
    cfg, _ = configuration(tmp_path)
    old = tmp_path / 'old'; run_experiment(cfg, old)
    provenance_path = old / 'provenance.json'
    provenance = json.loads(provenance_path.read_text())
    source_path = old / 'source/experiments/e1/engine.py'
    source_path.write_text(source_path.read_text() + '\n# earlier archived version\n')
    provenance['sources']['experiments/e1/engine.py'] = file_hash(source_path)
    provenance['source_sha256'] = content_hash(provenance['sources'])
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match='source'):
        replay_experiment(old, tmp_path / 'strict-failure')
    run_frozen_experiment(cfg_for(cfg, 'edf_backfill'), old, tmp_path / 'new')
    assert json.loads((tmp_path / 'new/run_status.json').read_text())['state'] == 'complete'


@pytest.mark.parametrize('change', ['capacity', 'sla', 'window', 'source_path'])
def test_frozen_entry_cannot_change_shared_experiment_contract(tmp_path, change):
    from experiments.e1.run import run_experiment, run_frozen_experiment
    cfg, _ = configuration(tmp_path)
    old = tmp_path / 'old'; run_experiment(cfg, old)
    bad = cfg_for(cfg, 'edf_backfill')
    if change == 'capacity': bad['datacenter']['total_cores'] *= 2
    if change == 'sla': bad['workload']['sla_multiplier'] = 4
    if change == 'window': bad['end'] = '2023-08-01T05:45:00Z'
    if change == 'source_path': bad['environment']['carbon_path'] = 'another-source.csv'
    with pytest.raises(ValueError, match='shared|window|config'):
        run_frozen_experiment(bad, old, tmp_path / 'bad')


def test_compare_runs_three_policies_once_and_reports_completion_context(tmp_path):
    from experiments.e1.run import run_experiment
    from experiments.e1.compare import compare_experiments
    cfg, _ = configuration(tmp_path)
    cfg['workload']['sla_multiplier'] = 3
    old = tmp_path / 'old'; run_experiment(cfg, old)
    compare_experiments(old, tmp_path / 'comparison', threshold_g_per_kwh=80)
    table = pd.read_csv(tmp_path / 'comparison/comparison.csv')
    assert list(table['policy']) == ['immediate_fifo', 'edf_backfill', 'carbon_threshold_fifo']
    assert set(table['arrived']) == {1}
    assert 'capacity_wait_intervals' in table and 'cores_fraction_mean' in table
    info = json.loads((tmp_path / 'comparison/fairness.json').read_text())
    assert info['identical_frozen_inputs'] is True
    assert info['threshold_g_per_kwh'] == 80
    assert info['window_carbon_is_not_completion_normalized'] is True
    assert info['fifo_legacy_regression']['passed'] is True


def test_historical_e1_1a_fifo_legacy_fields_and_numbers_unchanged():
    from experiments.e1.config import ROOT
    from experiments.e1.engine import LocalExecutor
    from experiments.e1.workload import FrozenWorkload
    from experiments.e1.environment import FrozenEnvironment
    from experiments.e1.power import UpstreamPowerModel
    old = ROOT / 'experiments/e1/artifacts/smoke-007'
    if not old.exists(): pytest.skip('Historical local artifact is not shipped with source; compare CLI checks it when supplied')
    cfg = json.loads((old / 'resolved_config.json').read_text())
    workload = FrozenWorkload.load(old / 'workload.json')
    environment = FrozenEnvironment.load(old / 'environment.json')
    dc = cfg['datacenter']
    result = LocalExecutor(workload, environment, (dc['total_cores'], dc['total_gpus'], dc['total_mem']),
                           UpstreamPowerModel(dc, old / 'dc_config.json')).run()
    saved = [json.loads(line) for line in (old / 'ledger.jsonl').read_text().splitlines()]
    assert list(result.tasks) == saved
    assert result.summary == json.loads((old / 'summary.json').read_text())
    csv = pd.read_csv(old / 'intervals.csv')
    for key in ('used_cores', 'used_gpus', 'used_mem', 'it_power_kw', 'hvac_power_kw',
                'total_energy_kwh', 'carbon_kg'):
        assert [r[key] for r in result.steps] == pytest.approx(list(csv[key]), rel=1e-14, abs=1e-12)


def test_comparison_preserves_explicit_empty_fifo_policy_params(tmp_path):
    from experiments.e1.run import run_experiment
    from experiments.e1.compare import compare_experiments
    cfg, _ = configuration(tmp_path)
    cfg['policy_params'] = {}
    base = tmp_path / 'base'; run_experiment(cfg, base)
    compare_experiments(base, tmp_path / 'comparison', threshold_g_per_kwh=80)
    assert (base / 'resolved_config.json').read_bytes() == (
        tmp_path / 'comparison/immediate_fifo/resolved_config.json').read_bytes()


@pytest.mark.parametrize('operation', ['replay', 'frozen'])
@pytest.mark.parametrize('damage', ['unlisted', 'altered'])
def test_current_archive_requires_all_policy_outputs_even_for_frozen_import(tmp_path, operation, damage):
    from experiments.e1.run import run_experiment, run_frozen_experiment, replay_experiment
    cfg, _ = configuration(tmp_path)
    base = tmp_path / 'base'; run_experiment(cfg, base)
    if damage == 'unlisted':
        path = base / 'provenance.json'
        provenance = json.loads(path.read_text())
        del provenance['output_hashes']['policy_events.jsonl']
        path.write_text(json.dumps(provenance))
    else:
        (base / 'policy_events.jsonl').write_text('altered diagnostic output\n')
    with pytest.raises(ValueError, match='manifest|hash'):
        if operation == 'replay':
            replay_experiment(base, tmp_path / 'bad')
        else:
            run_frozen_experiment(cfg_for(cfg, 'edf_backfill'), base, tmp_path / 'bad')
    assert json.loads((tmp_path / 'bad/run_status.json').read_text())['state'] == 'failed'
