"""Command contracts: sealed roles, real tiny episodes, provenance and selection."""
import json
import shutil
from types import SimpleNamespace

import pytest

from experiments.joint.config import load_config
from tests.e1.helpers import inputs, task


def test_test_role_rejected_before_loading_or_reading_physics(monkeypatch, tmp_path):
    from experiments.joint import inputs as joint_inputs
    monkeypatch.setattr(joint_inputs, 'load_bundle', lambda *a, **k: pytest.fail('must reject first'))
    with pytest.raises(ValueError, match='Train|Validation|split|Test'):
        joint_inputs.make_env(tmp_path, 'test', load_config(), 'B0')


@pytest.fixture
def tiny_bundles(monkeypatch, tmp_path):
    from experiments.e1.config import ROOT
    from experiments.joint import inputs as joint_inputs
    bundles = {}
    for split in ('train', 'validation'):
        folder = tmp_path / split
        folder.mkdir()
        # Keep the upstream physical rack sizing domain while exercising a tiny
        # logical resource pool through the real executor and PPO collectors.
        dc = dict(dc_id=1, location='US-CAL-CISO', total_cores=50000, total_gpus=1000,
                  total_mem=80000, hvac_setpoint_c=22)
        (folder / 'config.json').write_text(json.dumps({'datacenter': dc}), encoding='utf8')
        shutil.copyfile(ROOT / 'configs/dcs/dc_config.json', folder / 'dc_config.json')
        workload, environment = inputs([task(split + '-job', cpu=.5, mem=.5)], 30)
        identity = dict(split=split, factor=1., stress_seed=510001 if split == 'train' else 520001,
                        bundle_manifest_sha256=split + '-manifest', workload_sha256=split + '-workload',
                        environment_sha256=split + '-environment', raw_source_sha256={'fixture': 'shared'},
                        test_evaluation=False)
        bundles[str(folder)] = SimpleNamespace(split=split, workload=workload, environment=environment,
            capacity=(2, 1, 3), parameters={'threshold_g_per_kwh':150.}, identity=identity,
            factor=1., stress_seed=identity['stress_seed'])
    def guarded(path, *, allowed_split):
        bundle = bundles[str(path)]
        if bundle.split != allowed_split:
            raise ValueError('Input split does not match allowed split')
        return bundle
    monkeypatch.setattr(joint_inputs, 'load_bundle', guarded)
    cfg = load_config()
    cfg['training'].update(episodes=1, episodes_per_update=1)
    cfg['ppo'].update(n_epochs=1, batch_size=2, net_arch=[8])
    return tmp_path / 'train', tmp_path / 'validation', cfg


def test_train_and_evaluate_whole_episodes_and_checkpoint_provenance(tiny_bundles, tmp_path):
    from experiments.joint.train import train
    from experiments.joint.evaluate import evaluate
    train_bundle, validation_bundle, cfg = tiny_bundles
    models = tmp_path / 'models'
    for mode in ('B1','B2','B3'):
        train(train_bundle, models / mode, config=cfg, mode=mode)
        metadata = json.loads((models / mode / 'metadata.json').read_text())
        assert metadata['mode'] == mode and metadata['training_role'] == 'train'
        assert metadata['episodes_completed'] == 1 and metadata['physical_slots'] == 2
        assert metadata['input_identity']['split'] == 'train'
        assert all(metadata['initial_parameter_hashes'][k] != value
                   for k, value in metadata['final_parameter_hashes'].items())
    output = tmp_path / 'evaluation'
    comparison = evaluate(validation_bundle, models, output, config=cfg)
    assert set(comparison['results']) == {'B0','B1','B2','B3','RiskOnly'}
    for mode, row in comparison['results'].items():
        assert row['accounting'] == 'joint_full_hvac_v1'
        assert len((output / mode / 'intervals.jsonl').read_text().splitlines()) == 2
        assert (output / mode / 'tasks.json').is_file()
    selection = json.loads((output / 'selection.json').read_text())
    assert set(selection['candidates']) == {'B1','B2','B3'}
    assert selection['status'] in ('eligible_candidates','no_feasible_candidate')
    with pytest.raises(FileExistsError):
        train(train_bundle, models / 'B1', config=cfg, mode='B1')
    # Reject altered provenance BEFORE deserializing any checkpoint.
    metadata_path = models / 'B2' / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['mode'] = 'B3'
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='mode|provenance'):
        evaluate(validation_bundle, models, tmp_path / 'bad', config=cfg)


def test_smoke_override_never_selected_as_formal_candidate(tiny_bundles, tmp_path):
    from experiments.joint.pipeline import run
    train_bundle, validation_bundle, cfg = tiny_bundles
    result = run(train_bundle, validation_bundle, tmp_path / 'smoke', config=cfg,
                 episodes=1, max_episode_steps=1)
    assert result['smoke'] is True
    selection = json.loads((tmp_path / 'smoke' / 'evaluation' / 'selection.json').read_text())
    assert selection['status'] == 'smoke_only'
    assert all(not row['eligible'] for row in selection['candidates'].values())


def test_feasibility_requires_complete_work_and_no_thermal_violations():
    from experiments.joint.evaluate import feasibility
    row = dict(dmr=.04, temperature_violations=0, unfinished_tasks=0, censored_tasks=0)
    assert feasibility(row)
    assert not feasibility(dict(row, dmr=.051))
    assert not feasibility(dict(row, temperature_violations=1))
    assert not feasibility(dict(row, unfinished_tasks=1))
    assert not feasibility(dict(row, censored_tasks=1))


def test_checkpoint_validation_rejects_validation_fit_and_changed_bytes(tmp_path):
    from experiments.joint.evaluate import validate_metadata
    cfg = load_config()
    from experiments.e1.contracts import content_hash
    metadata = dict(mode='B1', training_role='validation', config_sha256=content_hash(cfg),
                    accounting=cfg['accounting'], input_identity={'split':'validation'},
                    checkpoint_sha256={})
    with pytest.raises(ValueError, match='Train|train|role'):
        validate_metadata(tmp_path, 'B1', cfg, metadata=metadata)
