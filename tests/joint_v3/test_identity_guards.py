"""Frozen identity checks must finish before an authorized Test attempt begins."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.e1.contracts import content_hash, file_hash
from experiments.joint_v2.checkpoint import atomic_json
from experiments.joint_v3 import VERSION
from experiments.joint_v3.checkpoint import Store, code_identity
from experiments.joint_v3.config import defaults


def real_training(folder, *, embedded_change=False, source_identity=None):
    from experiments.joint_v3.ci_forecaster import CIForecaster
    from experiments.joint_v3.mappo import RACSystem
    from tests.joint_v3.test_carbon import train_scenario
    cfg = defaults()
    embedded = deepcopy(cfg)
    if embedded_change:
        embedded['carbon']['kappa'] = 0.
    meta = dict(config_identity=content_hash(cfg), data_identity='synthetic-data',
                code_identity=source_identity or code_identity()['sha256'], test_access=False)
    atomic_json(folder/'config.json', cfg)
    atomic_json(folder/'identity.json', meta)
    system = RACSystem(cfg['seed'], cfg['ppo'])
    forecaster = CIForecaster(**cfg['forecast']).fit([train_scenario(100)])
    Store(folder/'checkpoints', meta['config_identity'], meta['data_identity'], meta['code_identity']).save(
        dict(config=embedded, system=system.export_state(), forecaster=forecaster.export_state()),
        ('latest', 'best_feasible'))
    return cfg


def test_model_config_tamper_is_rejected_before_weight_deserialization(tmp_path, monkeypatch):
    import torch
    from experiments.joint_v3.evaluate import load_model
    folder = tmp_path/'training'
    cfg = real_training(folder)
    cfg['carbon']['kappa'] = 0.
    atomic_json(folder/'config.json', cfg)
    def forbidden(*args, **kwargs):
        pytest.fail('Changed runtime config reached weight deserialization')
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(ValueError, match='config'):
        load_model(folder)


def test_model_checkpoint_config_must_equal_external_frozen_config(tmp_path):
    from experiments.joint_v3.evaluate import load_model
    folder = tmp_path/'training'
    real_training(folder, embedded_change=True)
    with pytest.raises(ValueError, match='config'):
        load_model(folder)


def test_source_guard_cannot_be_disabled_before_deserializing_weights(tmp_path, monkeypatch):
    import torch
    from experiments.joint_v3.evaluate import load_model
    folder = tmp_path/'training'
    real_training(folder, source_identity='previous-source')
    def forbidden(*args, **kwargs):
        pytest.fail('Changed source reached weight deserialization')
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(ValueError, match='source'):
        load_model(folder, check_code=False)


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    from experiments.joint_v3 import test_once
    from experiments.joint_v3.pipeline import prepare
    pool = tmp_path/'pool'
    atomic_json(pool/'manifest.json', dict(test_access=False, scenarios=[
        dict(scenario_id='train', role='train'), dict(scenario_id='dev', role='development')]))
    v2 = tmp_path/'chosen-v2'
    v2.mkdir()
    v2_weights = v2/'synthetic-v2.pt'
    v2_weights.write_bytes(b'synthetic episode52 fixture, never deserialized')
    atomic_json(v2/'best_feasible.json', dict(completed_episodes=52, file=v2_weights.name,
                                             sha256=file_hash(v2_weights)))
    protocol_path = prepare(pool, tmp_path/'campaign', v2_checkpoint=v2)
    protocol = json.loads(protocol_path.read_text())
    root = protocol_path.parent
    for job in protocol['jobs']:
        training = root/job['training']
        cfg = json.loads((root/job['config']).read_text())
        atomic_json(training/'config.json', cfg)
        meta = dict(config_identity=job['config_identity'], data_identity='data-'+job['job_id'],
                    code_identity=protocol['source_identity'], test_access=False)
        atomic_json(training/'identity.json', meta)
        atomic_json(training/'run-result.json', dict(seed=job['seed'], variant=job['variant'],
            physical_steps=65536, elapsed_seconds=120., completed_episodes=10,
            last_run=dict(stop_reason='step_budget'), test_access=False))
        weights = training/'checkpoints'/'synthetic.pt'
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b'synthetic fixture, never deserialized')
        atomic_json(training/'checkpoints'/'best_feasible.json', dict(meta, schema=VERSION,
            file=weights.name, sha256=file_hash(weights), completed_episodes=10))
        methods = ('B0', 'B1', 'B2', 'B3', 'B4', 'B5') if job['variant'] == 'full' else ('B5',)
        rows = [dict(scenario_id=scene, comparison_method=method, seed=job['seed'], variant=job['variant'],
                     scenario_role='development', arrived=1, completed=1, deadline_misses=0,
                     carbon_kg=1.) for scene in protocol['evaluation']['scenario_ids'] for method in methods]
        atomic_json(root/'jobs'/job['job_id']/'evaluation'/'results.json',
                    dict(complete=True, results=rows, test_access=False))
    # Only heavyweight model decoding is replaced. All files, hashes, protocol,
    # config, budgets and comparison inventory are checked by production code.
    def model_boundary(training, alias='best_feasible', **kwargs):
        cfg = json.loads((Path(training)/'config.json').read_text())
        return object(), object(), cfg, dict(config=cfg, completed_episodes=10)
    monkeypatch.setattr(test_once, 'load_model', model_boundary)
    monkeypatch.setattr(test_once, 'load_v2', lambda path: {})
    return root, protocol


def freeze(campaign, tmp_path):
    from experiments.joint_v3.test_once import freeze_campaign
    output = tmp_path/'freeze.json'
    freeze_campaign(campaign[0], output)
    return output


def authorization(freeze_path, tmp_path):
    path = tmp_path/'human-authorization.json'
    atomic_json(path, dict(authorize_september_test=True, freeze_sha256=file_hash(freeze_path)))
    return path


def test_freeze_rejects_interrupted_budget_instead_of_stamping_completion(campaign, tmp_path):
    root, protocol = campaign
    result = root/protocol['jobs'][0]['training']/'run-result.json'
    atomic_json(result, dict(physical_steps=0, elapsed_seconds=1., completed_episodes=0,
                            last_run=dict(stop_reason='interrupted_uncommitted_episode_rolled_back')))
    with pytest.raises(ValueError, match='budget|training'):
        freeze(campaign, tmp_path)
    assert not (tmp_path/'freeze.json').exists()


def test_freeze_rejects_empty_completed_comparison_inventory(campaign, tmp_path):
    root, protocol = campaign
    atomic_json(root/'jobs'/protocol['jobs'][0]['job_id']/'evaluation'/'results.json',
                dict(complete=True, results=[], test_access=False))
    with pytest.raises(ValueError, match='evaluation|comparison'):
        freeze(campaign, tmp_path)


def test_freeze_rejects_changed_training_source_identity(campaign, tmp_path):
    root, protocol = campaign
    identity = root/protocol['jobs'][0]['training']/'identity.json'
    meta = json.loads(identity.read_text())
    meta['code_identity'] = 'old-source'
    atomic_json(identity, meta)
    with pytest.raises(ValueError, match='identity|source'):
        freeze(campaign, tmp_path)


def test_freeze_rejects_modified_protocol_before_model_loading(campaign, tmp_path):
    root, protocol = campaign
    protocol['source_identity'] = 'old-source'
    atomic_json(root/'protocol.json', protocol)
    with pytest.raises(ValueError, match='protocol|source'):
        freeze(campaign, tmp_path)


def test_freeze_preserves_chosen_v2_and_binds_selected_checkpoint_identities(campaign, tmp_path):
    root, protocol = campaign
    frozen = json.loads(freeze(campaign, tmp_path).read_text())
    assert frozen['campaign_root'] == str(root.resolve())
    assert frozen['v2_checkpoint'] == protocol['evaluation']['v2_checkpoint']
    assert frozen['v2_pointer_sha256'] == file_hash(Path(frozen['v2_checkpoint'])/'best_feasible.json')
    assert len(frozen['models']) == 3
    for model in frozen['models']:
        meta = json.loads((Path(model['training'])/'identity.json').read_text())
        assert model['config_identity'] == meta['config_identity']
        assert model['data_identity'] == meta['data_identity']
        assert model['code_identity'] == meta['code_identity']


@pytest.mark.parametrize('changed', ['model-config', 'model-bytes', 'training-budget', 'comparisons',
                                    'v2-pointer', 'protocol', 'window', 'freeze-source'])
def test_authorized_run_revalidates_every_guard_before_test_access(campaign, tmp_path, monkeypatch, changed):
    from experiments.joint_v3 import test_once
    root, protocol = campaign
    freeze_path = freeze(campaign, tmp_path)
    frozen = json.loads(freeze_path.read_text())
    training = Path(frozen['models'][0]['training'])
    if changed == 'model-config':
        path = training/'config.json'
        cfg = json.loads(path.read_text()); cfg['carbon']['kappa'] = 0.
        atomic_json(path, cfg)
    elif changed == 'model-bytes':
        (training/'checkpoints'/'synthetic.pt').write_bytes(b'changed')
    elif changed == 'training-budget':
        atomic_json(training/'run-result.json', dict(physical_steps=0, elapsed_seconds=1.,
                    last_run=dict(stop_reason='interrupted_uncommitted_episode_rolled_back')))
    elif changed == 'comparisons':
        atomic_json(root/'jobs'/protocol['jobs'][0]['job_id']/'evaluation'/'results.json',
                    dict(complete=True, results=[], test_access=False))
    elif changed == 'v2-pointer':
        atomic_json(Path(frozen['v2_checkpoint'])/'best_feasible.json', dict(completed_episodes=51))
    elif changed == 'protocol':
        protocol['source_identity'] = 'old-source'
        atomic_json(root/'protocol.json', protocol)
    elif changed == 'window':
        frozen['test_window_metadata'][0] = '2023-09-01T00:00:00Z'
        atomic_json(freeze_path, frozen)
    elif changed == 'freeze-source':
        frozen['source_identity'] = 'old-source'
        atomic_json(freeze_path, frozen)
    auth = authorization(freeze_path, tmp_path)
    marker, output = tmp_path/'attempt.json', tmp_path/'test-output'
    source = tmp_path/'missing-test-source'
    from experiments.e1 import workload, selection
    def forbidden(*args, **kwargs):
        pytest.fail('A failed identity guard reached a raw Test loader')
    monkeypatch.setattr(workload, 'load_workload', forbidden)
    monkeypatch.setattr(selection, 'load_temporal_environment', forbidden)
    with pytest.raises(ValueError):
        test_once.run(freeze_path, auth, source, output, marker_path=marker)
    assert not marker.exists()
    assert not output.exists()


def test_valid_synthetic_freeze_requires_human_authorization_and_consumes_once(campaign, tmp_path):
    from experiments.joint_v3.test_once import authorize
    freeze_path = freeze(campaign, tmp_path)
    marker = tmp_path/'attempt.json'
    with pytest.raises(ValueError, match='sealed'):
        authorize(freeze_path, None, marker)
    assert not marker.exists()
    auth = authorization(freeze_path, tmp_path)
    frozen = authorize(freeze_path, auth, marker)
    assert frozen['campaign_complete'] is True
    assert json.loads(marker.read_text())['status'] == 'attempted'
    with pytest.raises(FileExistsError):
        authorize(freeze_path, auth, marker)


def test_changed_raw_test_window_metadata_fails_before_marker(campaign, tmp_path, monkeypatch):
    from experiments.joint_v3 import test_once
    freeze_path = freeze(campaign, tmp_path)
    repo = tmp_path/'metadata-only-repo'
    metadata = repo/'experiments/e1/configs/e1_5b_ppo.json'
    atomic_json(metadata, dict(windows=dict(test=[
        '2023-09-01T00:00:00Z', '2023-09-03T00:00:00Z', '2023-10-03T00:00:00Z'])))
    monkeypatch.setattr(test_once, '__file__', str(repo/'experiments/joint_v3/test_once.py'))
    marker = tmp_path/'attempt.json'
    with pytest.raises(ValueError, match='window'):
        test_once.authorize(freeze_path, authorization(freeze_path, tmp_path), marker)
    assert not marker.exists()
