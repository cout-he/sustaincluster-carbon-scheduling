import copy
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def module():
    return importlib.import_module('experiments.e1.diagnostic_experiment')


def config():
    return json.loads((ROOT/'experiments/e1/configs/e1_7a.json').read_text())


def test_registered_protocol_accepted():
    assert module().validate_config(config())['allowed_split'] == 'train'


@pytest.mark.parametrize('key,value', [('allowed_split','test'), ('test_access',True),
    ('validation_selection',True), ('threshold_g_per_kwh',200), ('step_minutes',5),
    ('capacity',[5000,1000,80000]), ('sla_multiplier',2), ('stress_seed',99)])
def test_protocol_changes_rejected(key, value):
    cfg = config()
    cfg[key] = value
    with pytest.raises(ValueError):
        module().validate_config(cfg)


def test_solver_budget_is_registered_and_finite():
    cfg = config()
    cfg['A']['solver']['max_seconds'] = 10000
    with pytest.raises(ValueError):
        module().validate_config(cfg)


def test_malformed_manifest_hash_rejected():
    cfg = config()
    cfg['bundle_manifest_sha256']['1'] += 'b'
    with pytest.raises(ValueError, match='SHA'):
        module().validate_config(cfg)


def test_frozen_manifest_tampering_and_wrong_role_rejected(tmp_path):
    from experiments.e1.contracts import content_hash
    payload = dict(schema='e1-7a-selection-manifest-v1', split='test', test_access=False)
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(dict(payload=payload, sha256=content_hash(payload))))
    with pytest.raises(ValueError, match='Train'):
        module().load_selection_manifest(path)
    payload.update(split='train', config=config(), config_sha256='bad')
    path.write_text(json.dumps(dict(payload=payload, sha256='bad')))
    with pytest.raises(ValueError, match='hash'):
        module().load_selection_manifest(path)


def test_config_validator_does_not_mutate_input():
    cfg = config()
    before = copy.deepcopy(cfg)
    module().validate_config(cfg)
    assert cfg == before


@pytest.mark.parametrize('section,key,value', [
    ('C','candidate_occurrence',2), ('C','factors',[2.0]),
    ('C','high_resource_threshold',.7), ('B','delta_minutes',30),
    ('B','ci_bins_g_per_kwh',[0,200])])
def test_all_nested_registration_fields_are_bound(section,key,value):
    cfg=config()
    cfg[section][key]=value
    with pytest.raises(ValueError,match='configuration'):
        module().validate_config(cfg)


@pytest.mark.parametrize('stage', ['run_baselines','run_offline','run_carbon','run_counterfactual'])
def test_every_actual_stage_rejects_test_before_output_or_input_access(tmp_path,stage):
    from experiments.e1.contracts import content_hash
    payload=dict(schema='e1-7a-selection-manifest-v1',split='test',test_access=False)
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps(dict(payload=payload,sha256=content_hash(payload))))
    output=tmp_path/'forbidden'
    with pytest.raises(ValueError,match='Train'):
        getattr(module(),stage)(path,output)
    assert not output.exists()


def test_historical_control_clock_uses_instant_not_timezone_spelling():
    assert module()._same_utc_instant('2023-05-03T14:00:00+00:00', '2023-05-03T22:00:00+08:00')
    assert not module()._same_utc_instant('2023-05-03T14:00:00+00:00', '2023-05-03T14:15:00+00:00')
    with pytest.raises(ValueError, match='aware'):
        module()._same_utc_instant('2023-05-03T14:00:00', '2023-05-03T14:00:00+00:00')
