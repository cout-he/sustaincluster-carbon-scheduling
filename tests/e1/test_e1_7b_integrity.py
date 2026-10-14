from copy import deepcopy
from pathlib import Path
import pytest
from experiments.e1.carbon_gain_experiment import validate_protocol,load_frozen,scientific_diagnostics
from experiments.e1.contracts import content_hash,write_json

def test_test_role_rejected_before_payload_read():
    from experiments.e1.carbon_gain_experiment import load_registered_bundle
    with pytest.raises(ValueError,match='Test|role'):load_registered_bundle({},'test','1')

def test_frozen_parameters_tamper_and_role(tmp_path):
    from .test_online_power_surrogate import model
    payload=dict(schema='e1-7b-calibration-v1',fit_role='train',test_access=False,model=model().to_dict(),
                 moderate_threshold_kg=.01,config_sha256='abc')
    p=tmp_path/'freeze.json';write_json(p,dict(payload=payload,sha256=content_hash(payload)))
    assert load_frozen(p)['moderate_threshold_kg']==.01
    payload['fit_role']='validation';write_json(p,dict(payload=payload,sha256=content_hash(payload)))
    with pytest.raises(ValueError):load_frozen(p)

def test_runtime_not_part_of_scientific_replay_hash():
    rows=[dict(task_id='a',gain_total_kg=.1,elapsed_seconds=.2,decision_seconds=.3)]
    clean=scientific_diagnostics(rows)
    assert clean==[dict(task_id='a',gain_total_kg=.1)]

def test_online_modules_do_not_import_offline_oracle():
    root=Path(__file__).parents[2]/'experiments/e1'
    for name in ('online_ci_forecast','online_power_surrogate','carbon_gain','deadline_guard','carbon_gain_policy','carbon_gain_rl_env'):
        source=(root/(name+'.py')).read_text()
        assert 'from .carbon_opportunity' not in source and 'from .offline_schedule' not in source

def test_frozen_config_no_parameter_search():
    import json
    cfg=json.loads((Path(__file__).parents[2]/'experiments/e1/configs/e1_7b_cgd_rl_v1.json').read_text())
    validate_protocol(cfg)
    bad=deepcopy(cfg);bad['threshold_g_per_kwh']=200
    with pytest.raises(ValueError):validate_protocol(bad)
