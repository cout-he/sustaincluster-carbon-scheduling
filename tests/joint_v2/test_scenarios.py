"""V2 source identity, role isolation and reproducible scenario sampling."""
from dataclasses import replace
import json

import pandas as pd
import pytest

from experiments.joint_v2.scenarios import TrainingScenario, fingerprint_bucket, load_scenario
from experiments.joint_v2.scenario_sampler import ScenarioSampler
from tests.e1.helpers import inputs, task


def scenario(tmp_path, name='a', role='train', pressure=.2):
    from experiments.e1.config import ROOT
    workload, environment = inputs([task(name)], 30)
    return TrainingScenario(name, role, workload, environment, (2,1,3),
        {'threshold_g_per_kwh':161.075},
        dict(dc_id=1, location='US-CAL-CISO', total_cores=50000, total_gpus=1000,
             total_mem=80000, hvac_setpoint_c=22), ROOT / 'configs/dcs/dc_config.json',
        dict(role=role, source_fragment_id=name, source_bucket_ids=[name],
             resource_pressure_factor=1., sla_multiplier=1.5),
        dict(pressure_proxy=pressure, load_category='low' if pressure < .5 else 'high',
             deadline_category='loose', ci_category='stable', temperature_category='cool'))


def test_repeated_bucket_fingerprint_uses_original_payload_and_all_job_content():
    row = list(task('source').raw_fields)
    row[0] = 'original-job'
    row[3] = pd.Timestamp('1970-01-26T00:00Z')
    first = fingerprint_bucket([row])
    assert first == fingerprint_bucket([list(row)])
    changed = list(row); changed[5] = 71
    assert first != fingerprint_bucket([changed])


def test_saved_scenario_checks_role_before_any_payload_then_checks_hashes(tmp_path):
    current = scenario(tmp_path)
    folder = tmp_path / 'frozen'
    current.save(folder)
    restored = load_scenario(folder, allowed_role='train')
    assert restored.workload.sha256 == current.workload.sha256
    assert restored.identity == current.identity
    assert restored.role == restored.split == 'train'
    (folder / 'workload.json').write_text('DO NOT READ')
    with pytest.raises(ValueError, match='role'):
        load_scenario(folder, allowed_role='development')
    with pytest.raises(ValueError, match='hash|identity'):
        load_scenario(folder, allowed_role='train')
    with pytest.raises(ValueError, match='Test|role'):
        load_scenario(folder, allowed_role='test')


def test_sampler_state_reproduces_next_draws_and_never_samples_heldout(tmp_path):
    scenarios = [scenario(tmp_path, chr(97+i), pressure=.2 if i < 3 else 1.2) for i in range(4)]
    sampler = ScenarioSampler(scenarios, seed=17)
    prefix = [sampler.sample(stage=1).scenario_id for _ in range(20)]
    assert len(set(prefix)) > 1
    state = sampler.state_dict()
    expected = [sampler.sample(stage=2).scenario_id for _ in range(50)]
    restored = ScenarioSampler(scenarios, seed=999)
    restored.load_state_dict(state)
    assert expected == [restored.sample(stage=2).scenario_id for _ in range(50)]
    with pytest.raises(ValueError, match='train|Train'):
        ScenarioSampler([scenario(tmp_path, 'heldout', role='development')], seed=17)


def test_sampler_caps_high_pressure_probability_and_rejects_pool_change(tmp_path):
    scenarios = [scenario(tmp_path, 'easy'), scenario(tmp_path, 'heavy', pressure=2.)]
    sampler = ScenarioSampler(scenarios, seed=42, config={'max_high_pressure_probability':.15})
    draws = [sampler.sample(stage=2).scenario_id for _ in range(5000)]
    assert draws.count('heavy') / len(draws) < .19
    changed = ScenarioSampler([scenario(tmp_path, 'other')], seed=42)
    with pytest.raises(ValueError, match='pool|identity'):
        changed.load_state_dict(sampler.state_dict())


def test_sampler_state_json_roundtrip(tmp_path):
    scenarios = [scenario(tmp_path, 'a'),scenario(tmp_path, 'b')]
    sampler = ScenarioSampler(scenarios, seed=72)
    for _ in range(7): sampler.sample()
    encoded = json.loads(json.dumps(sampler.state_dict()))
    expected = [sampler.sample().scenario_id for _ in range(15)]
    restored = ScenarioSampler(scenarios, seed=1)
    restored.load_state_dict(encoded)
    assert expected == [restored.sample().scenario_id for _ in range(15)]
