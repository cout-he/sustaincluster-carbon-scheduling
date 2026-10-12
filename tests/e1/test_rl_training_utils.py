"""Split isolation and immutable input contracts before any PPO sampling."""
import json
from dataclasses import asdict

import pytest

from experiments.e1.rl_env import ObservationNormalization, RewardConfig
from .test_stepwise import scenario
from .helpers import KnownPower

def test_test_bundle_rejected_before_missing_payload_is_read(tmp_path):
    from experiments.e1.rl_training_utils import load_bundle
    (tmp_path/'bundle.json').write_text(json.dumps({'split':'test'}))
    with pytest.raises(ValueError,match='Test|test'):
        load_bundle(tmp_path,allowed_split='train')

def test_wrong_split_rejected_before_missing_payload_is_read(tmp_path):
    from experiments.e1.rl_training_utils import load_bundle
    (tmp_path/'bundle.json').write_text(json.dumps({'split':'validation'}))
    with pytest.raises(ValueError,match='split'):
        load_bundle(tmp_path,allowed_split='train')

def test_freezer_rejects_test_without_creating_output(tmp_path):
    from experiments.e1.rl_training_utils import freeze_bundle
    with pytest.raises(ValueError,match='Test|test'):
        freeze_bundle({},'test',2,530001,tmp_path/'out')
    assert not (tmp_path/'out').exists()

def test_env_factory_preserves_default_reward_and_fresh_physical_resets():
    from experiments.e1.rl_training_utils import ExperimentBundle,make_env_factory
    w,e=scenario();created=[]
    class Tracked(KnownPower):
        def __init__(self):created.append(self)
    b=ExperimentBundle('train',w,e,(2,1,3),Tracked,{'threshold_g_per_kwh':150},510001,1,{})
    factory=make_env_factory(b,{'reward':asdict(RewardConfig()),'normalization':asdict(ObservationNormalization())})
    env=factory();env.reset(seed=7);env.step(0);first=env.executor.power_model
    env.reset(seed=7);assert env.executor.power_model is not first
    assert env.reward_config==RewardConfig() and env.stop_at_drain
    assert len(created)==2
    env.close()

def test_training_factory_rejects_validation_and_test_inputs():
    from experiments.e1.rl_training_utils import ExperimentBundle,make_env_factory
    w,e=scenario()
    for split in ('validation','test'):
        b=ExperimentBundle(split,w,e,(2,1,3),KnownPower,{'threshold_g_per_kwh':150},1,1,{})
        with pytest.raises(ValueError,match='train|Test|test'):make_env_factory(b,{})

def fixture_bundle(folder,split='train',payload_split=None):
    from experiments.e1.rl_training_utils import SPLIT_WINDOWS,PAYLOAD_FILES,VERSION
    from experiments.e1.selection import TemporalEnvironment
    from experiments.e1.environment import EnvironmentalSample
    from experiments.e1.workload import FrozenWorkload
    from experiments.e1.contracts import utc_time,interval_times,write_json,file_hash,content_hash
    from experiments.e1.config import ROOT
    from experiments.e1.cars_experiment import FIXED_PARAMETERS
    from copy import deepcopy
    import shutil
    cfg=json.loads((ROOT/'experiments/e1/configs/e1_5b_ppo.json').read_text())
    start,end,cap=map(utc_time,SPLIT_WINDOWS[payload_split or split])
    w=FrozenWorkload(start,end,cfg['source_sha256']['workload'],5,1.5,2020,())
    e=TemporalEnvironment(start,cap,cfg['source_sha256']['carbon'],cfg['source_sha256']['weather'],tuple(
        EnvironmentalSample(t,100,20,18) for t in interval_times(start,cap)))
    folder.mkdir();w.save(folder/'workload.json');e.save(folder/'environment.json')
    write_json(folder/'config.json',cfg);shutil.copyfile(ROOT/'configs/dcs/dc_config.json',folder/'dc_config.json')
    params=deepcopy(FIXED_PARAMETERS)
    params.update(threshold_g_per_kwh=161.075,arrival_start=start.isoformat(),arrival_end=end.isoformat(),stress_seed=510001)
    write_json(folder/'stress.json',{});write_json(folder/'parameters.json',params)
    manifest=dict(schema=VERSION,split=split,factor=1,stress_seed=510001,tasks=0,
        capacity=[50000,1000,80000],test_evaluation=False,config_sha256=content_hash(cfg),
        workload_sha256=w.sha256,environment_sha256=e.sha256,
        files={name:file_hash(folder/name) for name in PAYLOAD_FILES})
    write_json(folder/'bundle.json',manifest)
    return manifest

def test_train_and_validation_are_distinct_frozen_inputs(tmp_path):
    from experiments.e1.rl_training_utils import load_bundle
    fixture_bundle(tmp_path/'train');fixture_bundle(tmp_path/'val','validation')
    a=load_bundle(tmp_path/'train',allowed_split='train')
    b=load_bundle(tmp_path/'val',allowed_split='validation')
    assert a.workload.sha256!=b.workload.sha256
    assert a.environment.sha256!=b.environment.sha256
    assert a.environment.end<b.environment.start

def test_renamed_validation_payload_cannot_be_used_as_train(tmp_path):
    from experiments.e1.rl_training_utils import load_bundle
    fixture_bundle(tmp_path/'wrong','train',payload_split='validation')
    with pytest.raises(ValueError,match='UTC bounds'):load_bundle(tmp_path/'wrong',allowed_split='train')

def test_payload_tampering_fails_before_physics(tmp_path):
    from experiments.e1.rl_training_utils import load_bundle
    fixture_bundle(tmp_path/'bundle')
    (tmp_path/'bundle'/'dc_config.json').write_text('{"modified":true}')
    with pytest.raises(ValueError,match='hash mismatch'):load_bundle(tmp_path/'bundle',allowed_split='train')

def test_freeze_refuses_existing_output_before_opening_raw_inputs(tmp_path):
    from experiments.e1.rl_training_utils import freeze_bundle,SPLIT_WINDOWS
    cfg={'test_evaluation':False,'threshold_g_per_kwh':161.075,'windows':SPLIT_WINDOWS,
         'stress_seeds':{'train':[510001]},'precheck_factors':[1]}
    with pytest.raises(FileExistsError):freeze_bundle(cfg,'train',1,510001,tmp_path)

def test_resealed_config_cannot_silently_change_sla(tmp_path):
    from experiments.e1.rl_training_utils import load_bundle
    from experiments.e1.contracts import write_json,content_hash,file_hash
    b=tmp_path/'bundle';manifest=fixture_bundle(b)
    cfg=json.loads((b/'config.json').read_text());cfg['sla_multiplier']=2
    write_json(b/'config.json',cfg);manifest['config_sha256']=content_hash(cfg)
    manifest['files']['config.json']=file_hash(b/'config.json');write_json(b/'bundle.json',manifest)
    with pytest.raises(ValueError,match='SLA|sla'):load_bundle(b,allowed_split='train')

def test_resealed_parameters_cannot_silently_change_risk_rule(tmp_path):
    from experiments.e1.rl_training_utils import load_bundle
    from experiments.e1.contracts import write_json,file_hash
    b=tmp_path/'bundle';manifest=fixture_bundle(b)
    p=json.loads((b/'parameters.json').read_text());p['risk_breakpoints']=[0.7,1]
    write_json(b/'parameters.json',p);manifest['files']['parameters.json']=file_hash(b/'parameters.json')
    write_json(b/'bundle.json',manifest)
    with pytest.raises(ValueError,match='parameters|risk'):load_bundle(b,allowed_split='train')

def test_resealed_physics_cannot_change_original_formulas(tmp_path):
    from experiments.e1.rl_training_utils import load_bundle
    from experiments.e1.contracts import write_json,file_hash
    b=tmp_path/'bundle';manifest=fixture_bundle(b)
    write_json(b/'dc_config.json',{'different_physics':True})
    manifest['files']['dc_config.json']=file_hash(b/'dc_config.json');write_json(b/'bundle.json',manifest)
    with pytest.raises(ValueError,match='physical'):load_bundle(b,allowed_split='train')
