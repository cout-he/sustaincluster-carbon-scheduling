"""Checkpoint atomicity, operational gates, and full-episode resume semantics."""
import json
import random

import numpy as np
import pytest

from experiments.joint_v2.checkpoint import CheckpointStore, global_rng_state, restore_global_rng
from experiments.joint_v2.config import load_config
from experiments.joint_v2.selection import aggregate, update_lambda, advance_curriculum


def summary(**changes):
    result = dict(scenario_role='development', scenario_id='s', arrived=100, completed=100,
        unfinished=0, deadline_right_censored=0, temperature_violations=0, dmr=.04,
        deadline_misses=4, carbon_kg=100., incremental_carbon_kg=10.,
        fixed_window_complete=True, all_arrivals_observed=True, complete_episode_for_constraint=True)
    result.update(changes)
    return result


def test_best_feasible_rejects_censored_or_incomplete_work_and_wrong_roles():
    assert aggregate([summary()])['feasible']
    for changes in ({'unfinished':1}, {'deadline_right_censored':1}, {'temperature_violations':1},
                    {'dmr':.051}, {'all_arrivals_observed':False}, {'fixed_window_complete':False}):
        assert not aggregate([summary(**changes)])['feasible']
    with pytest.raises(ValueError):
        aggregate([summary(scenario_role='train')])


def test_lambda_is_bounded_complete_train_only_and_curriculum_needs_success():
    cfg = load_config()
    value, audit = update_lambda(10., summary(scenario_role='train', dmr=.15), {}, cfg['constraint'])
    assert value == 12. and audit['applied']
    value, audit = update_lambda(4999., summary(scenario_role='train', dmr=1.), {}, cfg['constraint'])
    assert value == 5000.
    value, audit = update_lambda(10., summary(scenario_role='train', complete_episode_for_constraint=False), {}, cfg['constraint'])
    assert value == 10. and not audit['applied']
    with pytest.raises(ValueError):
        update_lambda(10., summary(), {}, cfg['constraint'])
    state = dict(stage=0, success_streak=0)
    for _ in range(10):
        state = advance_curriculum(state, summary(dmr=.2), cfg['curriculum'])
    assert state['stage'] == 0
    state = advance_curriculum(state, summary(), cfg['curriculum'])
    assert state['stage'] == 0
    assert advance_curriculum(state, summary(), cfg['curriculum'])['stage'] == 1


def test_atomic_checkpoint_pointers_reject_tampering_and_orphan_generations(tmp_path):
    store = CheckpointStore(tmp_path, config_identity='config', data_identity='data')
    store.save(dict(completed_episodes=1, payload=[1,2]), ('latest','best_deadline'))
    old = json.loads((tmp_path/'latest.json').read_text())
    (tmp_path/'orphan.tmp').write_bytes(b'interrupted write')
    assert store.load()['completed_episodes'] == 1
    store.save(dict(completed_episodes=2, payload=[3]), ('latest',))
    assert store.load()['completed_episodes'] == 2
    assert store.load('best_deadline')['completed_episodes'] == 1
    with pytest.raises(ValueError):
        CheckpointStore(tmp_path,config_identity='wrong',data_identity='data').load()
    pointer = json.loads((tmp_path/'latest.json').read_text())
    (tmp_path/pointer['file']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='hash'):
        store.load()
    assert store.load('best_deadline')['completed_episodes'] == old['completed_episodes']


def test_global_rng_restores_python_numpy_and_torch():
    import torch
    state = global_rng_state()
    a = (random.random(),float(np.random.random()),torch.rand(3))
    restore_global_rng(state)
    b = (random.random(),float(np.random.random()),torch.rand(3))
    assert a[:2] == b[:2]
    assert torch.equal(a[2],b[2])


def test_joint_resume_matches_uninterrupted_agents_sampler_constraint_and_logs(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from experiments.e1.config import ROOT
    from tests.e1.helpers import inputs,task
    from experiments.joint_v2 import train_v2
    from experiments.joint_v2.ppo import IPPOAgent
    pool=tmp_path/'pool';pool.mkdir()
    (pool/'manifest.json').write_text('{"fixture":"controlled resume"}')
    (pool/'config.json').write_text('{"sampler":{}}')
    def scenario(name,role,load):
        workload,environment=inputs([task(name,duration=15,cpu=load,mem=.5,sla=2)],120)
        from dataclasses import replace
        workload=replace(workload,end=workload.start+__import__('pandas').Timedelta(minutes=30))
        return SimpleNamespace(scenario_id=name,role=role,workload=workload,environment=environment,
            capacity=(2,1,3),parameters={'threshold_g_per_kwh':150.},
            datacenter=dict(location='US-CAL-CISO',total_cores=50000,total_gpus=1000,total_mem=80000,hvac_setpoint_c=22),
            model_path=ROOT/'configs/dcs/dc_config.json',
            identity=dict(source_fragment_id=name,sha256=name),
            statistics=dict(capacity_pressure=.2,synthetic=False,augmentation={}))
    train=[scenario('first','train',.5),scenario('second','train',.7)]
    development=[scenario('development','development',.6)]
    monkeypatch.setattr(train_v2,'_role_sets',lambda _: (train,development))
    cfg=load_config();cfg['monitor']['tensorboard']=False
    cfg['ppo'].update(n_epochs=1,batch_size=4,net_arch=[8])
    cfg['validation']['heldout_final_check']=False
    full=train_v2.run(pool,tmp_path/'full',config=cfg,total_steps=100,wall_seconds=90,
                      max_episodes=3,skip_pretrain=True)
    train_v2.run(pool,tmp_path/'resumed',config=cfg,total_steps=100,wall_seconds=90,
                 max_episodes=1,skip_pretrain=True)
    resumed=train_v2.run(pool,tmp_path/'resumed',config=cfg,resume='latest',
        total_steps=100,wall_seconds=90,max_episodes=3,skip_pretrain=True)
    for key in ('counters','sampler_draws','curriculum','deadline_multiplier','component_hashes'):
        assert full[key]==resumed[key],key
    assert full['completed_episodes']==resumed['completed_episodes']==3
    def saved(name):
        pointer=json.loads((tmp_path/name/'checkpoints/latest.json').read_text())
        store=CheckpointStore(tmp_path/name/'checkpoints',config_identity=pointer['config_identity'],data_identity=pointer['data_identity'])
        return store.load()
    a,b=saved('full'),saved('resumed')
    assert a['sampler']==b['sampler']
    assert [r['scenario_id'] for r in a['episodes']]==[r['scenario_id'] for r in b['episodes']]
    for role in ('task','cooling'):
        first,second=IPPOAgent.from_state(a['agents'][role]),IPPOAgent.from_state(b['agents'][role])
        observation=np.zeros(first.obs_dim,dtype=np.float32);mask=np.ones(first.action_dim,dtype=bool)
        assert [first.act(observation,mask).action for _ in range(20)]==[second.act(observation,mask).action for _ in range(20)]
    assert (tmp_path/'resumed/monitor/training-curves.png').is_file()
    assert len(json.loads((tmp_path/'resumed/monitor/episodes.json').read_text()))==3


def test_step_budget_partial_episode_is_not_a_successful_terminal(monkeypatch,tmp_path):
    # Reuse the tiny real-plant fixture from the environment contracts.
    from experiments.joint_v2 import train_v2
    from .test_env import fixture_scenario
    train=fixture_scenario()
    train.role='train'
    development=fixture_scenario();development.role='development'
    monkeypatch.setattr(train_v2,'_role_sets',lambda _:([train],[development]))
    pool=tmp_path/'pool';pool.mkdir();(pool/'manifest.json').write_text('{}');(pool/'config.json').write_text('{"sampler":{}}')
    cfg=load_config();cfg['monitor']['tensorboard']=False;cfg['validation']['heldout_final_check']=False
    result=train_v2.run(pool,tmp_path/'run',config=cfg,total_steps=1,wall_seconds=60,skip_pretrain=True)
    assert result['completed_episodes']==0 and result['counters']['physical_steps']==0
    assert result['last_run']['partial_episode']['updates_applied'] is False
    assert result['sampler_draws']==0 and result['best_feasible'] is None
