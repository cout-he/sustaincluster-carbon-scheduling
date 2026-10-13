"""Actual PPO and scene-selection contracts for the new four-action controller."""
from dataclasses import replace
import gzip
import json

import numpy as np
import pytest

from .helpers import KnownPower, inputs, task


def bundles(*, tracked_power=None):
    from experiments.e1.rl_training_utils import ExperimentBundle
    first = [task('a', duration=45, sla=3),
             task('b', arrival=15, duration=15, sla=2),
             task('c', arrival=30, cpu=.25, mem=.25, duration=15, sla=2)]
    workload, _ = inputs(first, 90)
    _, environment = inputs([], 180)
    second = replace(workload, tasks=workload.tasks + (task('d', arrival=30, duration=15),))
    factory = tracked_power or KnownPower
    parameters = {'threshold_g_per_kwh':161.075, 'history_minutes':120}
    return tuple(ExperimentBundle('train', w, environment, (1,1,1), factory,
        parameters, 510001, factor, {'fixture':str(factor)})
        for w,factor in ((workload,1.5),(second,2.)))


def configuration(group='plain'):
    from experiments.e1.rl_env import ObservationNormalization, RewardConfig
    from dataclasses import asdict
    return dict(schema='e1-6b-adaptive-ppo-v1',group=group,
        ppo=dict(learning_rate=3e-4,n_steps=16,batch_size=16,n_epochs=2,
                 gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=0.,vf_coef=.5,
                 max_grad_norm=.5,normalize_advantage=True,target_kl=None),
        policy_kwargs={'net_arch':[64,64]},
        reward=asdict(RewardConfig()),normalization=asdict(ObservationNormalization()),
        runtime=dict(device='cpu',torch_threads=1,deterministic_algorithms=True))


def factory(group='plain',seed=55101,*,truncated=False):
    from experiments.e1.adaptive_train import EpisodeSceneEnv, GROUPS
    state,filtering = GROUPS[group]
    return lambda:EpisodeSceneEnv(bundles(),risk_state=state,risk_filter=filtering,
        scene_seed=seed,environment_kwargs={'max_episode_steps':2} if truncated else {})


def run(out,group='plain',seed=55101,*,truncated=False,fail_on_truncation=True):
    from experiments.e1.adaptive_train import train_adaptive_ppo
    cfg=configuration(group)
    cfg['runtime']['fail_on_truncation']=fail_on_truncation
    return train_adaptive_ppo(factory(group,seed,truncated=truncated),cfg,
        seed=seed,total_timesteps=64,output_dir=out)


def rows(out):
    with gzip.open(out/'steps.jsonl.gz','rt',encoding='utf8') as stream:
        return [json.loads(line) for line in stream]


def test_scene_seed_offsets_alternation_and_reset_restarts_same_sequence():
    # Random selection, action-dependent selection, or an ignored offset must fail.
    from experiments.e1.adaptive_train import EpisodeSceneEnv
    env=EpisodeSceneEnv(bundles(),risk_state=False,risk_filter=False,scene_seed=55101)
    selections=[]
    for i in range(5):
        observation,info=env.reset(seed=55101 if i==0 else None)
        selections.append(info['scene']['factor'])
        assert observation.shape==(83,)
    assert selections==[2.,1.5,2.,1.5,2.]
    _,info=env.reset(seed=55101)
    assert info['scene']['factor']==2.
    env.close()


@pytest.mark.parametrize('split',['validation','test'])
def test_scene_roles_rejected_before_any_physical_factory_is_called(split):
    # Label or role guards cannot silently train on a held-out cohort.
    from experiments.e1.adaptive_train import EpisodeSceneEnv
    def forbidden():raise AssertionError('physical factory called before role validation')
    invalid=tuple(replace(b,split=split,power_factory=forbidden) for b in bundles())
    with pytest.raises(ValueError,match='Train|train|Test|test'):
        EpisodeSceneEnv(invalid,risk_state=False,risk_filter=False,scene_seed=1)


def test_scene_reset_recreates_physics_and_step_preserves_it():
    # Rebuilding physics each step or retaining old-scene physics across reset fails.
    from experiments.e1.adaptive_train import EpisodeSceneEnv
    created=[]
    class Tracked(KnownPower):
        def __init__(self):created.append(self)
    env=EpisodeSceneEnv(bundles(tracked_power=Tracked),risk_state=True,risk_filter=True,scene_seed=0)
    env.reset(seed=0); first=env.executor.power_model
    env.step(0);assert env.executor.power_model is first
    env.reset();assert env.executor.power_model is not first and len(created)==2
    assert env.current_scene['factor']==2.
    env.close()


def test_real_training_updates_parameters_and_records_all_four_actions(tmp_path):
    from stable_baselines3 import PPO
    result=run(tmp_path/'trained')
    assert result['num_timesteps']==64
    assert result['parameter_sha256_before']!=result['parameter_sha256_after']
    assert result['deterministic_roundtrip'] and result['optimizer_state_entries']>0
    model=PPO.load(tmp_path/'trained'/'model.zip',device='cpu')
    assert model.observation_space.shape==(83,) and model.action_space.n==4
    assert model.policy.optimizer.state_dict()['state']
    trace=rows(tmp_path/'trained')
    assert len(trace)==64 and all(len(r['observation'])==83 for r in trace)
    assert set(r['action'] for r in trace)<={0,1,2,3}
    assert result['action_counts']==[sum(r['action']==a for r in trace) for a in range(4)]
    assert sum(result['scene_steps'].values())==64
    updates=json.loads((tmp_path/'trained'/'updates.json').read_text())
    assert updates and all(np.isfinite(u['train/loss']) for u in updates)


def test_training_trace_is_actual_preaction_state_and_filter_events(tmp_path):
    run(tmp_path/'trained','combined')
    env=factory('combined')();obs,_=env.reset(seed=55101)
    for row in rows(tmp_path/'trained'):
        np.testing.assert_array_equal(row['observation'],obs)
        assert row['scene']==env.current_scene
        obs,reward,terminated,truncated,info=env.step(row['action'])
        assert row['reward']==reward and row['reward_terms']==info['reward']
        assert row['interval']==info['interval']
        assert row['policy_diagnostics']==info['policy_diagnostics']
        assert row['risk_counts']==info['risk_counts']
        assert row['started_ids']==env.executor.policy_steps[-1]['started_ids']
        assert row['capacity_waiting_ids']==env.executor.policy_steps[-1]['capacity_waiting_ids']
        assert row['policy_deferred_ids']==env.executor.policy_steps[-1]['policy_deferred_ids']
        assert row['terminated']==terminated and row['truncated']==truncated
        env.executor.check_ownership()
        if terminated or truncated:obs,_=env.reset()
    env.close()


def test_same_seed_reproduces_parameters_scene_sequence_actions_and_rewards(tmp_path):
    first=run(tmp_path/'first','combined',seed=55102)
    second=run(tmp_path/'second','combined',seed=55102)
    assert first['parameter_sha256_after']==second['parameter_sha256_after']
    columns=('scene','action','observation','reward','reward_terms','policy_diagnostics')
    assert [[r[c] for c in columns] for r in rows(tmp_path/'first')]==[
        [r[c] for c in columns] for r in rows(tmp_path/'second')]


def test_four_groups_have_same_parameter_count_initialization_and_budget(tmp_path):
    # A smaller Plain network or group-specific hidden dimensions/budget breaks fairness.
    results=[run(tmp_path/group,group) for group in ('plain','state','shield','combined')]
    assert len({r['parameter_count'] for r in results})==1
    assert len({r['parameter_sha256_before'] for r in results})==1
    assert all(r['num_timesteps']==64 for r in results)
    for group in ('plain','shield'):
        assert all(all(v==0. for v in r['observation'][40:]) for r in rows(tmp_path/group))


def test_formal_training_rejects_censoring_but_diagnostic_keeps_terminal_liability(tmp_path):
    with pytest.raises(RuntimeError,match='truncat'):
        run(tmp_path/'formal',truncated=True)
    failure=json.loads((tmp_path/'formal'/'failure.json').read_text())
    assert failure['error_type']=='RuntimeError'
    run(tmp_path/'diagnostic',truncated=True,fail_on_truncation=False)
    trace=rows(tmp_path/'diagnostic');seen=set()
    for row in trace:
        misses=row['reward_terms']['new_deadline_miss_ids']
        assert not seen.intersection(misses);seen.update(misses)
        if row['truncated']:
            assert row['reward_terms']['unfinished_tasks']>0
            assert row['reward_terms']['terminal_cost']>0
        if row['terminated'] or row['truncated']:seen=set()


def test_overwrite_unaligned_budget_and_group_mismatch_rejected(tmp_path):
    from experiments.e1.adaptive_train import train_adaptive_ppo
    old=tmp_path/'old';old.mkdir();(old/'marker').write_text('keep')
    with pytest.raises(FileExistsError):run(old)
    assert (old/'marker').read_text()=='keep'
    with pytest.raises(ValueError,match='n_steps'):
        train_adaptive_ppo(factory(),configuration(),seed=55101,total_timesteps=65,output_dir=tmp_path/'bad')
    assert not (tmp_path/'bad').exists()
    with pytest.raises(ValueError,match='group|switch|filter|state'):
        train_adaptive_ppo(factory('shield'),configuration('plain'),seed=55101,
            total_timesteps=64,output_dir=tmp_path/'wrong')


def test_training_factory_rejects_changed_reward_and_creates_correct_group():
    from experiments.e1.adaptive_train import make_training_factory
    cfg=configuration('state')
    env=make_training_factory(bundles(),cfg,group='state',seed=55101)()
    obs,info=env.reset(seed=55101)
    assert obs.shape==(83,) and info['risk_state'] and not info['risk_filter']
    env.close()
    cfg['reward']['gamma']=11
    with pytest.raises(ValueError,match='reward|Reward'):
        make_training_factory(bundles(),cfg,group='state',seed=55101)


def test_training_metadata_binds_real_train_scenes_source_and_budget(tmp_path):
    # Caller-supplied scene labels or absent source versions cannot identify a model.
    result=run(tmp_path/'trained')
    recorded=json.loads((tmp_path/'trained'/'training_config.json').read_text())
    assert [i['factor'] for i in recorded['training_identities']]==[1.5,2.]
    assert all(i['split']=='train' for i in recorded['training_identities'])
    assert result['training_identities']==recorded['training_identities']
    assert len(result['input_budget_sha256'])==64
    source=json.loads((tmp_path/'trained'/'source-freeze.json').read_text())
    assert source['source_sha256']==result['source_sha256']
    assert 'experiments/e1/adaptive_train.py' in source['sources']
    provenance=json.loads((tmp_path/'trained'/'provenance.json').read_text())
    assert provenance['formal_test_evaluated'] is False
    assert provenance['training_split']=='train'
    assert provenance['input_budget_sha256']==result['input_budget_sha256']


def test_cli_rejects_a_bad_declared_train_manifest_before_training(tmp_path):
    # A stale or substituted bundle manifest must not be accepted by CLI.
    from experiments.e1.adaptive_train import main
    from .test_rl_training_utils import fixture_bundle
    fixture_bundle(tmp_path/'bundle')
    cfg=configuration()
    cfg['scenes']=[dict(split='train',factor=1.,input=str(tmp_path/'bundle'),manifest_sha256='0'*64)]*2
    cfgpath=tmp_path/'config.json';cfgpath.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='manifest|identity'):
        main(['--config',str(cfgpath),'--group','plain','--seed','55101',
              '--steps','64','--output',str(tmp_path/'out')])
    assert not (tmp_path/'out').exists()
