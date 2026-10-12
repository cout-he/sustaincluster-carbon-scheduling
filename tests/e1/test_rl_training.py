"""PPO training boundary contracts over the real, unchanged E1 execution kernel."""
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from .helpers import KnownPower, inputs, task


def tiny_factory(*, truncated=False):
    """Physical formula is isolated; task execution and rewards remain real."""
    from experiments.e1.rl_env import CarbonSchedulingEnv
    workload, environment = inputs([
        task('a', duration=45), task('b', arrival=15, duration=15, sla=1),
        task('c', arrival=30, cpu=.25, mem=.25, duration=15)], 90)
    def create():
        return CarbonSchedulingEnv(workload, environment, (1, 1, 1), KnownPower,
            {'threshold_g_per_kwh':161.075}, stop_at_drain=True,
            max_episode_steps=2 if truncated else None)
    return create


def tiny_config():
    return {
        'ppo':{'learning_rate':3e-4, 'n_steps':16, 'batch_size':16,
            'n_epochs':2, 'gamma':.99, 'gae_lambda':.95, 'clip_range':.2,
            'ent_coef':0., 'vf_coef':.5, 'max_grad_norm':.5,
            'normalize_advantage':True},
        'policy_kwargs':{'net_arch':[64,64]},
        'algorithm':'PPO-v0', 'runtime':{'device':'cpu','torch_threads':1}}


def run(directory, *, seed=55101, steps=64, truncated=False):
    from experiments.e1.rl_train import train_ppo
    config=tiny_config()
    if truncated: config['runtime']['fail_on_truncation']=False
    return train_ppo(tiny_factory(truncated=truncated), config, seed=seed,
        total_timesteps=steps, output_dir=directory)


def trace(directory):
    with gzip.open(directory/'steps.jsonl.gz', 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def test_training_changes_parameters_and_checkpoint_keeps_optimizer(tmp_path):
    # A learn() omission, fresh untrained checkpoint or dropped optimizer must fail.
    from stable_baselines3 import PPO
    result=run(tmp_path/'trained')
    assert result['num_timesteps']==64
    assert result['parameter_sha256_before'] != result['parameter_sha256_after']
    assert result['deterministic_roundtrip'] is True
    loaded=PPO.load(tmp_path/'trained'/'model.zip', device='cpu')
    assert loaded.policy.optimizer.state_dict()['state']
    assert result['optimizer_state_entries'] > 0
    assert len(result['checkpoint_sha256'])==64
    rows=trace(tmp_path/'trained')
    assert len(rows)==64 and set(r['action'] for r in rows)<={0,1,2}
    assert result['action_counts']==[sum(r['action']==a for r in rows) for a in range(3)]
    updates=json.loads((tmp_path/'trained'/'updates.json').read_text())
    assert updates and all(np.isfinite(u['train/loss']) for u in updates)


def test_saved_model_has_exact_deterministic_predictions_after_reload(tmp_path):
    from stable_baselines3 import PPO
    result=run(tmp_path/'trained')
    loaded=PPO.load(tmp_path/'trained'/'model.zip', device='cpu')
    records=json.loads((tmp_path/'trained'/'roundtrip.json').read_text())
    observations=np.asarray(records['observations'], dtype=np.float32)
    actions,_=loaded.predict(observations, deterministic=True)
    assert actions.tolist()==records['actions_before_save']==records['actions_after_load']
    assert result['deterministic_roundtrip']


def test_same_seed_repeats_parameters_actions_and_rewards_on_same_cpu(tmp_path):
    # Unseeded initialization/sampling or additional RNG use breaks this contract.
    first=run(tmp_path/'first',seed=55102)
    second=run(tmp_path/'second',seed=55102)
    assert first['parameter_sha256_after']==second['parameter_sha256_after']
    one,two=trace(tmp_path/'first'),trace(tmp_path/'second')
    assert [(r['action'],r['reward'],r['observation']) for r in one] == [
        (r['action'],r['reward'],r['observation']) for r in two]


def test_actor_trace_is_predecision_online_state_and_real_interval_accounting(tmp_path):
    result=run(tmp_path/'trained')
    rows=trace(tmp_path/'trained')
    env=tiny_factory()(); expected,_=env.reset(seed=55101)
    for row in rows:
        np.testing.assert_array_equal(row['observation'], expected)
        assert row['online_state']['pending_tasks']==env.observation_raw['pending_tasks']
        expected,reward,terminated,truncated,info=env.step(row['action'])
        assert row['reward']==reward and row['reward_terms']==info['reward']
        assert row['interval']==info['interval']
        assert row['terminated']==terminated and row['truncated']==truncated
        env.executor.check_ownership()
        assert sum(info['observed_counts'][s] for s in ('pending','running','completed'))==info['observed_counts']['arrived']
        assert all(info['interval'][key]<=1 for key in ('used_cores','used_gpus','used_mem'))
        if terminated or truncated: expected,_=env.reset()
    env.close()
    assert result['reward_components']['carbon_cost']==pytest.approx(sum(r['reward_terms']['carbon_cost'] for r in rows))
    assert result['reward_components']['deadline_cost']==pytest.approx(sum(r['reward_terms']['deadline_cost'] for r in rows))


def test_training_preserves_censored_backlog_penalty_and_deduplicated_misses(tmp_path):
    run(tmp_path/'truncated', truncated=True)
    rows=trace(tmp_path/'truncated')
    endings=[r for r in rows if r['truncated']]
    assert endings and all(r['reward_terms']['unfinished_tasks']>0 for r in endings)
    assert all(r['reward_terms']['terminal_cost']>0 for r in endings)
    episode_ids=[]
    for row in rows:
        episode_ids.extend(row['reward_terms']['new_deadline_miss_ids'])
        assert len(episode_ids)==len(set(episode_ids))
        if row['terminated'] or row['truncated']: episode_ids=[]


def test_training_rejects_overwrite_and_unaligned_rollout_budget(tmp_path):
    from experiments.e1.rl_train import train_ppo
    destination=tmp_path/'existing';destination.mkdir()
    marker=destination/'old.txt';marker.write_text('preserve')
    with pytest.raises(FileExistsError): run(destination)
    assert marker.read_text()=='preserve'
    with pytest.raises(ValueError,match='n_steps'):
        run(tmp_path/'unaligned', steps=65)
    assert not (tmp_path/'unaligned').exists()


def test_training_rejects_undeclared_accelerator_without_creating_output(tmp_path):
    from experiments.e1.rl_train import train_ppo
    configuration=tiny_config();configuration['runtime']['device']='cuda'
    with pytest.raises(ValueError,match='cpu'):
        train_ppo(tiny_factory(),configuration,seed=55101,total_timesteps=64,
                  output_dir=tmp_path/'gpu')
    assert not (tmp_path/'gpu').exists()


def test_formal_training_aborts_on_safety_censoring_and_preserves_failure(tmp_path):
    # SB3 time-limit bootstrapping must not silently train on censored real cases.
    from experiments.e1.rl_train import train_ppo
    with pytest.raises(RuntimeError,match='truncat'):
        train_ppo(tiny_factory(truncated=True),tiny_config(),seed=55101,
                  total_timesteps=64,output_dir=tmp_path/'unsafe')
    failure=json.loads((tmp_path/'unsafe'/'failure.json').read_text())
    assert failure['error_type']=='RuntimeError'
    endings=[r for r in trace(tmp_path/'unsafe') if r['truncated']]
    assert len(endings)==1 and endings[0]['reward_terms']['terminal_cost']>0


def test_declared_optional_target_kl_and_partial_budget_tail_are_recorded(tmp_path):
    # Config integration must not drop declared hyperparameters or hide partial work.
    from experiments.e1.rl_train import train_ppo
    config=tiny_config();config['ppo']['target_kl']=None
    result=train_ppo(tiny_factory(),config,seed=55101,total_timesteps=64,
                     output_dir=tmp_path/'candidate')
    assert result['peak_rss_kib']>0
    assert result['budget_ends_mid_episode']
    tail=result['partial_episode']
    assert tail['length']==result['current_partial_episode_steps']>0
    assert tail['environment_terminated'] is False and tail['environment_truncated'] is False
    assert sum(tail['observed_counts'][s] for s in ('pending','running','completed'))==tail['observed_counts']['arrived']
    rows=trace(tmp_path/'candidate')[-tail['length']:]
    assert tail['episode_return']==pytest.approx(sum(row['reward'] for row in rows))
    assert tail['carbon_kg']==pytest.approx(sum(row['reward_terms']['carbon_kg'] for row in rows))
    config['ppo']['target_kl']=-1
    with pytest.raises(ValueError,match='target_kl'):
        train_ppo(tiny_factory(),config,seed=55101,total_timesteps=64,
                  output_dir=tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()
