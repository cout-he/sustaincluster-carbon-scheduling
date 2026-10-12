"""Ordinary, seeded CPU PPO baseline over the unchanged E1-5A environment.

Checkpoints and detailed logs are local research artifacts.  The actor receives
only the existing 40-dimensional online observation, never diagnostic records.
"""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

import gymnasium as gym
import numpy as np

from .contracts import canonical_bytes, file_hash, write_json


def parameter_hash(policy):
    """Canonical CPU parameter hash, independent of ZIP metadata or timestamps."""
    digest=hashlib.sha256()
    for name,tensor in sorted(policy.state_dict().items()):
        array=tensor.detach().cpu().contiguous().numpy()
        digest.update(canonical_bytes({'name':name,'dtype':str(array.dtype),'shape':list(array.shape)}))
        digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


def _json_value(value):
    if isinstance(value,dict): return {str(k):_json_value(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [_json_value(v) for v in value]
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    return value


class _TrainingTrace(gym.Wrapper):
    """Write the pre-action sensor and actual transition without changing either."""
    def __init__(self,env,stream,*,fail_on_truncation):
        super().__init__(env)
        self.stream=stream
        self.fail_on_truncation=fail_on_truncation
        self.steps=0
        self.action_counts=Counter()
        self.component_values={key:[] for key in ('carbon_cost','wait_cost','deadline_cost','terminal_cost')}
        self.episodes=[]
        self.probe_observations=[]
        self._episode=0

    def reset(self,**kwargs):
        observation,info=self.env.reset(**kwargs)
        self._observation=observation.copy()
        self._episode_length=0
        self._episode_return=0.
        self._episode_components={key:0. for key in self.component_values}
        self._episode_raw={key:0. for key in ('carbon_kg','wait_task_minutes','new_deadline_misses')}
        self._last_info=_json_value(info)
        return observation,info

    def step(self,action):
        before=self._observation.copy()
        raw=self.env.unwrapped.observation_raw
        online={key:_json_value(raw[key]) for key in (
            'ci_g_per_kwh','utilization','pending_tasks','running_tasks',
            'history_rho_resources','history_effective_minutes','history_cold',
            'buffer_steps','urgent_tasks','deadline_due_tasks','solo_latest_start_passed_tasks')}
        result=self.env.step(action)
        observation,reward,terminated,truncated,info=result
        self._last_info=_json_value(info)
        self.env.unwrapped.executor.check_ownership()
        self.steps+=1
        self.action_counts[int(action)]+=1
        self._episode_length+=1
        self._episode_return+=reward
        for key in self.component_values:
            value=float(info['reward'][key])
            self.component_values[key].append(value)
            self._episode_components[key]+=value
        for key in self._episode_raw: self._episode_raw[key]+=info['reward'][key]
        row=dict(step=self.steps,episode=self._episode,action=int(action),
                 observation=before.tolist(),online_state=online,reward=float(reward),
                 reward_terms=_json_value(info['reward']),interval=_json_value(info['interval']),
                 terminated=bool(terminated),truncated=bool(truncated))
        self.stream.write(json.dumps(row,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
        if len(self.probe_observations)<256: self.probe_observations.append(before.tolist())
        self._observation=observation.copy()
        if terminated or truncated:
            self.episodes.append(dict(episode=self._episode,ending_step=self.steps,
                length=self._episode_length,episode_return=self._episode_return,
                terminated=bool(terminated),truncated=bool(truncated),
                **self._episode_raw,**self._episode_components,
                summary=_json_value(info['final_summary'])))
            self._episode+=1
        if truncated and self.fail_on_truncation:
            raise RuntimeError('Formal PPO episode reached a truncation/safety limit; '
                               'censored transitions cannot silently use time-limit bootstrapping')
        return result


def _validated_configuration(config,seed,total_timesteps):
    if not isinstance(config,dict): raise ValueError('config must be a mapping')
    config=deepcopy(config)
    runtime=config.get('runtime',{})
    if runtime.get('device','cpu')!='cpu': raise ValueError('This declared PPO baseline requires cpu')
    if runtime.get('torch_threads',1)!=1: raise ValueError('This reproducible baseline requires one torch thread')
    if runtime.get('deterministic_algorithms',True) is not True:
        raise ValueError('This baseline requires deterministic_algorithms=True')
    if not isinstance(runtime.get('fail_on_truncation',True),bool):
        raise ValueError('fail_on_truncation must be bool')
    for name,value in (('seed',seed),('total_timesteps',total_timesteps)):
        if isinstance(value,bool) or not isinstance(value,int) or value<(0 if name=='seed' else 1):
            raise ValueError(f'{name} must be a nonnegative seed or positive step count')
    ppo=config.get('ppo',{})
    required={'learning_rate','n_steps','batch_size','n_epochs','gamma','gae_lambda','clip_range',
              'ent_coef','vf_coef','max_grad_norm','normalize_advantage'}
    if not required.issubset(ppo) or set(ppo)-required-{'target_kl'}:
        raise ValueError('Explicit PPO hyperparameters required; unsupported options rejected')
    target=ppo.get('target_kl')
    if target is not None and (isinstance(target,bool) or not isinstance(target,(int,float))
                              or not math.isfinite(target) or target<=0):
        raise ValueError('target_kl must be None or finite positive numeric')
    for name in ('n_steps','batch_size','n_epochs'):
        if isinstance(ppo[name],bool) or not isinstance(ppo[name],int) or ppo[name]<1:
            raise ValueError(f'{name} must be a positive integer')
    if ppo['n_steps']<2 or ppo['batch_size']<2 or ppo['n_steps']%ppo['batch_size']:
        raise ValueError('n_steps must be a multiple of batch_size; both must exceed one')
    if total_timesteps%ppo['n_steps']:
        raise ValueError('total_timesteps must be an exact multiple of n_steps; no silent budget rounding')
    if config.get('policy_kwargs',{})!={'net_arch':[64,64]}:
        raise ValueError('This baseline freezes a [64, 64] MlpPolicy architecture')
    return config


def train_ppo(env_factory,config,*,seed,total_timesteps,output_dir):
    """Train a new candidate; no existing output or environment is overwritten.

    The factory must create the already verified E1 environment on the declared
    Train bundle.  CLI use enforces the split before opening frozen documents.
    SB3 saves policy, optimizer and algorithm state in its standard checkpoint.
    Reproduction is scoped to identical inputs, dependencies and single-thread
    CPU runtime; the checkpoint ZIP byte hash is a file identity, not a cross-run
    floating-point reproducibility claim. Formal full-Drain training rejects any
    censored episode; disabling fail_on_truncation is reserved for explicit tiny
    diagnostics of the environment's existing backlog reward.
    """
    config=_validated_configuration(config,seed,total_timesteps)
    if not callable(env_factory): raise ValueError('A fresh environment factory is required')
    output=Path(output_dir)
    if output.exists(): raise FileExistsError(f'Research output already exists: {output}')
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True,exist_ok=False)
    frozen_config=dict(config,seed=seed,total_timesteps=total_timesteps)
    write_json(output/'training_config.json',frozen_config)
    write_json(output/'dependencies.json',{
        name:importlib.metadata.version(name) for name in ('torch','stable-baselines3','gymnasium','numpy')})
    updates=[]
    class UpdateLog(BaseCallback):
        def _on_step(self): return True
        def _capture(self):
            values=self.model.logger.name_to_value
            if 'train/loss' not in values: return
            update={key:float(value) for key,value in values.items() if key.startswith('train/') and np.isscalar(value)}
            if any(not math.isfinite(value) for value in update.values()):
                raise RuntimeError('Nonfinite PPO update metric')
            update['environment_steps']=int(self.num_timesteps)
            if not updates or update.get('train/n_updates')!=updates[-1].get('train/n_updates'):
                updates.append(update)
        def _on_rollout_start(self): self._capture()
        def _on_training_end(self): self._capture()

    started=time.perf_counter()
    stream=gzip.open(output/'steps.jsonl.gz','wt',encoding='utf-8')
    vector=None
    model=None
    trace=None
    try:
        raw_env=env_factory()
        if raw_env.observation_space.shape!=(40,) or raw_env.action_space!=gym.spaces.Discrete(3):
            raw_env.close()
            raise ValueError('The verified E1 40-feature / Discrete(3) environment is required')
        trace=_TrainingTrace(raw_env,stream,fail_on_truncation=config.get('runtime',{}).get('fail_on_truncation',True))
        vector=DummyVecEnv([lambda:Monitor(trace,filename=str(output/'monitor.csv'))])
        model=PPO('MlpPolicy',vector,seed=seed,device='cpu',verbose=0,
                  policy_kwargs=config['policy_kwargs'],**config['ppo'])
        before=parameter_hash(model.policy)
        model.learn(total_timesteps=total_timesteps,callback=UpdateLog(),progress_bar=False)
        after=parameter_hash(model.policy)
        if model.num_timesteps!=total_timesteps or trace.steps!=total_timesteps:
            raise RuntimeError('Actual PPO step budget differs from declared budget')
        if before==after: raise RuntimeError('PPO completed without a parameter update')
        model.save(output/'model.zip')
        observations=np.asarray(trace.probe_observations,dtype=np.float32)
        actions_before,_=model.predict(observations,deterministic=True)
        loaded=PPO.load(output/'model.zip',device='cpu')
        actions_after,_=loaded.predict(observations,deterministic=True)
        roundtrip=bool(np.array_equal(actions_before,actions_after) and parameter_hash(loaded.policy)==after)
        if not roundtrip: raise RuntimeError('Checkpoint deterministic roundtrip failed')
        optimizer_entries=len(loaded.policy.optimizer.state_dict()['state'])
        if not optimizer_entries: raise RuntimeError('Checkpoint omitted trained optimizer state')
        write_json(output/'roundtrip.json',dict(observations=observations.tolist(),
            actions_before_save=actions_before.tolist(),actions_after_load=actions_after.tolist()))
        write_json(output/'updates.json',updates)
        with (output/'updates.csv').open('w',encoding='utf-8',newline='') as update_stream:
            columns=sorted({key for row in updates for key in row})
            writer=csv.DictWriter(update_stream,fieldnames=columns)
            writer.writeheader();writer.writerows(updates)
        write_json(output/'episodes.json',trace.episodes)
        with (output/'episodes.csv').open('w',encoding='utf-8',newline='') as episode_stream:
            fields=['episode','ending_step','length','episode_return','terminated','truncated',
                    'carbon_kg','wait_task_minutes','new_deadline_misses',
                    'carbon_cost','wait_cost','deadline_cost','terminal_cost']
            writer=csv.DictWriter(episode_stream,fieldnames=fields)
            writer.writeheader()
            writer.writerows({key:row[key] for key in fields} for row in trace.episodes)
        stream.close()
        partial=not (trace.episodes and trace.episodes[-1]['ending_step']==trace.steps)
        partial_record=None
        if partial and trace._episode_length:
            partial_record=dict(length=trace._episode_length,
                episode_return=trace._episode_return,**trace._episode_raw,**trace._episode_components,
                observed_counts=trace._last_info['observed_counts'],
                environment_terminated=False,environment_truncated=False,
                ending_reason='PPO environment-step budget; unfinished rollout remains bootstrapped, '
                              'not an environment truncation or complete evaluation')
        try:
            import resource
            peak_rss=float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if sys.platform=='darwin': peak_rss/=1024
        except ImportError:
            peak_rss=None
        summary=dict(seed=seed,num_timesteps=int(model.num_timesteps),
            elapsed_seconds=time.perf_counter()-started,completed_episodes=len(trace.episodes),
            peak_rss_kib=peak_rss,peak_rss_scope='process high-water mark; includes prior allocations',
            current_partial_episode_steps=trace._episode_length if partial else 0,
            partial_episode=partial_record,
            parameter_sha256_before=before,parameter_sha256_after=after,
            checkpoint_sha256=file_hash(output/'model.zip'),optimizer_state_entries=optimizer_entries,
            deterministic_roundtrip=roundtrip,action_counts=[trace.action_counts[i] for i in range(3)],
            reward_components={key:math.fsum(values) for key,values in trace.component_values.items()},
            budget_ends_mid_episode=bool(trace._episode_length and partial),
            config_sha256=file_hash(output/'training_config.json'),
            actor_inputs='unchanged E1-5A 40-dimensional online observation only',
            reproducibility_scope='same input/dependencies, deterministic single-thread CPU; no cross-platform promise')
        summary['files']={path.name:file_hash(path) for path in sorted(output.iterdir()) if path.is_file()}
        write_json(output/'training_summary.json',summary)
        return summary
    except Exception as exc:
        if trace is not None: write_json(output/'episodes.json',trace.episodes)
        write_json(output/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),
            num_timesteps=getattr(model,'num_timesteps',0)))
        raise
    finally:
        stream.close()
        if vector is not None: vector.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',required=True)
    parser.add_argument('--config',required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--steps',type=int,required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    from .rl_training_utils import load_bundle,make_env_factory
    configuration=json.loads(Path(args.config).read_text(encoding='utf-8'))
    bundle=load_bundle(args.bundle,allowed_split='train')
    configuration['training_identity']={'split':bundle.split,'stress_seed':bundle.stress_seed,
                                   'factor':bundle.factor,'identity':bundle.identity}
    result=train_ppo(make_env_factory(bundle,configuration),configuration,seed=args.seed,
                     total_timesteps=args.steps,output_dir=args.output)
    print(json.dumps({key:result[key] for key in ('seed','num_timesteps','elapsed_seconds',
        'parameter_sha256_after','checkpoint_sha256','action_counts')},sort_keys=True))


if __name__=='__main__': main()
