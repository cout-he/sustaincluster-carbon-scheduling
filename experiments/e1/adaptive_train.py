"""Independent E1-6B four-action PPO training; old PPO-v0 stays unchanged.

Only immutable Train scenes enter the sampler. Detailed traces and checkpoints
are private local artifacts, and the actor sees only the online 83-vector.
"""
import argparse
from collections import Counter
from copy import deepcopy
import csv
from dataclasses import asdict
import gzip
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

import gymnasium as gym
import numpy as np

from .contracts import content_hash, file_hash, write_json
from .rl_env import ObservationNormalization, RewardConfig
from .rl_train import _json_value, _validated_configuration, parameter_hash
from .task_risk_state import BudgetParameters

VERSION = 'e1-6b-adaptive-ppo-v1'
GROUPS = {'plain':(False,False), 'state':(True,False),
          'shield':(False,True), 'combined':(True,True)}
OBSERVATION_SIZE = 83


def _seed(value):
    if isinstance(value,bool) or not isinstance(value,int) or value < 0:
        raise ValueError('scene seed must be a nonnegative integer')
    return value


def _group(value):
    if value not in GROUPS:
        raise ValueError('group must be plain/state/shield/combined')
    return value


def _definitions(config):
    reward=RewardConfig(**config.get('reward',asdict(RewardConfig())))
    normalization=ObservationNormalization(**config.get('normalization',asdict(ObservationNormalization())))
    if reward!=RewardConfig() or normalization!=ObservationNormalization():
        raise ValueError('E1-6B freezes the shared Reward and engineering normalization')
    supplied=config.get('risk_parameters',asdict(BudgetParameters()))
    if supplied!=asdict(BudgetParameters()):
        raise ValueError('E1-6A task-risk parameters must remain frozen')
    if config.get('threshold_g_per_kwh',161.075)!=161.075:
        raise ValueError('Frozen CI threshold must be 161.075')
    return reward,normalization


class EpisodeSceneEnv(gym.Env):
    """Alternate two Train scenes, with seed%2 as the initial scene offset.

    1:1 describes episode selection, not environment-step occupancy. Reset with
    an explicit seed restarts the sequence; ordinary automatic resets advance.
    Every reset owns a fresh physical adapter, retained across that episode.
    """
    metadata={'render_modes':[]}
    render_mode=None

    def __init__(self,bundles,*,risk_state,risk_filter,scene_seed,environment_kwargs=None):
        super().__init__()
        if not isinstance(risk_state,bool) or not isinstance(risk_filter,bool):
            raise ValueError('risk state/filter switches must be bool')
        self.bundles=tuple(bundles)
        if len(self.bundles)!=2 or tuple(b.factor for b in self.bundles)!=(1.5,2.):
            raise ValueError('Train scene table must be the ordered 1.5/2 pair')
        if any(b.split!='train' for b in self.bundles):
            raise ValueError('Only Train scenes are permitted; Validation/Test are forbidden')
        if any(b.capacity!=self.bundles[0].capacity for b in self.bundles):
            raise ValueError('Train scenes must share CPU/GPU/MEM capacity')
        if any(b.parameters.get('threshold_g_per_kwh')!=161.075 for b in self.bundles):
            raise ValueError('Train scenes must preserve fixed CI threshold')
        self.risk_state,self.risk_filter=risk_state,risk_filter
        self.scene_seed=_seed(scene_seed)
        self.environment_kwargs=deepcopy(environment_kwargs or {})
        if set(self.environment_kwargs)-{'reward_config','normalization','max_episode_steps'}:
            raise ValueError('Undeclared environment override')
        self._next_episode=0
        self._active=None
        self.current_scene=None
        self.action_space=gym.spaces.Discrete(4)
        self.observation_space=gym.spaces.Box(-1.,1.,shape=(OBSERVATION_SIZE,),dtype=np.float32)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.scene_seed=_seed(seed)
            self._next_episode=0
        self.close()
        index=(self.scene_seed%2+self._next_episode)%2
        bundle=self.bundles[index]
        self.current_scene=dict(index=index,episode_index=self._next_episode,split='train',
            factor=bundle.factor,stress_seed=bundle.stress_seed,
            workload_sha256=bundle.workload.sha256,environment_sha256=bundle.environment.sha256,
            bundle_identity_sha256=content_hash(bundle.identity))
        self._next_episode+=1
        from .adaptive_rl_env import AdaptiveCarbonSchedulingEnv
        self._active=AdaptiveCarbonSchedulingEnv(bundle.workload,bundle.environment,bundle.capacity,
            bundle.power_factory,bundle.parameters,risk_state=self.risk_state,
            risk_filter=self.risk_filter,stop_at_drain=True,**self.environment_kwargs)
        observation,info=self._active.reset(seed=seed,options=options)
        self.action_space.seed(seed)
        return observation,self._scene_info(info)

    def _scene_info(self,info):
        return dict(info,scene=deepcopy(self.current_scene))

    def step(self,action):
        if self._active is None:
            raise RuntimeError('reset() is required before a scene step')
        observation,reward,terminated,truncated,info=self._active.step(action)
        return observation,reward,terminated,truncated,self._scene_info(info)

    @property
    def executor(self):
        return None if self._active is None else self._active.executor

    @property
    def observation_raw(self):
        if self._active is None:
            raise RuntimeError('reset() is required before observation access')
        return self._active.observation_raw

    def close(self):
        if self._active is not None:
            self._active.close()
            self._active=None


def make_training_factory(bundles,config,*,group,seed):
    """The caller must bind real bundle bytes through the existing split loader."""
    state,filtering=GROUPS[_group(group)]
    reward,normalization=_definitions(config)
    bundles=tuple(bundles)
    if any(b.split!='train' for b in bundles):
        raise ValueError('Training requires Train bundles; Test/Validation are forbidden')
    _seed(seed)
    return lambda:EpisodeSceneEnv(bundles,risk_state=state,risk_filter=filtering,
        scene_seed=seed,environment_kwargs=dict(reward_config=reward,normalization=normalization))


class _AdaptiveTrace(gym.Wrapper):
    def __init__(self,env,stream,*,fail_on_truncation):
        super().__init__(env)
        self.stream,self.fail_on_truncation=stream,fail_on_truncation
        self.steps=0
        self.action_counts=Counter();self.scene_steps=Counter();self.scene_episodes=Counter()
        self.component_values={key:[] for key in ('carbon_cost','wait_cost','deadline_cost','terminal_cost')}
        self.episodes=[];self.probe_observations=[]
        self._episode=0

    def reset(self,**kwargs):
        observation,info=self.env.reset(**kwargs)
        self._observation=observation.copy()
        self._episode_length=0;self._episode_return=0.
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
        scene=deepcopy(self.env.unwrapped.current_scene)
        result=self.env.step(action)
        observation,reward,terminated,truncated,info=result
        self.env.unwrapped.executor.check_ownership()
        self.env.unwrapped.executor.pool.inventory()
        self._last_info=_json_value(info)
        self.steps+=1;self.action_counts[int(action)]+=1
        self.scene_steps[str(scene['factor'])]+=1
        self._episode_length+=1;self._episode_return+=reward
        for key in self.component_values:
            value=float(info['reward'][key])
            self.component_values[key].append(value);self._episode_components[key]+=value
        for key in self._episode_raw:self._episode_raw[key]+=info['reward'][key]
        queues=self.env.unwrapped.executor.policy_steps[-1]
        row=dict(step=self.steps,episode=self._episode,scene=scene,action=int(action),
            observation=before.tolist(),online_state=online,reward=float(reward),
            reward_terms=_json_value(info['reward']),interval=_json_value(info['interval']),
            policy_diagnostics=_json_value(info['policy_diagnostics']),risk_counts=_json_value(info['risk_counts']),
            started_ids=queues['started_ids'],policy_deferred_ids=queues['policy_deferred_ids'],
            capacity_waiting_ids=queues['capacity_waiting_ids'],
            terminated=bool(terminated),truncated=bool(truncated))
        self.stream.write(json.dumps(row,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
        if len(self.probe_observations)<256:self.probe_observations.append(before.tolist())
        self._observation=observation.copy()
        if terminated or truncated:
            self.episodes.append(dict(episode=self._episode,scene=scene,ending_step=self.steps,
                length=self._episode_length,episode_return=self._episode_return,
                terminated=bool(terminated),truncated=bool(truncated),
                **self._episode_raw,**self._episode_components,summary=_json_value(info['final_summary'])))
            self.scene_episodes[str(scene['factor'])]+=1;self._episode+=1
        if truncated and self.fail_on_truncation:
            raise RuntimeError('Formal PPO episode reached a truncation/safety limit; censoring cannot silently bootstrap')
        return result


def train_adaptive_ppo(env_factory,config,*,seed,total_timesteps,output_dir):
    """Train a fresh four-action candidate with exact on-policy step budget."""
    config=_validated_configuration(config,seed,total_timesteps)
    if config.get('schema')!=VERSION:
        raise ValueError('New adaptive PPO schema is required')
    group=_group(config.get('group'))
    _definitions(config)
    if not callable(env_factory):raise ValueError('A fresh environment factory is required')
    output=Path(output_dir)
    if output.exists():raise FileExistsError('Research output already exists: '+str(output))
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    raw_env=env_factory()
    if (raw_env.observation_space.shape!=(OBSERVATION_SIZE,) or raw_env.action_space!=gym.spaces.Discrete(4)
        or (raw_env.unwrapped.risk_state,raw_env.unwrapped.risk_filter)!=GROUPS[group]):
        raw_env.close()
        raise ValueError('Group switches or 83-feature/Discrete(4) environment differ')
    config['training_identities']=[dict(split=b.split,factor=b.factor,
        workload_sha256=b.workload.sha256,environment_sha256=b.environment.sha256,
        identity=deepcopy(b.identity)) for b in raw_env.unwrapped.bundles]
    input_budget=dict(training_identities=config['training_identities'],seed=seed,
        total_timesteps=total_timesteps,scene_order=[1.5,2.],scene_offset=seed%2,
        sampling='alternating episodes',observation_size=OBSERVATION_SIZE,action_size=4)
    input_budget_sha256=content_hash(input_budget)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'training_config.json',dict(config,seed=seed,total_timesteps=total_timesteps))
    write_json(output/'dependencies.json',{name:importlib.metadata.version(name)
        for name in ('torch','stable-baselines3','gymnasium','numpy')})
    from .cars_experiment import _snapshot_sources
    from .run import _git
    from .temporal import experiment_sources
    sources=experiment_sources()
    source_sha256=content_hash(sources)
    _snapshot_sources(output,sources)
    write_json(output/'source-freeze.json',dict(sources=sources,source_sha256=source_sha256))
    write_json(output/'input-budget.json',input_budget)
    write_json(output/'provenance.json',dict(version=VERSION,training_split='train',group=group,
        python=sys.version,command=sys.argv,formal_test_evaluated=False,
        git=dict(head=_git('rev-parse','HEAD'),branch=_git('branch','--show-current')),
        source_sha256=source_sha256,input_budget_sha256=input_budget_sha256,
        config_sha256=file_hash(output/'training_config.json')))
    updates=[]
    class UpdateLog(BaseCallback):
        def _on_step(self):return True
        def _capture(self):
            values=self.model.logger.name_to_value
            if 'train/loss' not in values:return
            update={key:float(value) for key,value in values.items()
                    if key.startswith('train/') and np.isscalar(value)}
            if any(not math.isfinite(v) for v in update.values()):raise RuntimeError('Nonfinite PPO update metric')
            update['environment_steps']=int(self.num_timesteps)
            if not updates or update.get('train/n_updates')!=updates[-1].get('train/n_updates'):updates.append(update)
        def _on_rollout_start(self):self._capture()
        def _on_training_end(self):self._capture()
    started=time.perf_counter()
    stream=gzip.open(output/'steps.jsonl.gz','wt',encoding='utf8')
    vector=None;model=None;trace=None
    try:
        trace=_AdaptiveTrace(raw_env,stream,fail_on_truncation=config.get('runtime',{}).get('fail_on_truncation',True))
        vector=DummyVecEnv([lambda:Monitor(trace,filename=str(output/'monitor.csv'))])
        model=PPO('MlpPolicy',vector,seed=seed,device='cpu',verbose=0,
                  policy_kwargs=config['policy_kwargs'],**config['ppo'])
        before=parameter_hash(model.policy)
        parameter_count=sum(p.numel() for p in model.policy.parameters())
        model.learn(total_timesteps=total_timesteps,callback=UpdateLog(),progress_bar=False)
        after=parameter_hash(model.policy)
        if model.num_timesteps!=total_timesteps or trace.steps!=total_timesteps:
            raise RuntimeError('Actual PPO step budget differs from declared budget')
        if before==after:raise RuntimeError('PPO completed without a parameter update')
        if experiment_sources()!=sources:
            raise RuntimeError('Source changed during frozen PPO training')
        model.save(output/'model.zip')
        observations=np.asarray(trace.probe_observations,dtype=np.float32)
        actions_before,_=model.predict(observations,deterministic=True)
        loaded=PPO.load(output/'model.zip',device='cpu')
        actions_after,_=loaded.predict(observations,deterministic=True)
        roundtrip=bool(np.array_equal(actions_before,actions_after) and parameter_hash(loaded.policy)==after)
        if not roundtrip:raise RuntimeError('Checkpoint deterministic roundtrip failed')
        optimizer_entries=len(loaded.policy.optimizer.state_dict()['state'])
        if not optimizer_entries:raise RuntimeError('Checkpoint omitted trained optimizer state')
        write_json(output/'roundtrip.json',dict(observations=observations.tolist(),
            actions_before_save=actions_before.tolist(),actions_after_load=actions_after.tolist()))
        write_json(output/'updates.json',updates)
        with (output/'updates.csv').open('w',encoding='utf8',newline='') as stream_csv:
            writer=csv.DictWriter(stream_csv,fieldnames=sorted({k for u in updates for k in u}))
            writer.writeheader();writer.writerows(updates)
        write_json(output/'episodes.json',trace.episodes)
        with (output/'episodes.csv').open('w',encoding='utf8',newline='') as stream_csv:
            fields=['episode','ending_step','factor','length','episode_return','terminated','truncated',
                    'carbon_kg','wait_task_minutes','new_deadline_misses',
                    'carbon_cost','wait_cost','deadline_cost','terminal_cost']
            writer=csv.DictWriter(stream_csv,fieldnames=fields);writer.writeheader()
            writer.writerows({key:(row['scene']['factor'] if key=='factor' else row[key]) for key in fields}
                            for row in trace.episodes)
        stream.close()
        partial=not (trace.episodes and trace.episodes[-1]['ending_step']==trace.steps)
        partial_record=None
        if partial and trace._episode_length:
            partial_record=dict(scene=_json_value(trace.env.unwrapped.current_scene),length=trace._episode_length,
                episode_return=trace._episode_return,**trace._episode_raw,**trace._episode_components,
                observed_counts=trace._last_info['observed_counts'],
                environment_terminated=False,environment_truncated=False,
                ending_reason='PPO step budget; unfinished rollout bootstrapped, not environment truncation or evaluation')
        try:
            import resource
            peak_rss=float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if sys.platform=='darwin':peak_rss/=1024
        except ImportError:peak_rss=None
        summary=dict(schema=VERSION,group=group,risk_state=GROUPS[group][0],risk_filter=GROUPS[group][1],
            observation_size=OBSERVATION_SIZE,action_size=4,parameter_count=parameter_count,
            training_identities=config['training_identities'],input_budget_sha256=input_budget_sha256,
            source_sha256=source_sha256,
            seed=seed,num_timesteps=int(model.num_timesteps),elapsed_seconds=time.perf_counter()-started,
            completed_episodes=len(trace.episodes),peak_rss_kib=peak_rss,
            peak_rss_scope='process high-water mark; includes prior allocations',
            current_partial_episode_steps=trace._episode_length if partial else 0,partial_episode=partial_record,
            parameter_sha256_before=before,parameter_sha256_after=after,
            checkpoint_sha256=file_hash(output/'model.zip'),optimizer_state_entries=optimizer_entries,
            deterministic_roundtrip=roundtrip,action_counts=[trace.action_counts[i] for i in range(4)],
            scene_steps=dict(sorted(trace.scene_steps.items())),scene_episodes=dict(sorted(trace.scene_episodes.items())),
            scene_sampling='alternating ordered Train 1.5/2; seed%2 offset; 1:1 episode selection, not step occupancy',
            reward_components={key:math.fsum(values) for key,values in trace.component_values.items()},
            budget_ends_mid_episode=bool(trace._episode_length and partial),
            config_sha256=file_hash(output/'training_config.json'),
            actor_inputs='83-dimensional online vector only; risk block zeroed when disabled',
            reproducibility_scope='same inputs/dependencies; deterministic single-thread CPU; no cross-platform promise')
        summary['files']={path.name:file_hash(path) for path in sorted(output.iterdir()) if path.is_file()}
        write_json(output/'training_summary.json',summary)
        return summary
    except Exception as exc:
        if trace is not None:write_json(output/'episodes.json',trace.episodes)
        write_json(output/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),
                                             num_timesteps=getattr(model,'num_timesteps',0)))
        raise
    finally:
        stream.close()
        if vector is not None:vector.close()
        else:raw_env.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--group',choices=tuple(GROUPS),required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--steps',type=int,required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    config_path=Path(args.config)
    config=json.loads(config_path.read_text(encoding='utf8'))
    from .config import ROOT
    from .rl_training_utils import load_bundle
    scenes=config['scenes']
    if len(scenes)!=2 or any(s.get('split')!='train' for s in scenes):
        raise ValueError('Only the frozen Train scene table can enter PPO')
    bundles=[]
    for scene in scenes:
        path=Path(scene['input'])
        if not path.is_absolute():path=ROOT/path
        bundle=load_bundle(path,allowed_split='train')
        if bundle.factor!=scene['factor']:raise ValueError('Declared Train scene factor differs')
        if 'manifest_sha256' in scene and scene['manifest_sha256']!=file_hash(path/'bundle.json'):
            raise ValueError('Declared Train manifest identity differs')
        for key,actual in (('bundle_manifest_sha256',bundle.identity['bundle_manifest_sha256']),
                           ('workload_sha256',bundle.workload.sha256),('environment_sha256',bundle.environment.sha256)):
            if key in scene and scene[key]!=actual:raise ValueError('Declared scene identity differs: '+key)
        bundles.append(bundle)
    config=deepcopy(config);config['group']=args.group
    config['preregistered_config_sha256']=file_hash(config_path)
    config['training_identity']={'split':'train','scenes':[b.identity for b in bundles]}
    result=train_adaptive_ppo(make_training_factory(bundles,config,group=args.group,seed=args.seed),
        config,seed=args.seed,total_timesteps=args.steps,output_dir=args.output)
    print(json.dumps({k:result[k] for k in ('group','seed','num_timesteps','elapsed_seconds',
        'parameter_sha256_after','checkpoint_sha256','action_counts','scene_steps')},sort_keys=True))


if __name__=='__main__':main()
