"""E1-5B finite-cohort PPO evaluation without access to the reserved Test split.

All modes use the existing executor and physical adapter. A group first drains
each policy, then accounts for idle energy with each policy's still-live model
until the latest group endpoint. Actor inputs are the current 40 sensors only.
"""
import argparse
import copy
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

from .cars import make_cars_policy
from .contracts import content_hash, file_hash, finite, write_json
from .drain_metrics import phase_summary
from .engine import LocalExecutor
from .rl_env import (ACTION_POLICIES, CarbonSchedulingEnv, OBSERVATION_FIELDS,
                     ObservationNormalization, RewardConfig)
from .run import _fresh_run, _write_result
from .temporal import write_csv

EVALUATION_SCHEMA='e1-5b-common-window-evaluation-v1'


@dataclass(frozen=True)
class EvaluationBundle:
    """Small programmatic bundle; root ExperimentBundle is also accepted."""
    split: str
    workload: object
    environment: object
    capacity: tuple
    power_factory: object
    parameters: dict
    stress_seed: int
    factor: float
    identity: dict=field(default_factory=dict)
    reward_config: object=None
    normalization: object=None


@dataclass(frozen=True)
class EvaluationCandidate:
    name: str
    predictor: object
    seed: int=None
    checkpoint_sha256: str=None
    training_steps: int=None
    training_elapsed_seconds: float=None
    training_config_sha256: str=None


def _allowed_split(split):
    # Call before reading checkpoints, creating outputs, or starting physics.
    if split not in ('train','validation'):
        raise ValueError('E1-5B prohibits Test evaluation; only train/validation are allowed')


def load_ppo_candidate(checkpoint, metadata, *, split='validation'):
    """Load a checkpoint bound to its immutable Train protocol and artifacts."""
    _allowed_split(split)
    required={'name','seed','training_steps','checkpoint_sha256'}
    if (not required.issubset(metadata) or
        re.fullmatch(r'[0-9a-f]{64}',str(metadata.get('checkpoint_sha256',''))) is None):
        raise ValueError('Candidate metadata requires an explicit checkpoint SHA256, seed and steps')
    checkpoint=Path(checkpoint)
    digest=file_hash(checkpoint)
    if metadata['checkpoint_sha256']!=digest:
        raise ValueError('Checkpoint hash mismatch')
    parent=checkpoint.parent
    summary_path,config_path=parent/'training_summary.json',parent/'training_config.json'
    if not summary_path.is_file() or not config_path.is_file():
        raise ValueError('Checkpoint needs sibling training summary and configuration')
    summary=json.loads(summary_path.read_text(encoding='utf-8'))
    files=summary.get('files',{})
    if not {'training_config.json','model.zip','dependencies.json'}.issubset(files):
        raise ValueError('Training artifact hash manifest is incomplete')
    if any(Path(name).name!=name or name in ('.','..') or
           re.fullmatch(r'[0-9a-f]{64}',str(sha)) is None for name,sha in files.items()):
        raise ValueError('Unsafe training artifact hash manifest')
    for name,sha in files.items():
        if file_hash(parent/name)!=sha:raise ValueError('Training artifact hash mismatch: '+name)
    config_sha=file_hash(config_path)
    if (summary.get('config_sha256')!=config_sha or summary.get('checkpoint_sha256')!=digest
        or files['model.zip']!=digest
        or metadata.get('training_config_sha256',config_sha)!=config_sha):
        raise ValueError('Training configuration or checkpoint hash binding mismatch')
    config=json.loads(config_path.read_text(encoding='utf-8'))
    if any((metadata[key]!=summary.get(field) or metadata[key]!=config.get(key if key=='seed'
              else 'total_timesteps')) for key,field in (('seed','seed'),('training_steps','num_timesteps'))):
        raise ValueError('Candidate metadata differs from recorded training seed or steps')
    try:
        valid_protocol=(config['training_identity']['split']=='train'
            and config['schema']=='e1-5b-ppo-v0' and config['threshold_g_per_kwh']==161.075
            and RewardConfig(**config['reward'])==RewardConfig()
            and ObservationNormalization(**config['normalization'])==ObservationNormalization()
            and config['runtime'].get('device','cpu')=='cpu'
            and config['runtime'].get('torch_threads',1)==1
            and config['runtime'].get('deterministic_algorithms',True) is True)
    except (KeyError,TypeError,ValueError):valid_protocol=False
    if not valid_protocol:raise ValueError('Checkpoint training protocol differs from frozen PPO-v0 Train rules')
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    import gymnasium as gym
    from stable_baselines3 import PPO
    model=PPO.load(str(checkpoint),device='cpu')
    from .rl_train import parameter_hash
    if (model.observation_space.shape!=(40,) or model.action_space!=gym.spaces.Discrete(3)
        or model.num_timesteps!=metadata['training_steps'] or model.seed!=metadata['seed']
        or parameter_hash(model.policy)!=summary.get('parameter_sha256_after')):
        raise ValueError('Loaded checkpoint schema or trained parameter identity differs')
    def predict(observation):
        action,_=model.predict(observation,deterministic=True)
        return action
    return EvaluationCandidate(metadata['name'],predict,metadata['seed'],digest,
        metadata['training_steps'],summary['elapsed_seconds'],config_sha)


def _candidate_meta(candidate):
    return dict(seed=candidate.seed,checkpoint_sha256=candidate.checkpoint_sha256,
                training_steps=candidate.training_steps,
                training_elapsed_seconds=candidate.training_elapsed_seconds,
                training_config_sha256=candidate.training_config_sha256)


def _new_env(bundle):
    return CarbonSchedulingEnv(bundle.workload,bundle.environment,bundle.capacity,
        bundle.power_factory,bundle.parameters,reward_config=bundle.reward_config,
        normalization=bundle.normalization,stop_at_drain=True)


def _scalar_action(value):
    array=np.asarray(value)
    if array.size!=1 or array.dtype.kind not in ('i','u'):
        raise ValueError('Predictor action must be one integer in 0/1/2')
    selected=int(array.reshape(-1)[0])
    if selected not in (0,1,2):
        raise ValueError('Predictor action must be one integer in 0/1/2')
    return selected


def _rollout(env,candidate):
    observation,_=env.reset(seed=0 if candidate.seed is None else candidate.seed)
    trace=[]
    inference_seconds=0.
    while True:
        # Metadata are logged separately; only these online sensors reach actor.
        now=env.executor.observed_end
        current=copy.deepcopy(env.observation_raw)
        began=time.perf_counter()
        selected=_scalar_action(candidate.predictor(observation.copy()))
        inference_seconds+=time.perf_counter()-began
        sensors=observation.copy()
        observation,reward,terminated,truncated,info=env.step(selected)
        rhos=[r for r in current['history_rho_resources'] if r is not None]
        row=dict(step=len(trace),time=now.isoformat(),action=selected,mode=ACTION_POLICIES[selected],
            ci_g_per_kwh=current['ci_g_per_kwh'],
            ci_above_threshold=current['ci_g_per_kwh']>env.parameters['threshold_g_per_kwh'],
            pending_tasks=current['pending_tasks'],running_tasks=current['running_tasks'],
            urgent_tasks=current['urgent_tasks'],deadline_due_tasks=current['deadline_due_tasks'],
            history_rho=max(rhos) if rhos else None,history_cold=current['history_cold'],
            risk_buffer_steps=current['buffer_steps'],reward=reward,
            terminated=terminated,truncated=truncated,
            observation_min=float(sensors.min()),observation_max=float(sensors.max()),
            observation_saturated_count=int(np.count_nonzero(abs(sensors)>=.99)))
        for resource,value in zip(('cpu','gpu','mem'),current['utilization']):
            row[resource+'_utilization']=value
        for name,value in zip(OBSERVATION_FIELDS,sensors):row['obs_'+name]=float(value)
        for name,value in info['reward'].items():
            if isinstance(value,(list,tuple,dict)):
                value=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
            row['reward_'+name]=value
        trace.append(row)
        env.executor.check_ownership()
        env.executor.pool.inventory()
        if terminated or truncated:
            return env.executor.result(),trace,inference_seconds


def select_candidate(rows,*,max_deadline_violation_rate):
    """Predeclared SLA-first rule; retain every candidate, including failures."""
    maximum=finite(max_deadline_violation_rate,'SLA acceptance rate')
    if maximum>1:raise ValueError('SLA acceptance rate must be <=1')
    audit=[]
    eligible=[]
    names=[]
    for row in rows:
        name=row['policy']
        if name in names:raise ValueError('Duplicate candidate name')
        names.append(name)
        complete=(row['arrived']>0 and row['completed']==row['arrived']
                  and row['pending']==row['running']==0)
        miss=row['deadline_violation_rate_observed']
        accepted=bool(complete and miss is not None and finite(miss,'DMR')<=maximum)
        audit.append(dict(policy=name,seed=row.get('seed'),complete_cohort=complete,
            deadline_violation_rate=miss,carbon_kg=row['carbon_kg'],eligible=accepted))
        if accepted:
            eligible.append(row)
    eligible.sort(key=lambda r:(r['carbon_kg'],r['deadline_violation_rate_observed'],
        r.get('seed') if r.get('seed') is not None else -1,
        r.get('checkpoint_sha256') or '',r['policy']))
    chosen=eligible[0] if eligible else None
    return dict(status='selected' if chosen else 'no_eligible_candidate',
        selected_policy=chosen['policy'] if chosen else None,
        candidates_considered=len(rows),max_deadline_violation_rate=maximum,
        criterion='complete finite cohort; DMR <= fixed limit; then common carbon, DMR, seed, checkpoint SHA',
        candidate_audit=audit,uses_training_return=False)


def evaluate_group(bundle,output_dir,*,candidates=(),include_baselines=True,
                   fixed_actions=(),max_deadline_violation_rate=.05):
    """Run one frozen split/load group and return rows, evidence, and results.

    Full ledgers and trajectories remain local archives. Export only reviewed
    small summaries to a publication checkout. No predictor receives bundle.
    """
    _allowed_split(bundle.split)
    candidates=tuple(candidates)
    fixed_actions=tuple(fixed_actions)
    if any(isinstance(a,bool) or a not in (0,1,2) for a in fixed_actions):
        raise ValueError('Fixed actions must be integer 0/1/2')
    if len(set(fixed_actions))!=len(fixed_actions):raise ValueError('Duplicate fixed action')
    names=([*ACTION_POLICIES] if include_baselines else [])
    names+=['fixed_action_'+str(a) for a in fixed_actions]+[c.name for c in candidates]
    if not names or len(set(names))!=len(names) or any(
            re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',name) is None for name in names):
        raise ValueError('Evaluation requires unique safe policy names')
    if fixed_actions and not include_baselines:raise ValueError('Fixed equivalence needs direct baselines')
    maximum=finite(max_deadline_violation_rate,'SLA acceptance rate')
    if maximum>1:raise ValueError('SLA acceptance rate must be <=1')
    reward=bundle.reward_config or RewardConfig()
    normalization=bundle.normalization or ObservationNormalization()
    reward_definition=dict(reward) if isinstance(reward,dict) else asdict(reward)
    normalization_definition=(dict(normalization) if isinstance(normalization,dict)
                              else asdict(normalization))
    baseline_input_hash=content_hash(dict(workload=bundle.workload.sha256,
        environment=bundle.environment.sha256,capacity=bundle.capacity,
        parameters=bundle.parameters,identity=bundle.identity,
        reward_config=reward_definition,normalization=normalization_definition))

    def run(out):
        live={};own={};traces={};metadata={};runtime={};closers=[]
        try:
            for name in ACTION_POLICIES if include_baselines else ():
                began=time.perf_counter()
                model=bundle.power_factory()
                closer=getattr(model,'close',None)
                if closer is not None:closers.append(closer)
                ex=LocalExecutor(bundle.workload,bundle.environment,bundle.capacity,model,
                    make_cars_policy(name,bundle.parameters,bundle.workload.start,bundle.capacity),drain=True)
                live[name]=ex;own[name]=ex.run(stop_at_drain=True)
                runtime[name]=dict(run_seconds=time.perf_counter()-began,inference_seconds=0.)
                metadata[name]=dict(seed=None,checkpoint_sha256=None,training_steps=None,
                                    training_elapsed_seconds=None,training_config_sha256=None)
            all_candidates=tuple(EvaluationCandidate('fixed_action_'+str(a),lambda obs,a=a:a)
                                 for a in fixed_actions)+candidates
            for candidate in all_candidates:
                began=time.perf_counter()
                env=_new_env(bundle);closers.append(env.close)
                result,trace,inference=_rollout(env,candidate)
                live[candidate.name]=env.executor;own[candidate.name]=result
                traces[candidate.name]=trace;metadata[candidate.name]=_candidate_meta(candidate)
                runtime[candidate.name]=dict(run_seconds=time.perf_counter()-began,
                                             inference_seconds=inference)
            common=max(ex.observed_end for ex in live.values())
            results={name:ex.extend_idle(common) if ex.drained_at is not None else own[name]
                     for name,ex in live.items()}
            rows=[];common_rows=[];action_rows={};phases={}
            for name,result in results.items():
                ex=live[name];folder=out/name;folder.mkdir()
                _write_result(folder,result)
                phase=dict(fixed=phase_summary(result,bundle.workload.end),
                    own=phase_summary(own[name],pd.Timestamp(own[name].summary['observation_end'])),
                    common=phase_summary(result,common))
                write_json(folder/'phases.json',phase);phases[name]=phase
                counts={a:sum(r['action']==a for r in traces.get(name,())) for a in range(3)}
                if name in ACTION_POLICIES:
                    counts[ACTION_POLICIES.index(name)]=len(own[name].steps)
                for phase_name,summary in phase.items():
                    row=dict(policy=name,phase=phase_name,split=bundle.split,factor=bundle.factor,
                        stress_seed=bundle.stress_seed,**metadata[name],**runtime[name],
                        own_observation_end=ex.drained_at.isoformat() if ex.drained_at is not None
                                            else own[name].summary['observation_end'],
                        drain_hours=(ex.drained_at-bundle.workload.end).total_seconds()/3600
                                    if ex.drained_at is not None else None,
                        termination='all_completed' if ex.drained_at is not None else 'safety_cap_incomplete',
                        action_0_count=counts[0],action_1_count=counts[1],action_2_count=counts[2],
                        **summary)
                    rows.append(row)
                    if phase_name=='common':common_rows.append(row)
                if name in traces:
                    write_csv(folder/'actions.csv',traces[name]);action_rows[name]=traces[name]
                    write_json(folder/'reward_summary.json',dict(
                        reward_sum=math.fsum(t['reward'] for t in traces[name]),
                        component_costs={key:math.fsum(t['reward_'+key] for t in traces[name])
                          for key in ('carbon_cost','wait_cost','deadline_cost','terminal_cost')},
                        own_episode_end=own[name].summary['observation_end'],
                        reward_horizon='own drain only; common idle tail is evaluated as physical metrics'))
            all_complete=all(r['completed']==len(bundle.workload.tasks)
                             and r['pending']==r['running']==0 for r in common_rows)
            resource_hours={key:[r[key+'_hours'] for r in common_rows]
                            for key in ('cores','gpus','mem')}
            conserved=all(all(math.isclose(v,values[0],rel_tol=1e-12,abs_tol=1e-8) for v in values)
                          for values in resource_hours.values())
            if all_complete and not conserved:raise RuntimeError('Completed cohort resource hours differ')
            expected={s.task_id:s.to_dict() for s in bundle.workload.tasks}
            for result in results.values():
                if len(result.tasks)!=len(expected) or {t['task_id'] for t in result.tasks}!=set(expected):
                    raise RuntimeError('Evaluation task conservation violated')
                if any({k:t[k] for k in expected[t['task_id']]}!=expected[t['task_id']] for t in result.tasks):
                    raise RuntimeError('Evaluation altered immutable task features')
            fairness=dict(version=EVALUATION_SCHEMA,split=bundle.split,
                same_frozen_input_sha256=baseline_input_hash,identical_frozen_inputs=True,
                workload_sha256=bundle.workload.sha256,environment_sha256=bundle.environment.sha256,
                capacity=list(bundle.capacity),parameters=bundle.parameters,stress_seed=bundle.stress_seed,
                reward_config=reward_definition,normalization=normalization_definition,
                factor=bundle.factor,arrived=len(expected),arrival_end=bundle.workload.end.isoformat(),
                safety_end=bundle.environment.end.isoformat(),common_end=common.isoformat(),
                all_tasks_completed=all_complete,equal_resource_hours=conserved,
                equal_workload_carbon_comparison_valid=all_complete and conserved,
                resource_hours=resource_hours,idle_included=True,policy_future_ci_access=False,
                formal_test_evaluated=False)
            equivalence={}
            for action in fixed_actions:
                fixed,direct=results['fixed_action_'+str(action)],results[ACTION_POLICIES[action]]
                exact=asdict(fixed)==asdict(direct)
                equivalence[str(action)]=dict(policy=ACTION_POLICIES[action],all_execution_fields_exact=exact,
                    comparison='all dataclass result fields, exact numbers; no tolerance')
                if not exact:raise RuntimeError('Fixed mode execution differs from original rule')
            candidate_names={c.name for c in candidates}
            selection=select_candidate([r for r in common_rows if r['policy'] in candidate_names],
                                       max_deadline_violation_rate=maximum)
            write_csv(out/'comparison.csv',rows);write_csv(out/'common_summary.csv',common_rows)
            write_json(out/'fairness.json',fairness);write_json(out/'equivalence.json',equivalence)
            write_json(out/'selection.json',selection)
            output_hashes={str(p.relative_to(out)).replace('\\','/'):file_hash(p)
                for p in sorted(out.rglob('*')) if p.is_file() and p.name!='run_status.json'}
            write_json(out/'output_hashes.json',output_hashes)
            return dict(rows=rows,common_rows=common_rows,fairness=fairness,equivalence=equivalence,
                        selection=selection,results=results,action_rows=action_rows,output_hashes=output_hashes)
        finally:
            for close in reversed(closers):close()
    return _fresh_run(output_dir,run)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--split',choices=('train','validation'),required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--candidate-manifest',type=Path)
    parser.add_argument('--fixed-actions',default='')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from .rl_training_utils import load_bundle
    bundle=load_bundle(args.bundle,allowed_split=args.split)
    config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    candidates=[]
    if args.candidate_manifest:
        for item in json.loads(args.candidate_manifest.read_text(encoding='utf-8')):
            checkpoint=Path(item['checkpoint'])
            if not checkpoint.is_absolute():checkpoint=args.candidate_manifest.parent/checkpoint
            candidates.append(load_ppo_candidate(checkpoint,item,split=args.split))
    fixed=tuple(int(a) for a in args.fixed_actions.split(',') if a)
    limit=config['selection']['maximum_deadline_miss_rate']
    result=evaluate_group(bundle,args.output,candidates=candidates,fixed_actions=fixed,
                          max_deadline_violation_rate=limit)
    print(json.dumps(dict(fairness=result['fairness'],selection=result['selection']),indent=2))


if __name__=='__main__':main()
