"""Four-action development evaluation, complete Drain and physical idle tails.

Test is sealed. Actors receive only observations; scientific replay uses saved
action tapes explicitly as replay evidence, never as an online policy input.
"""
import argparse
import copy
import csv
from dataclasses import asdict, dataclass
import gzip
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .adaptive_rl_env import AdaptiveCarbonSchedulingEnv, RL_SCHEMA
from .cars import make_cars_policy
from .contracts import content_hash, file_hash, write_json, canonical_bytes
from .drain_metrics import phase_summary
from .engine import LocalExecutor
from .rl_evaluate import EvaluationCandidate, load_ppo_candidate, _new_env, _allowed_split
from .run import _fresh_run, _write_result
from .task_risk import make_task_risk_policy
from .task_risk_experiment import PARAMETERS, _input_identity
from .temporal import write_csv, experiment_sources
from .cars_experiment import _snapshot_sources

BASELINES=('edf_backfill','edf_carbon_threshold','cars_risk_only','task_risk_full')
GROUP_SWITCHES={'plain':(False,False),'state':(True,False),
                'shield':(False,True),'combined':(True,True)}
SCHEMA='e1-6b-common-evaluation-v1'


def validate_checkpoint_protocol(config,summary):
    """Compare the registered protocol, independent of Validation performance."""
    frozen=json.loads((Path(__file__).parent/'configs/e1_6b_ppo.json').read_text(encoding='utf8'))
    keys=('schema','threshold_g_per_kwh','risk_parameters','ppo','policy_kwargs','reward',
          'normalization','runtime','scene_sampling','observation_schema',
          'observation_dimensions','action_slack_minutes')
    if any(config.get(k)!=frozen[k] for k in keys):raise ValueError('Registered training protocol differs')
    identities=config.get('training_identities',[])
    group=config.get('group')
    if (group not in GROUP_SWITCHES or len(identities)!=2
        or [i.get('factor') for i in identities]!=[1.5,2]
        or any(i.get('split')!='train' for i in identities)
        or summary.get('training_identities')!=identities
        or summary.get('group')!=group or summary.get('seed')!=config.get('seed')
        or summary.get('num_timesteps')!=config.get('total_timesteps')
        or (summary.get('risk_state'),summary.get('risk_filter'))!=GROUP_SWITCHES[group]
        or summary.get('observation_size')!=83 or summary.get('action_size')!=4):
        raise ValueError('Train identities, budget or ablation schema differ')
    return True


@dataclass(frozen=True)
class AdaptiveCandidate:
    name: str
    group: str
    predictor: object
    seed: int=None
    checkpoint_sha256: str=None
    training_steps: int=None
    training_elapsed_seconds: float=None


def load_adaptive_candidate(checkpoint, *, split='validation', name=None):
    _allowed_split(split)
    p=Path(checkpoint);folder=p.parent
    summary=json.loads((folder/'training_summary.json').read_text(encoding='utf8'))
    cfg=json.loads((folder/'training_config.json').read_text(encoding='utf8'))
    validate_checkpoint_protocol(cfg,summary)
    files=summary.get('files',{})
    if not {'model.zip','training_config.json','dependencies.json'}.issubset(files):
        raise ValueError('Incomplete training artifact manifest')
    for f,h in files.items():
        if Path(f).name!=f or file_hash(folder/f)!=h:raise ValueError('Training hash mismatch')
    group=cfg.get('group')
    identities=cfg.get('training_identities',[])
    if (cfg.get('schema')!='e1-6b-adaptive-ppo-v1' or group not in GROUP_SWITCHES
        or cfg.get('threshold_g_per_kwh')!=161.075
        or cfg.get('risk_parameters')!={k:v for k,v in PARAMETERS.items() if k!='threshold_g_per_kwh'}
        or len(identities)!=2 or any(i.get('split')!='train' for i in identities)
        or summary.get('checkpoint_sha256')!=file_hash(p)):
        raise ValueError('Invalid frozen adaptive Train protocol')
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    from stable_baselines3 import PPO
    from .rl_train import parameter_hash
    model=PPO.load(str(p),device='cpu')
    if (model.observation_space.shape!=(83,) or model.action_space.n!=4
        or parameter_hash(model.policy)!=summary['parameter_sha256_after']
        or model.num_timesteps!=summary['num_timesteps'] or model.seed!=summary['seed']):
        raise ValueError('Checkpoint schema or parameter identity mismatch')
    def predict(obs):return model.predict(obs,deterministic=True)[0]
    return AdaptiveCandidate(name or group+'_'+str(summary['seed']),group,predict,
        summary['seed'],file_hash(p),summary['num_timesteps'],summary['elapsed_seconds'])


def scientific_hashes(results):
    # No wall-clock, checkpoint container timestamps or provenance in this hash.
    return {name:content_hash(asdict(result)) for name,result in results.items()}


def _rollout(env,candidate,out):
    obs,_=env.reset(seed=candidate.seed or 0)
    actions=[];inference=0.;counts=dict(candidate_checks=0,solo_eligible=0,filter_veto=0,deferred=0)
    with gzip.open(out/'policy-diagnostics.jsonl.gz','wb') as stream:
        while True:
            raw=copy.deepcopy(env.observation_raw)
            now=env.executor.observed_end
            began=time.perf_counter();value=np.asarray(candidate.predictor(obs.copy()))
            inference+=time.perf_counter()-began
            n=env.action_space.n
            if value.size!=1 or value.dtype.kind not in ('i','u') or not 0<=int(value.item())<n:
                raise ValueError('Actor must return a valid scalar integer action')
            action=int(value.item());sensors=obs.copy()
            obs,reward,terminated,truncated,info=env.step(action)
            details=env.executor.policy_steps[-1]
            for row in info.get('policy_diagnostics',()):stream.write(canonical_bytes(row)+b'\n')
            for key in counts:counts[key]+=info.get('risk_counts',{}).get(key,0)
            actions.append(dict(time=now.isoformat(),action=action,reward=reward,
                reward_event_time=info['time'],
                ci_g_per_kwh=raw['ci_g_per_kwh'],pending=raw['pending_tasks'],running=raw['running_tasks'],
                observation_sha256=content_hash(sensors.tolist()),
                started_ids=json.dumps(details['started_ids']),deferred_ids=json.dumps(details['policy_deferred_ids']),
                capacity_waiting_ids=json.dumps(details['capacity_waiting_ids']),
                new_deadline_miss_ids=json.dumps(info['reward']['new_deadline_miss_ids']),
                **info.get('risk_counts',{})))
            env.executor.check_ownership();env.executor.pool.inventory()
            if terminated or truncated:return env.executor.result(),actions,inference,counts


def evaluate_adaptive_group(bundle,output_dir,*,candidates=(),external=()):
    _allowed_split(bundle.split)
    if bundle.parameters.get('threshold_g_per_kwh')!=161.075:raise ValueError('Frozen CI threshold differs')
    names=list(BASELINES)+[c.name for c in candidates]+[c.name for c in external]
    if len(names)!=len(set(names)):raise ValueError('Duplicate method names')
    if any(c.group not in GROUP_SWITCHES for c in candidates):raise ValueError('Invalid group')
    identity=_input_identity(bundle);sources=experiment_sources()
    def run(out):
        _snapshot_sources(out,sources)
        write_json(out/'input-identity.json',identity)
        write_json(out/'source-freeze.json',dict(sources=sources,source_sha256=content_hash(sources)))
        live={};own={};meta={};traces={};closers=[]
        try:
            for name in BASELINES:
                began=time.perf_counter();power=bundle.power_factory()
                closer=getattr(power,'close',None)
                if closer:closers.append(closer)
                policy=(make_task_risk_policy(name,PARAMETERS,bundle.workload.start,bundle.capacity)
                        if name=='task_risk_full' else make_cars_policy(name,bundle.parameters,bundle.workload.start,bundle.capacity))
                ex=LocalExecutor(bundle.workload,bundle.environment,bundle.capacity,power,policy,drain=True)
                own[name]=ex.run(stop_at_drain=True);live[name]=ex
                meta[name]=dict(group='heuristic',seed=None,run_seconds=time.perf_counter()-began,
                               inference_seconds=0.,candidate_checks=0,solo_eligible=0,filter_veto=0,deferred=0,
                               checkpoint_sha256=None,training_steps=None,training_elapsed_seconds=None)
            for c in tuple(candidates)+tuple(external):
                folder=out/c.name;folder.mkdir();began=time.perf_counter()
                if isinstance(c,AdaptiveCandidate):
                    state,filtered=GROUP_SWITCHES[c.group]
                    env=AdaptiveCarbonSchedulingEnv(bundle.workload,bundle.environment,bundle.capacity,
                        bundle.power_factory,PARAMETERS,risk_state=state,risk_filter=filtered,
                        reward_config=bundle.reward_config,normalization=bundle.normalization,stop_at_drain=True)
                    group=c.group
                else:env=_new_env(bundle);group='legacy_3action'
                closers.append(env.close)
                result,tape,inference,counts=_rollout(env,c,folder)
                live[c.name]=env.executor;own[c.name]=result;traces[c.name]=tape
                meta[c.name]=dict(group=group,seed=c.seed,run_seconds=time.perf_counter()-began,
                    inference_seconds=inference,checkpoint_sha256=c.checkpoint_sha256,
                    training_steps=c.training_steps,training_elapsed_seconds=c.training_elapsed_seconds,**counts)
            common=max(ex.observed_end for ex in live.values())
            results={name:ex.extend_idle(common) if ex.drained_at is not None else own[name] for name,ex in live.items()}
            rows=[];common_rows=[]
            for name,result in results.items():
                folder=out/name;folder.mkdir(exist_ok=True);_write_result(folder,result)
                if name in traces:write_csv(folder/'actions.csv',traces[name])
                for phase,end,data in [('fixed',bundle.workload.end,result),
                    ('own',pd.Timestamp(own[name].summary['observation_end']),own[name]),('common',common,result)]:
                    row=dict(policy=name,phase=phase,split=bundle.split,factor=bundle.factor,**meta[name],
                        drain_hours=(live[name].drained_at-bundle.workload.end).total_seconds()/3600 if live[name].drained_at else None,
                        own_observation_end=own[name].summary['observation_end'],
                        termination='all_completed' if live[name].drained_at else 'safety_cap_incomplete',
                        **{f'action_{a}_count':sum(r['action']==a for r in traces.get(name,())) for a in range(4)},
                        **phase_summary(data,end))
                    rows.append(row)
                    if phase=='common':common_rows.append(row)
            complete=all(r['completed']==len(bundle.workload.tasks) and r['pending']==r['running']==0 for r in common_rows)
            hours={k:[r[k+'_hours'] for r in common_rows] for k in ('cores','gpus','mem')}
            conserved=all(all(math.isclose(x,v[0],rel_tol=1e-12,abs_tol=1e-8) for x in v) for v in hours.values())
            expected={s.task_id:s.to_dict() for s in bundle.workload.tasks}
            for result in results.values():
                if (len(result.tasks)!=len(expected) or {r['task_id'] for r in result.tasks}!=set(expected)
                    or any({k:r[k] for k in expected[r['task_id']]}!=expected[r['task_id']] for r in result.tasks)):
                    raise RuntimeError('Task conservation/features changed')
            if complete and not conserved:raise RuntimeError('Resource hours changed')
            fairness=dict(schema=SCHEMA,split=bundle.split,workload_sha256=bundle.workload.sha256,
                environment_sha256=bundle.environment.sha256,capacity=list(bundle.capacity),common_end=common.isoformat(),
                arrival_end=bundle.workload.end.isoformat(),arrived=len(expected),same_frozen_input_sha256=content_hash(identity),
                all_tasks_completed=complete,equal_resource_hours=conserved,resource_hours=hours,
                equal_workload_carbon_comparison_valid=complete and conserved,idle_included=True,
                policy_future_ci_access=False,formal_test_evaluated=False,development_validation=bundle.split=='validation')
            write_csv(out/'comparison.csv',rows);write_csv(out/'common_summary.csv',common_rows)
            write_json(out/'fairness.json',fairness);write_json(out/'scientific_hashes.json',scientific_hashes(results))
            write_json(out/'output_hashes.json',{p.relative_to(out).as_posix():file_hash(p) for p in out.rglob('*')
                if p.is_file() and p.name not in ('run_status.json','output_hashes.json')})
            if experiment_sources()!=sources:raise RuntimeError('Source changed during evaluation')
            return dict(rows=rows,common_rows=common_rows,fairness=fairness,results=results,actions=traces)
        finally:
            for close in reversed(closers):close()
    return _fresh_run(output_dir,run)


def replay_adaptive_group(archive,output_dir,*,bundle):
    """Exact current-source replay using sealed historical actor action tapes."""
    _allowed_split(bundle.split)
    archive=Path(archive)
    from .task_risk_experiment import _verify_hashes
    _verify_hashes(archive,json.loads((archive/'output_hashes.json').read_text()))
    freeze=json.loads((archive/'source-freeze.json').read_text())
    if freeze['sources']!=experiment_sources():raise ValueError('Replay source hash differs; use the archived source snapshot')
    identity=json.loads((archive/'input-identity.json').read_text())
    if identity!=_input_identity(bundle):raise ValueError('Replay frozen input identity differs')
    with (archive/'common_summary.csv').open(encoding='utf8',newline='') as s:rows=list(csv.DictReader(s))
    candidates=[];external=[]
    for row in rows:
        if row['policy'] in BASELINES:continue
        with (archive/row['policy']/'actions.csv').open(encoding='utf8',newline='') as s:tape=list(csv.DictReader(s))
        def actor_for(records):
            index=0
            def predict(obs):
                nonlocal index
                r=records[index];index+=1
                if content_hash(obs.tolist())!=r['observation_sha256']:raise RuntimeError('Replay observation hash differs')
                return int(r['action'])
            return predict
        seed=int(row['seed']) if row['seed'] else None
        if row['group']=='legacy_3action':
            external.append(EvaluationCandidate(row['policy'],actor_for(tape),seed))
        else:candidates.append(AdaptiveCandidate(row['policy'],row['group'],actor_for(tape),seed))
    result=evaluate_adaptive_group(bundle,output_dir,candidates=candidates,external=external)
    expected=json.loads((archive/'scientific_hashes.json').read_text())
    exact=scientific_hashes(result['results'])==expected
    report=dict(scientific_outputs_exact=exact,task_ledger_resources_energy_carbon=True,
                replay_kind='recorded action tape; not a new online algorithm or training',formal_test_evaluated=False)
    write_json(Path(output_dir)/'replay-verification.json',report)
    if not exact:raise RuntimeError('Scientific replay differs')
    result['replay']=report
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True);p.add_argument('--split',choices=('train','validation'),required=True)
    p.add_argument('--candidates',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    from .rl_training_utils import load_bundle
    bundle=load_bundle(a.bundle,allowed_split=a.split)
    new=[];old=[]
    for item in json.loads(Path(a.candidates).read_text(encoding='utf8')):
        if item['kind']=='adaptive':new.append(load_adaptive_candidate(item['checkpoint'],split=a.split,name=item['name']))
        elif item['kind']=='legacy':old.append(load_ppo_candidate(item['checkpoint'],item,split=a.split))
        else:raise ValueError('Unknown checkpoint kind')
    result=evaluate_adaptive_group(bundle,a.output,candidates=new,external=old)
    print(json.dumps(result['fairness'],sort_keys=True))


if __name__=='__main__':main()
