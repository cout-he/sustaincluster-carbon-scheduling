"""Bounded, resumable B3 pretraining and multi-scenario joint optimization."""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import time

import numpy as np

from experiments.e1.contracts import content_hash, file_hash
from experiments.joint.env import TASK_FIELDS, COOLING_FIELDS
from experiments.joint.train import source_provenance

from . import VERSION
from .checkpoint import CheckpointStore, atomic_json, global_rng_state, restore_global_rng
from .config import load_config
from .env import V2Env
from .monitor import Monitor, episode_record, step_record, capacity_audit, plot
from .ppo import IPPOAgent, SlotTransition
from .scenario_sampler import ScenarioSampler
from .scenarios import load_pool
from .selection import aggregate, advance_curriculum, update_lambda


def _code_provenance():
    value=source_provenance()
    root=Path(__file__).resolve().parents[2]
    for relative in ('rl_components/task.py',):
        value['code_sha256'][relative]=file_hash(root/relative)
    for path in sorted(Path(__file__).parent.rglob('*')):
        if path.suffix in ('.py','.yaml'):
            value['code_sha256'][path.relative_to(root).as_posix()]=file_hash(path)
    value['code_manifest_sha256']=content_hash(value['code_sha256'])
    return value


def _role_sets(pool):
    train=load_pool(pool,roles=('train',))
    development=load_pool(pool,roles=('development',))
    if not train or not development:
        raise ValueError('Both Train and disjoint development validation scenarios are required')
    return train,development


def _snapshot(state,agents,sampler):
    state['agents']={name:agent.export_state() for name,agent in agents.items()}
    state['sampler']=sampler.state_dict()
    state['global_rng']=global_rng_state()
    return state


def _restore_best_aliases(store,state):
    # latest is the transaction boundary. Roll back a best pointer that was
    # published just before a crash prevented publication of the new latest.
    for name in ('best_feasible','best_deadline'):
        pointer=state.get(name+'_pointer')
        path=store.root/(name+'.json')
        if pointer: atomic_json(path,pointer)
        elif path.exists(): path.unlink()


def evaluate_scenarios(agents,scenarios,config,deadline_multiplier,*,deadline=None):
    results=[]
    for scenario in scenarios:
        if scenario.role not in ('development','validation'):
            raise ValueError('Evaluation is restricted to development or final Validation roles')
        env=V2Env(scenario,config,evaluation=True)
        try:
            env.set_deadline_multiplier(deadline_multiplier);env.reset(seed=config['seed'])
            while not env.done:
                if deadline is not None and time.monotonic()>=deadline:
                    return None
                env.step(*(lambda obs,mask,agent=agents[role]:agent.act(obs,mask,deterministic=True)
                           for role in ('task','cooling')))
            results.append(env.summary())
        finally:env.close()
    return results


def _update_best(state,validation,store,agents,sampler):
    score=aggregate(validation)
    record=dict(episode=state['completed_episodes'],**score)
    state['validations'].append(dict(record,scenario_summaries=validation))
    improved=[]
    if state['best_deadline'] is None or tuple(score['deadline_key'])<tuple(state['best_deadline']['deadline_key']):
        state['best_deadline']=record;improved.append('best_deadline')
    if score['feasible'] and (state['best_feasible'] is None or score['carbon_kg']<state['best_feasible']['carbon_kg']):
        state['best_feasible']=record;improved.append('best_feasible')
    if improved:
        pointer=store.save(_snapshot(state,agents,sampler),tuple(improved))
        for name in improved:state[name+'_pointer']=pointer
    return score


def run(pool,output,*,config=None,resume=None,total_steps=None,wall_seconds=None,
        max_episodes=None,skip_pretrain=False,heldout_final=None):
    started=time.monotonic()
    import torch
    cfg=load_config(config)
    limit=cfg['budget']['total_physical_steps'] if total_steps is None else total_steps
    seconds=cfg['budget']['max_wall_seconds'] if wall_seconds is None else wall_seconds
    episode_limit=cfg['budget']['max_episodes'] if max_episodes is None else max_episodes
    if isinstance(limit,bool) or not isinstance(limit,int) or limit<1:
        raise ValueError('A positive total physical-step budget is required')
    if isinstance(seconds,bool) or not math.isfinite(seconds) or seconds<=0:
        raise ValueError('A finite positive wall-clock budget is required')
    if episode_limit is not None and (isinstance(episode_limit,bool) or not isinstance(episode_limit,int) or episode_limit<1):
        raise ValueError('Optional episode limit must be positive')
    deadline=started+seconds
    train,development=_role_sets(pool)
    development=sorted(development,key=lambda s:s.scenario_id)[:cfg['validation']['max_development_scenarios']]
    pool_path=Path(pool);manifest=pool_path/'manifest.json' if pool_path.is_dir() else pool_path
    data_identity=content_hash(dict(manifest_sha256=file_hash(manifest),
        train=[s.identity for s in sorted(train,key=lambda s:s.scenario_id)],
        development=[s.identity for s in development]))
    provenance=_code_provenance();code_identity=provenance['code_manifest_sha256']
    out=Path(output)
    if resume is None and out.exists():raise FileExistsError('Refusing to overwrite V2 output: '+str(out))
    if resume is not None and not out.is_dir():raise FileNotFoundError(out)
    out.mkdir(parents=True,exist_ok=True)
    store=CheckpointStore(out/'checkpoints',config_identity=content_hash(cfg),data_identity=data_identity)
    frozen_pool_config=json.loads((manifest.parent/'config.json').read_text(encoding='utf8'))
    sampler=ScenarioSampler(train,cfg['seed'],config=frozen_pool_config['sampler'])
    if resume is not None:
        state=store.load(resume)
        if state['code_identity']!=code_identity:raise ValueError('Resume source-code identity changed')
        agents={name:IPPOAgent.from_state(value) for name,value in state['agents'].items()}
        sampler.load_state_dict(state['sampler']);restore_global_rng(state['global_rng'])
        _restore_best_aliases(store,state)
    else:
        random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
        agents=dict(task=IPPOAgent(len(TASK_FIELDS),2,cfg['ppo'],cfg['seed']+1),
                    cooling=IPPOAgent(len(COOLING_FIELDS),3,cfg['ppo'],cfg['seed']+2))
        initial={role:agent.component_hashes() for role,agent in agents.items()}
        atomic_json(out/'training_config.json',cfg)
        atomic_json(out/'provenance.json',dict(provenance,schema=VERSION,data_identity=data_identity,
            pool_manifest_sha256=file_hash(manifest),pool=str(manifest.resolve()),test_access=False))
        state=dict(schema=VERSION,code_identity=code_identity,config=cfg,data_identity=data_identity,
            initial_component_hashes=initial,pretrain={},pretraining_stages={},pretraining_skipped=bool(skip_pretrain),
            completed_episodes=0,curriculum=dict(stage=0,success_streak=0),
            deadline_multiplier=cfg['constraint']['initial_lambda'],
            counters=dict(physical_steps=0,task_actions=0,cooling_actions=0,processed_tasks=0,
                          task_ppo_updates=0,cooling_ppo_updates=0),
            episodes=[],step_history=[],dropped_step_history=0,validations=[],
            best_feasible=None,best_deadline=None,best_feasible_pointer=None,best_deadline_pointer=None,
            elapsed_seconds=0.,capacity_audits={})
        store.save(_snapshot(state,agents,sampler))
    elapsed_before=state['elapsed_seconds']
    monitor=Monitor(out/'monitor',tensorboard=cfg['monitor']['tensorboard'])
    monitor.flush(state)
    steps_before=state['counters']['physical_steps']
    attempted_steps=0;stop_reason='budget_reached';partial=None
    try:
        from .pretraining import run_pretraining
        pretraining_complete=run_pretraining(state,agents,sampler,store,train,cfg,out,deadline)
        if not pretraining_complete:stop_reason='pretraining_wall_budget_rolled_back'
        while pretraining_complete and state['counters']['physical_steps']<limit:
            if episode_limit is not None and state['completed_episodes']>=episode_limit:
                stop_reason='episode_limit';break
            if time.monotonic()>=deadline:stop_reason='wall_clock_limit';break
            scenario=sampler.sample(stage=state['curriculum']['stage'])
            env=V2Env(scenario,cfg)
            trajectory={role:[] for role in agents};episode_steps=[]
            episode_number=state['completed_episodes']+1
            try:
                env.set_deadline_multiplier(state['deadline_multiplier']);env.reset(seed=cfg['seed'])
                while not env.done:
                    if time.monotonic()>=deadline or state['counters']['physical_steps']+len(episode_steps)>=limit:
                        partial=dict(scenario_id=scenario.scenario_id,physical_steps=len(episode_steps),
                                     reason='wall_or_step_budget_mid_episode',updates_applied=False)
                        stop_reason=partial['reason'];break
                    outcome=env.step(agents['task'].act,agents['cooling'].act);attempted_steps+=1
                    for role,boundary,decisions in (
                        ('task',outcome.task_boundary_observation,outcome.task_decisions),
                        ('cooling',outcome.cooling_boundary_observation,outcome.cooling_decisions)):
                        trajectory[role].append(SlotTransition(boundary,decisions,outcome.reward,outcome.done,
                                                               agents[role].value(boundary)))
                    episode_steps.append(step_record(outcome,episode=episode_number,
                        global_step=state['counters']['physical_steps']+len(episode_steps)+1,
                        scenario_id=scenario.scenario_id))
                if partial:break
                summary=env.summary()
            finally:env.close()
            # Each agent gets the same single reward for each physical slot.
            updates={role:agent.update(trajectory[role],bootstrap_value=0.,deadline=deadline) for role,agent in agents.items()}
            if time.monotonic()>=deadline:
                stop_reason='wall_clock_limit_during_update_rolled_back'
                partial=dict(scenario_id=scenario.scenario_id,physical_steps=len(episode_steps),
                             reason=stop_reason,updates_committed=False)
                break
            value,constraint=update_lambda(state['deadline_multiplier'],summary,scenario.statistics,cfg['constraint'])
            state['deadline_multiplier']=value
            state['curriculum']=advance_curriculum(state['curriculum'],summary,cfg['curriculum'])
            count=state['counters'];count['physical_steps']+=len(episode_steps)
            count['task_actions']+=summary['task_action_count'];count['cooling_actions']+=summary['cooling_action_count']
            count['processed_tasks']+=summary['completed'];count['task_ppo_updates']+=updates['task']['optimizer_steps']
            count['cooling_ppo_updates']+=updates['cooling']['optimizer_steps']
            state['completed_episodes']=episode_number
            elapsed=elapsed_before+time.monotonic()-started
            state['elapsed_seconds']=elapsed
            state['episodes'].append(episode_record(episode_number,scenario,summary,updates,constraint,
                                                    state['curriculum'],count,elapsed))
            state['step_history'].extend(episode_steps)
            overflow=max(0,len(state['step_history'])-cfg['monitor']['max_step_history'])
            if overflow:state['step_history']=state['step_history'][overflow:];state['dropped_step_history']+=overflow
            if scenario.scenario_id not in state['capacity_audits']:
                state['capacity_audits'][scenario.scenario_id]=capacity_audit(scenario)
            should_validate=episode_number==1 or episode_number%cfg['validation']['every_episodes']==0
            if should_validate and time.monotonic()<deadline:
                validation=evaluate_scenarios(agents,development,cfg,value,deadline=deadline)
                if validation is not None:_update_best(state,validation,store,agents,sampler)
            speed=(count['physical_steps']-steps_before)/max(1e-9,time.monotonic()-started)
            state['progress']=dict(physical_steps_per_second=speed,
                estimated_seconds_to_step_budget=(limit-count['physical_steps'])/speed if speed else None,
                wall_seconds_remaining=max(0.,deadline-time.monotonic()),
                committed_episodes=episode_number,distinct_scenarios=len(sampler.counts),
                distinct_source_fragments=len({row['source_fragment_id'] for row in state['episodes']}),
                synthetic_episode_fraction=sum(r['synthetic'] for r in state['episodes'])/len(state['episodes']))
            store.save(_snapshot(state,agents,sampler))
            monitor.flush(state)
            atomic_json(out/'progress.json',state['progress'])
            print(json.dumps(dict(phase='joint_v2',episode=episode_number,scenario_id=scenario.scenario_id,
                committed_physical_steps=count['physical_steps'],dmr=summary['dmr'],completed=summary['completed'],
                unfinished=summary['unfinished'],incremental_carbon_kg=summary['incremental_carbon_kg'],
                lambda_d=value,curriculum_stage=state['curriculum']['stage'],
                actor_updates={role:updates[role].get('actor_optimizer_steps',updates[role].get('actor_steps')) for role in agents},
                seconds_this_run=time.monotonic()-started)),flush=True)
    except KeyboardInterrupt:
        stop_reason='interrupted_uncommitted_episode_rolled_back'
    finally:
        monitor.close()
    # Restore the last committed generation even if sampling/collection/update
    # was interrupted. Partial trajectories never become shorter fake episodes.
    state=store.load('latest');_restore_best_aliases(store,state)
    agents={name:IPPOAgent.from_state(value) for name,value in state['agents'].items()}
    sampler.load_state_dict(state['sampler']);restore_global_rng(state['global_rng'])
    final_pointer=store.save(_snapshot(state,agents,sampler),('latest','final'))
    monitor=Monitor(out/'monitor',tensorboard=cfg['monitor']['tensorboard'])
    monitor.flush(state);monitor.close();plot(out/'monitor')
    # July's original Validation is a one-off final check; never used to select
    # checkpoints, promote curriculum, tune lambda, or apply gradient updates.
    check=cfg['validation']['heldout_final_check'] if heldout_final is None else heldout_final
    heldout_path=out/'heldout-validation-once.json'
    if check and state['completed_episodes'] and not heldout_path.exists() and time.monotonic()<deadline:
        # Publish the immutable generation BEFORE accessing July. This durable
        # attempt marker survives a crash and prevents repeated held-out prefixes.
        heldout_record=dict(role='final_check_only_no_selection',checkpoint=final_pointer,
            code_identity=code_identity,results=None,status='attempted',test_access=False)
        atomic_json(heldout_path,heldout_record)
        state['heldout_attempted']=True
        try:
            heldout=load_pool(pool,roles=('validation',))
            heldout_results=evaluate_scenarios(agents,heldout,cfg,state['deadline_multiplier'],deadline=deadline)
            heldout_record.update(results=heldout_results,
                status='completed' if heldout_results is not None else 'budget_exhausted')
        except KeyboardInterrupt:
            heldout_record['status']='interrupted'
            stop_reason='interrupted_during_final_validation'
        finally:
            atomic_json(heldout_path,heldout_record)
    # Final elapsed accounting includes monitoring, plotting and held-out work.
    # A running plant step/checkpoint write is atomic and cannot be preempted.
    state['elapsed_seconds']=elapsed_before+time.monotonic()-started
    state['last_run']=dict(stop_reason=stop_reason,attempted_physical_steps=attempted_steps,
        partial_episode=partial,wall_seconds=time.monotonic()-started,step_budget=limit,wall_budget_seconds=seconds,
        wall_budget_scope='training setup, pretraining, joint optimization and evaluation; atomic finalization may overrun')
    store.save(_snapshot(state,agents,sampler),('latest','final'))
    state['last_run']['wall_seconds']=time.monotonic()-started
    result=dict(schema=VERSION,output=str(out.resolve()),completed_episodes=state['completed_episodes'],
        counters=state['counters'],distinct_train_scenarios=len(state['sampler']['counts']),
        sampler_draws=state['sampler']['draws'],curriculum=state['curriculum'],
        deadline_multiplier=state['deadline_multiplier'],last_run=state['last_run'],
        best_feasible=state['best_feasible'],best_deadline=state['best_deadline'],
        selection_status='feasible_candidate' if state['best_feasible'] else 'no_feasible_model',
        component_hashes={role:agent.component_hashes() for role,agent in agents.items()},
        progress=state.get('progress',{}),heldout_validation=json.loads(heldout_path.read_text()) if heldout_path.exists() else None,
        test_access=False)
    atomic_json(out/'run-result.json',result)
    return result


def parser():
    command=argparse.ArgumentParser(description=__doc__)
    command.add_argument('--pool',required=True);command.add_argument('--output',required=True)
    command.add_argument('--config');command.add_argument('--resume',nargs='?',const='latest')
    command.add_argument('--total-steps',type=int);command.add_argument('--wall-seconds',type=float)
    command.add_argument('--max-episodes',type=int);command.add_argument('--skip-pretrain',action='store_true')
    command.add_argument('--no-heldout-final',action='store_true')
    return command


def main(argv=None):
    args=parser().parse_args(argv)
    run(args.pool,args.output,config=args.config,resume=args.resume,total_steps=args.total_steps,
        wall_seconds=args.wall_seconds,max_episodes=args.max_episodes,skip_pretrain=args.skip_pretrain,
        heldout_final=False if args.no_heldout_final else None)


if __name__=='__main__':main()
