"""One preregistered task-joint PPO pilot, with original V2 responsibility."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import subprocess
from time import monotonic

import torch

from experiments.platform_v1.data import file_sha
from experiments.platform_v2.inventory import load_config,build_inventory
from experiments.platform_v2.environment import run as run_v2
from experiments.platform_v2.verification import verify_result
from experiments.sca_ppo.e0_evidence import percentile
from experiments.sca_ppo.e1_runner import write_json,jsonl_line,run_episode
from experiments.sca_ppo.e2_data import load_authorized
from experiments.sca_ppo.e2_pilot import model_fingerprint,write_csv,_compact,_valid,rollout as macro_rollout
from experiments.sca_ppo.ppo import PPOTrainer
from .joint_ppo import JointPPOTrainer
from .task_environment import TaskEpisode,STATE_SCHEMA

ROOT=Path(__file__).resolve().parents[2]
PLAN_PATH=ROOT/'docs/research/task-ppo-v1/pilot-config.json'


def validate_plan(plan):
    admission=ROOT/plan['data_admission_path']
    if file_sha(admission)!=plan['data_admission_sha256']:raise ValueError('Admission bytes changed')
    admitted=json.loads(admission.read_text(encoding='utf-8'))
    if any(plan[key]!=admitted[key] for key in ('train_ids','holdout_ids','inputs')):
        raise ValueError('Unapproved input grouping or identities')
    if (plan['schema']!='task-ppo-v1-pilot-v1' or plan['capacities']!=['OM','OH']
        or plan['state_schema']!=STATE_SCHEMA['version']
        or plan['training_episodes']!=32 or plan['maximum_training_steps']!=9216
        or plan['seed']!=20261021 or plan['ppo']['seed']!=plan['seed']
        or (plan['ppo']['input_dim'],plan['ppo']['action_dim'],plan['ppo']['hidden_sizes'])!=(48,2,[64,64])
        or plan['ppo']['gamma']!=1 or plan['ppo']['lambda_gae']!=.95 or plan['ppo']['entropy_coef']!=0
        or (plan['episode_minutes'],plan['step_minutes'],plan['clear_minutes'],plan['sla'])!=(4320,15,3960,1.5)
        or (plan['reward']['c_scale_kg'],plan['reward']['lambda_miss'])!=(100,.01)
        or len(plan['training_sequence'])!=32
        or Counter(row['capacity'] for row in plan['training_sequence'])!=Counter(OM=16,OH=16)
        or any(row['fragment'] not in plan['train_ids'] for row in plan['training_sequence'])
        or set(plan['train_ids'])&set(plan['holdout_ids'])):
        raise ValueError('Pilot differs from preregistered physical/joint PPO contract')
    for relative,expected in plan['protected_source_sha256'].items():
        if file_sha(ROOT/relative)!=expected:raise ValueError('Protected source changed: '+relative)
    if file_sha(plan['historical']['checkpoint'])!=plan['historical']['checkpoint_sha256']:
        raise ValueError('Historical checkpoint changed')


def rollout(trainer,scenario,inv,cfg,idle,directory,plan,deterministic,label,capacity):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    episode=TaskEpisode(scenario,inv,cfg,idle,plan['reward']['c_scale_kg'],plan['reward']['lambda_miss'])
    episode.reset();trainer.model.eval();started=monotonic()
    batch={key:[] for key in ('states','task_states','task_actions','log_probs','values','rewards','terminated')}
    counts=[0,0];probabilities=[];spreads=[];slot_seconds=[];mixed=0
    try:
        with (directory/'decisions.jsonl').open('w',encoding='utf-8') as stream:
            done=False
            while not done:
                begin=monotonic();state,candidates=episode.prepare_slot()
                with torch.no_grad():_,value=trainer.model(state)
                choices=[];states=[];actions=[];logs=[];slot_probabilities=[]
                minute=episode.env.minute
                for identity in candidates:
                    task_state=episode.next_state()
                    with torch.no_grad():action,log_prob,_=trainer.model.act(task_state,deterministic=deterministic)
                    binary=int(action);log=float(log_prob)
                    p1=math.exp(log) if binary else 1-math.exp(log)
                    episode.propose(binary)
                    choices.append(dict(task_id=identity,state=task_state,action=binary,log_prob=log,defer_probability=p1))
                    states.append(task_state);actions.append(binary);logs.append(log)
                    counts[binary]+=1;probabilities.append(p1);slot_probabilities.append(p1)
                reward,done,info=episode.execute_slot()
                elapsed=monotonic()-begin;slot_seconds.append(elapsed)
                joint_log=math.fsum(logs) if logs else None
                batch['states'].append(state);batch['task_states'].append(states);batch['task_actions'].append(actions)
                batch['log_probs'].append(joint_log);batch['values'].append(float(value))
                batch['rewards'].append(reward);batch['terminated'].append(done)
                if len(slot_probabilities)>1:spreads.append(max(slot_probabilities)-min(slot_probabilities))
                mixed+=bool(actions and min(actions)!=max(actions))
                safe=('proposed','approved','effective_policy_wait','capacity_wait','actual_started',
                      'reference_cg_pass','failure_latched','failure_code','new_irreversible_miss_ids',
                      'row','idle_carbon_kg','incremental_carbon_kg','carbon_reward','miss_reward','new_miss_count',
                      'internal_candidate_checks','internal_calendar_plans')
                jsonl_line(stream,dict(minute=minute,value_state=state,value=float(value),joint_log_prob=joint_log,
                    choices=choices,reward=reward,slot_seconds=elapsed,
                    **{key:info[key] for key in safe if key in info}))
        result=episode.finish();summary=result['summary'];checks=[c for r in episode.records for c in r['checks']]
        reasons=Counter(reason for check in checks for reason in check.reasons)
        summary.update(method=label,fragment=scenario.scenario_id,capacity=capacity,
            binary_action_counts=counts,task_decisions=sum(counts),sampled_slots=sum(bool(s) for s in batch['task_states']),
            forced_slots=sum(not s for s in batch['task_states']),attempts=len(checks),
            approved=sum(c.approved for c in checks),rejected=sum(not c.approved for c in checks),
            rejection_reason_counts=dict(reasons),
            defer_probability_mean=math.fsum(probabilities)/len(probabilities) if probabilities else 0.,
            within_slot_probability_spread_mean=math.fsum(spreads)/len(spreads) if spreads else 0.,
            mixed_binary_slots=mixed,slot_p50_seconds=percentile(slot_seconds,.5),
            slot_p95_seconds=percentile(slot_seconds,.95),slot_max_seconds=max(slot_seconds),
            internal_candidate_checks=sum(record['internal_candidate_checks'] for record in episode.records),
            internal_calendar_plans=sum(record['internal_calendar_plans'] for record in episode.records),
            elapsed_seconds=monotonic()-started)
        write_json(directory/'result.json',result)
        try:audit=verify_result(result,scenario,inv,cfg,1.5)
        except Exception as exc:audit=dict(errors=['Saved verifier error: '+repr(exc)])
        summary['verification_pass']=not audit['errors']
        if audit['errors']:
            summary['protocol_valid']=False
            summary['failure_reasons']=[*summary.get('failure_reasons',()),'SAVED_V2_VERIFICATION_FAILED']
        write_json(directory/'verification.json',audit);write_json(directory/'result.json',result)
        return batch,dict(summary)
    except Exception as exc:
        write_json(directory/'error.json',dict(error=repr(exc),minute=episode.env.minute,
            task_ledger=episode.env.ledger,past_steps=episode.env.steps,adapter_records=episode.adapter.records,
            physical_rollout=batch,current_choices=locals().get('choices',[]),
            partial_task_decisions=(getattr(episode,'_slot',None) or {}).get('decisions',[])))
        raise


def paired_gate(rows,expected):
    if len(rows)!=expected or any(not _valid(row) for row in rows):raise RuntimeError('Incomplete comparison responsibility')
    if max(row['inlet_final_c'] for row in rows)-min(row['inlet_final_c'] for row in rows)>.05:
        raise RuntimeError('Unfair terminal temperature')
    if max(row['it_kwh'] for row in rows)-min(row['it_kwh'] for row in rows)>1e-7:
        raise RuntimeError('Unfair paired IT energy')


def execute(output):
    plan=json.loads(PLAN_PATH.read_text(encoding='utf-8'));validate_plan(plan)
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('Commit the design, configuration, implementation and tests before the sole real pilot')
    output=Path(output).resolve();local=(ROOT/'research_notes/task-ppo-v1').resolve()
    if not output.is_relative_to(local):raise ValueError('Large logs/checkpoints must remain in ignored local directory')
    local.mkdir(parents=True,exist_ok=True)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    metadata=dict(schema='task-ppo-v1-execution',source_sha=source,protocol_sha256=file_sha(PLAN_PATH),
        data_admission_sha256=plan['data_admission_sha256'],seed=plan['seed'],torch_version=torch.__version__,
        device='cpu',status='running',training_steps=0,completed_training_episodes=0,errors=[],
        source_sha256={str(p.relative_to(ROOT)).replace('\\','/'):file_sha(p)
            for p in sorted((ROOT/'experiments/task_ppo').glob('*.py'))})
    marker=local/'pilot-once.json'
    with marker.open('x',encoding='utf-8') as stream:json.dump(metadata,stream,indent=2)
    output.mkdir(exist_ok=False);torch.set_num_threads(1);started=monotonic();training=[];evaluation=[];trainer=None
    try:
        cfg=load_config();inventories={cap:build_inventory(cfg,cap) for cap in plan['capacities']}
        scenarios={fragment:load_authorized(Path(plan['pool'])/fragment,plan['inputs'][fragment])
            for fragment in plan['train_ids']+plan['holdout_ids']}
        backgrounds={}
        for fragment,scenario in scenarios.items():
            for cap,inv in inventories.items():
                idle=run_v2(scenario,inv,cfg,'edf','fixed',1.5,idle=True)
                audit=verify_result(idle,scenario,inv,cfg,1.5,idle=True)
                write_json(output/'backgrounds'/f'{fragment}__{cap}'/'result.json',idle)
                write_json(output/'backgrounds'/f'{fragment}__{cap}'/'verification.json',audit)
                if audit['errors']:raise RuntimeError('Common idle verification failed')
                backgrounds[fragment,cap]=idle
        trainer=JointPPOTrainer(plan['ppo'])
        metadata['initial_model_fingerprint']=model_fingerprint(trainer.model)
        checkpoint_meta=dict(source_sha=source,protocol_sha256=file_sha(PLAN_PATH),
            data_admission_sha256=plan['data_admission_sha256'],training_sequence=plan['training_sequence'],
            completed_episodes=0,training_steps=0)
        trainer.save(output/'checkpoints/initial.pt',checkpoint_meta)
        historical=PPOTrainer.load(plan['historical']['checkpoint'])
        metadata['historical_model_fingerprint']=model_fingerprint(historical.model)
        for fragment in plan['holdout_ids']:
            for cap,inv in inventories.items():
                scenario=scenarios[fragment];idle=backgrounds[fragment,cap]
                group=output/'evaluations'/f'{fragment}__{cap}';rows=[]
                for label,arm in [('EDF-CG','A'),('CarbonRule-DDF-CG','C')]:
                    row=run_episode(scenario,inv,cfg,arm,group/label,
                        dict(fragment=fragment,capacity=cap,source_sha=source),
                        dict(action_probe_minutes=[],stop_decision_seconds=30))
                    row.update(common_idle_carbon_kg=idle['summary']['carbon_kg'],
                        common_incremental_carbon_kg=row['carbon_kg']-idle['summary']['carbon_kg'])
                    row['reward_sum']=-row['common_incremental_carbon_kg']/100.-.01*row['deadline_missed']
                    saved=json.loads((group/label/'result.json').read_text());saved['summary'].update(row)
                    write_json(group/label/'result.json',saved);rows.append(row)
                    evaluation.append(_compact(row));write_csv(output/'evaluation-summary.csv',evaluation)
                    if not _valid(row):raise RuntimeError('Baseline responsibility failed; no training')
                _,row=macro_rollout(historical,scenario,inv,cfg,idle,group/'MacroPPO-history',plan,True,'MacroPPO-history')
                row.update(capacity=cap,historical_only=True)
                saved=json.loads((group/'MacroPPO-history'/'result.json').read_text());saved['summary'].update(row)
                write_json(group/'MacroPPO-history'/'result.json',saved);rows.append(row)
                evaluation.append(_compact(row));write_csv(output/'evaluation-summary.csv',evaluation)
                if not _valid(row):raise RuntimeError('Historical responsibility failed; no training')
                _,row=rollout(trainer,scenario,inv,cfg,idle,group/'TaskPPO-initial',plan,True,'TaskPPO-initial',cap)
                rows.append(row);evaluation.append(_compact(row))
                write_csv(output/'evaluation-summary.csv',evaluation);paired_gate(rows,4)
                print(f'Task-PPO initial evaluation {fragment[6:14]} {cap} PASS',flush=True)
        if model_fingerprint(trainer.model)!=metadata['initial_model_fingerprint']:
            raise RuntimeError('Initial evaluation changed model parameters')
        if model_fingerprint(historical.model)!=metadata['historical_model_fingerprint']:
            raise RuntimeError('Historical evaluation changed model parameters')
        metadata['initial_evaluation_passed']=True;trainer=JointPPOTrainer.load(output/'checkpoints/initial.pt')
        for index,case in enumerate(plan['training_sequence'],1):
            fragment,cap=case['fragment'],case['capacity']
            batch,row=rollout(trainer,scenarios[fragment],inventories[cap],cfg,backgrounds[fragment,cap],
                output/'train'/f'episode-{index:03d}',plan,False,'TaskPPO-training',cap)
            metadata['training_steps']+=len(batch['states']);metadata['completed_training_episodes']=index
            row.update(episode=index,source_sha=source)
            training.append(_compact(dict(row,update_performed=False)))
            write_csv(output/'training-metrics.csv',training)
            if not _valid(row):
                raise RuntimeError('Training responsibility failed; preserve episode and stop without update')
            update=trainer.update(batch)
            row.update(update,update_performed=True);training[-1]=_compact(row)
            if update['actor_parameter_delta_l2']<=0 or update['parameter_delta_l2']<=0:
                raise RuntimeError('Actor/network weights did not update')
            checkpoint_meta.update(completed_episodes=index,training_steps=metadata['training_steps'])
            path=output/'checkpoints'/f'episode-{index:03d}.pt';fingerprint=model_fingerprint(trainer.model)
            trainer.save(path,checkpoint_meta);restored=JointPPOTrainer.load(path)
            if (model_fingerprint(restored.model)!=fingerprint or restored.update_count!=index
                or restored.optimizer_steps!=trainer.optimizer_steps):raise RuntimeError('Checkpoint resume mismatch')
            trainer=restored
            write_csv(output/'training-metrics.csv',training);write_json(output/'execution.json',metadata)
            print(f'Task-PPO train{index}/32 {fragment[6:14]} {cap} reward={row["reward_sum"]:.6f} miss={row["deadline_missed"]} decisions={row["task_decisions"]}',flush=True)
        if metadata['training_steps']!=9216:raise AssertionError('Physical training budget mismatch')
        trainer.save(output/'checkpoints/final.pt',checkpoint_meta);trainer=JointPPOTrainer.load(output/'checkpoints/final.pt')
        metadata['final_model_fingerprint']=model_fingerprint(trainer.model)
        for fragment in plan['holdout_ids']:
            for cap,inv in inventories.items():
                _,row=rollout(trainer,scenarios[fragment],inv,cfg,backgrounds[fragment,cap],
                    output/'evaluations'/f'{fragment}__{cap}'/'TaskPPO-final',plan,True,'TaskPPO-final',cap)
                evaluation.append(_compact(row));write_csv(output/'evaluation-summary.csv',evaluation)
                paired_gate([r for r in evaluation if r['fragment']==fragment and r['capacity']==cap],5)
                print(f'Task-PPO final evaluation {fragment[6:14]} {cap} PASS',flush=True)
        if model_fingerprint(trainer.model)!=metadata['final_model_fingerprint']:
            raise RuntimeError('Final frozen evaluation changed parameters')
        metadata.update(status='completed',optimizer_updates=trainer.update_count,optimizer_steps=trainer.optimizer_steps)
    except Exception as exc:
        metadata['status']='failed';metadata['errors'].append(repr(exc))
        if trainer is not None:
            try:trainer.save(output/'checkpoints/failed-state.pt',dict(status='failed',errors=metadata['errors']))
            except Exception as checkpoint_exc:metadata['errors'].append('Failed-state saving: '+repr(checkpoint_exc))
    metadata.update(elapsed_seconds=monotonic()-started,evaluation_episodes=len(evaluation),evaluation_steps=len(evaluation)*288,
        background_episodes=len(list((output/'backgrounds').glob('*/result.json'))),
        checkpoints={str(p.relative_to(output)).replace('\\','/'):file_sha(p) for p in sorted((output/'checkpoints').glob('*.pt'))})
    write_csv(output/'training-metrics.csv',training);write_csv(output/'evaluation-summary.csv',evaluation)
    write_json(output/'execution.json',metadata);write_json(marker,metadata)
    print(json.dumps(metadata,indent=2),flush=True)
    return metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(ROOT/'research_notes/task-ppo-v1/pilot-001'))
    args=parser.parse_args()
    if execute(args.output)['status']!='completed':raise SystemExit(2)
