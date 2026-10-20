"""One preregistered CPU PyTorch pilot; no search or automatic failure retries."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from time import monotonic

import torch

from experiments.platform_v1.data import file_sha
from experiments.platform_v2.environment import run as run_v2
from experiments.platform_v2.inventory import load_config, build_inventory
from experiments.platform_v2.verification import verify_result
from .e1_runner import write_json, jsonl_line, run_episode
from .e2_data import load_authorized
from .ppo import PPOTrainer
from .rl_environment import RLEpisode, STATE_SCHEMA

ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT/'docs/research/sca-ppo/E2-pilot-config.json'
ADMISSION = ROOT/'docs/research/sca-ppo/e2-results/data-admission.json'


def model_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode()); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_plan(plan):
    if (plan.get('schema') != 'sca-ppo-v02-e2-pilot-v1'
            or not 1 <= plan.get('training_episodes',0) <= 32
            or plan.get('maximum_training_steps') != 9216 or plan.get('episode_minutes') != 4320
            or plan.get('step_minutes') != 15 or plan.get('capacity') != 'OM'
            or plan.get('sla') != 1.5 or plan.get('clear_minutes') != 3960
            or plan.get('device') != 'cpu' or plan.get('torch_threads') != 1
            or plan.get('ppo',{}).get('gamma') != 1 or plan['ppo'].get('action_dim') != 7
            or plan['ppo'].get('hidden_sizes') != [64,64] or plan['ppo'].get('input_dim') != 27
            or plan.get('reward',{}).get('c_scale_kg',0) <= 0
            or plan.get('state_schema') != STATE_SCHEMA['version']
            or len(plan.get('training_sequence',[])) != plan['training_episodes']
            or not set(plan['training_sequence']) <= set(plan['train_ids'])
            or set(plan['train_ids']) & set(plan['holdout_ids'])):
        raise ValueError('Pilot violates frozen contract, data split or single-seed budget')
    if file_sha(ADMISSION) != plan['data_admission_sha256']:
        raise ValueError('Committed data admission changed')
    admitted=json.loads(ADMISSION.read_text(encoding='utf-8'))
    if any(plan[key] != admitted[key] for key in ('train_ids','holdout_ids','inputs')):
        raise ValueError('Pilot split/inputs differ from the committed data admission')
    original = json.loads((ROOT/'docs/research/sca-ppo/E1-protocol.json').read_text(encoding='utf-8'))
    for relative, expected in {**original['e0_source_sha256'],**original['frozen_platform_source_sha256']}.items():
        if file_sha(ROOT/relative) != expected: raise ValueError('Protected source changed: '+relative)


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})


def rollout(trainer, scenario, inv, cfg, idle, directory, plan, deterministic, label):
    directory = Path(directory); directory.mkdir(parents=True,exist_ok=False)
    episode = RLEpisode(scenario,inv,cfg,idle,plan['reward']['c_scale_kg'],plan['reward']['lambda_miss'])
    episode.reset(); trainer.model.eval()
    batch = {k:[] for k in ('states','actions','log_probs','values','rewards','terminated','policy_sampled')}
    counts=[0]*7; started=monotonic()
    try:
        with (directory/'decisions.jsonl').open('w',encoding='utf-8') as stream:
            done=False
            while not done:
                state, eligible = episode.prepare()
                with torch.no_grad():
                    if eligible:
                        action, log_prob, value = trainer.model.act(state, deterministic=deterministic)
                        chosen=int(action); saved_action=chosen; saved_log=float(log_prob)
                        counts[chosen]+=1
                    else:
                        _, value=trainer.model(state)
                        chosen=0; saved_action=-1; saved_log=None
                reward, done, info=episode.step(chosen,policy_sampled=eligible)
                batch['states'].append(state);batch['actions'].append(saved_action)
                batch['log_probs'].append(saved_log);batch['values'].append(float(value))
                batch['rewards'].append(reward);batch['terminated'].append(done)
                batch['policy_sampled'].append(eligible)
                jsonl_line(stream,dict(minute=episode.env.minute-15,selected_action=chosen,
                    policy_sampled=eligible,log_prob=saved_log,value=float(value),reward=reward,
                    **{k:v for k,v in episode.records[-1].items()
                       if k not in ('minute','selected_action','policy_sampled','log_prob','value','reward','checks')}))
        result=episode.finish();summary=result['summary']
        checks=[check for record in episode.records for check in record['checks']]
        reasons={}
        for check in checks:
            for reason in check.reasons:reasons[reason]=reasons.get(reason,0)+1
        summary.update(method=label,fragment=scenario.scenario_id,capacity='OM',
            selected_action_counts=counts,sampled_slots=sum(batch['policy_sampled']),
            forced_noop_slots=288-sum(batch['policy_sampled']),elapsed_seconds=monotonic()-started,
            attempts=len(checks),approved=sum(check.approved for check in checks),
            rejected=sum(not check.approved for check in checks),rejection_reason_counts=reasons,
            proposed=sum(len(record['proposed']) for record in episode.records),
            effective_policy_wait_events=sum(len(record['effective_policy_wait']) for record in episode.records),
            capacity_wait_events=sum(len(record['capacity_wait']) for record in episode.records),
            actual_started_events=sum(len(record['actual_started']) for record in episode.records))
        # Save the full real account before auditing; failed responsibility is never discarded.
        write_json(directory/'result.json',result)
        try:audit=verify_result(json.loads((directory/'result.json').read_text(encoding='utf-8')),scenario,inv,cfg,1.5)
        except Exception as exc:audit=dict(errors=['Preserved result verifier error: '+repr(exc)])
        summary['verification_pass']=not audit['errors']
        if audit['errors']:
            summary['protocol_valid']=False
            summary['failure_reasons']=[*summary.get('failure_reasons',()),'SAVED_V2_VERIFICATION_FAILED']
        write_json(directory/'verification.json',audit);write_json(directory/'result.json',result)
        return batch,dict(summary)
    except Exception as exc:
        write_json(directory/'error.json',dict(error=repr(exc),minute=episode.env.minute,
            task_ledger=episode.env.ledger,past_steps=episode.env.steps,
            adapter_records=episode.adapter.records,rollout=batch))
        raise


def _compact(summary):
    # No per-task identity lists in publication tables.
    return {k:v for k,v in summary.items() if k not in ('irreversible_miss_ids',)}


def _valid(summary):
    return (summary.get('protocol_valid',True) and summary.get('verification_pass')
        and not summary.get('failure_latched') and summary.get('actual_common_idle_tail_valid')
        and summary.get('unfinished') == 0 and summary.get('deadline_censored') == 0)


def check_pairs(evaluations, fragments, expected_methods):
    for fragment in fragments:
        rows=[row for row in evaluations if row['fragment']==fragment]
        if len(rows)!=expected_methods or max(row['inlet_final_c'] for row in rows)-min(row['inlet_final_c'] for row in rows)>.05:
            raise RuntimeError('Paired terminal-temperature gate failed')
        if max(row['it_kwh'] for row in rows)-min(row['it_kwh'] for row in rows)>1e-7:
            raise RuntimeError('Paired IT-energy conservation failed')


def execute(output):
    plan=json.loads(PLAN_PATH.read_text(encoding='utf-8'));validate_plan(plan)
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('Commit admission/config/implementation before the sole real pilot')
    output=Path(output).resolve()
    if not output.is_relative_to((ROOT/'research_notes/sca-ppo').resolve()):
        raise ValueError('Checkpoints and large logs must remain under ignored research_notes/sca-ppo')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    torch.set_num_threads(1)
    metadata=dict(schema='sca-ppo-e2-pilot-execution-v1',source_sha=source,protocol_sha256=file_sha(PLAN_PATH),
        data_admission_sha256=file_sha(ADMISSION),status='running',training_steps=0,completed_training_episodes=0,
        evaluations=[],errors=[],seed=plan['seed'],torch_version=torch.__version__,device='cpu',
        source_sha256={str(p.relative_to(ROOT)).replace('\\','/'):file_sha(p)
                       for p in sorted((ROOT/'experiments/sca_ppo').glob('*.py'))})
    output.parent.mkdir(parents=True,exist_ok=True)
    marker=output.parent/'E2-pilot-once.json'
    with marker.open('x',encoding='utf-8') as stream:json.dump(metadata,stream,indent=2)
    output.mkdir(exist_ok=False);started=monotonic();metrics=[];evaluations=[]
    trainer=None
    try:
        cfg=load_config();inv=build_inventory(cfg,'OM')
        scenarios={f:load_authorized(Path(plan['pool'])/f,plan['inputs'][f])
                   for f in plan['train_ids']+plan['holdout_ids']}
        backgrounds={}
        for f,s in scenarios.items():
            idle=run_v2(s,inv,cfg,'edf','fixed',1.5,idle=True)
            audit=verify_result(idle,s,inv,cfg,1.5,idle=True)
            write_json(output/'backgrounds'/f/'result.json',idle)
            write_json(output/'backgrounds'/f/'verification.json',audit)
            if audit['errors']:raise RuntimeError('Common background verification failed')
            backgrounds[f]=idle
        trainer=PPOTrainer(plan['ppo'])
        metadata['initial_model_fingerprint']=model_fingerprint(trainer.model)
        checkpoint_meta=dict(protocol_sha256=file_sha(PLAN_PATH),source_sha=source,completed_episodes=0,
            training_steps=0,training_sequence=plan['training_sequence'],data_admission_sha256=file_sha(ADMISSION))
        trainer.save(output/'checkpoints/initial.pt',checkpoint_meta)

        # Deterministic initial evaluation is saved before any optimizer update.
        for f in plan['holdout_ids']:
            s=scenarios[f];group=output/'evaluations'/f
            for label,arm in [('EDF-CG','A'),('CarbonRule-DDF-CG','C')]:
                row=run_episode(s,inv,cfg,arm,group/label,
                    dict(fragment=f,capacity='OM',source_sha=source),
                    dict(action_probe_minutes=[],stop_decision_seconds=30))
                row.update(method=label,common_idle_carbon_kg=backgrounds[f]['summary']['carbon_kg'],
                    common_incremental_carbon_kg=row['carbon_kg']-backgrounds[f]['summary']['carbon_kg'])
                row['reward_sum']=-row['common_incremental_carbon_kg']/plan['reward']['c_scale_kg']-plan['reward']['lambda_miss']*row['deadline_missed']
                path=group/label/'result.json';saved=json.loads(path.read_text());saved['summary'].update(row);write_json(path,saved)
                evaluations.append(_compact(row))
                if not _valid(row):raise RuntimeError('Initial comparison failed responsibility gate; no training')
            _,row=rollout(trainer,s,inv,cfg,backgrounds[f],group/'PPO-initial',plan,True,'PPO-initial')
            evaluations.append(_compact(row))
            if not _valid(row):raise RuntimeError('Initial policy failed responsibility gate; no training')
        check_pairs(evaluations,plan['holdout_ids'],3)
        metadata['initial_evaluation_passed']=True
        if model_fingerprint(trainer.model) != metadata['initial_model_fingerprint']:
            raise RuntimeError('Initial evaluation changed network parameters')
        write_csv(output/'evaluation-summary.csv',evaluations)
        # Restore exact RNG/optimizer state from the initial checkpoint before learning.
        trainer=PPOTrainer.load(output/'checkpoints/initial.pt')
        for index,f in enumerate(plan['training_sequence'],1):
            batch,row=rollout(trainer,scenarios[f],inv,cfg,backgrounds[f],
                              output/'train'/f'episode-{index:03d}',plan,False,'PPO-training')
            metadata['training_steps']+=len(batch['states'])
            metadata['completed_training_episodes']=index
            row.update(episode=index,source_sha=source)
            if not _valid(row):
                metrics.append(_compact(dict(row,update_performed=False)))
                write_csv(output/'training-metrics.csv',metrics)
                raise RuntimeError('Training responsibility gate failed; no update or subsequent episode')
            update=trainer.update(batch)
            if update['actor_parameter_delta_l2'] <= 0 or update['parameter_delta_l2'] <= 0:
                raise RuntimeError('Network policy parameters did not update')
            row.update(update,update_performed=True)
            metrics.append(_compact(row));write_csv(output/'training-metrics.csv',metrics)
            checkpoint_meta.update(completed_episodes=index,training_steps=metadata['training_steps'])
            path=output/'checkpoints'/f'episode-{index:03d}.pt'
            expected=model_fingerprint(trainer.model)
            trainer.save(path,checkpoint_meta);restored=PPOTrainer.load(path)
            if (model_fingerprint(restored.model)!=expected or restored.update_count!=index
                    or restored.optimizer_steps!=trainer.optimizer_steps):
                raise RuntimeError('Complete checkpoint round-trip failed')
            trainer=restored
            metadata['latest_checkpoint']=str(path.relative_to(output))
            write_json(output/'execution.json',metadata)
            print(f"E2 train {index}/32 {f[6:14]} return={row['reward_sum']:.6f} miss={row['deadline_missed']} actor_delta={update['actor_parameter_delta_l2']:.6g}",flush=True)
        if metadata['training_steps']>9216:raise AssertionError('Training budget exceeded')
        trainer.save(output/'checkpoints/final.pt',checkpoint_meta)
        trainer=PPOTrainer.load(output/'checkpoints/final.pt')
        metadata['final_model_fingerprint']=model_fingerprint(trainer.model)
        for f in plan['holdout_ids']:
            _,row=rollout(trainer,scenarios[f],inv,cfg,backgrounds[f],
                output/'evaluations'/f/'PPO-final',plan,True,'PPO-final')
            evaluations.append(_compact(row));write_csv(output/'evaluation-summary.csv',evaluations)
            if not _valid(row):raise RuntimeError('Final policy responsibility gate failed')
        if model_fingerprint(trainer.model)!=metadata['final_model_fingerprint']:
            raise RuntimeError('Final frozen evaluation changed network parameters')
        # Same fixed thermal mass, tasks and terminal request across paired methods.
        check_pairs(evaluations,plan['holdout_ids'],4)
        metadata['status']='completed';metadata['optimizer_updates']=trainer.update_count
        metadata['optimizer_steps']=trainer.optimizer_steps
    except Exception as exc:
        metadata['status']='failed';metadata['errors'].append(repr(exc))
        if trainer is not None:
            try:trainer.save(output/'checkpoints/failed-state.pt',dict(status='failed',errors=metadata['errors']))
            except Exception as checkpoint_exc:metadata['errors'].append('Failure checkpoint: '+repr(checkpoint_exc))
    metadata.update(elapsed_seconds=monotonic()-started,evaluation_episodes=len(evaluations),
        evaluation_steps=288*len(evaluations),background_episodes=len(list((output/'backgrounds').glob('*/result.json'))),
        evaluations=evaluations,checkpoints={str(p.relative_to(output)).replace('\\','/'):file_sha(p)
            for p in sorted((output/'checkpoints').glob('*.pt'))})
    write_csv(output/'training-metrics.csv',metrics);write_csv(output/'evaluation-summary.csv',evaluations)
    write_json(output/'execution.json',metadata)
    write_json(marker,{k:v for k,v in metadata.items() if k not in ('evaluations','source_sha256')})
    print(json.dumps({k:metadata[k] for k in ('status','training_steps','completed_training_episodes','evaluation_episodes','elapsed_seconds','errors')},indent=2),flush=True)
    return metadata


def evaluate_checkpoint(checkpoint, output):
    """Explicit frozen-policy replay only; never selects or updates a model.

    Full resume state is exposed by PPOTrainer.load. A stopped pilot is not
    automatically resumed; any further training requires a new authorization.
    """
    plan=json.loads(PLAN_PATH.read_text(encoding='utf-8'));validate_plan(plan)
    output=Path(output).resolve()
    if not output.is_relative_to((ROOT/'research_notes/sca-ppo').resolve()):
        raise ValueError('Evaluation logs must remain under ignored research_notes/sca-ppo')
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    trainer=PPOTrainer.load(checkpoint)
    if trainer.config != plan['ppo']:
        raise ValueError('Checkpoint training configuration differs from preregistration')
    before=model_fingerprint(trainer.model);rows=[]
    cfg=load_config();inv=build_inventory(cfg,'OM')
    for fragment in plan['holdout_ids']:
        scenario=load_authorized(Path(plan['pool'])/fragment,plan['inputs'][fragment])
        idle=run_v2(scenario,inv,cfg,'edf','fixed',1.5,idle=True)
        _,row=rollout(trainer,scenario,inv,cfg,idle,output/fragment,plan,True,'PPO-frozen-replay')
        rows.append(_compact(row));write_csv(output/'evaluation-summary.csv',rows)
        if not _valid(row):raise RuntimeError('Frozen evaluation failed complete-responsibility gate')
    if model_fingerprint(trainer.model)!=before:raise AssertionError('Evaluation changed model')
    write_json(output/'evaluation-identity.json',dict(checkpoint_sha256=file_sha(checkpoint),
        protocol_sha256=file_sha(PLAN_PATH),model_fingerprint=before,updates_performed=0))
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(ROOT/'research_notes/sca-ppo/e2-pilot-001'))
    parser.add_argument('--evaluate-checkpoint',help='Frozen held-out replay only; no training/update')
    args=parser.parse_args()
    if args.evaluate_checkpoint:evaluate_checkpoint(args.evaluate_checkpoint,args.output)
    elif execute(args.output)['status']!='completed':raise SystemExit(2)
