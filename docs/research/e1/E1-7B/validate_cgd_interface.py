"""Finite Train RL rollouts and tiny synthetic SB3 smoke; no research training."""
import argparse,json,time
from pathlib import Path
import numpy as np
from experiments.e1.carbon_gain_experiment import CONFIG,load_frozen,load_registered_bundle
from experiments.e1.carbon_gain_rl_env import CarbonGainSchedulingEnv
from experiments.e1.online_ci_forecast import extract_prehistory
from experiments.e1.online_power_surrogate import PowerSurrogate
from experiments.e1.config import ROOT
from experiments.e1.contracts import file_hash,write_json,content_hash
from experiments.e1.contracts import canonical_bytes
from experiments.e1.run import _fresh_run,_write_result
from experiments.e1.temporal import write_csv

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',type=Path,required=True);ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    cfg=json.loads(CONFIG.read_text());frozen=load_frozen(args.freeze);bundle=load_registered_bundle(cfg,'train','1')
    model=PowerSurrogate.from_dict(frozen['model']);prior=extract_prehistory(ROOT/'data/carbon_intensity/US-CAL-CISO/2023/US-CAL-CISO_2023_hourly.csv',bundle.workload.start,expected_sha256=bundle.environment.carbon_source_sha256)
    common=json.loads((args.reference/'load-1/fairness.json').read_text())['common_end']
    def action(out):
        summaries=[];equivalence=[];all_steps=[]
        for mode,policy in ((0,'edf_backfill'),(1,'m2_moderate'),(2,'m2'),('random',None)):
            live=CarbonGainSchedulingEnv(bundle.workload,bundle.environment,bundle.capacity,bundle.power_factory,bundle.parameters,
                model,prehistory=prior,moderate_threshold_kg=frozen['moderate_threshold_kg'],stop_at_drain=True,
                reward_config=bundle.reward_config,normalization=bundle.normalization)
            before=time.perf_counter();live.reset(seed=77101);rewards=[];counts=[0,0,0];informative=0;distinct12=0
            interactions=[];actual_mode12_checks=actual_mode12_different=actual_deferred_steps=0
            while True:
                raw=live.observation_raw['carbon_gain'];rows=raw.get('candidates',[])
                aggressive={r['task_id'] for r in rows if r.get('gain_total_kg') is not None and r['gain_total_kg']>0}
                moderate={r['task_id'] for r in rows if r.get('gain_total_kg') is not None and frozen['moderate_threshold_kg'] is not None and r['gain_total_kg']>frozen['moderate_threshold_kg']}
                informative+=bool(aggressive);distinct12+=aggressive!=moderate
                selected=int(live.action_space.sample()) if mode=='random' else mode;counts[selected]+=1
                obs,r,term,trunc,info=live.step(selected);rewards.append(info['reward'])
                assert np.isfinite(obs).all() and live.observation_space.contains(obs)
                diagnostics=info['policy_diagnostics']
                actual_deferred_steps+=any(d['defer'] for d in diagnostics)
                # On an action1/2 pass, the first different eligible decision
                # proves different started IDs; no difference proves identical
                # one-pass allocations by induction. This is not an assessment
                # of future episode performance. Action0 skips estimates.
                exact12=None
                if selected in (1,2):
                    valid=[d for d in diagnostics if d['status']=='ok']
                    aset={d['task_id'] for d in valid if d['gain_total_kg']>0}
                    mset={d['task_id'] for d in valid if d['gain_total_kg']>frozen['moderate_threshold_kg']}
                    exact12=aset==mset;actual_mode12_checks+=1;actual_mode12_different+=not exact12
                interactions.append(dict(time=info['interval']['interval_start'],action=selected,
                    observation_after=obs.tolist(),reward=r,reward_components=info['reward'],
                    candidate_diagnostics=diagnostics,mode1_2_equivalent_this_pass=exact12))
                all_steps.append(dict(mode=str(mode),step=sum(counts),time=info['interval']['interval_start'],action=selected,
                    reward=r,actual_carbon_kg=info['reward']['carbon_kg'],new_misses=info['reward']['new_deadline_misses'],
                    prepass_positive=len(aggressive),prepass_moderate=len(moderate),actual_pass_mode1_2_equivalent=exact12))
                if term or trunc:
                    if trunc:raise RuntimeError('Full Train RL rollout truncated')
                    break
            own=live.executor.result();result=live.executor.extend_idle(common)
            target=out/('action-'+str(mode));target.mkdir();_write_result(target,result)
            with (target/'interactions.jsonl').open('wb') as stream:
                for row in interactions:stream.write(canonical_bytes(row)+b'\n')
            if policy:
                for name in ('ledger.jsonl','intervals.csv'):
                    actual=file_hash(target/name);expected=file_hash(args.reference/'load-1'/policy/name)
                    if actual!=expected:raise RuntimeError('Fixed action not equivalent '+str(mode)+'/'+name)
                    equivalence.append(dict(action=mode,reference_policy=policy,file=name,sha256=actual))
            assert sum(t['new_deadline_misses'] for t in rewards)==result.summary['deadline_missed_completed']
            summaries.append(dict(mode=str(mode),**result.summary,own_drain=own.summary['drained_at'],
                actions=counts,steps=sum(counts),opportunity_steps=informative,action1_2_distinct_steps=distinct12,
                actual_mode1_2_checked_steps=actual_mode12_checks,actual_mode1_2_different_steps=actual_mode12_different,
                actual_policy_deferred_steps=actual_deferred_steps,
                reward_misses=sum(t['new_deadline_misses'] for t in rewards),wall_seconds=time.perf_counter()-before))
            print(json.dumps(dict(mode=mode,completed=result.summary['completed'],missed=result.summary['deadline_missed_completed'])),flush=True)
            live.close()
        write_csv(out/'rollout-steps.csv',all_steps)
        # Determinism replay uses the same full finite cohort and random sequence.
        write_json(out/'interface-result.json',dict(schema='e1-7b-cgd-rl-v1',observation_dim=54,actions=3,
            fixed_equivalence=equivalence,rollouts=summaries,formal_ppo=False,test_access=False,
            source_script_sha256=file_hash(Path(__file__)),frozen_sha256=file_hash(args.freeze)))
        write_json(out/'output-hashes.json',{p.relative_to(out).as_posix():file_hash(p) for p in out.rglob('*')
            if p.is_file() and p.name!='run_status.json'})
    _fresh_run(args.output,action)

if __name__=='__main__':main()
