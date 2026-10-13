"""Read-only E1-6B smoke artifact audit; creates a new gate receipt only."""
import gzip
import hashlib
import json
import math
from pathlib import Path


ROOT=Path(__file__).resolve().parent
GROUPS=('plain','state','shield','combined')


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def load(path):return json.loads(path.read_text(encoding='utf8'))


def finite(value):
    if isinstance(value,float):return math.isfinite(value)
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,(tuple,list)):return all(finite(v) for v in value)
    return True


def main():
    destination=ROOT/'smoke-gate-001.json'
    if destination.exists():raise FileExistsError(destination)
    summaries=[]
    for group in GROUPS:
        folder=ROOT/f'smoke-{group}-001'
        summary=load(folder/'training_summary.json')
        configuration=load(folder/'training_config.json')
        checks={name:sha(folder/name)==value for name,value in summary['files'].items()}
        sources=load(folder/'source-freeze.json')['sources']
        source_checks={name:sha(folder/'source'/name)==value for name,value in sources.items()}
        assert checks and all(checks.values()) and source_checks and all(source_checks.values())
        assert summary['group']==group and summary['num_timesteps']==1024 and summary['seed']==55100
        assert configuration['schema']=='e1-6b-adaptive-ppo-v1'
        assert len(configuration['training_identities'])==2
        assert [i['factor'] for i in configuration['training_identities']]==[1.5,2.]
        assert all(i['split']=='train' for i in configuration['training_identities'])
        assert summary['parameter_sha256_before']!=summary['parameter_sha256_after']
        assert summary['deterministic_roundtrip'] and summary['optimizer_state_entries']>0
        assert summary['elapsed_seconds']<=720 and summary['peak_rss_kib']/1024**2<=4
        assert summary['parameter_count']==19397 and summary['observation_size']==83 and summary['action_size']==4
        updates=load(folder/'updates.json')
        assert len(updates)==4 and finite(updates)
        assert updates[-1]['train/n_updates']==40 and updates[-1]['environment_steps']==1024
        seen=set();action_counts=[0]*4;scene_steps={};episode_factors={}
        miss_count=0;filter_counts={'candidate_checks':0,'solo_eligible':0,'filter_veto':0,'deferred':0}
        with gzip.open(folder/'steps.jsonl.gz','rt',encoding='utf8') as stream:
            for index,line in enumerate(stream,1):
                row=json.loads(line)
                assert row['step']==index and finite(row) and len(row['observation'])==83
                assert 0<=row['action']<4 and not row['truncated']
                assert row['scene']['split']=='train'
                factor=row['scene']['factor'];key=str(factor)
                assert factor==(1.5 if row['episode']%2==0 else 2.)
                assert row['scene']['episode_index']==row['episode']
                action_counts[row['action']]+=1;scene_steps[key]=scene_steps.get(key,0)+1
                episode_factors[row['episode']]=factor
                assert row['reward']==-(row['reward_terms']['carbon_cost']+row['reward_terms']['wait_cost']+
                    row['reward_terms']['deadline_cost']+row['reward_terms']['terminal_cost'])
                misses=row['reward_terms']['new_deadline_miss_ids']
                assert not seen.intersection(misses);seen.update(misses);miss_count+=len(misses)
                for count in filter_counts:filter_counts[count]+=row['risk_counts'][count]
                assert row['risk_counts']['candidate_checks']==len(row['policy_diagnostics'])
                assert row['risk_counts']['filter_veto']==sum(d['filter_veto'] for d in row['policy_diagnostics'])
                assert row['risk_counts']['deferred']==len(row['policy_deferred_ids'])
                assert not set(row['started_ids']).intersection(row['policy_deferred_ids'])
                assert not set(row['started_ids']).intersection(row['capacity_waiting_ids'])
                if group in ('plain','shield'):assert all(v==0. for v in row['observation'][40:])
                if group in ('plain','state'):assert not row['risk_counts']['filter_veto']
                if row['terminated']:seen=set()
        assert index==1024 and action_counts==summary['action_counts'] and scene_steps==summary['scene_steps']
        assert len(episode_factors)==summary['completed_episodes']+int(summary['budget_ends_mid_episode'])
        assert summary['reward_components']['terminal_cost']==0.
        summaries.append(dict(group=group,exit_code=0,num_timesteps=summary['num_timesteps'],
            elapsed_seconds=summary['elapsed_seconds'],peak_rss_gib=summary['peak_rss_kib']/1024**2,
            completed_episodes=summary['completed_episodes'],partial_steps=summary['current_partial_episode_steps'],
            action_counts=action_counts,scene_steps=scene_steps,parameter_count=summary['parameter_count'],
            initial_parameter_sha256=summary['parameter_sha256_before'],
            final_parameter_sha256=summary['parameter_sha256_after'],checkpoint_sha256=summary['checkpoint_sha256'],
            input_budget_sha256=summary['input_budget_sha256'],source_sha256=summary['source_sha256'],
            preregistered_config_sha256=configuration['preregistered_config_sha256'],
            artifact_files_verified=len(checks),source_files_verified=len(source_checks),
            steps_all_finite=True,first_miss_duplicate_count=0,new_deadline_misses=miss_count,
            filter_counts=filter_counts,optimizer_state_entries=summary['optimizer_state_entries']))
    for key in ('initial_parameter_sha256','parameter_count','input_budget_sha256','source_sha256',
                'preregistered_config_sha256'):
        assert len({r[key] for r in summaries})==1,key
    receipt=dict(schema='e1-6b-smoke-functional-cost-gate-v1',passed=True,
        formal_training_started=False,formal_test_evaluated=False,
        criterion='Train-only;1024 steps;finite;updated;roundtrip;wall<=720s;RSS<=4GiB;shared parameters/budget/source',
        concurrency='independent Python processes; at most two smoke jobs concurrent; other local tests may overlap',
        interpretation='functional/cost gate only; rewards and outcomes did not select coefficients, group or budget',
        source_version_note='early smoke source retained unchanged; later audited source revisions require a new formal freeze',
        groups=summaries)
    destination.write_text(json.dumps(receipt,sort_keys=True,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'passed':True,'receipt':str(destination),'groups':len(summaries)},sort_keys=True))


if __name__=='__main__':main()
