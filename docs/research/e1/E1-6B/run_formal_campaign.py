"""Fixed-budget Train-only campaign supervisor; root authorizes actual launch.

Use in the configured WSL Python runtime. This script creates fresh receipts,
never replaces old outputs, and stops scheduling on failure/source mutation.
"""
import argparse
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
NOTES=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

from experiments.e1.contracts import canonical_bytes, content_hash, file_hash, write_json
from experiments.e1.temporal import experiment_sources
from experiments.e1.cars_experiment import _snapshot_sources

GROUPS=('plain','state','shield','combined')
SEEDS=(55101,55102,55103)
STEPS=20480
MAX_WORKERS=3


def stamp():return datetime.now(timezone.utc).isoformat()


def read(path):return json.loads(path.read_text(encoding='utf8'))


def finite(value):
    if isinstance(value,float):return math.isfinite(value)
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,(tuple,list)):return all(finite(v) for v in value)
    return True


def validate_model(folder,group,seed,source_sha):
    s=read(folder/'training_summary.json');cfg=read(folder/'training_config.json')
    assert s['group']==group and s['seed']==seed and s['num_timesteps']==STEPS
    assert s['source_sha256']==source_sha
    assert s['parameter_count']==19397 and s['observation_size']==83 and s['action_size']==4
    assert s['parameter_sha256_before']!=s['parameter_sha256_after'] and s['deterministic_roundtrip']
    assert s['optimizer_state_entries']>0 and s['elapsed_seconds']<=14400 and s['peak_rss_kib']/1024**2<=6
    assert [i['factor'] for i in cfg['training_identities']]==[1.5,2.]
    assert all(i['split']=='train' for i in cfg['training_identities'])
    assert all(file_hash(folder/name)==sha for name,sha in s['files'].items())
    updates=read(folder/'updates.json')
    assert len(updates)==80 and updates[-1]['train/n_updates']==800 and finite(updates)
    assert updates[-1]['environment_steps']==STEPS
    episodes=read(folder/'episodes.json')
    assert all(e['terminated'] and not e['truncated'] and
        e['summary']['completed']==e['summary']['arrived'] and
        e['summary']['pending']==e['summary']['running']==0 for e in episodes)
    counts=[0]*4;scene_steps={};seen=set();risk={key:0 for key in ('candidate_checks','solo_eligible','filter_veto','deferred')}
    with gzip.open(folder/'steps.jsonl.gz','rt',encoding='utf8') as stream:
        for index,line in enumerate(stream,1):
            r=json.loads(line)
            assert r['step']==index and finite(r) and len(r['observation'])==83 and not r['truncated']
            assert r['scene']['split']=='train' and r['scene']['episode_index']==r['episode']
            expected_factor=(1.5,2.)[(seed%2+r['episode'])%2]
            assert r['scene']['factor']==expected_factor
            assert 0<=r['action']<4
            counts[r['action']]+=1;factor=str(r['scene']['factor'])
            scene_steps[factor]=scene_steps.get(factor,0)+1
            misses=r['reward_terms']['new_deadline_miss_ids']
            assert not seen.intersection(misses);seen.update(misses)
            assert r['risk_counts']['candidate_checks']==len(r['policy_diagnostics'])
            assert r['risk_counts']['deferred']==len(r['policy_deferred_ids'])
            assert not set(r['started_ids']).intersection(r['capacity_waiting_ids'])
            assert not set(r['started_ids']).intersection(r['policy_deferred_ids'])
            for key in risk:risk[key]+=r['risk_counts'][key]
            if group in ('plain','shield'):assert all(v==0. for v in r['observation'][40:])
            if group in ('plain','state'):assert r['risk_counts']['filter_veto']==0
            if r['terminated']:seen=set()
    assert index==STEPS and counts==s['action_counts'] and scene_steps==s['scene_steps']
    return dict(group=group,seed=seed,steps=STEPS,elapsed_seconds=s['elapsed_seconds'],
        peak_rss_gib=s['peak_rss_kib']/1024**2,parameter_count=s['parameter_count'],
        initial_parameter_sha256=s['parameter_sha256_before'],final_parameter_sha256=s['parameter_sha256_after'],
        checkpoint_sha256=s['checkpoint_sha256'],source_sha256=s['source_sha256'],
        input_budget_sha256=s['input_budget_sha256'],config_sha256=s['config_sha256'],
        action_counts=s['action_counts'],scene_steps=s['scene_steps'],scene_episodes=s['scene_episodes'],
        completed_episodes=s['completed_episodes'],partial_steps=s['current_partial_episode_steps'],
        risk_counts=risk,all_steps_finite=True,first_miss_duplicate_count=0,updates=80,epochs=800)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(NOTES/'formal-campaign-001'))
    args=parser.parse_args(argv)
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    cfgpath=ROOT/'experiments/e1/configs/e1_6b_ppo.json'
    cfg=read(cfgpath);smoke_path=NOTES/'smoke-gate-001.json';smoke=read(smoke_path)
    if not smoke.get('passed'):raise ValueError('Smoke functional/cost gate not passed')
    if (cfg['schema']!='e1-6b-adaptive-ppo-v1' or cfg['rl_seeds']!=list(SEEDS)
        or cfg['budgets']['formal']!=STEPS or cfg['test_evaluation'] is not False
        or cfg['scene_sampling']['rule']!='seed_mod_2_initial_then_alternate'
        or any(s['split']!='train' for s in cfg['scenes'])):
        raise ValueError('Frozen Train campaign protocol differs')
    sources=experiment_sources();source_sha=content_hash(sources);config_sha=file_hash(cfgpath)
    from experiments.e1.rl_training_utils import load_bundle
    training_identities=[]
    for scene in cfg['scenes']:
        path=ROOT/scene['input']
        if file_hash(path/'bundle.json')!=scene['manifest_sha256']:
            raise ValueError('Frozen Train bundle identity changed')
        bundle=load_bundle(path,allowed_split='train')
        if bundle.factor!=scene['factor']:raise ValueError('Frozen Train factor changed')
        training_identities.append(dict(split=bundle.split,factor=bundle.factor,
            workload_sha256=bundle.workload.sha256,environment_sha256=bundle.environment.sha256,
            identity=bundle.identity))
    budget_hashes={str(seed):content_hash(dict(training_identities=training_identities,
        seed=seed,total_timesteps=STEPS,scene_order=[1.5,2.],scene_offset=seed%2,
        sampling='alternating episodes',observation_size=83,action_size=4)) for seed in SEEDS}
    jobs=[]
    for group in GROUPS:
        for seed in SEEDS:
            folder=NOTES/f'train-{group}-{seed}-001'
            log=NOTES/f'formal-{group}-{seed}-001.txt'
            if folder.exists() or log.exists():raise FileExistsError(folder if folder.exists() else log)
            command=[sys.executable,'-m','experiments.e1.adaptive_train','--config',str(cfgpath),
                '--group',group,'--seed',str(seed),'--steps',str(STEPS),'--output',str(folder)]
            jobs.append(dict(group=group,seed=seed,output=str(folder),log=str(log),command=command))
    out.mkdir(parents=True,exist_ok=False)
    _snapshot_sources(out,sources)
    shutil.copyfile(cfgpath,out/'preregistered-config.json')
    write_json(out/'source-freeze.json',dict(sources=sources,source_sha256=source_sha))
    manifest=dict(schema='e1-6b-fixed-formal-campaign-v1',created_at_utc=stamp(),jobs=jobs,
        groups=list(GROUPS),seeds=list(SEEDS),steps_per_model=STEPS,total_steps=12*STEPS,
        max_workers=MAX_WORKERS,source_sha256=source_sha,config_sha256=config_sha,
        training_identities=training_identities,input_budget_sha256_by_seed=budget_hashes,
        smoke_gate_sha256=file_hash(smoke_path),smoke_source_sha256=smoke['groups'][0]['source_sha256'],
        smoke_revision_note='Earlier schema-identical smoke retained; audited loader/replay/analysis/tests added before formal freeze',
        training_splits=['train'],formal_test_evaluated=False,validation_training=False)
    write_json(out/'campaign.json',manifest)
    progress=(out/'progress.jsonl').open('wb')
    completed=[];running={};next_job=0;failed=False;failure_reason=None
    def event(value):
        progress.write(canonical_bytes(dict(time_utc=stamp(),**value))+b'\n');progress.flush()
        print(json.dumps(value,sort_keys=True),flush=True)
    def start(job):
        stream=Path(job['log']).open('w',encoding='utf8')
        p=subprocess.Popen(job['command'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
            text=True,encoding='utf8')
        running[p.pid]=(p,stream,job,time.monotonic())
        event(dict(event='started',group=job['group'],seed=job['seed'],pid=p.pid))
    last_source_check=0.
    try:
        while running or (next_job<len(jobs) and not failed):
            now=time.monotonic()
            if now-last_source_check>=5:
                if experiment_sources()!=sources or file_hash(cfgpath)!=config_sha:
                    failed=True;failure_reason='source_or_protocol_changed'
                    event(dict(event='aborting',reason=failure_reason))
                    for p,_,_,_ in running.values():p.terminate()
                last_source_check=now
            while not failed and len(running)<MAX_WORKERS and next_job<len(jobs):
                start(jobs[next_job]);next_job+=1
            for pid,(p,stream,job,began) in list(running.items()):
                if p.poll() is None:
                    if time.monotonic()-began>14400:
                        failed=True;failure_reason='model_wall_limit';p.terminate()
                        event(dict(event='limit_exceeded',group=job['group'],seed=job['seed']))
                    continue
                wall=time.monotonic()-began;stream.close();del running[pid]
                row=dict(group=job['group'],seed=job['seed'],exit_code=p.returncode,
                         supervisor_wall_seconds=wall,output=job['output'],log=job['log'])
                if p.returncode==0:
                    try:row.update(validate_model(Path(job['output']),job['group'],job['seed'],source_sha))
                    except Exception as exc:
                        failed=True;failure_reason='artifact_validation_failed'
                        row['audit_error']=type(exc).__name__+': '+str(exc)
                else:failed=True;failure_reason=failure_reason or 'child_failed'
                completed.append(row);event(dict(event='finished',**row))
                write_json(out/'progress-receipt.json',dict(completed=completed,
                    running=[j for _,_,j,_ in running.values()],pending=jobs[next_job:],failed=failed))
            if running:time.sleep(.5)
        if not failed and len(completed)==12:
            for seed in SEEDS:
                same=[r for r in completed if r['seed']==seed]
                for key in ('initial_parameter_sha256','input_budget_sha256','parameter_count'):
                    assert len({r[key] for r in same})==1,(seed,key)
                assert all(r['input_budget_sha256']==budget_hashes[str(seed)] for r in same)
            assert len({r['source_sha256'] for r in completed})==1
            assert experiment_sources()==sources and file_hash(cfgpath)==config_sha
        receipt=dict(schema=manifest['schema'],completed_at_utc=stamp(),passed=not failed and len(completed)==12,
            failure_reason=failure_reason,models=completed,unstarted_jobs=jobs[next_job:],
            source_sha256=source_sha,config_sha256=config_sha,smoke_gate_sha256=file_hash(smoke_path),
            total_actual_steps=sum(r.get('steps',0) for r in completed),validation_training=False,
            formal_test_evaluated=False)
        write_json(out/'campaign-receipt.json',receipt)
        event(dict(event='campaign_complete',passed=receipt['passed'],models=len(completed)))
        if not receipt['passed']:raise RuntimeError('Fixed Train campaign failed; all evidence preserved')
        return receipt
    except Exception as exc:
        write_json(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),
            failure_reason=failure_reason,completed=completed,unstarted_jobs=jobs[next_job:]))
        raise
    finally:
        for p,stream,_,_ in running.values():
            if p.poll() is None:p.terminate()
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            stream.close()
        progress.close()


if __name__=='__main__':main()
