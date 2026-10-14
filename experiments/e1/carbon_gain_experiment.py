"""Frozen CGD V1 calibration, development evaluation and exact scientific replay."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from .cars import make_cars_policy
from .carbon_gain_policy import CarbonGainEDF
from .config import ROOT
from .contracts import content_hash, file_hash, write_json, canonical_bytes, utc_time
from .drain_metrics import phase_summary
from .engine import LocalExecutor
from .online_ci_forecast import extract_prehistory,evaluate_forecasts
from .online_power_surrogate import PowerSurrogate,fit_surrogate,evaluate_surrogate
from .rl_training_utils import load_bundle
from .run import _fresh_run,_write_result,source_hashes,dependencies,_git
from .temporal import write_csv

VERSION='e1-7b-cgd-rl-v1'
CONFIG=ROOT/'experiments/e1/configs/e1_7b_cgd_rl_v1.json'
EXTERNAL=('edf_backfill','edf_carbon_threshold','cars_risk_only')
BASELINE_FILES=('ledger.jsonl','events.jsonl','intervals.csv','summary.json','policy_ledger.jsonl',
                'policy_events.jsonl','policy_intervals.jsonl','policy_summary.json')

def validate_protocol(config):
    fixed=dict(schema=VERSION,test_access=False,formal_ppo_training=False,capacity=[50000,1000,80000],
        threshold_g_per_kwh=161.075,sla_multiplier=1.5,resource_multiplier=5,step_minutes=15,
        history_minutes=120,cold_buffer_steps=2,risk_buffer_steps=[0,1,2],forecast_method='seasonal',
        forecast_max_steps=96,hvac_alpha=0,hvac_tail_steps=0,weather_forecast='current_persistence',
        background='known_running_releases_only',unavailable_fallback='execute_now',
        moderate_quantile=.75,gain_threshold_kg=0,factors=[1.,1.5,2.])
    if any(config.get(k)!=v for k,v in fixed.items()):raise ValueError('Frozen CGD protocol differs')
    if set(config['inputs'])!={'train','validation'}:raise ValueError('Test or undeclared input role')
    return config

def load_registered_bundle(config,split,label):
    if split not in ('train','validation'):raise ValueError('Test and undeclared roles forbidden')
    item=config['inputs'][split][label];folder=ROOT/item['path']
    if file_hash(folder/'bundle.json')!=item['sha256']:raise ValueError('Registered frozen cohort identity differs')
    return load_bundle(folder,allowed_split=split)

def load_frozen(path):
    doc=json.loads(Path(path).read_text(encoding='utf8'));p=doc['payload']
    if p.get('schema')!='e1-7b-calibration-v1' or p.get('fit_role')!='train' or p.get('test_access') is not False:
        raise ValueError('Calibration must be frozen Train only')
    if content_hash(p)!=doc['sha256']:raise ValueError('Calibration identity mismatch')
    PowerSurrogate.from_dict(p['model'])
    return p

def scientific_diagnostics(rows):
    return [{k:v for k,v in r.items() if k not in ('elapsed_seconds','decision_seconds')} for r in rows]

def _sources():
    hashes=source_hashes()
    hashes.update({p.relative_to(ROOT).as_posix():file_hash(p) for p in (ROOT/'tests/e1').glob('*.py')})
    hashes[CONFIG.relative_to(ROOT).as_posix()]=file_hash(CONFIG)
    return dict(sorted(hashes.items()))

def _provenance(out,config,inputs,initial_sources):
    if _sources()!=initial_sources:raise RuntimeError('Source changed during frozen run')
    files={p.relative_to(out).as_posix():file_hash(p) for p in out.rglob('*') if p.is_file()
           and p.name not in ('provenance.json','run_status.json','runtime.json','profiling.jsonl')}
    write_json(out/'provenance.json',dict(version=VERSION,config_sha256=content_hash(config),
        sources=initial_sources,source_sha256=content_hash(initial_sources),inputs=inputs,
        command=sys.argv,python=sys.version,dependencies=dependencies(),output_hashes=files,
        original_git=dict(head=_git('rev-parse','HEAD'),branch=_git('branch','--show-current')),
        publication_base=config['publication_base'],test_access=False))

def _reference_rows(reference,policy):
    hashes=json.loads((reference/'output_hashes.json').read_text())
    target=policy+'/intervals.csv'
    if hashes.get(target)!=file_hash(reference/target):raise ValueError('Historical physical interval hash mismatch')
    return pd.read_csv(reference/target).to_dict('records')

class _CalibrationEDF(CarbonGainEDF):
    """Observe opportunities at actual EDF starts, never change its actions."""
    def __init__(self,*args,**kw):
        super().__init__(*args,**kw);self.set_action(0);self.opportunities=[]
    def decide(self,spec,context):
        if self.guard.check(spec,context.now)['allowed']:
            r=self._estimate(spec)
            self.opportunities.append(dict(time=context.now.isoformat(),task_id=spec.task_id,**r))
        return super().decide(spec,context)

def calibrate(config,outpath):
    config=validate_protocol(config)
    def action(out):
        sources=_sources();bundles=[];it_rows=[];hvac_rows=[];input_ids=[]
        for factor in config['factors']:
            label=format(factor,'g');bundle=load_registered_bundle(config,'train',label);bundles.append(bundle)
            reference=ROOT/config['inputs']['train'][label]['reference'];input_ids.append(bundle.identity)
            for policy in EXTERNAL:
                rows=_reference_rows(reference,policy)
                hvac_rows.extend(rows)
                if label=='1' and policy=='edf_backfill':it_rows=rows
        model,fit=fit_surrogate(it_rows,hvac_rows,capacity=config['capacity'],role='train')
        write_json(out/'fit-report.json',fit)
        opportunities=[];forecasts=[];points=[];prehistory_hashes={}
        for bundle in bundles:
            label=format(bundle.factor,'g');folder=out/('load-'+label);folder.mkdir()
            raw=ROOT/json.loads((ROOT/config['inputs']['train'][label]['path']/'config.json').read_text())['paths']['carbon']
            prior=extract_prehistory(raw,bundle.workload.start,expected_sha256=bundle.environment.carbon_source_sha256)
            prior_json=[dict(time=r['time'].isoformat(),ci=r['ci'],source_time=r['source_time'].isoformat()) for r in prior]
            write_json(folder/'prehistory.json',prior_json);prehistory_hashes[label]=content_hash(prior_json)
            p=_CalibrationEDF('m2',bundle.workload.start,bundle.capacity,model,prehistory=prior)
            ex=LocalExecutor(bundle.workload,bundle.environment,bundle.capacity,bundle.power_factory(),p,True)
            ex.run(stop_at_drain=True)
            rows=[dict(factor=bundle.factor,**r) for r in p.opportunities]
            opportunities.extend(rows)
            samples=[dict(time=s.time,ci=s.ci_g_per_kwh) for s in bundle.environment.samples if s.time<bundle.workload.end]
            score=evaluate_forecasts(samples,prior)
            forecasts.extend(dict(factor=bundle.factor,**r) for r in score['summary'])
            if label=='1':points=score['points']
            print(json.dumps(dict(stage='calibration',factor=bundle.factor,candidates=len(rows))),flush=True)
        positive=[r['gain_total_kg'] for r in opportunities if r['status']=='ok' and r['gain_total_kg']>0]
        moderate=float(np.quantile(positive,.75)) if positive else None
        payload=dict(schema='e1-7b-calibration-v1',fit_role='train',test_access=False,
            config_sha256=content_hash(config),model=model.to_dict(),moderate_threshold_kg=moderate,
            positive_samples=len(positive),moderate_definition=config['moderate_source'],
            input_identities=input_ids,prehistory_sha256=prehistory_hashes,
            fit_report_sha256=file_hash(out/'fit-report.json'),source_sha256=content_hash(sources))
        write_json(out/'freeze.json',dict(payload=payload,sha256=content_hash(payload)))
        write_csv(out/'forecast-summary.csv',forecasts);write_csv(out/'forecast-points.csv',points)
        with (out/'opportunities.jsonl').open('wb') as stream:
            for r in scientific_diagnostics(opportunities):stream.write(canonical_bytes(r)+b'\n')
        _provenance(out,config,input_ids,sources)
        return payload
    return _fresh_run(outpath,action)

def make_gain_policy(name,bundle,model,prehistory,moderate):
    if name in EXTERNAL:return make_cars_policy(name,bundle.parameters,bundle.workload.start,bundle.capacity)
    mode='m2' if name.startswith('m2') else name
    p=CarbonGainEDF(mode,bundle.workload.start,bundle.capacity,model,prehistory=prehistory,
        forecast_method='persistence' if name=='m2_persistence' else 'seasonal',moderate_threshold_kg=moderate)
    if name=='m2_moderate':p.set_action(1)
    return p

def _write_jsonl(path,rows):
    with path.open('wb') as stream:
        for row in rows:stream.write(canonical_bytes(row)+b'\n')

def run_matrix(config,frozenpath,split,outpath):
    config=validate_protocol(config)
    if split not in ('train','validation'):raise ValueError('Test forbidden')
    frozen=load_frozen(frozenpath)
    if frozen['config_sha256']!=content_hash(config):raise ValueError('Frozen model/protocol mismatch')
    def action(out):
        sources=_sources();model=PowerSurrogate.from_dict(frozen['model']);allrows=[];allids=[];regressions=[]
        write_json(out/'freeze.json',dict(payload=frozen,sha256=content_hash(frozen)))
        write_json(out/'config.json',config)
        for factor in config['factors']:
            label=format(factor,'g');bundle=load_registered_bundle(config,split,label);allids.append(bundle.identity)
            folder=out/('load-'+label);folder.mkdir()
            cfg=json.loads((ROOT/config['inputs'][split][label]['path']/'config.json').read_text())
            prior=extract_prehistory(ROOT/cfg['paths']['carbon'],bundle.workload.start,expected_sha256=bundle.environment.carbon_source_sha256)
            write_json(folder/'input-identity.json',bundle.identity)
            prior_json=[dict(time=r['time'].isoformat(),ci=r['ci'],source_time=r['source_time'].isoformat()) for r in prior]
            write_json(folder/'prehistory.json',prior_json)
            reference=ROOT/config['inputs'][split][label]['reference']
            fairness=json.loads((reference/'fairness.json').read_text());historical_end=utc_time(fairness['common_end'],True)
            executors={};own={};runtimes={}
            for name in config['policies']:
                started=time.perf_counter();p=make_gain_policy(name,bundle,model,prior,frozen['moderate_threshold_kg'])
                ex=LocalExecutor(bundle.workload,bundle.environment,bundle.capacity,bundle.power_factory(),p,True)
                own[name]=ex.run(stop_at_drain=True);executors[name]=ex;runtimes[name]=time.perf_counter()-started
                if ex.drained_at is None or own[name].summary['completed']!=len(bundle.workload.tasks):
                    raise RuntimeError('Incomplete drain: no equal-workload conclusion')
                if name in EXTERNAL:
                    hist=folder/'history-controls'/name;hist.mkdir(parents=True)
                    _write_result(hist,ex.extend_idle(historical_end))
                    expected=json.loads((reference/'output_hashes.json').read_text())
                    for filename in BASELINE_FILES:
                        actual=file_hash(hist/filename);old=expected[name+'/'+filename]
                        if file_hash(reference/name/filename)!=old or actual!=old:
                            raise RuntimeError('Historical numeric regression changed '+split+'/'+label+'/'+name+'/'+filename)
                        regressions.append(dict(split=split,factor=factor,policy=name,file=filename,sha256=actual))
                print(json.dumps(dict(stage=split,factor=factor,policy=name,
                    completed=own[name].summary['completed'],missed=own[name].summary['deadline_missed_completed'],
                    drain=ex.drained_at.isoformat(),seconds=runtimes[name])),flush=True)
            common=max(historical_end,max(ex.drained_at for ex in executors.values()))
            results={n:ex.extend_idle(common) for n,ex in executors.items()}
            resource_totals={r:[phase_summary(v,common)[r+'_hours'] for v in results.values()] for r in ('cores','gpus','mem')}
            if not all(all(math.isclose(v,vs[0],abs_tol=1e-8,rel_tol=1e-12) for v in vs) for vs in resource_totals.values()):
                raise RuntimeError('Finite cohort resource-hours differ')
            write_json(folder/'fairness.json',dict(identical_inputs=True,all_completed=True,idle_tail=True,
                common_end=common.isoformat(),resource_hours=resource_totals,input_identity=bundle.identity,
                prehistory_sha256=content_hash(prior_json),test_access=False))
            for name,result in results.items():
                p=executors[name].policy;target=folder/name;target.mkdir();_write_result(target,result)
                summary=phase_summary(result,common);fixed=phase_summary(result,bundle.workload.end)
                write_json(target/'fixed-window.json',fixed);write_json(target/'own-drain.json',phase_summary(own[name],executors[name].drained_at))
                diag=getattr(p,'diagnostics',[]);_write_jsonl(target/'diagnostics.jsonl',scientific_diagnostics(diag))
                _write_jsonl(target/'profiling.jsonl',[dict(task_id=r['task_id'],time=r['time'],elapsed_seconds=r.get('elapsed_seconds',0),
                    decision_seconds=r.get('decision_seconds',0)) for r in diag])
                times=[r.get('elapsed_seconds',0.) for r in diag if r.get('status')=='ok']
                predict=[r for r in diag if r.get('status')=='ok']
                gain_diff=sum((r['gain_it_kg']>0)!=(r['gain_total_kg']>0) for r in predict)
                it_carbon=math.fsum(s['it_energy_kwh']*s['ci_g_per_kwh']/1000 for s in result.steps)
                hvac_carbon=math.fsum(s['hvac_energy_kwh']*s['ci_g_per_kwh']/1000 for s in result.steps)
                row=dict(split=split,factor=factor,policy=name,**summary,it_carbon_kg=it_carbon,hvac_carbon_kg=hvac_carbon,
                    drain_hours=(executors[name].drained_at-bundle.workload.end).total_seconds()/3600,
                    common_end=common.isoformat(),wall_seconds=runtimes[name],
                    prediction_calls=sum(r.get('status') not in ('not_evaluated',None) for r in diag),
                    prediction_valid_calls=len(predict),
                    estimate_unavailable=sum(r.get('status') not in ('ok','not_evaluated',None) for r in diag),
                    forecast_cold_calls=sum(bool(r.get('forecast_cold',False)) for r in diag),
                    outside_fit_calls=sum(bool(r.get('outside_fit',False)) for r in diag),
                    it_total_sign_disagreements=gain_diff,mean_candidate_seconds=math.fsum(times)/len(times) if times else 0.,
                    max_candidate_seconds=max(times,default=0.))
                allrows.append(row);write_json(target/'runtime.json',dict(wall_seconds=runtimes[name],prediction_count=len(times)))
                quality=evaluate_surrogate(model,result.steps);write_json(target/'surrogate-errors.json',quality['summary'])
            # Pair changes describe different policy trajectories, not single-action causal effects.
            base={t['task_id']:t for t in results['edf_backfill'].tasks};pairs=[]
            for name,result in results.items():
                for t in result.tasks:
                    b=base[t['task_id']];pairs.append(dict(policy=name,task_id=t['task_id'],edf_start=b['start_time'],
                        start=t['start_time'],shift_minutes=(pd.Timestamp(t['start_time'])-pd.Timestamp(b['start_time'])).total_seconds()/60,
                        edf_outcome=b['deadline_outcome'],outcome=t['deadline_outcome']))
            write_csv(folder/'task-start-comparison.csv',pairs)
        write_csv(out/'comparison.csv',allrows);write_json(out/'historical-regression.json',dict(passed=True,byte_exact_files=regressions))
        _provenance(out,config,allids,sources)
        return allrows
    return _fresh_run(outpath,action)

def replay_matrix(previous,outpath):
    previous=Path(previous);prov=json.loads((previous/'provenance.json').read_text())
    if json.loads((previous/'run_status.json').read_text())['state']!='complete':raise ValueError('Incomplete source run')
    if prov['sources']!=_sources() or prov['python']!=sys.version or prov['dependencies']!=dependencies():
        raise ValueError('Frozen source/runtime mismatch')
    for name,sha in prov['output_hashes'].items():
        if file_hash(previous/name)!=sha:raise ValueError('Scientific output tampered '+name)
    cfg=json.loads((previous/'config.json').read_text());split=prov['inputs'][0]['split']
    rows=run_matrix(cfg,previous/'freeze.json',split,outpath)
    new=Path(outpath)
    # Runtime-bearing comparison is excluded from byte-exact science comparison.
    same={name:file_hash(new/name)==sha for name,sha in prov['output_hashes'].items() if name!='comparison.csv'}
    if not all(same.values()):raise RuntimeError('Replay scientific mismatch')
    a=pd.read_csv(previous/'comparison.csv');b=pd.read_csv(new/'comparison.csv')
    cols=[c for c in a.columns if c not in ('wall_seconds','mean_candidate_seconds','max_candidate_seconds')]
    if not a[cols].equals(b[cols]):raise RuntimeError('Replay summary numeric mismatch')
    write_json(new/'replay-receipt.json',dict(passed=True,byte_exact_files=len(same),runtime_fields_excluded=True))
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('calibrate','evaluate','replay'))
    parser.add_argument('--config',type=Path,default=CONFIG);parser.add_argument('--freeze',type=Path)
    parser.add_argument('--split',choices=('train','validation'));parser.add_argument('--previous',type=Path)
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    config=json.loads(a.config.read_text())
    if a.command=='calibrate':calibrate(config,a.output)
    elif a.command=='evaluate':run_matrix(config,a.freeze,a.split,a.output)
    else:replay_matrix(a.previous,a.output)

if __name__=='__main__':main()
