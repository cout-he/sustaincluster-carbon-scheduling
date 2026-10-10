"""Finite-cohort temporal shifting: genuine drain, common idle accounting, strict replay."""
import argparse
import copy
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import pandas as pd

from .config import ROOT, load_config, resolve_path, validate_config
from .contracts import content_hash, file_hash, write_json
from .drain_metrics import phase_rows, phase_summary
from .engine import LocalExecutor
from .policies import POLICY_NAMES, CarbonThresholdFIFO, make_policy
from .power import UpstreamPowerModel
from .run import _fresh_run, _git, _write_result, dependencies, source_hashes
from .selection import CI_BASIS, TemporalEnvironment, load_temporal_environment, select_window
from .workload import FrozenWorkload, load_workload

VERSION='e1-2a-finite-cohort-drain-v1'


def experiment_sources():
    files=source_hashes()
    files.update({str(p.relative_to(ROOT)).replace('\\','/'):file_hash(p)
                  for p in sorted((ROOT/'tests/e1').glob('*.py'))})
    return dict(sorted(files.items()))


def validate_temporal_config(cfg,workload,environment):
    cfg=copy.deepcopy(cfg)
    if cfg['environment']['interpolation'] not in ('linear_utc',CI_BASIS):
        raise ValueError('Unsupported CI basis')
    cfg['environment']['interpolation']='linear_utc'
    cfg=validate_config(cfg);cfg['environment']['interpolation']=CI_BASIS
    if cfg['policy']!='immediate_fifo': raise ValueError('Temporal reference config must be FIFO')
    if (pd.Timestamp(cfg['start']),pd.Timestamp(cfg['end']))!=(workload.start,workload.end):
        raise ValueError('Arrival config/workload window mismatch')
    if environment.start!=workload.start or environment.end<workload.end:
        raise ValueError('Drain environment coverage excludes arrival window')
    if (workload.resource_multiplier,workload.sla_multiplier,workload.trace_year)!=(
            cfg['workload']['resource_multiplier'],cfg['workload']['sla_multiplier'],cfg['workload']['trace_year']):
        raise ValueError('Frozen workload/config mismatch')
    if any(t.sla_multiplier!=workload.sla_multiplier or t.dc_id!=1 for t in workload.tasks):
        raise ValueError('Frozen task/config SLA or DC mismatch')
    return cfg


def paired_shifts(results,environment):
    ci={s.time:s.ci_g_per_kwh for s in environment.samples}
    base={t['task_id']:t for t in results['immediate_fifo'].tasks}
    rows=[]
    for policy,result in results.items():
        for t in result.tasks:
            b=base[t['task_id']]
            bs=pd.Timestamp(b['start_time']) if b['start_time'] else None
            ps=pd.Timestamp(t['start_time']) if t['start_time'] else None
            rows.append(dict(policy=policy,task_id=t['task_id'],fifo_start=b['start_time'],
                policy_start=t['start_time'],fifo_start_ci=ci.get(bs),policy_start_ci=ci.get(ps),
                shift_minutes=(ps-bs).total_seconds()/60 if bs is not None and ps is not None else None,
                state=t['state'],deadline_outcome=t['deadline_outcome']))
    return rows


def write_csv(path,rows):
    if not rows: raise ValueError('Cannot write an empty comparison')
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def _execute(out,cfg,workload,environment,model_path,threshold,selection=None):
    cfg=validate_temporal_config(cfg,workload,environment)
    threshold=CarbonThresholdFIFO(threshold).threshold_g_per_kwh
    if not isinstance(environment,TemporalEnvironment):
        raise ValueError('E1-2A requires an explicit causal TemporalEnvironment; legacy CI cannot be relabeled')
    hashes=experiment_sources()
    for name,sha in hashes.items():
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
        if file_hash(target)!=sha: raise RuntimeError('Source changed during snapshot')
    write_json(out/'resolved_config.json',cfg)
    write_json(out/'design.json',dict(version=VERSION,arrival_end=workload.end.isoformat(),
        safety_end=environment.end.isoformat(),threshold_g_per_kwh=threshold,ci_basis=CI_BASIS,
        common_end_rule='maximum observed grid drain boundary, or safety cap if any incomplete',
        idle_rule='same persistent physical model, zero CPU/GPU/MEM, real corresponding weather/CI'))
    write_json(out/'selection.json',selection)
    workload.save(out/'workload.json');environment.save(out/'environment.json')
    shutil.copyfile(model_path,out/'dc_config.json')
    dc=cfg['datacenter'];capacity=(dc['total_cores'],dc['total_gpus'],dc['total_mem'])
    executors,own,configs={},{},{}
    for policy in POLICY_NAMES:
        p=copy.deepcopy(cfg);p['policy']=policy;p.pop('policy_params',None)
        if policy=='carbon_threshold_fifo': p['policy_params']={'threshold_g_per_kwh':threshold}
        configs[policy]=p
        ex=LocalExecutor(workload,environment,capacity,UpstreamPowerModel(dc,out/'dc_config.json'),
                         policy=make_policy(p),drain=True)
        executors[policy]=ex;own[policy]=ex.run(stop_at_drain=True)
    common=max(ex.observed_end for ex in executors.values())
    results={p:(ex.extend_idle(common) if ex.drained_at is not None else own[p]) for p,ex in executors.items()}
    phases,comparison,series={},[],[]
    for policy,result in results.items():
        folder=out/policy;folder.mkdir()
        _write_result(folder,result);write_json(folder/'resolved_config.json',configs[policy])
        ex=executors[policy]
        phases[policy]=dict(fixed=phase_summary(result,workload.end),
            own=phase_summary(own[policy],pd.Timestamp(own[policy].summary['observation_end'])),
            common=phase_summary(result,common))
        for name,summary in phases[policy].items():
            comparison.append(dict(policy=policy,phase=name,drained_at=ex.drained_at.isoformat() if ex.drained_at else None,
                drain_hours=(ex.drained_at-workload.end).total_seconds()/3600 if ex.drained_at else None,
                termination='all_completed' if ex.drained_at else 'safety_cap_incomplete',**summary))
        write_json(folder/'phases.json',phases[policy])
        # Store fixed-cutoff states without future starts/completions masquerading as observed.
        write_json(folder/'fixed_ledger.json',phase_rows(result,workload.end))
        write_json(folder/'run_status.json',dict(state='complete'))
        cumulative=0.0
        for step in result.steps:
            cumulative=math.fsum((cumulative,step['carbon_kg']))
            series.append(dict(policy=policy,**step,cumulative_carbon_kg=cumulative,
                               phase='arrival' if pd.Timestamp(step['interval_start'])<workload.end else 'drain'))
    completed=all(phases[p]['common']['completed']==len(workload.tasks) for p in POLICY_NAMES)
    resource_hours={r:[phases[p]['common'][r+'_hours'] for p in POLICY_NAMES] for r in ('cores','gpus','mem')}
    conserved=all(all(math.isclose(v,values[0],rel_tol=1e-12,abs_tol=1e-8) for v in values)
                  for values in resource_hours.values())
    if completed and not conserved: raise RuntimeError('Completed cohort resource-hours differ')
    task_features=[{k:t[k] for k in workload.tasks[0].to_dict()} for t in results['immediate_fifo'].tasks] if workload.tasks else []
    for p,result in results.items():
        features=[{k:t[k] for k in workload.tasks[0].to_dict()} for t in result.tasks] if workload.tasks else []
        if features!=task_features: raise RuntimeError('Policy altered frozen task features')
    write_json(out/'fairness.json',dict(version=VERSION,identical_frozen_inputs=True,
        identical_shared_config=True,all_tasks_completed=completed,equal_resource_hours=conserved,
        equal_workload_carbon_comparison_valid=completed and conserved,resource_hours=resource_hours,
        arrived=len(workload.tasks),arrival_end=workload.end.isoformat(),common_end=common.isoformat(),
        safety_end=environment.end.isoformat(),threshold_g_per_kwh=threshold,
        threshold_source='selection calibration' if selection else 'explicit fixture literal',
        deadline_denominator='same finite arrived cohort; censored tasks retained',
        idle_included=True,policy_future_ci_access=False))
    write_csv(out/'comparison.csv',comparison);write_csv(out/'timeseries.csv',series)
    write_csv(out/'paired_task_shifts.csv',paired_shifts(results,environment))
    if experiment_sources()!=hashes: raise RuntimeError('Source changed during execution')
    scientific={str(p.relative_to(out)).replace('\\','/'):file_hash(p) for p in out.rglob('*') if p.is_file()
        and 'source' not in p.relative_to(out).parts and p.name not in ('run_status.json','provenance.json')}
    write_json(out/'provenance.json',dict(version=VERSION,git=dict(head=_git('rev-parse','HEAD'),
        branch=_git('branch','--show-current'),status=_git('status','--porcelain')),
        command=sys.argv,python=sys.version,dependencies=dependencies(),sources=hashes,source_sha256=content_hash(hashes),
        input_hashes=dict(workload_snapshot=workload.sha256,environment_snapshot=environment.sha256,
            workload_source=workload.source_sha256,carbon_source=environment.carbon_source_sha256,
            weather_source=environment.weather_source_sha256,dc_config=file_hash(out/'dc_config.json')),
        output_hashes=scientific))
    return phases


def run_comparison(cfg,workload,environment,model_path,output_dir,threshold_g_per_kwh,selection=None):
    return _fresh_run(output_dir,lambda out:_execute(out,cfg,workload,environment,model_path,
                                                   threshold_g_per_kwh,selection))


def replay_comparison(previous_dir,output_dir):
    previous=Path(previous_dir)
    def action(out):
        if json.loads((previous/'run_status.json').read_text())['state']!='complete':
            raise ValueError('Cannot replay incomplete temporal comparison')
        prov=json.loads((previous/'provenance.json').read_text())
        if prov['version']!=VERSION or prov['sources']!=experiment_sources() or content_hash(prov['sources'])!=prov['source_sha256']:
            raise ValueError('Temporal replay source/version mismatch')
        if sys.version!=prov['python'] or dependencies()!=prov['dependencies']:
            raise ValueError('Temporal replay Python/dependencies mismatch')
        required={'resolved_config.json','design.json','selection.json','workload.json','environment.json','dc_config.json',
                  'comparison.csv','timeseries.csv','paired_task_shifts.csv','fairness.json'}
        required.update(f'{p}/{name}' for p in POLICY_NAMES for name in (
            'ledger.jsonl','events.jsonl','intervals.csv','summary.json','policy_ledger.jsonl','policy_events.jsonl',
            'policy_intervals.jsonl','policy_summary.json','resolved_config.json','phases.json','fixed_ledger.json'))
        if set(prov['output_hashes'])!=required: raise ValueError('Temporal output manifest incomplete')
        for name,sha in prov['sources'].items():
            if file_hash(previous/'source'/name)!=sha: raise ValueError('Temporal archived source hash mismatch')
        for name,sha in prov['output_hashes'].items():
            if file_hash(previous/name)!=sha: raise ValueError('Temporal output hash mismatch')
        cfg=json.loads((previous/'resolved_config.json').read_text())
        design=json.loads((previous/'design.json').read_text())
        workload=FrozenWorkload.load(previous/'workload.json');env=TemporalEnvironment.load(previous/'environment.json')
        if dict(workload_snapshot=workload.sha256,environment_snapshot=env.sha256,workload_source=workload.source_sha256,
                carbon_source=env.carbon_source_sha256,weather_source=env.weather_source_sha256,
                dc_config=file_hash(previous/'dc_config.json'))!=prov['input_hashes']:
            raise ValueError('Temporal input manifest hash mismatch')
        result=_execute(out,cfg,workload,env,previous/'dc_config.json',design['threshold_g_per_kwh'],
                        json.loads((previous/'selection.json').read_text()))
        for name,sha in prov['output_hashes'].items():
            if file_hash(out/name)!=sha: raise RuntimeError(f'Temporal replay mismatch: {name}')
        return result
    return _fresh_run(output_dir,action)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-config',type=Path,default=ROOT/'experiments/e1/configs/e1_2a_selection.json')
    parser.add_argument('--config',type=Path,default=ROOT/'experiments/e1/configs/single_dc.yaml')
    parser.add_argument('--replay',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.replay:
        result=replay_comparison(args.replay,args.output)
    else:
        def action(out):
            cfg=load_config(args.config);rule=json.loads(args.selection_config.read_text(encoding='utf-8-sig'))
            selection=select_window(cfg,rule)
            chosen=selection['selected'];cfg['start'],cfg['end']=chosen['start'],chosen['end']
            start,end=pd.Timestamp(cfg['start']),pd.Timestamp(cfg['end'])
            # Literal predeclared safety cap, not adapted to policy performance.
            safety=end+pd.Timedelta(days=30)
            w=cfg['workload'];dc=cfg['datacenter'];e=cfg['environment']
            workload=load_workload(resolve_path(w['path']),start,end,w['resource_multiplier'],w['sla_multiplier'],1,w['trace_year'])
            env=load_temporal_environment(resolve_path(e['carbon_path']),resolve_path(e['weather_path']),start,safety)
            selection['workload_count']=len(workload.tasks)
            return _execute(out,cfg,workload,env,resolve_path(dc['dc_config_file']),selection['threshold_g_per_kwh'],selection)
        result=_fresh_run(args.output,action)
    print(json.dumps({p:phases['common'] for p,phases in result.items()},indent=2))
    print(f'Artifacts: {args.output.resolve()}')


if __name__=='__main__': main()
