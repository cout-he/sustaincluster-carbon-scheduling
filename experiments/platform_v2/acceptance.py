"""FINAL-P: freeze then execute exactly one 36-case Train integration batch."""
import argparse
from datetime import datetime,timezone
import itertools
import json
import math
from pathlib import Path
import subprocess
import time
from experiments.platform_v1.acceptance import csv_write,write_json,claim_once,FRAGMENTS
from experiments.platform_v1.data import load_train,file_sha,canonical_sha
from .inventory import load_config,build_inventory
from .environment import run
from .verification import verify_saved

DOC=Path('docs/research/platform-v2-final');PROTOCOL=DOC/'acceptance-protocol.json'
GUARD=Path('research_notes/platform-v2/FINAL-acceptance-once.json')
CAPACITIES=('OM','OH','OL');POLICIES=('edf','threshold_edf');COOLING=('fixed','variable')


def case_matrix():return tuple(itertools.product(FRAGMENTS,CAPACITIES,POLICIES,COOLING))


def attach_background(row,common,matched):
    for prefix,idle in [('common',common),('matched',matched)]:
        row[prefix+'_idle_carbon_kg']=idle['carbon_kg'];row[prefix+'_idle_total_kwh']=idle['total_kwh']
        row[prefix+'_incremental_carbon_kg']=row['carbon_kg']-idle['carbon_kg']
        row[prefix+'_incremental_kwh']=row['total_kwh']-idle['total_kwh']


def source_hashes():
    paths=[]
    for directory in ['experiments/platform_v1','experiments/platform_v2','tests/platform_v2']:
        paths.extend(p for p in Path(directory).rglob('*') if p.suffix in ['.py','.json'])
    return {p.as_posix():file_sha(p) for p in sorted(paths)}


def freeze(pool):
    if PROTOCOL.exists():raise FileExistsError('Already frozen')
    cfg=load_config();inputs={}
    for identity in FRAGMENTS:
        scenario=load_train(Path(pool)/'scenarios'/identity,72)
        for name in CAPACITIES:
            cap=build_inventory(cfg,name).capacity
            if any(any(x>c for x,c in zip(t.demand,cap)) for t in scenario.tasks):raise ValueError('Intrinsic task overflow; never remove')
        inputs[identity]=dict(**scenario.provenance,tasks=len(scenario.tasks),task_identity_sha256=canonical_sha([(t.task_id,t.arrival_minute,t.duration_minutes,t.demand) for t in scenario.tasks]))
    plan=dict(schema='FINAL-P-single-batch-v2',frozen_utc=datetime.now(timezone.utc).isoformat(),
              cases=[list(x) for x in case_matrix()],workload_cases=36,idle_cases=18,sla=1.5,
              arrival_hours=24,responsibility_hours=72,terminal_common_hours=6,
              source_hashes=source_hashes(),inputs=inputs,
              common_background='fixed20+same safety filter, same fragment and installed capacity, full72h; matched-controller background additional only',
              criteria=dict(max_heat_residual_kj=1e-5,max_integral_error=1e-7,max_inlet_c=27+1e-8,hard_inlet_c=28.,
                            max_terminal_pair_k=.05,max_it_pair_kwh=1e-7,
                            require_complete_tasks=True,require_no_resource_overflow=True,require_feasible_controls=True,
                            no_required_carbon_improvement=True,no_required_oh_dmr=True,no_joint_superiority_claim=True),
              prohibitions=['no second integration invocation','no post-result tuning','no forbidden data','no learning','no task/deadline mutation'])
    write_json(PROTOCOL,plan);print(dict(protocol_sha256=file_sha(PROTOCOL),tasks={f:x['tasks'] for f,x in inputs.items()},cases=36,idle=18))


def execute(pool,output):
    plan=json.loads(PROTOCOL.read_text(encoding='utf-8'));cfg=load_config()
    if plan['cases']!=[list(x) for x in case_matrix()] or plan['source_hashes']!=source_hashes():raise ValueError('Frozen source/matrix changed')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Commit before acceptance')
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();start=time.monotonic();output=Path(output)
    claim_once(GUARD,dict(status='running',source_sha=head,protocol_sha256=file_sha(PROTOCOL),output=str(output)))
    if output.exists():raise FileExistsError('Output exists, once remains consumed')
    output.mkdir(parents=True);scenarios={};summaries=[];idles=[];errors=[]
    try:
        for f in FRAGMENTS:scenarios[f]=load_train(Path(pool)/'scenarios'/f,72,plan['inputs'][f])
        for f,c,hvac in itertools.product(FRAGMENTS,CAPACITIES,COOLING):
            result=run(scenarios[f],build_inventory(cfg,c),cfg,'edf',hvac,1.5,idle=True)
            idles.append(dict(fragment=f,capacity=c,cooling=hvac,**result['summary']))
            csv_write(output/'local-detail'/f'idle__{f}__{c}__{hvac}-steps.csv',result['steps'])
        idle_index={(r['fragment'],r['capacity'],r['cooling']):r for r in idles}
        for n,(f,c,policy,hvac) in enumerate(case_matrix(),1):
            identity=dict(case_id=f'{f}__{c}__{policy}__{hvac}',fragment=f,capacity=c,sla=1.5,policy=policy,cooling=hvac,
                          source_workload_sha256=scenarios[f].provenance['workload_sha256'])
            try:
                result=run(scenarios[f],build_inventory(cfg,c),cfg,policy,hvac,1.5)
                summary=result['summary'];attach_background(summary,idle_index[f,c,'fixed'],idle_index[f,c,hvac])
                summary['schedule_sha256']=canonical_sha([(t['task_id'],t['start_minute'],t['finish_minute']) for t in result['tasks']])
                summary['resource_overflow_steps']=sum(any(s['used_'+k]>v+1e-9 for k,v in zip(['cpu','gpu','mem'],build_inventory(cfg,c).capacity)) for s in result['steps'])
                summaries.append(dict(**identity,status='completed',**summary))
                csv_write(output/'local-detail'/(identity['case_id']+'-tasks.csv'),result['tasks'])
                csv_write(output/'local-detail'/(identity['case_id']+'-steps.csv'),result['steps'])
            except Exception as exc:
                errors.append(dict(case_id=identity['case_id'],error=repr(exc)));summaries.append(dict(**identity,status='failed'))
            if n%6==0:print(f'FINAL-P {n}/36',flush=True)
    except Exception as exc:errors.append(dict(batch_error=repr(exc)))
    csv_write(output/'case-summary.csv',summaries);csv_write(output/'idle-background.csv',idles)
    valid=[r for r in summaries if r.get('status')=='completed'];index={(r['fragment'],r['capacity'],r['policy'],r['cooling']):r for r in valid}
    deferrals=[];cooling=[]
    fields=['dmr','capacity_wait_minutes','policy_wait_minutes','mean_jct_minutes','queue_peak','competition_minutes','start_gpu_peak','it_kwh','hvac_kwh','total_kwh','carbon_kg','common_incremental_carbon_kg','thermal_storage_kwh','inlet_final_c']
    for f,c,h in itertools.product(FRAGMENTS,CAPACITIES,COOLING):
        a=index.get((f,c,'edf',h));b=index.get((f,c,'threshold_edf',h))
        if not a or not b:continue
        row=dict(fragment=f,capacity=c,cooling=h)
        for k in fields:row['immediate_'+k]=a[k];row['threshold_'+k]=b[k];row['delta_'+k]=b[k]-a[k]
        deferrals.append(row)
    for f,c,p in itertools.product(FRAGMENTS,CAPACITIES,POLICIES):
        a=index.get((f,c,p,'fixed'));b=index.get((f,c,p,'variable'))
        if not a or not b:continue
        row=dict(fragment=f,capacity=c,policy=p,same_task_schedule=a['schedule_sha256']==b['schedule_sha256'])
        for k in fields:row['fixed_'+k]=a[k];row['variable_'+k]=b[k];row['delta_'+k]=b[k]-a[k]
        cooling.append(row)
    csv_write(output/'paired-deferral.csv',deferrals);csv_write(output/'paired-cooling.csv',cooling)
    # Separate reader reconstructs resource occupancy and integrates the saved
    # component powers, carbon and heat for workload AND idle traces.
    try:independent=verify_saved(output,scenarios,cfg,plan)
    except Exception as exc:independent=dict(accepted=False,errors=[dict(error=repr(exc))])
    write_json(output/'independent-verification.json',independent)
    lim=plan['criteria'];allrows=valid+idles
    checks=dict(matrix_complete=len(valid)==36 and len(idles)==18 and not errors,
                tasks_complete=len(valid)==36 and all(r['arrived']==plan['inputs'][r['fragment']]['tasks'] and r['unfinished']==0 for r in valid),
                resource_limits=bool(valid) and all(r['resource_overflow_steps']==0 for r in valid),
                heat_balance=bool(allrows) and all(max(r['max_balance_residual_kj'],r['max_chiller_balance_residual_kj'])<=lim['max_heat_residual_kj'] for r in allrows),
                independent_saved_accounting=independent['accepted'],
                carbon_integral=independent['accepted'] and bool(valid) and all(r['carbon_recalculation_error_kg']<=lim['max_integral_error'] for r in valid),
                thermal_safety=bool(allrows) and all(r['inlet_max_c']<=lim['max_inlet_c'] and r['cooling_infeasible_steps']==0 and r['performance_domain_invalid_steps']==0 for r in allrows),
                phase_accounting=bool(valid) and all(abs(r['arrival_total_kwh']+r['drain_total_kwh']-r['total_kwh'])<1e-7 and abs(r['arrival_carbon_kg']+r['drain_carbon_kg']-r['carbon_kg'])<1e-7 for r in valid),
                same_cooling_schedules=len(cooling)==18 and all(r['same_task_schedule'] for r in cooling),
                conserved_it_energy=len(cooling)==18 and len(deferrals)==18 and all(abs(r['delta_it_kwh'])<=lim['max_it_pair_kwh'] for r in cooling+deferrals),
                terminal_fairness=len(cooling)==18 and len(deferrals)==18 and all(abs(r['delta_inlet_final_c'])<=lim['max_terminal_pair_k'] for r in cooling+deferrals),
                unchanged_source=plan['source_hashes']==source_hashes())
    metadata=dict(source_sha=head,source_hashes=source_hashes(),protocol_sha256=file_sha(PROTOCOL),integration_invocations=1,
                  elapsed_seconds=time.monotonic()-start,inputs=plan['inputs'],checks=checks,errors=errors,
                  independent_verification={k:v for k,v in independent.items() if k!='details'},accepted=all(checks.values()),
                  inventories={c:build_inventory(cfg,c).to_dict() for c in CAPACITIES})
    write_json(output/'execution.json',metadata);write_json(GUARD,dict(status='completed' if metadata['accepted'] else 'failed',accepted=metadata['accepted'],source_sha=head,protocol_sha256=file_sha(PROTOCOL),output=str(output)))
    print(json.dumps(dict(accepted=metadata['accepted'],checks=checks,errors=errors,elapsed_seconds=metadata['elapsed_seconds']),indent=2))
    return metadata


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pool',required=True);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze',action='store_true');g.add_argument('--execute-once',action='store_true');p.add_argument('--output',default='research_notes/platform-v2/acceptance-001');a=p.parse_args()
    if a.freeze:freeze(a.pool)
    elif not execute(a.pool,a.output)['accepted']:raise SystemExit(2)
