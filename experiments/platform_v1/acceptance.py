"""Freeze first, run one immutable acceptance matrix once, publish aggregates only."""
import argparse
import csv
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import subprocess
import time

from .data import canonical_sha, file_sha, load_train
from .environment import run
from .inventory import build_inventory, load_config

FRAGMENTS=('train-20230501T0500-r1-s1.5','train-20230302T0500-r1-s1.5','train-20230404T0500-r1-s1.5')
CAPACITIES=('OM','OH','OL')
POLICIES=('fifo','edf','threshold_fifo','threshold_edf')
COOLING=('fixed','variable')
DEFAULT_PROTOCOL=Path('docs/research/platform-v1/acceptance-protocol.json')
GLOBAL_GUARD=Path('research_notes/platform-v1/P1-acceptance-once.json')


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def csv_write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def case_matrix():return tuple(itertools.product(FRAGMENTS,CAPACITIES,(1.5,2.0),POLICIES,COOLING))


def claim_once(path,metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(metadata,f,indent=2)


def terminal_pairs_ok(deferrals,cooling_pairs,limit):
    return all(abs(r['delta_inlet_final_c'])<=limit for r in [*deferrals,*cooling_pairs])


def freeze(pool,protocol=DEFAULT_PROTOCOL):
    protocol=Path(protocol)
    if protocol.exists():raise FileExistsError('Protocol already frozen')
    inputs={}
    for identity in FRAGMENTS:
        root=Path(pool)/'scenarios'/identity
        m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        if m['role']!='train' or m['test_access'] is not False:raise ValueError('Only Train inputs may be frozen')
        inputs[identity]=dict(manifest_sha256=file_sha(root/'manifest.json'))
        for name in ['workload','environment']:
            sha=file_sha(root/(name+'.json'))
            if sha!=m['files'][name+'.json']:raise ValueError('Input file hash differs')
            inputs[identity][name+'_sha256']=sha
    plan=dict(schema='P1-one-batch-acceptance-v1',frozen_utc=datetime.now(timezone.utc).isoformat(),
              config_sha256=file_sha(Path(__file__).with_name('config.json')),
              fragments=inputs,cases=[list(x) for x in case_matrix()],workload_cases=144,idle_background_cases=18,
              responsibility_hours=72,arrival_hours=24,terminal_common_hours=6,
              threshold='median of prior observed <=24 h CI after 4 h warmup; no future values',
              acceptance=dict(task_completion=1.,max_step_heat_residual_kj=1e-5,max_carbon_error_kg=1e-7,
                              max_it_pair_difference_kwh=1e-7,max_terminal_pair_difference_k=.05,
                              max_filtered_inlet_c=27.+1e-8,hard_inlet_limit_c=28.,
                              rq1='capacity contention, policy deferral and changed same-sort burst/capacity-wait/DMR in at least one pair; no benefit requirement',
                              rq2='safe balanced load/thermal response and changed HVAC energy on identical task schedules; no benefit requirement'),
              prohibitions=['no data mutation or deduplication','no raw CSV investigation','no Development/Validation/Test payload','no training','no second integration batch or post-outcome parameter tuning'])
    write_json(protocol,plan)
    print(json.dumps(dict(frozen_protocol=str(protocol),sha256=file_sha(protocol),workload_cases=144,idle_cases=18)))


def source_hashes():
    return {str(p).replace('\\','/'):file_sha(p) for directory in ['experiments/platform_v1','tests/platform_v1']
            for p in sorted(Path(directory).rglob('*')) if p.suffix in ['.py','.json']}


def execute(pool,output,protocol=DEFAULT_PROTOCOL):
    protocol=Path(protocol);plan=json.loads(protocol.read_text(encoding='utf-8'));cfg=load_config()
    if plan['cases']!=[list(x) for x in case_matrix()] or file_sha(Path(__file__).with_name('config.json'))!=plan['config_sha256']:
        raise ValueError('Frozen matrix/config changed')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Commit and verify code/protocol before integration')
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    start=time.monotonic();output=Path(output)
    claim_once(GLOBAL_GUARD,dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),source_sha=head,protocol_sha256=file_sha(protocol),output=str(output)))
    if output.exists():raise FileExistsError('Refuse output overwrite; once marker remains consumed')
    output.mkdir(parents=True)
    metadata=dict(source_sha=head,protocol_sha256=file_sha(protocol),source_hashes=source_hashes(),
                  started_utc=datetime.now(timezone.utc).isoformat(),integration_invocations=1,expected_workload_cases=144,expected_idle_cases=18)
    summaries=[];idle_rows=[];errors=[];scenarios={}
    try:
        for identity in FRAGMENTS:scenarios[identity]=load_train(Path(pool)/'scenarios'/identity,72,plan['fragments'][identity])
        for fragment,scenario in scenarios.items():
            for capacity in CAPACITIES:
                inventory=build_inventory(cfg,capacity)
                for cooling in COOLING:
                    result=run(scenario,inventory,cfg,'fifo',cooling,1.5,idle=True)
                    idle_rows.append(dict(fragment=fragment,capacity=capacity,cooling=cooling,**result['summary']))
        idle_index={(r['fragment'],r['capacity'],r['cooling']):r for r in idle_rows}
        for number,(fragment,capacity,sla,policy,cooling) in enumerate(case_matrix(),1):
            if time.monotonic()-start>600:raise TimeoutError('Single acceptance batch exceeded 600 s; no retry authorized')
            case_id=f'{fragment}__{capacity}__s{sla}__{policy}__{cooling}'
            identity=dict(case_id=case_id,fragment=fragment,capacity=capacity,sla=sla,policy=policy,cooling=cooling,
                          source_workload_sha256=scenarios[fragment].provenance['workload_sha256'])
            try:
                result=run(scenarios[fragment],build_inventory(cfg,capacity),cfg,policy,cooling,sla)
                summary=result['summary'];idle=idle_index[fragment,capacity,cooling]
                summary.update(idle_total_kwh=idle['total_kwh'],idle_carbon_kg=idle['carbon_kg'],
                               incremental_kwh=summary['total_kwh']-idle['total_kwh'],
                               incremental_carbon_kg=summary['carbon_kg']-idle['carbon_kg'])
                schedule=[(r['task_id'],r['start_minute'],r['finish_minute']) for r in result['tasks']]
                summary['schedule_sha256']=canonical_sha(schedule)
                row=dict(identity,status='completed',**summary)
                summaries.append(row)
                csv_write(output/'local-detail'/f'{case_id}-tasks.csv',result['tasks'])
                csv_write(output/'local-detail'/f'{case_id}-steps.csv',result['steps'])
                if any(not math.isfinite(v) for v in summary.values() if isinstance(v,(int,float))):raise ValueError('Nonfinite metric')
                if any(s['used_'+n]>c+1e-9 for s in result['steps'] for n,c in zip(['cpu','gpu','mem'],build_inventory(cfg,capacity).capacity)):
                    raise ValueError('Resource overflow')
            except Exception as exc:
                errors.append(dict(case_id=case_id,error=repr(exc)))
                if not summaries or summaries[-1]['case_id']!=case_id:summaries.append(dict(identity,status='failed'))
            if number%12==0:print(f'Acceptance progress {number}/144',flush=True)
    except Exception as exc:errors.append(dict(batch_error=repr(exc)))
    csv_write(output/'case-summary.csv',summaries);csv_write(output/'idle-background.csv',idle_rows)
    valid=[r for r in summaries if r.get('status')=='completed']
    index={(r['fragment'],r['capacity'],r['sla'],r['policy'],r['cooling']):r for r in valid}
    deferrals=[];cooling_pairs=[]
    for f,c,sla,sort,hvac in itertools.product(FRAGMENTS,CAPACITIES,(1.5,2.),('fifo','edf'),COOLING):
        a=index.get((f,c,sla,sort,hvac));b=index.get((f,c,sla,'threshold_'+sort,hvac))
        if a is None or b is None:continue
        row=dict(fragment=f,capacity=c,sla=sla,sort=sort,cooling=hvac)
        for field in ['dmr','capacity_wait_minutes','capacity_wait_tasks','policy_wait_minutes','mean_jct_minutes','queue_peak','start_cpu_peak','start_gpu_peak','competition_minutes','it_kwh','hvac_kwh','total_kwh','carbon_kg','incremental_carbon_kg','inlet_final_c','thermal_storage_kwh']:
            row['immediate_'+field]=a[field];row['threshold_'+field]=b[field];row['delta_'+field]=b[field]-a[field]
        row['terminal_debt_compressor_equivalent_kwh']=abs(row['delta_thermal_storage_kwh'])/cfg['thermal']['cop_min']
        deferrals.append(row)
    for f,c,sla,policy in itertools.product(FRAGMENTS,CAPACITIES,(1.5,2.),POLICIES):
        a=index.get((f,c,sla,policy,'fixed'));b=index.get((f,c,sla,policy,'variable'))
        if a is None or b is None:continue
        row=dict(fragment=f,capacity=c,sla=sla,policy=policy,same_task_schedule=a['schedule_sha256']==b['schedule_sha256'])
        for field in ['it_kwh','hvac_kwh','carbon_kg','inlet_final_c','thermal_storage_kwh','compressor_kwh','safety_overrides']:
            row['fixed_'+field]=a[field];row['variable_'+field]=b[field];row['delta_'+field]=b[field]-a[field]
        row['terminal_debt_compressor_equivalent_kwh']=abs(row['delta_thermal_storage_kwh'])/cfg['thermal']['cop_min']
        cooling_pairs.append(row)
    csv_write(output/'paired-deferral.csv',deferrals);csv_write(output/'paired-cooling.csv',cooling_pairs)
    limits=plan['acceptance']
    checks=dict(matrix_complete=len(valid)==144 and len(idle_rows)==18 and not errors,
                tasks_complete=bool(valid) and all(r['arrived']==len(scenarios[r['fragment']].tasks) and r['unfinished']==0 for r in valid),
                heat_balance=bool(valid) and all(r['max_balance_residual_kj']<=limits['max_step_heat_residual_kj'] for r in valid+idle_rows),
                carbon_integral=bool(valid) and all(r['carbon_recalculation_error_kg']<=limits['max_carbon_error_kg'] for r in valid),
                thermal_safety=bool(valid) and all(r['inlet_max_c']<=limits['max_filtered_inlet_c'] for r in valid+idle_rows),
                identical_cooling_schedules=len(cooling_pairs)==72 and all(r['same_task_schedule'] for r in cooling_pairs),
                terminal_pairs=len(cooling_pairs)==72 and len(deferrals)==72 and terminal_pairs_ok(deferrals,cooling_pairs,limits['max_terminal_pair_difference_k']),
                it_energy_conservation=len(deferrals)==72 and all(abs(r['delta_it_kwh'])<=limits['max_it_pair_difference_kwh'] for r in deferrals) and all(abs(r['delta_it_kwh'])<=limits['max_it_pair_difference_kwh'] for r in cooling_pairs),
                rq1_contention=any(r['capacity_wait_minutes']>0 for r in valid),
                rq1_active_deferral=any(r['policy_wait_minutes']>0 for r in valid),
                rq1_changed_risk_or_concentration=any(r['delta_capacity_wait_minutes']>0 or r['delta_dmr']>0 or r['delta_start_gpu_peak']>0 or r['delta_start_cpu_peak']>0 for r in deferrals),
                rq2_control_effect=any(abs(r['delta_hvac_kwh'])>1e-6 for r in cooling_pairs))
    metadata.update(finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,
                    completed_workload_cases=len(valid),completed_idle_cases=len(idle_rows),errors=errors,checks=checks,
                    accepted=all(checks.values()),source_inputs={f:s.provenance for f,s in scenarios.items()},
                    inventories={name:build_inventory(cfg,name).to_dict() for name in [*CAPACITIES,'OM-legacy']})
    write_json(output/'execution.json',metadata)
    write_json(GLOBAL_GUARD,dict(status='completed' if metadata['accepted'] else 'failed',source_sha=head,protocol_sha256=file_sha(protocol),output=str(output),accepted=metadata['accepted']))
    print(json.dumps(dict(accepted=metadata['accepted'],checks=checks,errors=errors,elapsed_seconds=metadata['elapsed_seconds']),indent=2))
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool',required=True);p.add_argument('--protocol',default=str(DEFAULT_PROTOCOL))
    p.add_argument('--freeze',action='store_true');p.add_argument('--execute-once',action='store_true')
    p.add_argument('--output',default='research_notes/platform-v1/acceptance-001')
    args=p.parse_args()
    if args.freeze==args.execute_once:p.error('Select exactly one of --freeze or --execute-once')
    if args.freeze:freeze(args.pool,args.protocol)
    else:
        metadata=execute(args.pool,args.output,args.protocol)
        if not metadata['accepted']:raise SystemExit(2)


if __name__=='__main__':main()
