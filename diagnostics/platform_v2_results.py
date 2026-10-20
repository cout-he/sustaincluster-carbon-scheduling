"""Read saved FINAL-P results only; no simulator, policies or data loader.

Derived tables describe the single batch, not independent statistical samples.
Local task/step details and downloaded references are never copied to Git.
"""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import shutil
from experiments.platform_v1.acceptance import csv_write,write_json
from experiments.platform_v1.data import file_sha


def read(path):
    with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def main(output,public):
    output=Path(output);public=Path(public);public.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((output/'execution.json').read_text(encoding='utf-8'))
    audit=json.loads((output/'independent-verification.json').read_text(encoding='utf-8'))
    assert metadata['accepted'] and audit['accepted']
    rows=read(output/'case-summary.csv');idle=read(output/'idle-background.csv')
    def total(subset,key):return math.fsum(float(r[key]) for r in subset)
    pooled=[];phases=[];bottlenecks=[]
    for c,p,h in itertools.product(['OM','OH','OL'],['edf','threshold_edf'],['fixed','variable']):
        subset=[r for r in rows if (r['capacity'],r['policy'],r['cooling'])==(c,p,h)]
        assert len(subset)==3
        n=total(subset,'arrived')
        row=dict(capacity=c,policy=p,cooling=h,fragments=3,tasks=int(n),source_sha=metadata['source_sha'])
        for k in ['deadline_missed','capacity_wait_tasks','policy_wait_tasks','capacity_wait_minutes','policy_wait_minutes','competition_minutes',
                  'it_kwh','hvac_kwh','total_kwh','carbon_kg','common_idle_carbon_kg','common_incremental_carbon_kg',
                  'common_idle_total_kwh','common_incremental_kwh','safety_overrides','thermal_storage_kwh']:
            row[k]=total(subset,k)
        row.update(dmr=row['deadline_missed']/n,capacity_wait_fraction=row['capacity_wait_tasks']/n,
                   policy_wait_fraction=row['policy_wait_tasks']/n,
                   mean_jct_minutes=math.fsum(float(r['mean_jct_minutes'])*float(r['arrived']) for r in subset)/n,
                   queue_peak=max(float(r['queue_peak']) for r in subset),inlet_max_c=max(float(r['inlet_max_c']) for r in subset),
                   common_background_carbon_fraction=row['common_idle_carbon_kg']/row['carbon_kg'])
        pooled.append(row)
        for phase in ['arrival','drain']:
            part=dict(capacity=c,policy=p,cooling=h,period=phase,fragments=3,hours_per_fragment=24 if phase=='arrival' else 48,source_sha=metadata['source_sha'])
            for k in ['it_kwh','hvac_kwh','total_kwh','carbon_kg']:part[k]=total(subset,phase+'_'+k)
            for axis in ['cpu','gpu','mem']:
                part[axis+'_mean_util']=total(subset,phase+'_'+axis+'_mean_util')/3
                part[axis+'_peak_util']=max(float(r[phase+'_'+axis+'_peak_util']) for r in subset)
            phases.append(part)
        if h=='fixed':
            bottlenecks.append(dict(capacity=c,policy=p,tasks=int(n),
                                   **{k:total(subset,k) for k in ['blocked_cpu','blocked_gpu','blocked_mem','competition_minutes','capacity_wait_tasks','capacity_wait_minutes']},source_sha=metadata['source_sha']))
    csv_write(public/'pooled-summary.csv',pooled);csv_write(public/'phase-summary.csv',phases);csv_write(public/'resource-bottlenecks.csv',bottlenecks)
    v1path=Path('docs/research/platform-v1/evidence/case-summary.csv');v1=read(v1path)
    old={(r['fragment'],r['capacity'],r['policy'],r['cooling']):r for r in v1 if float(r['sla'])==1.5 and r['policy'] in ['edf','threshold_edf']}
    differences=[];components=['cpu','gpu','dimm','host_other','psu_loss','rack_switch','supply_fan','pump','compressor','condenser_fan']
    for r in rows:
        previous=old[r['fragment'],r['capacity'],r['policy'],r['cooling']]
        assert r['source_workload_sha256']==previous['source_workload_sha256'] and r['schedule_sha256']==previous['schedule_sha256']
        delta=dict(case_id=r['case_id'],fragment=r['fragment'],capacity=r['capacity'],policy=r['policy'],cooling=r['cooling'],
                   same_task_schedule=True,source_sha=metadata['source_sha'],v1_summary_sha256=file_sha(v1path))
        for k in components+['it','hvac','total']:
            delta['v1_'+k+'_kwh']=float(previous[k+'_kwh']);delta['v2_'+k+'_kwh']=float(r[k+'_kwh'])
            delta['delta_'+k+'_kwh']=float(r[k+'_kwh'])-float(previous[k+'_kwh'])
        for k in ['carbon_kg','thermal_storage_kwh','envelope_kwh','coil_kwh','inlet_final_c']:
            delta['v1_'+k]=float(previous[k]);delta['v2_'+k]=float(r[k]);delta['delta_'+k]=float(r[k])-float(previous[k])
        delta['delta_cooling_package_kwh']=delta['delta_compressor_kwh']+delta['delta_condenser_fan_kwh']
        delta['electric_decomposition_error_kwh']=math.fsum(delta['delta_'+k+'_kwh'] for k in components)-delta['delta_total_kwh']
        assert abs(delta['electric_decomposition_error_kwh'])<1e-7
        differences.append(delta)
    csv_write(public/'v1-v2-component-differences.csv',differences)
    weather=[]
    for f in metadata['inputs']:
        steps=read(output/'local-detail'/(f+'__OM__edf__fixed-steps.csv'))
        row=dict(fragment=f,source_sha=metadata['source_sha'],role='train',observations=len(steps),
                 low_ambient_boundary_steps=sum(x['low_ambient_boundary']=='True' for x in steps))
        for field in ['outdoor_c','ci_g_per_kwh','cop']:
            values=[float(x[field]) for x in steps];row[field+'_min']=min(values);row[field+'_max']=max(values);row[field+'_mean']=math.fsum(values)/len(values)
        weather.append(row)
    csv_write(public/'historical-input-summary.csv',weather)
    deferrals=read(output/'paired-deferral.csv');cooling=read(output/'paired-cooling.csv')
    paired=dict(deferral_pairs=len(deferrals),cooling_pairs=len(cooling),
                dmr_worse_pairs=sum(float(r['delta_dmr'])>1e-10 for r in deferrals),
                capacity_wait_worse_pairs=sum(float(r['delta_capacity_wait_minutes'])>0 for r in deferrals),
                queue_peak_worse_pairs=sum(float(r['delta_queue_peak'])>0 for r in deferrals),
                start_gpu_peak_worse_pairs=sum(float(r['delta_start_gpu_peak'])>1e-8 for r in deferrals),
                max_terminal_pair_k=max(abs(float(r['delta_inlet_final_c'])) for r in deferrals+cooling),
                max_it_pair_difference_kwh=max(abs(float(r['delta_it_kwh'])) for r in deferrals+cooling),
                deferral_total_carbon_change_pct_range=[min(100*float(r['delta_carbon_kg'])/float(r['immediate_carbon_kg']) for r in deferrals),max(100*float(r['delta_carbon_kg'])/float(r['immediate_carbon_kg']) for r in deferrals)],
                cooling_total_carbon_change_pct_range=[min(100*float(r['delta_carbon_kg'])/float(r['fixed_carbon_kg']) for r in cooling),max(100*float(r['delta_carbon_kg'])/float(r['fixed_carbon_kg']) for r in cooling)],
                v1_v2_total_energy_change_pct_range=[min(100*r['delta_total_kwh']/r['v1_total_kwh'] for r in differences),max(100*r['delta_total_kwh']/r['v1_total_kwh'] for r in differences)],
                v1_v2_schedule_matches=sum(r['same_task_schedule'] for r in differences),
                method='All fixed matrix pairs retained; no confidence intervals; same tasks reused, not 36 independent samples.')
    write_json(public/'paired-statistics.json',paired)
    for name in ['case-summary.csv','idle-background.csv','paired-deferral.csv','paired-cooling.csv','execution.json']:
        shutil.copyfile(output/name,public/name)
    maxima={}
    for detail in audit['details']:
        for key,value in detail['max_errors'].items():maxima[key]=max(maxima.get(key,0.),value)
    write_json(public/'independent-verification-summary.json',dict(**{k:v for k,v in audit.items() if k!='details'},max_errors=maxima))
    write_json(public/'result-methods.json',dict(source_sha=metadata['source_sha'],protocol_sha256=metadata['protocol_sha256'],
        data_identity=metadata['inputs'],units=dict(electricity='kWh_e',coil_evaporator_heat_rejection_envelope='kWh_th',
        thermal_storage='kWh_th (never add to electrical energy)',carbon='kgCO2e',resource_utilization='fraction of schedulable quota',
        blocked_counts='task-slot resource violations; CPU/GPU counts can overlap'),
        pooled_method='3 fragment sums; DMR and JCT task-weighted; utilisation equal hours; peak is max; cooling replicas never pooled together',
        comparison='V1 exact same SHA input, SLA, schedule and 72h window; component arithmetic decomposition, not physical causal attribution or real savings',
        hashes={p.name:file_sha(p) for p in sorted(public.iterdir()) if p.is_file() and p.name!='result-methods.json'}))
    print(json.dumps(paired,indent=2));print('Only saved data read; no simulation invoked.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='research_notes/platform-v2/acceptance-001');p.add_argument('--public',default='docs/research/platform-v2-final/evidence');a=p.parse_args()
    main(a.output,a.public)
