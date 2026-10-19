"""Independent arithmetic checks of SAVED acceptance files; never calls run()."""
import argparse
import csv
import json
import math
from pathlib import Path

from .acceptance import csv_write,write_json
from .data import file_sha


def read_csv(path):
    with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def independent_carbon(steps):
    values=[]
    for step in steps:
        value=float(step['total_kw'])*.25*float(step['ci_g_per_kwh'])/1000
        if not math.isclose(value,float(step['carbon_kg']),rel_tol=0,abs_tol=1e-10):
            raise ValueError('Saved step carbon differs from independently integrated power/CI')
        values.append(value)
    return math.fsum(values)


def verify(root,output):
    root=Path(root);out=Path(output);metadata=json.loads((root/'execution.json').read_text())
    rows=read_csv(root/'case-summary.csv');max_energy_error=0.;max_heat=0.;tasks_checked=0
    for path,sha in metadata['source_hashes'].items():
        if file_sha(path)!=sha:raise ValueError('Execution source changed after one batch: '+path)
    for case in rows:
        assert case['status']=='completed'
        tasks=read_csv(root/'local-detail'/(case['case_id']+'-tasks.csv'))
        steps=read_csv(root/'local-detail'/(case['case_id']+'-steps.csv'))
        assert len(tasks)==int(case['arrived']) and len({t['task_id'] for t in tasks})==len(tasks)
        assert len(steps)==288 and [float(s['minute']) for s in steps]==list(range(0,4320,15))
        capacity=metadata['inventories'][case['capacity']]['capacity'];thermal_c=metadata['inventories'][case['capacity']]['hosts']*500.
        misses=0;capacity_wait=0;policy_wait=0
        for t in tasks:
            assert t['state']=='completed'
            start=float(t['start_minute']);finish=float(t['finish_minute']);duration=finish-start
            deadline=float(t['deadline_minute']);arrival=deadline-float(case['sla'])*duration
            waits=float(t['capacity_wait_minutes'])+float(t['policy_wait_minutes'])
            assert math.isclose(start-arrival,waits,abs_tol=1e-8)
            assert math.isclose(finish-arrival,float(t['jct_minutes']),abs_tol=1e-8)
            missed=finish>deadline+1e-9
            assert (t['deadline_missed']=='True')==missed
            misses+=missed;capacity_wait+=float(t['capacity_wait_minutes']);policy_wait+=float(t['policy_wait_minutes'])
        assert misses==int(case['deadline_missed']) and capacity_wait==float(case['capacity_wait_minutes']) and policy_wait==float(case['policy_wait_minutes'])
        for s in steps:
            for name,c in zip(['cpu','gpu','mem'],capacity):assert 0<=float(s['used_'+name])<=c+1e-9
            assert float(s['observation_minute'])==float(s['minute'])//60*60
            assert float(s['inlet_peak_c'])<=27+1e-8
            for name in ['cpu_kw','gpu_kw','dimm_kw','psu_loss_kw','host_other_kw','rack_switch_kw','supply_fan_kw','pump_kw','compressor_kw','condenser_fan_kw']:
                assert math.isfinite(float(s[name])) and float(s[name])>=0
            residual=(float(s['it_kw'])+float(s['hvac_kw'])+float(s['envelope_kw'])-float(s['heat_rejection_kw']))*900-thermal_c*(float(s['inlet_end_c'])-float(s['inlet_start_c']))
            max_heat=max(max_heat,abs(residual));assert abs(residual)<=1e-5
        for summary_key,power_key,scale in [('it_kwh','it_kw',.25),('hvac_kwh','hvac_kw',.25),('total_kwh','total_kw',.25),('carbon_kg','carbon_kg',1.)]:
            value=independent_carbon(steps) if summary_key=='carbon_kg' else math.fsum(float(s[power_key])*scale for s in steps)
            error=abs(value-float(case[summary_key]));max_energy_error=max(max_energy_error,error)
            assert error<=1e-7
        tasks_checked+=len(tasks)
    assert len(rows)==144 and metadata['integration_invocations']==1
    pooled=[]
    for cap in ['OM','OH','OL']:
        for sla in [1.5,2.]:
            for policy in ['fifo','edf','threshold_fifo','threshold_edf']:
                for cooling in ['fixed','variable']:
                    a=[r for r in rows if r['capacity']==cap and float(r['sla'])==sla and r['policy']==policy and r['cooling']==cooling]
                    assert len(a)==3;n=sum(int(r['arrived']) for r in a)
                    item=dict(capacity=cap,sla=sla,policy=policy,cooling=cooling,fragments=3,tasks=n,
                              dmr_pct=sum(int(r['deadline_missed']) for r in a)/n*100,
                              capacity_wait_pct=sum(int(r['capacity_wait_tasks']) for r in a)/n*100,
                              mean_jct_minutes=sum(float(r['mean_jct_minutes'])*int(r['arrived']) for r in a)/n,
                              queue_peak=max(int(r['queue_peak']) for r in a),
                              competition_minutes_sum=sum(float(r['competition_minutes']) for r in a))
                    for k in ['it_kwh','hvac_kwh','total_kwh','carbon_kg','idle_carbon_kg','incremental_carbon_kg','capacity_wait_minutes','policy_wait_minutes']:
                        item[k]=sum(float(r[k]) for r in a)
                    pooled.append(item)
    out.mkdir(parents=True,exist_ok=True)
    for name in ['case-summary.csv','paired-deferral.csv','paired-cooling.csv','idle-background.csv','execution.json']:
        (out/name).write_bytes((root/name).read_bytes())
    csv_write(out/'pooled-summary.csv',pooled)
    verdict=dict(saved_cases_checked=144,saved_task_executions_checked=tasks_checked,saved_steps_checked=144*288,
                 max_integral_recalculation_error=max_energy_error,max_independent_heat_residual_kj=max_heat,
                 no_simulation_replay=True,source_files_equal_execution_hashes=True,
                 pooled_method='task-weight DMR/JCT, additive fixed-window energy/carbon, maximum individual queue, no confidence intervals',
                 experimental_acceptance=metadata['accepted'])
    write_json(out/'saved-result-verification.json',verdict)
    print(json.dumps(verdict,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',default='research_notes/platform-v1/acceptance-001')
    p.add_argument('--output',default='docs/research/platform-v1/evidence');args=p.parse_args();verify(args.input,args.output)


if __name__=='__main__':main()
