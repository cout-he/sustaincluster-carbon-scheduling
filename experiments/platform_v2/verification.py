"""Independent ledger/CSV accounting; never calls a simulator or transition.

All 36 workload AND 18 idle traces are checked before acceptance. Re-reading
saved traces later is verification, not another integration invocation.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from experiments.platform_v1.data import load_train,canonical_sha
from .inventory import load_config,build_inventory


def flag(value):return value is True or str(value).lower() in ['true','1']


def verify_result(result,scenario,inv,cfg,sla=1.5,idle=False):
    errors=[];maxima={};steps=result['steps'];summary=result['summary'];rows=result['tasks']
    def require(ok,label):
        if not ok and label not in errors:errors.append(label)
    def equal(a,b,label,tol=1e-7):
        a=float(a);b=float(b);error=abs(a-b)
        maxima[label]=max(maxima.get(label,0.),error)
        require(math.isfinite(a) and math.isfinite(b) and error<=tol,label)
    # Derive aggregate heat capacity independently of PhysicalModel/log fields.
    g=cfg['room'];t=cfg['thermal'];area=inv.racks*g['floor_m2_per_rack']
    c=area*g['height_m']*t['air_density_kg_m3']*t['air_cp_kj_kg_k']
    c+=(inv.cpu_only_hosts*g['cpu_host_mass_kg']+inv.gpu_hosts*g['gpu_host_mass_kg']+inv.racks*g['rack_mass_kg'])*g['equipment_cp_kj_kg_k']*g['equipment_coupling_fraction']
    c+=area*g['effective_slab_m']*g['concrete_density_kg_m3']*g['concrete_cp_kj_kg_k']
    equal(summary['thermal_capacity_kj_k'],c,'heat_capacity')
    expected=() if idle else scenario.tasks;tasks={x.task_id:x for x in expected}
    require(len(rows)==len(tasks) and len({r['task_id'] for r in rows})==len(rows) and set(r['task_id'] for r in rows)==set(tasks),'task_identity')
    horizon=15*len(scenario.observations);require(len(steps)==len(scenario.observations),'step_count')
    events={};misses=0;waits=[0.,0.];jcts=[]
    for row in rows:
        if row['task_id'] not in tasks:continue
        task=tasks[row['task_id']];start=float(row['start_minute']);finish=float(row['finish_minute'])
        require(row['state']=='completed','task_completion')
        require(start>=task.arrival_minute and start%15==0 and finish<=horizon,'task_time_domain')
        equal(finish-start,task.duration_minutes,'task_duration')
        deadline=task.arrival_minute+sla*task.duration_minutes
        equal(row['deadline_minute'],deadline,'synthetic_deadline')
        missed=finish>deadline+1e-9;misses+=missed
        require(flag(row['deadline_missed'])==missed and not flag(row['deadline_censored']),'deadline_classification')
        cap=float(row['capacity_wait_minutes']);policy=float(row['policy_wait_minutes'])
        require(cap>=0 and policy>=0 and cap%15==policy%15==0,'wait_units')
        equal(cap+policy,start-task.arrival_minute,'wait_identity')
        waits[0]+=cap;waits[1]+=policy;jcts.append(finish-task.arrival_minute)
        equal(row['jct_minutes'],jcts[-1],'jct_identity')
        release=15*math.ceil(finish/15-1e-12)
        for minute,sign in [(start,1),(release,-1)]:
            event=events.setdefault(minute,[[],[],[]])
            for axis in range(3):event[axis].append(sign*task.demand[axis])
    for key,value in [('arrived',len(tasks)),('completed',len(tasks)),('unfinished',0),('deadline_missed',misses),
                      ('dmr',misses/len(tasks) if tasks else 0.),('completion_rate',1.),
                      ('capacity_wait_minutes',waits[0]),('policy_wait_minutes',waits[1]),
                      ('mean_jct_minutes',math.fsum(jcts)/len(jcts) if jcts else 0.)]:equal(summary[key],value,'summary_'+key)
    if 'schedule_sha256' in summary:
        require(summary['schedule_sha256']==canonical_sha([(r['task_id'],float(r['start_minute']),float(r['finish_minute'])) for r in rows]),'schedule_hash')
    used=[0.,0.,0.];previous=cfg['thermal']['initial_inlet_c'];itparts=['cpu','gpu','dimm','host_other','psu_loss','rack_switch'];hvacparts=['supply_fan','pump','compressor','condenser_fan']
    integral_parts=itparts+hvacparts+['it','hvac','total','coil','heat_rejection','envelope','evaporator','package']
    integral={k:[] for k in integral_parts};carbon=[];storages=[];stepwait=[0,0];phase={p:dict(it=[],hvac=[],total=[],carbon=[]) for p in ['arrival','drain']}
    for index,row in enumerate(steps):
        if index>=len(scenario.observations):break
        sample=scenario.observations[index];minute=float(row['minute']);equal(minute,index*15,'step_grid')
        equal(row['ci_g_per_kwh'],sample.ci_g_per_kwh,'ci_observation');equal(row['outdoor_c'],sample.outdoor_c,'weather_observation')
        equal(row['observation_minute'],sample.observed_minute,'observation_time');require(sample.observed_minute<=minute,'causal_observation')
        for axis,k in enumerate(['cpu','gpu','mem']):
            used[axis]=math.fsum([used[axis],*events.get(minute,[[],[],[]])[axis]])
            equal(row['used_'+k],used[axis],'resource_ledger_'+k)
            require(-1e-7<=used[axis]<=inv.capacity[axis]+1e-7,'resource_bound_'+k)
        for k in integral_parts:integral[k].append(float(row[k+'_kw'])*.25)
        for k in itparts+hvacparts:require(math.isfinite(float(row[k+'_kw'])) and float(row[k+'_kw'])>=0,'power_nonnegative')
        it=math.fsum(float(row[k+'_kw']) for k in itparts);hvac=math.fsum(float(row[k+'_kw']) for k in hvacparts);total=it+hvac
        equal(row['it_kw'],it,'it_component_sum');equal(row['hvac_kw'],hvac,'hvac_component_sum');equal(row['total_kw'],total,'total_component_sum')
        p=cfg['power']
        equal(row['cpu_kw'],inv.cpu_objects*(p['cpu_idle_w']+(p['cpu_reference_full_w']-p['cpu_idle_w'])*used[0]/inv.installed_cores)/1000,'cpu_mapping')
        equal(row['gpu_kw'],inv.gpus*(p['gpu_idle_w']+(p['gpu_reference_full_w']-p['gpu_idle_w'])*used[1]/inv.gpus)/1000,'gpu_mapping')
        equal(row['dimm_kw'],inv.dimm_objects*p['dimm_base_w']/1000,'fixed_memory_power')
        dc=math.fsum(float(row[k+'_kw']) for k in ['cpu','gpu','dimm','host_other'])
        equal(row['psu_loss_kw'],dc*(1/p['psu_efficiency']-1),'psu_once')
        equal(row['package_kw'],float(row['compressor_kw'])+float(row['condenser_fan_kw']),'package_once')
        equal(row['evaporator_kw'],float(row['coil_kw'])+float(row['pump_kw']),'pump_heat')
        equal(row['heat_rejection_kw'],float(row['evaporator_kw'])+float(row['package_kw']),'chiller_heat')
        require(float(row['evaporator_kw'])<=float(row['available_capacity_kw'])+1e-7,'chiller_capacity')
        require(float(row['coil_kw'])<=float(row['coil_request_kw'])+1e-7,'coil_request_bound')
        equal(row['unmet_cooling_kw'],float(row['coil_request_kw'])-float(row['coil_kw']),'unmet_cooling')
        require(0<=float(row['water_flow_kg_s'])<=inv.chiller_modules*cfg['chiller']['unit_max_water_kg_s']+1e-7,'water_flow_bound')
        equal(row['inlet_start_c'],previous,'thermal_continuity')
        storage=c*(float(row['inlet_end_c'])-float(row['inlet_start_c']));storages.append(storage)
        equal(row['storage_delta_kj'],storage,'thermal_storage',1e-5)
        heat=(total+float(row['envelope_kw'])-float(row['heat_rejection_kw']))*900
        equal(heat,storage,'facility_first_law',1e-5)
        previous=float(row['inlet_end_c'])
        require(float(row['inlet_peak_c'])<=t['inlet_control_limit_c']+1e-8 and flag(row['cooling_feasible']) and flag(row['curve_domain_valid']),'thermal_safety')
        if minute>=horizon-cfg['evaluation']['terminal_common_hours']*60:equal(row['supply_requested_c'],cfg['evaluation']['fixed_supply_c'],'common_terminal_control')
        kg=total*.25*sample.ci_g_per_kwh/1000;carbon.append(kg);equal(row['carbon_kg'],kg,'step_carbon_integral')
        for j,key in enumerate(['capacity_wait_tasks','policy_wait_tasks']):stepwait[j]+=float(row[key])*15
        bucket=phase['arrival' if minute<1440 else 'drain']
        for k,value in [('it',it*.25),('hvac',hvac*.25),('total',total*.25),('carbon',kg)]:bucket[k].append(value)
    for k,values in integral.items():equal(summary[k+'_kwh'],math.fsum(values),'energy_integral_'+k)
    equal(summary['carbon_kg'],math.fsum(carbon),'carbon_integral')
    equal(summary['thermal_storage_kwh'],math.fsum(storages)/3600,'window_thermal_storage')
    equal(summary['inlet_final_c'],previous,'final_temperature')
    for j,key in enumerate(['capacity_wait_minutes','policy_wait_minutes']):equal(summary[key],stepwait[j],'step_'+key)
    for p,bucket in phase.items():
        for k,values in bucket.items():equal(summary[p+'_'+('carbon_kg' if k=='carbon' else k+'_kwh')],math.fsum(values),'phase_'+p+'_'+k)
    return dict(errors=errors,max_errors=maxima,tasks=len(rows),steps=len(steps),idle=idle)


def read_rows(path):
    with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def verify_saved(output,scenarios,cfg,plan):
    output=Path(output);cases=read_rows(output/'case-summary.csv');idles=read_rows(output/'idle-background.csv');details=[];errors=[]
    for rows,idle in [(cases,False),(idles,True)]:
        for row in rows:
            f=row['fragment'];name=row['capacity'];h=row['cooling']
            label=f'idle__{f}__{name}__{h}' if idle else row['case_id']
            try:
                result=dict(summary=row,steps=read_rows(output/'local-detail'/(label+'-steps.csv')),
                            tasks=[] if idle else read_rows(output/'local-detail'/(label+'-tasks.csv')))
                audit=verify_result(result,scenarios[f],build_inventory(cfg,name),cfg,plan['sla'],idle)
                details.append(dict(case_id=label,**audit))
                if audit['errors']:errors.append(dict(case_id=label,errors=audit['errors']))
            except Exception as exc:errors.append(dict(case_id=label,error=repr(exc)))
    idle_index={(r['fragment'],r['capacity'],r['cooling']):r for r in idles}
    for row in cases:
        for prefix,cooling in [('common','fixed'),('matched',row['cooling'])]:
            background=idle_index.get((row['fragment'],row['capacity'],cooling))
            if background is None:errors.append(dict(case_id=row['case_id'],error='Missing idle background'));continue
            for label,unit in [('carbon','kg'),('total','kwh')]:
                base=label+'_'+unit;inc=prefix+'_incremental_'+('carbon_kg' if label=='carbon' else 'kwh')
                if abs(float(row[base])-float(background[base])-float(row[inc]))>1e-7:
                    errors.append(dict(case_id=row['case_id'],error='Background subtraction '+inc))
    expected_cases={tuple(x) for x in plan['cases']}
    actual_cases={(r['fragment'],r['capacity'],r['policy'],r['cooling']) for r in cases}
    expected_idle={(x[0],x[1],x[3]) for x in plan['cases']}
    actual_idle={(r['fragment'],r['capacity'],r['cooling']) for r in idles}
    if len(cases)!=36 or len(idles)!=18 or actual_cases!=expected_cases or actual_idle!=expected_idle:errors.append(dict(error='Incomplete/duplicate/mutated saved matrix'))
    return dict(accepted=not errors and len(details)==54,errors=errors,workload_cases=len(cases),idle_cases=len(idles),
                task_executions=sum(d['tasks'] for d in details),step_rows=sum(d['steps'] for d in details),details=details)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pool',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    plan=json.loads(Path('docs/research/platform-v2-final/acceptance-protocol.json').read_text(encoding='utf-8'))
    scenarios={f:load_train(Path(a.pool)/'scenarios'/f,72,v) for f,v in plan['inputs'].items()}
    audit=verify_saved(a.output,scenarios,load_config(),plan)
    print(json.dumps({k:v for k,v in audit.items() if k!='details'},indent=2))
    if not audit['accepted']:raise SystemExit(2)
