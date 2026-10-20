"""One frozen synthetic batch; no task executor, trace loader or training imports.

Power is kW, heat flow kW_th, capacity kJ/K, time seconds. V1 is read-only.
"""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import subprocess
import time

from experiments.platform_v1.inventory import load_config, build_inventory
from experiments.platform_v1.physics import PhysicalModel

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/research/platform-v1-1'
MARKER=ROOT/'research_notes/platform-v1-1/once.json'
BASE='64aa6926b77acf507f03c471322cad63008287d4'
GEOMETRY=('coil_rating_kw','air_conductance_kw_k','airflow_m3_s','supply_fan_kw')
COMPONENTS=('it_kw','supply_fan_kw','pump_kw','compressor_kw','condenser_fan_kw',
            'hvac_kw','coil_kw','envelope_kw','heat_rejection_kw')


def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def variants(cfg,ranges):
    yield 'baseline',copy.deepcopy(cfg),'baseline',None
    for path,values in ranges.items():
        group,key=path.split('.')
        for value in values:
            other=copy.deepcopy(cfg);other[group][key]=value
            yield path+'='+str(value),other,path,value


def make_model(cfg,base,path):
    model=PhysicalModel(build_inventory(cfg,'OM'),cfg)
    if path.startswith('power.'):
        reference=PhysicalModel(build_inventory(base,'OM'),base)
        for attribute in GEOMETRY:setattr(model,attribute,getattr(reference,attribute))
    return model


def memory_rows(cfg,candidates,fractions):
    rows=[]
    for candidate in candidates:
        other=copy.deepcopy(cfg)
        other['host']['dimm_count']=candidate['dimm_count'];other['host']['dimm_gib']=candidate['dimm_gib']
        inv=build_inventory(other,'OM');model=PhysicalModel(inv,other)
        for fraction in fractions:
            powers=model.it_power(tuple(fraction*c for c in inv.capacity))
            rows.append(dict(candidate=candidate['name'],load_fraction=fraction,hosts=inv.hosts,
                             dimms=inv.dimm_objects,dimm_gib=candidate['dimm_gib'],
                             installed_mem_gb=inv.installed_mem_gb,os_reserved_gb=inv.reserved_mem_gb,
                             available_after_os_gb=inv.installed_mem_gb-inv.reserved_mem_gb,
                             schedulable_mem_gb=inv.capacity[2],quota_excluded_mem_gb=inv.quota_excluded_mem_gb,
                             capacity_ok=inv.quota_excluded_mem_gb>=0,**powers))
    return rows


def equilibrium(model,it_kw,outdoor_c,supply_c):
    """Independent piecewise solution of constant-load room heat balance."""
    k=model.air_conductance_kw_k;ua=model.ua_kw_k;q=it_kw+model.supply_fan_kw
    temp=(q+ua*outdoor_c+k*supply_c)/(k+ua)
    if temp>supply_c+model.coil_rating_kw/k:
        temp=outdoor_c+(q-model.coil_rating_kw)/ua if ua>0 else None
    if temp is not None and temp<supply_c:
        temp=outdoor_c+q/ua if ua>0 else None
    coil=None if temp is None else min(model.coil_rating_kw,max(0,k*(temp-supply_c)))
    demand_at_limit=q+ua*(outdoor_c-model.t['inlet_control_limit_c'])
    return dict(steady_inlet_c=temp,steady_coil_kw=coil,
                time_constant_minutes=model.c_kj_k/(k+ua)/60,
                steady_safe_supply_max_c=model.t['inlet_control_limit_c']-demand_at_limit/k,
                required_coil_at_27_kw=demand_at_limit,
                steady_coil_capacity_margin_kw=model.coil_rating_kw-demand_at_limit)


def simulate(model,demand,outdoor,request,mode,steps,initial,ci):
    parts=model.it_power(demand);temp=initial;rows=[];overrides=0;applied=[];status='completed'
    for _ in range(steps):
        try:
            if mode=='v1_safety_filter':
                action,row,override=model.safe_transition(temp,parts['it_kw'],outdoor,request)
            elif mode=='unfiltered':
                action=request;row=model.transition(temp,parts['it_kw'],outdoor,request);override=False
            else:raise ValueError('Unknown mode')
        except ValueError as error:
            if str(error)!='No safe cooling action at current state/load':raise
            status='no_safe_action';break
        if any(not math.isfinite(x) for x in row.values()):raise ArithmeticError('Nonfinite physical output')
        if any(row[x]<0 for x in ['hvac_kw','pump_kw','compressor_kw','supply_fan_kw','condenser_fan_kw']):
            raise ArithmeticError('Negative electric power')
        row['inlet_start_c']=temp;temp=row['inlet_end_c'];rows.append(row)
        applied.append(action);overrides+=int(override)
    out=dict(status=status,completed_steps=len(rows),safety_overrides=overrides,inlet_end_c=temp,
             inlet_peak_c=max([initial]+[r['inlet_peak_c'] for r in rows]),
             supply_applied_min_c=min(applied) if applied else None,
             supply_applied_max_c=max(applied) if applied else None,
             inlet_15min_c=rows[0]['inlet_end_c'] if rows else None,
             inlet_60min_c=rows[3]['inlet_end_c'] if len(rows)>=4 else None,
             storage_kwh_th=model.c_kj_k*(temp-initial)/3600,
             independent_balance_residual_kj=max([0.]+[abs((parts['it_kw']+r['hvac_kw']+r['envelope_kw']-r['heat_rejection_kw'])*900-model.c_kj_k*(r['inlet_end_c']-r['inlet_start_c'])) for r in rows]))
    for key in COMPONENTS:
        out[key+'_mean']=math.fsum(parts['it_kw'] if key=='it_kw' else r[key] for r in rows)/len(rows) if rows else None
    cops=[r['cop'] for r in rows]
    out.update(cop_min=min(cops) if cops else None,cop_mean=math.fsum(cops)/len(cops) if cops else None,
               cop_max=max(cops) if cops else None,
               cop_ceiling_fraction=sum(abs(x-model.t['cop_max'])<1e-9 for x in cops)/len(cops) if cops else None,
               cop_floor_fraction=sum(abs(x-model.t['cop_min'])<1e-9 for x in cops)/len(cops) if cops else None)
    complete=status=='completed' and len(rows)==steps
    for label,key in [('it','it_kw'),('hvac','hvac_kw')]:
        out[label+'_kwh']=out[key+'_mean']*steps*.25 if complete else None
    out['total_kwh']=out['it_kwh']+out['hvac_kwh'] if complete else None
    out['synthetic_carbon_kg']=out['total_kwh']*ci/1000 if complete else None
    out['electric_integral_error_kwh']=abs(out['total_kwh']-math.fsum((parts['it_kw']+r['hvac_kw'])*.25 for r in rows)) if complete else None
    return out


def claim_once(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(payload,stream,indent=2)


def csv_write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def freeze():
    protocol=DOC/'protocol.json'
    files=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE,'experiments/platform_v1','tests/platform_v1','docs/research/platform-v1'],cwd=ROOT,text=True).splitlines()
    protected={rel:digest(ROOT/rel) for rel in files}
    harness=['diagnostics/platform_v1_1/'+name for name in ['__init__.py','run.py','test_validation.py']]
    manifest=dict(base_sha=BASE,protocol_sha256=digest(protocol),protected_files=protected,
                  harness_files={rel:digest(ROOT/rel) for rel in harness},
                  created_utc=datetime.now(timezone.utc).isoformat())
    target=DOC/'frozen-manifest.json'
    if target.exists():raise FileExistsError('Freeze already exists; do not overwrite')
    write_json(target,manifest)
    print(json.dumps({'frozen_manifest_sha256':digest(target),'protected_files':len(protected)},indent=2))


def execute():
    manifest=json.loads((DOC/'frozen-manifest.json').read_text(encoding='utf-8'))
    if digest(DOC/'protocol.json')!=manifest['protocol_sha256']:raise ValueError('Protocol changed after freeze')
    for rel,sha in {**manifest['protected_files'],**manifest['harness_files']}.items():
        if digest(ROOT/rel)!=sha:raise ValueError('Frozen source changed: '+rel)
    if subprocess.check_output(['git','status','--porcelain','--','diagnostics/platform_v1_1','docs/research/platform-v1-1/protocol.json','docs/research/platform-v1-1/frozen-manifest.json'],cwd=ROOT,text=True).strip():
        raise ValueError('Commit harness and freeze before executing')
    protocol=json.loads((DOC/'protocol.json').read_text(encoding='utf-8'));cfg=load_config()
    start=time.perf_counter();stamp=datetime.now(timezone.utc).isoformat()
    source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    marker=dict(status='started',source_sha=source_sha,protocol_sha256=digest(DOC/'protocol.json'),started_utc=stamp)
    claim_once(MARKER,marker)
    outputs=DOC/'evidence';results=[]
    options=list(variants(cfg,protocol['oat_ranges']))
    for name,other,path,value in options:
        model=make_model(other,cfg,path)
        for load,outdoor,request,mode in itertools.product(protocol['load_fractions_of_schedulable_cpu_gpu_mem'],protocol['outdoor_c'],protocol['requested_supply_c'],protocol['modes']):
            demand=tuple(load*c for c in model.inv.capacity)
            summary=simulate(model,demand,outdoor,request,mode,int(protocol['hours']*4),protocol['initial_inlet_c'],protocol['synthetic_flat_ci_g_per_kwh'])
            results.append(dict(variant=name,changed_parameter=path,changed_value=value,load_fraction=load,
                                outdoor_c=outdoor,supply_request_c=request,mode=mode,
                                synthetic_ci_g_per_kwh=protocol['synthetic_flat_ci_g_per_kwh'],
                                coil_rating_kw=model.coil_rating_kw,supply_fan_design_kw=model.supply_fan_kw,
                                **equilibrium(model,model.it_power(demand)['it_kw'],outdoor,request),**summary))
        print(name+': '+str(len(results))+' trajectories',flush=True)
    memory=memory_rows(cfg,protocol['memory_static_only'],protocol['memory_load_fractions'])
    model=make_model(cfg,cfg,'baseline');capacity=[]
    for basis,outdoor in itertools.product(protocol['capacity_static_demand_basis'],protocol['capacity_static_outdoor_c']):
        inv=model.inv
        demand=inv.capacity if basis=='schedulable_full' else (inv.installed_cores,inv.gpus,inv.installed_mem_gb)
        it=model.it_power(demand)['it_kw'];r=equilibrium(model,it,outdoor,18)
        capacity.append(dict(basis=basis,outdoor_c=outdoor,it_kw=it,**r))
    csv_write(outputs/'synthetic-cases.csv',results);csv_write(outputs/'memory-static.csv',memory);csv_write(outputs/'capacity-static.csv',capacity)
    baseline={(r['load_fraction'],r['outdoor_c'],r['supply_request_c'],r['mode']):r for r in results if r['variant']=='baseline'}
    deltas=[]
    for r in results:
        if r['variant']=='baseline':continue
        b=baseline[(r['load_fraction'],r['outdoor_c'],r['supply_request_c'],r['mode'])]
        if r['status']!='completed' or b['status']!='completed':continue
        deltas.append(dict(variant=r['variant'],load_fraction=r['load_fraction'],outdoor_c=r['outdoor_c'],supply_request_c=r['supply_request_c'],mode=r['mode'],
                           it_energy_change_pct=100*(r['it_kwh']/b['it_kwh']-1),hvac_energy_change_pct=100*(r['hvac_kwh']/b['hvac_kwh']-1),
                           total_energy_and_flat_ci_carbon_change_pct=100*(r['total_kwh']/b['total_kwh']-1),
                           inlet_peak_change_k=r['inlet_peak_c']-b['inlet_peak_c'],storage_change_kwh_th=r['storage_kwh_th']-b['storage_kwh_th']))
    csv_write(outputs/'oat-differences.csv',deltas)
    checks=dict(expected_trajectories=len(options)*90,actual_trajectories=len(results),
                no_safe_action_count=sum(r['status']!='completed' for r in results),
                max_filtered_peak_c=max(r['inlet_peak_c'] for r in results if r['mode']=='v1_safety_filter'),
                max_first_law_residual_kj=max(r['independent_balance_residual_kj'] for r in results),
                max_integral_error_kwh=max(r['electric_integral_error_kwh'] or 0 for r in results),
                protected_files_unchanged=all(digest(ROOT/rel)==sha for rel,sha in manifest['protected_files'].items()),
                no_task_or_historical_input=True,experiment_interpretation='synthetic sensitivity only, not historical savings or calibration')
    metadata=dict(source_sha=source_sha,protocol_sha256=digest(DOC/'protocol.json'),manifest_sha256=digest(DOC/'frozen-manifest.json'),
                  python=platform.python_version(),started_utc=stamp,elapsed_seconds=time.perf_counter()-start,checks=checks,
                  output_sha256={p.name:digest(p) for p in outputs.glob('*.csv')})
    write_json(outputs/'execution.json',metadata)
    marker.update(status='completed',elapsed_seconds=metadata['elapsed_seconds'],trajectories=len(results));write_json(MARKER,marker)
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze',action='store_true');group.add_argument('--execute-once',action='store_true')
    args=parser.parse_args()
    if args.freeze:freeze()
    else:execute()
