"""Predeclared synthetic physical boundary checks. No historical input."""
import copy
import itertools
from pathlib import Path
from experiments.platform_v1.acceptance import csv_write,write_json
from .inventory import load_config,build_inventory
from .physics import PhysicalModel


def main():
    cfg=load_config();out=Path('docs/research/platform-v2-final/evidence');rows=[];failures=[];safe=[];design=[]
    for name in ['OM','OH','OL']:
        inv=build_inventory(cfg,name);model=PhysicalModel(inv,cfg)
        design.append(dict(**inv.to_dict(),floor_m2=model.floor_m2,volume_m3=model.volume_m3,
                           thermal_capacity_kj_k=model.c_kj_k,ua_kw_k=model.ua_kw_k,
                           **model.thermal_mass_components,it_design_kw=model.it_design_kw,
                           coil_rating_kw=model.coil_rating_kw,chiller_reference_kw=model.chiller.reference_kw,
                           supply_fan_kw=model.supply_fan_kw,water_flow_max_kg_s=model.water_flow_max_kg_s))
        for fractions,outdoor,supply in itertools.product(itertools.product([0.,.5,1.],repeat=3),[5,25,35,50],[18,22,26]):
            it=model.it_power(tuple(f*c for f,c in zip(fractions,inv.capacity)))
            r=model.transition(24,it['it_kw'],outdoor,supply)
            row=dict(capacity=name,cpu_fraction=fractions[0],gpu_fraction=fractions[1],mem_fraction=fractions[2],outdoor_c=outdoor,supply_c=supply,**it,**r)
            rows.append(row)
            if abs(r['balance_residual_kj'])>1e-5 or abs(r['chiller_balance_residual_kj'])>1e-5:failures.append('balance')
            if r['evaporator_kw']>r['available_capacity_kw']+1e-8:failures.append('capacity limit')
            if r['water_flow_kg_s']>model.water_flow_max_kg_s+1e-8:failures.append('water flow limit')
            if abs(r['package_kw']-r['compressor_kw']-r['condenser_fan_kw'])>1e-8:failures.append('package double count')
        for load,outdoor,supply in itertools.product([0.,.5,1.],[5,25,50],[18,22,26]):
            it=model.it_power(tuple(load*c for c in inv.capacity))['it_kw']
            action,r,_=model.safe_transition(24,it,outdoor,supply)
            safe.append(dict(capacity=name,load_fraction=load,outdoor_c=outdoor,request_c=supply,applied_c=action,**r))
            if r['cooling_feasible'] and r['inlet_peak_c']>27+1e-8:failures.append('unsafe marked safe')
            if not r['cooling_feasible'] and r['inlet_peak_c']<=27+1e-8 and r['curve_domain_valid']:failures.append('infeasibility not explained')
    # Monotonic individual-resource IT; memory is intentionally flat.
    for name in ['OM','OH','OL']:
        inv=build_inventory(cfg,name);model=PhysicalModel(inv,cfg)
        for axis in range(3):
            values=[]
            for f in [0,.25,.5,.75,1]:
                d=[0.,0.,0.];d[axis]=f*inv.capacity[axis];values.append(model.it_power(d)['it_kw'])
            if any(b<a-1e-9 for a,b in zip(values,values[1:])):failures.append('IT reversal')
    sensitivity=[]
    variations=[('baseline',None),('effective_slab_m',.01),('effective_slab_m',.04),('equipment_coupling_fraction',.5),('envelope_u_w_m2_k',.25),('envelope_u_w_m2_k',1.)]
    for parameter,value in variations:
        other=copy.deepcopy(cfg)
        if value is not None:other['room'][parameter]=value
        inv=build_inventory(other,'OM');m=PhysicalModel(inv,other)
        for load,outdoor,supply in itertools.product([.1,.5,.9],[5,25,50],[18,26]):
            temp=24;electric=0.;unmet=0.;peak=temp;overrides=0;infeasible=0
            it=m.it_power(tuple(load*c for c in inv.capacity))['it_kw']
            for _ in range(24):
                _,r,override=m.safe_transition(temp,it,outdoor,supply);electric+=(it+r['hvac_kw'])*.25
                unmet+=r['unmet_cooling_kw']*.25;peak=max(peak,r['inlet_peak_c']);temp=r['inlet_end_c']
                overrides+=override;infeasible+=not r['cooling_feasible']
            sensitivity.append(dict(parameter=parameter,value=value,load_fraction=load,outdoor_c=outdoor,request_c=supply,
                                    c_kj_k=m.c_kj_k,ua_kw_k=m.ua_kw_k,total_kwh=electric,unmet_kwh_th=unmet,
                                    inlet_peak_c=peak,final_c=temp,storage_kwh_th=m.c_kj_k*(temp-24)/3600,
                                    safety_overrides=overrides,infeasible_steps=infeasible,scope='synthetic six-hour constant load, not historical savings'))
    csv_write(out/'physical-boundaries.csv',rows);csv_write(out/'safe-boundaries.csv',safe)
    csv_write(out/'thermal-assumption-sensitivity.csv',sensitivity);write_json(out/'inventories.json',design)
    verdict=dict(synthetic=True,historical_data_read=False,boundary_points=len(rows),safe_probes=len(safe),sensitivity_trajectories=len(sensitivity),
                 max_balance_residual_kj=max(abs(r['balance_residual_kj']) for r in rows),
                 max_chiller_residual_kj=max(abs(r['chiller_balance_residual_kj']) for r in rows),
                 capacity_shortfall_seen=any(r['unmet_cooling_kw']>0 and r['inlet_end_c']>27 for r in rows),
                 safe_infeasibility_seen=any(not r['cooling_feasible'] for r in safe),failures=failures)
    write_json(out/'engineering-checks.json',verdict);print(verdict)
    if failures or not verdict['capacity_shortfall_seen'] or not verdict['safe_infeasibility_seen']:raise SystemExit(1)


if __name__=='__main__':main()
