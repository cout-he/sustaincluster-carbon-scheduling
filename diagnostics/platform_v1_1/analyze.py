"""Arithmetic postprocessing only: no physical transition or trace data reads."""
import copy
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/research/platform-v1-1'
EVIDENCE=DOC/'evidence'


def check_row(row,hours):
    if row['status']!='completed':return 0.,0.
    f=lambda k:float(row[k])
    hvac=math.fsum(f(k+'_kw_mean') for k in ['supply_fan','pump','compressor','condenser_fan'])
    errors=[abs(hvac-f('hvac_kw_mean')),abs(f('it_kw_mean')*hours-f('it_kwh')),
            abs(hvac*hours-f('hvac_kwh')),abs((f('it_kw_mean')+hvac)*hours-f('total_kwh')),
            abs(f('total_kwh')*f('synthetic_ci_g_per_kwh')/1000-f('synthetic_carbon_kg'))]
    balance=abs((f('it_kw_mean')+hvac+f('envelope_kw_mean')-f('heat_rejection_kw_mean'))*hours-f('storage_kwh_th'))*3600
    if max(errors)>1e-7:raise ValueError('Saved electrical units/integral mismatch')
    if balance>1e-5:raise ValueError('Saved whole-window heat balance mismatch')
    if row['mode']=='v1_safety_filter' and f('inlet_peak_c')>27.00000001:raise ValueError('Filtered thermal limit violated')
    return max(errors),balance


def read_csv(name):
    with (EVIDENCE/name).open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))


def write_csv(name,rows):
    with (EVIDENCE/name).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    execution=json.loads((EVIDENCE/'execution.json').read_text(encoding='utf-8'))
    manifest=json.loads((DOC/'frozen-manifest.json').read_text(encoding='utf-8'))
    for name,sha in execution['output_sha256'].items():
        if hashlib.sha256((EVIDENCE/name).read_bytes()).hexdigest()!=sha:raise ValueError('Original CSV changed: '+name)
    for name,sha in {**manifest['protected_files'],**manifest['harness_files']}.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('Frozen source changed: '+name)
    protocol=json.loads((DOC/'protocol.json').read_text(encoding='utf-8'))
    rows=read_csv('synthetic-cases.csv');deltas=read_csv('oat-differences.csv')
    keys={(r['variant'],r['load_fraction'],r['outdoor_c'],r['supply_request_c'],r['mode']) for r in rows}
    if len(keys)!=len(rows) or len(rows)!=3240:raise ValueError('Incomplete/duplicate matrix')
    errors=[check_row(r,protocol['hours']) for r in rows]
    baseline=[r for r in rows if r['variant']=='baseline']
    write_csv('baseline-safe.csv',[r for r in baseline if r['mode']=='v1_safety_filter'])
    summary=[]
    for name in dict.fromkeys(r['variant'] for r in deltas):
        group=[r for r in deltas if r['variant']==name and r['mode']=='v1_safety_filter']
        summary.append(dict(variant=name,conditions=len(group),
                            total_energy_flat_ci_carbon_min_pct=min(float(r['total_energy_and_flat_ci_carbon_change_pct']) for r in group),
                            total_energy_flat_ci_carbon_max_pct=max(float(r['total_energy_and_flat_ci_carbon_change_pct']) for r in group),
                            hvac_energy_min_pct=min(float(r['hvac_energy_change_pct']) for r in group),
                            hvac_energy_max_pct=max(float(r['hvac_energy_change_pct']) for r in group)))
    write_csv('oat-summary.csv',summary)
    # Derived equilibrium for the SAME existing cases, not another trajectory.
    # It applies the V1 algebra at the maximum steady-safe requested action.
    cfg=json.loads((ROOT/'experiments/platform_v1/config.json').read_text(encoding='utf-8'));steady=[]
    for r in rows:
        if r['mode']!='unfiltered' or r['variant'] not in ['baseline','thermal.ua_w_per_k_host=0']:continue
        t=copy.deepcopy(cfg['thermal'])
        if r['variant']!='baseline':t['ua_w_per_k_host']=0
        it=float(r['it_kw_mean']);fan=float(r['supply_fan_design_kw']);rating=float(r['coil_rating_kw'])
        k=rating/t['air_design_delta_k'];ua=t['ua_w_per_k_host']*167/1000;out=float(r['outdoor_c'])
        supply=min(float(r['supply_request_c']),float(r['steady_safe_supply_max_c']))
        if supply<18:raise ValueError('No equilibrium action for derived case')
        temp=(it+fan+ua*out+k*supply)/(k+ua);coil=k*(temp-supply)
        evap=supply-t['evaporator_approach_k'];cond=max(out+t['condenser_approach_k'],evap+t['minimum_lift_k'])
        cop=max(t['cop_min'],min(t['cop_max'],t['carnot_efficiency']*(evap+273.15)/(cond-evap)))
        pump=t['pump_pressure_pa']*coil/(t['water_cp_kj_kg_k']*t['water_delta_k']*t['water_density_kg_m3']*t['pump_efficiency']*1000)
        compressor=(coil+pump)/cop;speed=max(t['condenser_fan_min_speed'],min(1,(coil+pump)/rating))
        condenser=rating*t['condenser_fan_electric_fraction_of_coil_rating']*speed**3
        steady.append(dict(variant=r['variant'],load_fraction=r['load_fraction'],outdoor_c=out,
                           supply_request_c=r['supply_request_c'],derived_steady_safe_supply_c=supply,
                           steady_inlet_c=temp,cop=cop,it_kw=it,coil_kw=coil,
                           steady_hvac_kw=fan+pump+compressor+condenser,
                           derivation='algebra on existing batch case; not a new simulation'))
    write_csv('steady-derived.csv',steady)
    groups={}
    for r in rows:
        key=(r['variant'],r['outdoor_c'],r['supply_request_c'],r['mode'])
        groups.setdefault(key,[]).append(r)
    monotone_violations=[]
    for key,group in groups.items():
        group.sort(key=lambda r:float(r['load_fraction']))
        for first,last in zip(group,group[1:]):
            if float(last['total_kwh'])<float(first['total_kwh'])-1e-7:monotone_violations.append(key)
    verification=dict(saved_trajectories=len(rows),matrix_unique_complete=True,
                      original_csv_hashes_match=True,protected_and_executed_source_unchanged=True,
                      no_simulation_replay=True,postprocessing_only=True,
                      max_saved_electric_carbon_arithmetic_error=max(x[0] for x in errors),
                      max_whole_window_heat_residual_kj=max(x[1] for x in errors),
                      increasing_cofraction_load_total_energy_violations=len(monotone_violations),
                      interpretation='Arithmetic/mechanism checks, NOT real parameter calibration or historical algorithm savings')
    (EVIDENCE/'saved-verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
