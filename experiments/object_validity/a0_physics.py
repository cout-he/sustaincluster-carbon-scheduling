"""A0 deterministic physical probes, without workload, CI or weather payloads.

This module calls the unchanged V2 JointPowerModel.  All loads and weather are
hand specified diagnostic inputs, never observations or calibrated equipment.
No replacement power equation, trained model or optimization policy is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
import subprocess
import time

import pandas as pd

from experiments.e1.environment import EnvironmentalSample
from experiments.joint.power import JointPowerModel


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_FILES = (
    'configs/dcs/dc_config.json', 'utils/dc_config_reader.py',
    'utils/make_envs.py', 'envs/sustaindc/datacenter_model.py',
    'envs/sustaindc/dc_gym.py', 'experiments/joint/power.py',
    'experiments/e1/configs/e1_5b_ppo.json',
    'experiments/joint_v2/scenario_config.yaml',
    'experiments/joint_v2/training_config.yaml',
)
TEMPERATURES = (-5, 5, 15, 25, 35, 40)
SETPOINTS = (18, 19, 20, 21, 22)


def source_hashes(root=ROOT):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in PRODUCTION_FILES}


def effective_mapping(model):
    """Count actual constructed objects separately from input configuration."""
    d = model.dc_env.DC_Config
    racks = model.dc_env.dc.racks_list
    server_objects = sum(len(r.server_list) for r in racks)
    return dict(
        logical_cpu_cores=d.total_cores, logical_gpus=d.total_gpus,
        logical_memory_gb=d.total_mem_GB,
        declared_cpu_server_count=d.total_servers_cpu,
        declared_gpu_server_count=d.total_servers_gpu,
        declared_combined_server_count=d.total_servers,
        rack_count=len(racks), configured_cpu_cores_per_rack=d.CPUS_PER_RACK,
        configured_gpus_per_rack=d.GPUS_PER_RACK,
        declared_cpu_configs=sum(map(len, d.RACK_CPU_CONFIG)),
        declared_gpu_configs=sum(map(len, d.RACK_GPU_CONFIG)),
        instantiated_cpu_server_objects=server_objects,
        represented_cpu_cores=server_objects * d.CORES_PER_SERVER,
        instantiated_gpu_power_objects=sum(r.num_gpus for r in racks),
        gpu_power_object_to_logical_capacity_ratio=(
            sum(r.num_gpus for r in racks) / d.total_gpus),
        configured_gpus_per_server_for_sizing=d.GPUS_PER_SERVER,
        actual_gpu_objects_per_server=1,
        max_w_per_rack=d.MAX_W_PER_RACK,
        nominal_datacenter_capacity_mw=d.datacenter_capacity_mw,
        effective_ct_fan_reference_w=float(d.CT_FAN_REF_P),
        effective_ct_reference_airflow_m3_per_s=float(d.CT_REFRENCE_AIR_FLOW_RATE),
        loaded_chiller_cop=float(d.CHILLER_COP),
        active_chiller_rated_cop_literal=3.0,
        supply_approach_c=float(model.supply_approach_c),
        return_approach_c=float(d.RACK_RETURN_APPROACH_TEMP_LIST[0]),
        memory_background_w=.07 * d.total_mem_GB,
        rack_inlet_limit_c=float(model.rack_inlet_limit_c),
    )


def probe(model, cpu, gpu, mem, setpoint, ambient, family):
    """One algebraic, not elapsed-time, original-equation measurement."""
    model.set_setpoint(setpoint)
    sample = EnvironmentalSample(pd.Timestamp('2023-01-01T00:00:00Z'),
                                 100.0, float(ambient), .9 * float(ambient))
    reading = model.measure(cpu, gpu, mem, sample)
    env = model.dc_env
    row = dict(
        diagnostic_id=f'{family}:cpu={cpu:g}:gpu={gpu:g}:mem={mem:g}:'
                      f'sp={setpoint:g}:ambient={ambient:g}',
        original_scenario_id='NONE_HAND_SPECIFIED_PHYSICS_PROBE',
        family=family, cpu_utilization=cpu, gpu_utilization=gpu,
        mem_utilization=mem, setpoint_c=setpoint,
        ambient_dry_bulb_c=ambient, approximate_wet_bulb_c=.9 * ambient,
        cpu_kw=float(sum(env.rackwise_cpu_pwr) / 1000),
        gpu_kw=float(sum(env.rackwise_gpu_pwr) / 1000),
        server_fan_kw=float(sum(env.rackwise_itfan_pwr) / 1000),
        memory_background_kw=.07 * env.dc_memory_GB / 1000,
        **{key + '_kw': value for key, value in reading.components_kw.items()},
        hvac_kw=reading.hvac_kw, system_kw=reading.total_kw,
        crac_thermal_kw=reading.crac_thermal_kw,
        legacy_hvac_kw=reading.legacy_hvac_kw,
        legacy_system_kw=reading.legacy_total_kw,
        rack_inlet_max_c=reading.rack_inlet_max_c,
        rack_outlet_max_c=reading.rack_outlet_max_c,
        rack_inlet_margin_c=model.rack_inlet_limit_c - reading.rack_inlet_max_c,
        water_usage_litres_per_15min=float(env.water_usage),
    )
    return row


def parameter_rows(model):
    """Trace JSON declarations, override points and actually constructed values."""
    d = model.dc_env.DC_Config
    raw = json.loads((ROOT / 'configs/dcs/dc_config.json').read_text())
    rows = []

    def add(name, value, unit, evidence, status='runtime_confirmed', note=''):
        rows.append(dict(parameter=name, value=value, unit=unit,
                         evidence=evidence, status=status, interpretation=note))

    for key, value in effective_mapping(model).items():
        unit = ('W' if key.endswith('_w') else 'MW' if key.endswith('_mw')
                else 'degrees_C' if key.endswith('_c') else 'GB' if key.endswith('_gb')
                else 'm3/s' if 'airflow' in key else 'count_or_ratio')
        add(key, value, unit, 'DC_Config and actual JointPowerModel object inspection')
    add('json_declared_rack_count',
        raw['data_center_configuration']['NUM_ROWS'] *
        raw['data_center_configuration']['NUM_RACKS_PER_ROW'], 'racks',
        'configs/dcs/dc_config.json', note='Overridden by ceil(server_count/25): 32.')
    for k in ('CT_FAN_REF_P', 'CT_REFRENCE_AIR_FLOW_RATE', 'CHILLER_COP_BASE'):
        add('json_' + k, raw['hvac_configuration'][k],
            'W' if k == 'CT_FAN_REF_P' else 'm3/s' if 'FLOW' in k else 'dimensionless',
            'configs/dcs/dc_config.json', note=(
                'CT sizing overwrites fan reference with a thermal load and feeds it '
                'to both fan and chiller.' if k != 'CHILLER_COP_BASE' else
                'Loaded COP=5 is not passed into calculate_chiller_power; active rated_cop=3.'))
    for key in ('C_AIR', 'RHO_AIR', 'CRAC_FAN_REF_P',
                'CRAC_SUPPLY_AIR_FLOW_RATE_pu', 'CRAC_REFRENCE_AIR_FLOW_RATE_pu',
                'CW_PRESSURE_DROP', 'CW_WATER_FLOW_RATE', 'CW_PUMP_EFFICIENCY',
                'CT_PRESSURE_DROP', 'CT_WATER_FLOW_RATE', 'CT_PUMP_EFFICIENCY'):
        add(key, getattr(d, key), 'as_named_in_source',
            'configs/dcs/dc_config.json -> DC_Config -> calculate_HVAC_power')
    rack = model.dc_env.dc.racks_list[0]
    for key in ('m_coefficient', 'c_coefficient', 'it_slope'):
        add('vectorized_server_fan_' + key, getattr(rack, key), 'dimensionless',
            'envs/sustaindc/datacenter_model.py:Rack.server_and_fan_init',
            note='Hand-tuned multipliers in active vectorized equation; not empirically validated here.')
    add('vectorized_cpu_extra_temperature_slope', .05, '1/degrees_C',
        'envs/sustaindc/datacenter_model.py:Rack.compute_instantaneous_pwr_vecd',
        'source_confirmed', 'Added to fitted CPU slope; causes idle CPU power to exceed declared full-load rating.')
    add('fixed_crac_fan_electric_kw',
        d.CRAC_FAN_REF_P * (d.CRAC_SUPPLY_AIR_FLOW_RATE_pu /
                            d.CRAC_REFRENCE_AIR_FLOW_RATE_pu) ** 3 / 1000,
        'kW', 'calculate_HVAC_power', note='Does not scale with rack count or actual load.')
    add('fixed_each_water_pump_electric_kw',
        d.CW_PRESSURE_DROP * d.CW_WATER_FLOW_RATE / d.CW_PUMP_EFFICIENCY / 1000,
        'kW', 'calculate_HVAC_power', note='Same configured parameters are reused for chilled-water and tower pumps.')
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output, max_seconds=120, source_sha=None):
    began = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    before = source_hashes()
    if source_sha is None:
        source_sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if not re.fullmatch('[0-9a-f]{40}', source_sha):
        raise ValueError('source SHA must be a full 40-character Git commit SHA')
    cfg = json.loads((ROOT / 'experiments/e1/configs/e1_5b_ppo.json').read_text())['datacenter']
    model = JointPowerModel(cfg, ROOT / 'configs/dcs/dc_config.json')
    rows, complete = [], True
    mapping = effective_mapping(model)
    parameters = parameter_rows(model)
    try:
        families = (
            ('factorial', itertools.product((0, .5, 1), repeat=3)),
            ('diagonal', ((i / 20, i / 20, i / 20) for i in range(21))),
        )
        for family, loads in families:
            for (cpu, gpu, mem), ambient, setpoint in itertools.product(
                    loads, TEMPERATURES, SETPOINTS):
                if time.monotonic() - began >= max_seconds:
                    complete = False
                    break
                rows.append(probe(model, cpu, gpu, mem, setpoint, ambient, family))
            if not complete:
                break
        # Explicit background references at the same weather.  No CI series used.
        idle = {(r['setpoint_c'], r['ambient_dry_bulb_c']): r['system_kw']
                for r in rows if r['family'] == 'factorial' and
                (r['cpu_utilization'], r['gpu_utilization'], r['mem_utilization']) == (0, 0, 0)}
        for row in rows:
            key = (row['setpoint_c'], row['ambient_dry_bulb_c'])
            row['same_setpoint_idle_background_kw'] = idle.get(key, '')
            row['same_setpoint_idle_fraction_of_total'] = (
                idle[key] / row['system_kw'] if key in idle else '')
            ref = idle.get((18, row['ambient_dry_bulb_c']))
            row['idle18_reference_kw'] = ref if ref is not None else ''
            row['increment_above_idle18_kw'] = row['system_kw'] - ref if ref is not None else ''
            row['source_sha'] = source_sha
        write_csv(output / 'physical-probes.csv', rows)
        write_csv(output / 'effective-physical-configuration.csv', parameters)
        comparisons = []
        grouped = {}
        for row in rows:
            key = (row['family'], row['cpu_utilization'], row['gpu_utilization'],
                   row['mem_utilization'], row['ambient_dry_bulb_c'])
            grouped.setdefault(key, {})[row['setpoint_c']] = row
        for key, by_sp in grouped.items():
            if set(by_sp) != set(SETPOINTS):
                continue
            cold, warm = by_sp[18], by_sp[22]
            best = min(by_sp.values(), key=lambda x: x['system_kw'])
            comparisons.append(dict(
                family=key[0], cpu_utilization=key[1], gpu_utilization=key[2],
                mem_utilization=key[3], ambient_dry_bulb_c=key[4],
                system18_kw=cold['system_kw'], system22_kw=warm['system_kw'],
                delta_22_minus_18_kw=warm['system_kw'] - cold['system_kw'],
                delta_it_22_minus_18_kw=warm['it_kw'] - cold['it_kw'],
                delta_ct_fan_22_minus_18_kw=warm['ct_fan_kw'] - cold['ct_fan_kw'],
                delta_compressor_22_minus_18_kw=warm['compressor_kw'] - cold['compressor_kw'],
                best_setpoint_c=best['setpoint_c'], best_system_kw=best['system_kw'],
                saving_vs_22_percent=100 * (warm['system_kw'] - best['system_kw']) / warm['system_kw'],
                source_sha=source_sha,
            ))
        write_csv(output / 'setpoint-comparisons.csv', comparisons)
        crossings = []
        for ambient in TEMPERATURES:
            diag = sorted((r for r in comparisons if r['family'] == 'diagonal'
                           and r['ambient_dry_bulb_c'] == ambient),
                          key=lambda r: r['cpu_utilization'])
            brackets = []
            for lo, hi in zip(diag, diag[1:]):
                if lo['delta_22_minus_18_kw'] * hi['delta_22_minus_18_kw'] < 0:
                    brackets.append([lo['cpu_utilization'], hi['cpu_utilization']])
            crossings.append(dict(ambient_dry_bulb_c=ambient,
                                  diagonal_18_22_sign_change_load_brackets=brackets))
        cpu_endpoints = []
        for sp in (18, 22):
            rack = model.dc_env.dc.racks_list[0]
            model.set_setpoint(sp)
            for util in (0, 1):
                cpu_w, _, _ = rack.compute_instantaneous_pwr_vecd(
                    sp + model.supply_approach_c, util * 100, 0, 0)
                cpu_endpoints.append(dict(setpoint_c=sp, cpu_utilization=util,
                    cpu_per_server_w=float(cpu_w / rack.num_servers),
                    declared_cpu_idle_w=float(rack.idle_pwr[0]),
                    declared_cpu_full_load_w=float(rack.full_load_pwr[0]),
                    ratio_to_declared_full_load=float(cpu_w / sum(rack.full_load_pwr))))
        summary = dict(
            schema='a0-readonly-physical-diagnosis-v1', source_sha=source_sha,
            production_source_hashes=before,
            production_sources_unchanged=source_hashes() == before,
            original_scenario_id=None, data_identity='HAND_SPECIFIED_NO_TASK_CI_WEATHER_PAYLOAD',
            workload_ci_weather_payload_accessed=False,
            probe_semantics='algebraic stationary point, no task simulation, no time integration',
            synthetic_weather='dry bulb {-5,5,15,25,35,40} C; wet bulb = 0.9*dry bulb, inherited approximation, not psychrometric validation; wet bulb affects water estimate, not electrical equations',
            input_loads='factorial {0,.5,1}^3 plus diagonal {0,.05,...,1}^3',
            setpoints_c=list(SETPOINTS), completed=complete, rows=len(rows),
            expected_rows=1440, elapsed_seconds=time.monotonic() - began,
            configured_mapping=mapping,
            cpu_server_endpoints=cpu_endpoints,
            inlet_c_min=min((r['rack_inlet_max_c'] for r in rows), default=None),
            inlet_c_max=max((r['rack_inlet_max_c'] for r in rows), default=None),
            inlet_violation_count=sum(r['rack_inlet_margin_c'] < 0 for r in rows),
            system_kw_min=min((r['system_kw'] for r in rows), default=None),
            system_kw_max=max((r['system_kw'] for r in rows), default=None),
            above_nominal_1mw_points=sum(r['system_kw'] > 1000 for r in rows),
            crac_thermal_to_it_power_ratio_min=min(
                (r['crac_thermal_kw'] / r['it_kw'] for r in rows), default=None),
            crac_thermal_to_it_power_ratio_max=max(
                (r['crac_thermal_kw'] / r['it_kw'] for r in rows), default=None),
            same_setpoint_idle_fraction_min=min(
                (r['same_setpoint_idle_fraction_of_total'] for r in rows
                 if r['same_setpoint_idle_fraction_of_total'] != ''), default=None),
            same_setpoint_idle_fraction_max=max(
                (r['same_setpoint_idle_fraction_of_total'] for r in rows
                 if r['same_setpoint_idle_fraction_of_total'] != ''), default=None),
            probes_with_total_below_same_setpoint_idle=sum(
                r['same_setpoint_idle_fraction_of_total'] != '' and
                r['same_setpoint_idle_fraction_of_total'] > 1.000000001 for r in rows),
            probes_with_negative_increment_above_idle18=sum(
                r['increment_above_idle18_kw'] != '' and
                r['increment_above_idle18_kw'] < 0 for r in rows),
            illustrative_25c_diagonal_points=[
                {k: r[k] for k in ('cpu_utilization', 'gpu_utilization',
                 'mem_utilization', 'setpoint_c', 'cpu_kw', 'gpu_kw',
                 'server_fan_kw', 'memory_background_kw', 'it_kw',
                 'crac_thermal_kw', 'ct_fan_kw', 'compressor_kw', 'system_kw',
                 'same_setpoint_idle_fraction_of_total')}
                for r in rows if r['family'] == 'factorial' and
                r['ambient_dry_bulb_c'] == 25 and r['setpoint_c'] in (18, 22) and
                r['cpu_utilization'] == r['gpu_utilization'] == r['mem_utilization']],
            setpoint_22_wins_over_18_points=sum(r['delta_22_minus_18_kw'] < 0 for r in comparisons),
            setpoint_18_wins_over_22_points=sum(r['delta_22_minus_18_kw'] > 0 for r in comparisons),
            setpoint_comparison_count=len(comparisons), diagonal_crossings=crossings,
            evidence_interpretation=[
                'Executable and internally summed does not mean physically validated.',
                '1 MW comes from 6 W/core + 500 W/logical GPU + 2.5 W/GB; it is passed as metadata, not a cap on total power.',
                'CT sizing returns a thermal cooling load; make_dc_env stores that W_thermal value in CT_FAN_REF_P, later multiplying it as fan W_electric.',
                'Active vectorized CPU slope adds 0.05 per C to the fitted slope; idle and full-load exceed declared 384 W/server rating.',
                'CPU/GPU power is computed uniformly from rack 0 then broadcast to 32 racks; no per-device placement or thermal hotspots.',
                'Memory is fixed 0.07 W/GB; memory utilization affects server fan response only, not DRAM electric power.',
                'The 28 C inlet ceiling is satisfied structurally by setpoint + fixed 5.15 C approach; it is independent of weather/load here.',
                'Both pumps are fixed 379.31 W; CRAC fan is fixed about 32.4 W for the whole DC. No independent equipment calibration was performed.',
                'Setpoint crossings are outputs of these suspect equations and do not validate a real cooperative-control benefit.',
                'CRAC thermal removal is not constrained to IT heat and ranges 1.27 to 3.71 times IT power in this scan; no calibrated heat balance was established.',
                'Increasing memory-only utilization can reduce system electric power because memory heat is fixed while fan airflow changes outlet/CRAC load; negative incremental power is a model output, not evidence that memory work saves electricity.',
            ],
            units=dict(power='kW_electric except crac_thermal_kw=kW_thermal',
                       temperature='degrees C', utilization='fraction [0,1]',
                       water='litres/15 minutes; approximate and not validated'),
        )
        (output / 'physics-summary.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return summary
    finally:
        model.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-seconds', type=float, default=120)
    parser.add_argument('--source-sha', help='Full baseline commit, for Windows worktrees read under WSL; all production file hashes are also recorded.')
    args = parser.parse_args()
    summary = run(args.output, args.max_seconds, args.source_sha)
    print(json.dumps({k: summary[k] for k in ('completed', 'rows', 'elapsed_seconds',
                                            'source_sha', 'production_sources_unchanged')}))


if __name__ == '__main__':
    main()
