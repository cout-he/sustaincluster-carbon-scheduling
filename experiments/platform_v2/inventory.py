"""Counts and quotas, balanced memory sizing and whole reference chiller modules."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
from experiments.platform_v1.inventory import Inventory as BaseInventory


def load_config(path=None):
    c=json.loads(Path(path or Path(__file__).with_name('config.json')).read_text(encoding='utf-8'))
    if c['schema']!='single-dc-platform-v2.0' or c['evaluation']['step_minutes']!=15:raise ValueError('Unsupported platform')
    for part in ['cpu','gpu']:
        if not 0<=c['power'][part+'_idle_w']<c['power'][part+'_reference_full_w']:raise ValueError('Invalid power endpoints')
    if not 0<c['power']['psu_efficiency']<=1 or c['power']['dimm_base_w']<0:raise ValueError('Invalid power')
    if c['host']['dimm_count']!=16:raise ValueError('Preserve two sockets and all eight channels each')
    if not 0<=c['chiller']['package_condenser_fan_share']<1:raise ValueError('Invalid package split')
    if c['chiller']['plr_min']<=0 or c['chiller']['plr_max']!=1:raise ValueError('Invalid PLR boundary')
    for group,keys in [('room',['height_m','effective_slab_m','envelope_u_w_m2_k','equipment_coupling_fraction']),
                       ('thermal',['substep_seconds','water_delta_k','fan_efficiency','pump_efficiency'])]:
        if any(not math.isfinite(c[group][k]) or c[group][k]<=0 for k in keys):raise ValueError('Invalid physical parameter')
    return c


@dataclass(frozen=True)
class Inventory(BaseInventory):
    dimm_gib: int
    psu_objects: int
    chiller_modules: int


def build_inventory(c,name):
    cap=tuple(c['capacities'][name]);h=c['host'];p=c['power'];cores=h['sockets']*h['cores_per_socket']
    if cap[1]%h['gpu_per_host']:raise ValueError('GPU count must map to four-card hosts')
    hosts=max(math.ceil(cap[0]/(cores-h['reserved_cores'])),math.ceil(cap[1]/h['gpu_per_host']))
    gpu_hosts=int(cap[1]/h['gpu_per_host']);reserved=hosts*h['reserved_mem_gib']*2**30/1e9
    choice=next((g for g in h['supported_dimm_gib'] if hosts*h['dimm_count']*g*2**30/1e9-reserved>=cap[2]),None)
    if choice is None:raise ValueError('No supported balanced memory configuration covers quota')
    mem=hosts*h['dimm_count']*choice*2**30/1e9;racks=math.ceil(hosts/h['hosts_per_rack'])
    dc=hosts*h['sockets']*p['cpu_reference_full_w']+cap[1]*p['gpu_reference_full_w']+hosts*h['dimm_count']*p['dimm_base_w']+hosts*p['host_other_w']
    design=dc/p['psu_efficiency']/1000+racks*p['rack_switch_w']/1000
    modules=math.ceil(c['thermal']['coil_sizing_ratio']*design/c['chiller']['unit_reference_capacity_kw_th'])
    cpu_host_dc=h['sockets']*p['cpu_reference_full_w']+h['dimm_count']*p['dimm_base_w']+p['host_other_w']
    if cpu_host_dc>h['cpu_host_psu_output_w'] or cpu_host_dc+h['gpu_per_host']*p['gpu_reference_full_w']>h['gpu_host_psu_output_w']:
        raise ValueError('One redundant PSU cannot support design DC load')
    return Inventory(name,cap,1.,hosts,gpu_hosts,hosts-gpu_hosts,int(cap[1]),hosts*cores,
                     hosts*h['reserved_cores'],hosts*(cores-h['reserved_cores'])-cap[0],mem,reserved,
                     mem-reserved-cap[2],racks,hosts*h['sockets'],hosts*h['dimm_count'],choice,
                     hosts*h['psus_per_host'],modules)
