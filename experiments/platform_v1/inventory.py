"""Installed equipment, schedulable quotas and overhead use one versioned source."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


def load_config(path=None):
    cfg=json.loads(Path(path or Path(__file__).with_name('config.json')).read_text(encoding='utf-8'))
    if cfg['schema']!='single-dc-platform-v1.0' or cfg['evaluation']['step_minutes']!=15:
        raise ValueError('Unsupported platform or time step')
    p=cfg['power']
    for component in ['cpu','gpu','dimm']:
        if not 0<=p[component+'_idle_w']<p[component+'_reference_full_w']:
            raise ValueError('Invalid component endpoints')
    if not 0<p['psu_efficiency']<=1:raise ValueError('Invalid PSU efficiency')
    t=cfg['thermal']
    for key in ['heat_capacity_kj_per_k_host','coil_sizing_ratio','air_design_delta_k','air_density_kg_m3','air_cp_kj_kg_k','fan_efficiency','water_cp_kj_kg_k','water_delta_k','pump_efficiency','substep_seconds']:
        if not math.isfinite(t[key]) or t[key]<=0:raise ValueError('Invalid thermal coefficient: '+key)
    return cfg


@dataclass(frozen=True)
class Inventory:
    name: str
    capacity: tuple
    mapping_factor: float
    hosts: int
    gpu_hosts: int
    cpu_only_hosts: int
    gpus: int
    installed_cores: int
    reserved_cores: int
    quota_excluded_cores: int
    installed_mem_gb: float
    reserved_mem_gb: float
    quota_excluded_mem_gb: float
    racks: int
    cpu_objects: int
    dimm_objects: int

    def to_dict(self):return asdict(self)


def build_inventory(cfg,name):
    legacy=name.endswith('-legacy'); base=name.removesuffix('-legacy')
    factor=5 if legacy else 1
    cap=tuple(factor*x for x in cfg['capacities'][base])
    h=cfg['host']; cores=h['sockets']*h['cores_per_socket']
    hosts=max(math.ceil(cap[0]/(cores-h['reserved_cores'])),math.ceil(cap[1]/h['gpu_per_host']))
    if cap[1]%h['gpu_per_host']:raise ValueError('GPU logical capacity must map to whole four-card hosts')
    gpu_hosts=int(cap[1]/h['gpu_per_host'])
    mem=hosts*h['dimm_count']*h['dimm_gib']*2**30/1e9
    reserved_mem=hosts*h['reserved_mem_gib']*2**30/1e9
    if mem-reserved_mem<cap[2]:raise ValueError('Memory inventory cannot supply logical quota')
    return Inventory(name,cap,factor,hosts,gpu_hosts,hosts-gpu_hosts,int(cap[1]),hosts*cores,
                     hosts*h['reserved_cores'],hosts*(cores-h['reserved_cores'])-cap[0],
                     mem,reserved_mem,mem-reserved_mem-cap[2],math.ceil(hosts/h['hosts_per_rack']),
                     hosts*h['sockets'],hosts*h['dimm_count'])
