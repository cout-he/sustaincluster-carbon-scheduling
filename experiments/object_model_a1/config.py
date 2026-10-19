"""A1 Train-only immutable logical scenarios; no physical-model constructor.

Observed CPU/GPU equivalents are fluid usage proxies, not reservations. The
official units do not recover discarded plan_* fields or placement constraints.
"""
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from experiments.e1.contracts import content_hash, file_hash

MAPPINGS = {'legacy_usage_x5_pressure_v1': 5., 'observed_usage_x1_pressure_v1': 1.}
BASE_SOURCE_SHA = 'a754c5ee4bd3922cecd6993d9abb6bfbacdda7e0'


def load_candidate(path):
    path=Path(path)
    data=json.loads(path.read_text(encoding='utf8'))
    if data.get('schema')!='a1-logical-candidate-v1':
        raise ValueError('Unsupported candidate schema')
    if data.get('mapping_version') not in MAPPINGS:
        raise ValueError('Unsupported or unavailable mapping')
    if data.get('power_model')!='disabled' or data.get('physical_carbon_output') is not False:
        raise ValueError('A1 power model must be disabled')
    if data.get('allowed_roles')!=['train'] or data.get('sla_version')!='arrival_plus_multiplier_times_original_duration_v1':
        raise ValueError('Only Train and the unchanged SLA definition are supported')
    declarations={'pressure_factors':[.75,1.,1.25],'sla_multipliers':[1.5,2.],
        'ci_version':'unchanged_frozen_train_previous_hour_left_hold_v1',
        'base_source_sha':BASE_SOURCE_SHA,'final_research_object_frozen':False}
    if any(data.get(key)!=value for key,value in declarations.items()):
        raise ValueError('Candidate execution declaration differs from the A1 protocol')
    cap=data.get('logical_capacity',{})
    if set(cap)!= {'cpu','gpu','mem'} or any(not math.isfinite(float(v)) or float(v)<=0 for v in cap.values()):
        raise ValueError('Positive finite CPU/GPU/MEM capacity required')
    data=dict(data,config_sha256=file_hash(path))
    return data


@dataclass(frozen=True)
class LogicalScenario:
    scenario_id: str
    role: str
    workload: object
    environment: object
    capacity: tuple
    identity: dict
    parameters: dict
    statistics: dict


def adapt_scenario(source,candidate):
    """Create a new logical descriptor; retain raw task fields and environment.

    Source must be a role-checked frozen V2 Train scenario with its exact old
    demand mapping. Failing this assertion stops execution, rather than silently
    remapping an already transformed workload. No save-to-V2/physical path exists.
    """
    if source.role!='train' or source.identity.get('role')!='train':
        raise ValueError('Only Train scenarios may be adapted')
    if candidate['mapping_version'] not in MAPPINGS:
        raise ValueError('Unknown mapping')
    pressure=float(source.identity['resource_pressure_factor'])
    factor=MAPPINGS[candidate['mapping_version']]
    for task in source.workload.tasks:
        raw=task.raw_fields
        expected=(5*pressure*float(raw[5])/100,5*pressure*float(raw[6])/100,5*pressure*float(raw[7]))
        if not all(math.isclose(actual,wanted,rel_tol=1e-10,abs_tol=1e-9)
                   for actual,wanted in zip((task.cores_req,task.gpu_req,task.mem_req),expected)):
            raise ValueError('Frozen mapping differs from raw fields; refuse reinterpretation')
    ratio=factor/5
    tasks=source.workload.tasks if ratio==1 else tuple(replace(t,
        cores_req=t.cores_req*ratio,gpu_req=t.gpu_req*ratio,mem_req=t.mem_req*ratio)
        for t in source.workload.tasks)
    workload=replace(source.workload,tasks=tasks,resource_multiplier=factor*pressure)
    capacity=tuple(float(candidate['logical_capacity'][key]) for key in ('cpu','gpu','mem'))
    identity=dict(source.identity,source_scenario_id=source.scenario_id,
        source_scenario_identity_sha256=content_hash(source.identity),
        source_workload_sha256=source.workload.sha256,
        source_environment_sha256=source.identity.get('environment_sha256'),
        candidate_id=candidate['candidate_id'],candidate_config_sha256=candidate['config_sha256'],
        mapping_version=candidate['mapping_version'],sla_version=candidate['sla_version'],
        derived_workload_sha256=workload.sha256,physical_model_enabled=False)
    return LogicalScenario('a1-'+candidate['candidate_id']+'-'+source.scenario_id,
        'train',workload,source.environment,capacity,identity,
        dict(source.parameters,a1_power_model='disabled'),dict(source.statistics,a1_derived=True))
