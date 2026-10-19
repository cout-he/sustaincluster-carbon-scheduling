"""Only role-checked frozen Train inputs; never unpickle the annual workload."""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

COLUMNS=('job_name','start_time','end_time','start_dt','duration_min','cpu_usage','gpu_wrk_util','avg_mem','avg_gpu_wrk_mem','bandwidth_gb','weekday_name','weekday_num')


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def stamp(value):
    d=datetime.fromisoformat(value)
    if d.tzinfo is None:raise ValueError('Timestamp must be timezone aware')
    return d.astimezone(timezone.utc)


@dataclass(frozen=True)
class Task:
    task_id: str
    arrival_minute: float
    duration_minutes: float
    demand: tuple
    source_job_name: str
    source_row: int
    source_index: int


@dataclass(frozen=True)
class Observation:
    minute: float
    ci_g_per_kwh: float
    outdoor_c: float
    observed_minute: float


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    tasks: tuple
    observations: tuple
    provenance: dict
    arrival_window_minutes: float


def causal_observations(samples,start,hours):
    origin=stamp(start)
    if origin.minute or origin.second:raise ValueError('V1 cohort must start on an hourly boundary')
    hourly={}
    for row in samples:
        when=stamp(row['time'])
        if when.minute or when.second:continue  # discard interpolated quarter hours
        minute=(when-origin).total_seconds()/60
        if not 0<=minute<hours*60:continue
        if minute in hourly:raise ValueError('Duplicate hourly environmental endpoint')
        ci=float(row['ci_g_per_kwh']);outdoor=float(row['temperature_c'])
        if not math.isfinite(ci) or ci<0 or not math.isfinite(outdoor):raise ValueError('Invalid environmental value')
        hourly[minute]=(ci,outdoor)
    out=[]
    for minute in range(0,int(hours*60),15):
        boundary=minute//60*60
        if boundary not in hourly:raise ValueError('Missing past hourly environment observation')
        ci,outdoor=hourly[boundary]
        out.append(Observation(float(minute),ci,outdoor,float(boundary)))
    return tuple(out)


def load_train(root,hours=72,expected=None):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    # This check MUST precede every payload read/hash.
    if manifest.get('role')!='train' or manifest.get('test_access') is not False or not manifest.get('scenario_id','').startswith('train-'):
        raise ValueError('Only explicitly identified Train scenarios are authorized')
    if manifest.get('schema')!='joint-v2-training-scenario-v1':raise ValueError('Unsupported source schema')
    if expected and file_sha(root/'manifest.json')!=expected['manifest_sha256']:raise ValueError('Frozen Train manifest changed')
    loaded={};hashes={}
    for name in ['workload.json','environment.json']:
        sha=file_sha(root/name)
        if sha!=manifest['files'][name] or (expected and sha!=expected[name.removesuffix('.json')+'_sha256']):
            raise ValueError('Train payload hash mismatch: '+name)
        doc=json.loads((root/name).read_text(encoding='utf-8'))
        if canonical_sha(doc['payload'])!=doc['sha256']:raise ValueError('Snapshot content checksum mismatch')
        loaded[name]=doc['payload'];hashes[name.removesuffix('.json')+'_sha256']=sha
    w=loaded['workload.json'];e=loaded['environment.json']
    if tuple(w['columns'])!=COLUMNS or w['resource_multiplier']!=5:raise ValueError('Require legacy x5, pressure 1 source before Observed mapping')
    origin=stamp(w['start']);end=stamp(w['end'])
    if (end-origin).total_seconds()!=86400:raise ValueError('Expected unchanged 24 h arrival cohort')
    tasks=[]
    for item in w['tasks']:
        raw=item['raw_fields'];arrival=(stamp(item['arrival_time'])-origin).total_seconds()/60
        if len(raw)!=12:raise ValueError('Task schema differs')
        demand=(float(raw[5])/100,float(raw[6])/100,float(raw[7]))
        duration=float(raw[4])
        if not 0<=arrival<1440 or arrival%15 or not math.isfinite(duration) or duration<=0 or any(not math.isfinite(x) or x<0 for x in demand):
            raise ValueError('Invalid original task')
        if abs(duration-item['duration'])>1e-9 or any(not math.isclose(float(item[k]),5*v,rel_tol=1e-10,abs_tol=1e-9) for k,v in zip(['cores_req','gpu_req','mem_req'],demand)):
            raise ValueError('Frozen input mapping differs from raw usage')
        tasks.append(Task(item['task_id'],arrival,duration,demand,str(raw[0]),item['source_row'],item['source_task_index']))
    if len({t.task_id for t in tasks})!=len(tasks):raise ValueError('Duplicate stable task ID')
    observations=causal_observations(e['samples'],w['start'],hours)
    provenance=dict(hashes,manifest_sha256=file_sha(root/'manifest.json'),
                    annual_workload_sha256=w['source_sha256'],carbon_source_sha256=e['carbon_source_sha256'],
                    weather_source_sha256=e['weather_source_sha256'],
                    weather_source_timezone=e['weather_source_timezone'],
                    weather_source_utc_offset_seconds=e['weather_source_utc_offset_seconds'],
                    weather_source_time_basis=e['weather_time_basis'],
                    environment_transform='hour_boundary_observations_previous_hold_both_v1',
                    role='train',start=w['start'],arrival_end=w['end'])
    return Scenario(manifest['scenario_id'],tuple(tasks),observations,provenance,1440.)
