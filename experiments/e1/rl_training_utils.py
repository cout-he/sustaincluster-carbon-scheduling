"""Immutable Train/Validation inputs for PPO; Test is coverage-only in E1-5B."""
import argparse
import copy
import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from .cars_experiment import FIXED_PARAMETERS
from .config import ROOT, resolve_path
from .contracts import content_hash, file_hash, utc_time, write_json
from .power import UpstreamPowerModel
from .rl_env import CarbonSchedulingEnv, ObservationNormalization, RewardConfig
from .selection import TemporalEnvironment, calibrate_threshold, load_temporal_environment
from .stress import stress_workload
from .workload import FrozenWorkload, load_workload

VERSION='e1-5b-input-bundle-v1'
SPLIT_WINDOWS={
    'train':('2023-05-01T05:00:00Z','2023-05-03T05:00:00Z','2023-06-02T05:00:00Z'),
    'validation':('2023-07-08T05:00:00Z','2023-07-10T05:00:00Z','2023-08-09T05:00:00Z')}
PAYLOAD_FILES={'workload.json','environment.json','dc_config.json','config.json','stress.json','parameters.json'}
APPROVED_SOURCE_SHA256={
    'workload':'3ba8a8e0067288f8d2542752a292f7da04cb79305689ba145cc6060e96cb7f2e',
    'carbon':'44bafab775c6b0d4762b1dc971eb293cfea154f80799a5230f9268d32e1ad55b',
    'weather':'090bddc8a5f4e9edb5f3f193c6c147ed03841337b4bb17c7740f3b63a03cdaf5',
    'physical_model':'bbe439176605512523a3a0ecd2bd22f18fdb9abf6ae6e782217157beedbf705b'}

@dataclass(frozen=True)
class ExperimentBundle:
    split: str
    workload: object
    environment: object
    capacity: tuple
    power_factory: object
    parameters: dict
    stress_seed: int
    factor: float
    identity: dict
    reward_config: object=None
    normalization: object=None

def _require_split(split,allowed=None):
    if split=='test' or allowed=='test':raise ValueError('Test is forbidden in E1-5B training/evaluation')
    if split not in SPLIT_WINDOWS or (allowed is not None and split!=allowed):
        raise ValueError('Input split does not match allowed split: '+str(allowed))

def _bounds(split,workload,environment):
    start,end,cap=map(utc_time,SPLIT_WINDOWS[split])
    if (workload.start,workload.end,environment.start,environment.end)!=(start,end,start,cap):
        raise ValueError('Frozen payload UTC bounds do not match declared split')

def _fixed_config(cfg):
    if cfg.get('schema')!='e1-5b-ppo-v0' or cfg.get('test_evaluation') is not False:
        raise ValueError('PPO-v0 schema or Test rule differs')
    if cfg.get('threshold_g_per_kwh')!=161.075:raise ValueError('April threshold differs')
    if cfg.get('sla_multiplier')!=1.5:raise ValueError('SLA multiplier differs from frozen protocol')
    if cfg.get('resource_multiplier')!=5 or cfg.get('trace_year')!=2020:
        raise ValueError('Task resource or trace mapping differs')
    if RewardConfig(**cfg['reward'])!=RewardConfig() or ObservationNormalization(**cfg['normalization'])!=ObservationNormalization():
        raise ValueError('PPO-v0 reward or normalization differs')
    expected=dict(dc_id=1,location='US-CAL-CISO',total_cores=50000,total_gpus=1000,total_mem=80000,hvac_setpoint_c=22)
    if cfg['datacenter']!=expected:raise ValueError('Frozen datacenter differs')
    if cfg['source_sha256']!=APPROVED_SOURCE_SHA256:raise ValueError('Approved raw/physical source identities differ')

def _parameters(cfg,split,stress_seed):
    start,end,_=SPLIT_WINDOWS[split]
    params=copy.deepcopy(FIXED_PARAMETERS)
    params.update(threshold_g_per_kwh=161.075,arrival_start=utc_time(start).isoformat(),
                  arrival_end=utc_time(end).isoformat(),stress_seed=stress_seed)
    return params

def freeze_bundle(config,split,factor,stress_seed,output_dir):
    """Generate only the authorised split, verifying all raw source identities."""
    _require_split(split)
    if config.get('test_evaluation') is not False:raise ValueError('Test evaluation must remain disabled')
    if float(config['threshold_g_per_kwh'])!=161.075:raise ValueError('Fixed April CI threshold changed')
    if tuple(map(utc_time,config['windows'][split]))!=tuple(map(utc_time,SPLIT_WINDOWS[split])):
        raise ValueError('Configured split bounds changed')
    if stress_seed not in config['stress_seeds'][split]:raise ValueError('Unreserved stress seed')
    if float(factor) not in config['precheck_factors']:raise ValueError('Unregistered pressure factor')
    out=Path(output_dir)
    if out.exists():raise FileExistsError('Refusing to overwrite a frozen bundle')
    _fixed_config(config)
    paths={k:resolve_path(v) for k,v in config['paths'].items()}
    for name,sha in config['source_sha256'].items():
        if file_hash(paths[name])!=sha:raise ValueError('Raw source hash mismatch: '+name)
    if calibrate_threshold(pd.read_csv(paths['carbon']),*config['threshold_calibration'])!=161.075:
        raise ValueError('April calibration identity differs')
    start,end,cap=SPLIT_WINDOWS[split]
    base=load_workload(paths['workload'],start,end,config['resource_multiplier'],
                       config['sla_multiplier'],1,config['trace_year'])
    expected_tasks={'train':5542,'validation':8708}[split]
    if len(base.tasks)!=expected_tasks:raise ValueError('Frozen base cohort count differs from coverage audit')
    dc=config['datacenter']
    capacity=tuple(dc[k] for k in ('total_cores','total_gpus','total_mem'))
    if capacity!=(50000,1000,80000) or dc['hvac_setpoint_c']!=22:
        raise ValueError('Hardware or setpoint differs from E1 physical model')
    workload,stress=stress_workload(base,factor,stress_seed)
    environment=load_temporal_environment(paths['carbon'],paths['weather'],start,cap)
    _bounds(split,workload,environment)
    parameters=_parameters(config,split,stress_seed)
    out.mkdir(parents=True)
    workload.save(out/'workload.json');environment.save(out/'environment.json')
    shutil.copyfile(paths['physical_model'],out/'dc_config.json')
    write_json(out/'config.json',config);write_json(out/'stress.json',stress)
    write_json(out/'parameters.json',parameters)
    manifest=dict(schema=VERSION,split=split,factor=float(factor),stress_seed=stress_seed,
      tasks=len(workload.tasks),base_tasks=len(base.tasks),capacity=list(capacity),
      workload_sha256=workload.sha256,environment_sha256=environment.sha256,
      files={name:file_hash(out/name) for name in sorted(PAYLOAD_FILES)},
      config_sha256=content_hash(config),raw_source_sha256={k:file_hash(v) for k,v in paths.items()},
      threshold_g_per_kwh=161.075,test_evaluation=False)
    write_json(out/'bundle.json',manifest)
    return manifest

def load_bundle(path,*,allowed_split):
    """Reject Test/wrong roles before reading snapshots, then bind bytes and UTC."""
    root=Path(path)
    manifest=json.loads((root/'bundle.json').read_text(encoding='utf8'))
    _require_split(manifest.get('split'),allowed_split)
    if manifest.get('schema')!=VERSION or manifest.get('test_evaluation') is not False:
        raise ValueError('Frozen bundle schema or Test guard differs')
    if set(manifest['files'])!=PAYLOAD_FILES:raise ValueError('Frozen payload inventory differs')
    for name,sha in manifest['files'].items():
        if file_hash(root/name)!=sha:raise ValueError('Frozen payload hash mismatch: '+name)
    cfg=json.loads((root/'config.json').read_text(encoding='utf8'))
    if content_hash(cfg)!=manifest['config_sha256']:raise ValueError('Config identity mismatch')
    _fixed_config(cfg)
    workload=FrozenWorkload.load(root/'workload.json');environment=TemporalEnvironment.load(root/'environment.json')
    _bounds(manifest['split'],workload,environment)
    if workload.sha256!=manifest['workload_sha256'] or environment.sha256!=manifest['environment_sha256']:
        raise ValueError('Frozen input identity mismatch')
    if len(workload.tasks)!=manifest['tasks'] or cfg['test_evaluation'] is not False:
        raise ValueError('Frozen task count or Test guard differs')
    parameters=json.loads((root/'parameters.json').read_text(encoding='utf8'))
    if parameters!=_parameters(cfg,manifest['split'],manifest['stress_seed']):
        raise ValueError('Frozen risk/policy parameters differ')
    if tuple(manifest['capacity'])!=(50000,1000,80000):raise ValueError('Frozen capacity differs')
    if workload.resource_multiplier!=5 or workload.sla_multiplier!=1.5 or workload.trace_year!=2020:
        raise ValueError('Frozen task mapping or SLA differs')
    if file_hash(root/'dc_config.json')!=APPROVED_SOURCE_SHA256['physical_model']:
        raise ValueError('Frozen physical configuration differs')
    if (workload.source_sha256,environment.carbon_source_sha256,environment.weather_source_sha256)!=tuple(
           APPROVED_SOURCE_SHA256[k] for k in ('workload','carbon','weather')):
        raise ValueError('Frozen source identities differ')
    identity=dict(manifest,bundle_manifest_sha256=file_hash(root/'bundle.json'))
    factory=lambda:UpstreamPowerModel(cfg['datacenter'],root/'dc_config.json')
    return ExperimentBundle(manifest['split'],workload,environment,tuple(manifest['capacity']),factory,
        parameters,manifest['stress_seed'],manifest['factor'],identity,
        RewardConfig(**cfg['reward']),ObservationNormalization(**cfg['normalization']))

def make_env_factory(bundle,config):
    _require_split(bundle.split,'train')
    reward=RewardConfig(**config.get('reward',asdict(RewardConfig())))
    norm=ObservationNormalization(**config.get('normalization',asdict(ObservationNormalization())))
    if reward!=RewardConfig() or norm!=ObservationNormalization():
        raise ValueError('PPO-v0 must preserve E1-5A default reward and engineering scales')
    return lambda:CarbonSchedulingEnv(bundle.workload,bundle.environment,bundle.capacity,
      bundle.power_factory,bundle.parameters,reward_config=reward,normalization=norm,
      stop_at_drain=True)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--split',choices=('train','validation'),required=True)
    parser.add_argument('--factor',type=float,required=True);parser.add_argument('--stress-seed',type=int,required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    cfg=json.loads(Path(args.config).read_text(encoding='utf8'))
    print(json.dumps(freeze_bundle(cfg,args.split,args.factor,args.stress_seed,args.output)))

if __name__=='__main__':main()
