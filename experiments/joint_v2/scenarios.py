"""Versioned, deduplicated real-trace scenarios; September Test stays sealed.

Source task timestamps remain in 1970 inside the expanded 2020 pickle. They
and every raw job field identify the original bucket, independently of its
shifted annual date. Demand/SLA augmentation retains every original task.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import timedelta, timezone
import csv
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

from experiments.e1.contracts import content_hash, file_hash, interval_times, utc_time, write_json
from experiments.e1.environment import CI_COLUMN, EnvironmentalSample, FrozenEnvironment
from experiments.e1.rl_training_utils import APPROVED_SOURCE_SHA256, load_bundle
from experiments.e1.selection import TemporalEnvironment
from experiments.e1.workload import FrozenWorkload, TaskSpec, _raw_value


SCHEMA = 'joint-v2-training-scenario-v1'
ROLES = ('train', 'development', 'validation')
DEFAULT_CONFIG = Path(__file__).with_name('scenario_config.yaml')


@dataclass(frozen=True)
class TrainingScenario:
    scenario_id: str
    role: str
    workload: object
    environment: object
    capacity: tuple
    parameters: dict
    datacenter: dict
    model_path: Path
    identity: dict
    statistics: dict

    def __post_init__(self):
        if self.role not in ROLES or self.identity.get('role') != self.role:
            raise ValueError('Scenario role must be Train/development/Validation; Test is sealed')
        if not self.scenario_id or self.workload.start != self.environment.start:
            raise ValueError('Scenario ID or source time bounds differ')
        if self.environment.end < self.workload.end:
            raise ValueError('Environment cannot end before arrivals')
        object.__setattr__(self, 'model_path', Path(self.model_path))

    @property
    def split(self):
        return self.role

    @property
    def synthetic(self):
        return bool(self.statistics.get('synthetic', False))

    def save(self, path):
        out = Path(path)
        if out.exists():
            raise FileExistsError('Refusing to overwrite frozen V2 scenario: ' + str(out))
        out.mkdir(parents=True)
        self.workload.save(out / 'workload.json')
        self.environment.save(out / 'environment.json')
        shutil.copyfile(self.model_path, out / 'dc_config.json')
        details = dict(parameters=self.parameters, datacenter=self.datacenter,
                       capacity=list(self.capacity), identity=self.identity, statistics=self.statistics)
        write_json(out / 'scenario.json', details)
        files = {name: file_hash(out / name) for name in
                 ('workload.json','environment.json','dc_config.json','scenario.json')}
        manifest = dict(schema=SCHEMA, scenario_id=self.scenario_id, role=self.role,
                        environment_kind='temporal' if isinstance(self.environment,TemporalEnvironment) else 'linear',
                        identity_sha256=content_hash(self.identity), files=files, test_access=False)
        write_json(out / 'manifest.json', manifest)
        return out / 'manifest.json'


def load_scenario(path, allowed_role):
    if allowed_role not in ROLES:
        raise ValueError('Unsupported scenario role; Test is sealed')
    root = Path(path)
    if root.is_file():
        root = root.parent
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf8'))
    # This precedes every workload/environment/physical payload read.
    if manifest.get('role') != allowed_role:
        raise ValueError('Scenario role does not match allowed role')
    if manifest.get('schema') != SCHEMA or manifest.get('test_access') is not False:
        raise ValueError('Unsupported V2 scenario identity or Test rule')
    required = {'workload.json','environment.json','dc_config.json','scenario.json'}
    if set(manifest['files']) != required:
        raise ValueError('Unexpected scenario payload inventory')
    for name, sha in manifest['files'].items():
        if file_hash(root / name) != sha:
            raise ValueError('Frozen V2 scenario hash mismatch: ' + name)
    data = json.loads((root / 'scenario.json').read_text(encoding='utf8'))
    if content_hash(data['identity']) != manifest['identity_sha256']:
        raise ValueError('Frozen V2 scenario identity mismatch')
    workload = FrozenWorkload.load(root / 'workload.json')
    environment_loader = TemporalEnvironment if manifest.get('environment_kind') == 'temporal' else FrozenEnvironment
    environment = environment_loader.load(root / 'environment.json')
    return TrainingScenario(manifest['scenario_id'], allowed_role, workload, environment,
        tuple(data['capacity']), data['parameters'], data['datacenter'], root / 'dc_config.json',
        data['identity'], data['statistics'])


def fingerprint_bucket(matrix):
    """Hash unchanged original timestamps AND every field, preserving row order."""
    rows = [[_raw_value(v) for v in row] for row in matrix]
    return content_hash(dict(raw_columns=12, original_tasks=rows))


def _index_source(frame):
    if not {'interval_15m','tasks_matrix'} <= set(frame.columns) or frame.empty:
        raise ValueError('Original trace bucket schema is unavailable')
    indexed, original, first_repeats = {}, {}, []
    first_source = None
    for row_index, (stamp, matrix) in enumerate(zip(frame['interval_15m'], frame['tasks_matrix'])):
        stamp = utc_time(stamp, True)
        if stamp in indexed or not len(matrix):
            raise ValueError('Source needs unique nonempty recorded buckets')
        if any(len(row) != 12 for row in matrix):
            raise ValueError('Alibaba source requires its unchanged twelve raw fields')
        bucket = utc_time(matrix[0][3]).floor('15min')
        if any(utc_time(row[3]).floor('15min') != bucket for row in matrix):
            raise ValueError('Original raw timestamp bucket disagrees within matrix')
        fingerprint = fingerprint_bucket(matrix)
        key = bucket.isoformat()
        if key in original and original[key] != fingerprint:
            raise ValueError('A repeated original bucket changed job content')
        original[key] = fingerprint
        first_source = first_source or key
        if key == first_source:
            first_repeats.append(stamp)
        indexed[stamp] = dict(source_row=row_index, matrix=matrix,
                              source_bucket_id=key, fingerprint=fingerprint)
    source_times = [utc_time(key) for key in original]
    duration = max(source_times) - min(source_times) + pd.Timedelta(minutes=15)
    if len(first_repeats) > 1 and any(b-a != duration for a,b in zip(first_repeats, first_repeats[1:])):
        raise ValueError('Annual repetition is inconsistent with actual cropped trace duration')
    audit = dict(yearly_recorded_buckets=len(indexed), unique_original_nonempty_buckets=len(original),
        repeated_bucket_fraction=1-len(original)/len(indexed),
        original_start=min(source_times).isoformat(), original_end_inclusive=max(source_times).isoformat(),
        trace_duration_seconds=int(duration.total_seconds()),
        trace_duration_rule='actual last original bucket minus first plus 15 minutes',
        possible_buckets_per_period=int(duration/pd.Timedelta(minutes=15)),
        absent_buckets_per_period=int(duration/pd.Timedelta(minutes=15))-len(original),
        observed_full_repetitions=len(first_repeats),
        original_bucket_fingerprints_sha256=content_hash(original),
        all_repeated_bucket_contents_verified=True,
        source_is_repeated_shifted_trace=True, independent_yearly_collection=False)
    return indexed, audit


def _window_records(index, start, end):
    records = []
    for arrival in interval_times(start, end):
        query = arrival.replace(year=2020)
        record = index.get(query)
        if record is not None:
            records.append((arrival, record))
    return records


def _bucket_ids(records):
    return {record['source_bucket_id'] for _, record in records}


def _array_values(path, key):
    """Incremental JSON-array decoder. Consumers stop before held-out values.

    Only the named array is decoded. Reading past its authorized prefix is not
    required even though later unrelated arrays occur in the same source file.
    """
    pattern = re.compile(r'"' + re.escape(key) + r'"\s*:\s*\[')
    decoder = json.JSONDecoder()
    with Path(path).open('r', encoding='utf8') as stream:
        buffer = ''
        while True:
            chunk = stream.read(4096)
            if not chunk:
                raise ValueError('Weather array missing: ' + key)
            buffer += chunk
            match = pattern.search(buffer)
            if match:
                buffer = buffer[match.end():]
                break
            buffer = buffer[-256:]
        while True:
            buffer = buffer.lstrip(' \t\r\n,')
            if buffer.startswith(']'):
                return
            try:
                value, consumed = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = stream.read(4096)
                if not chunk:
                    raise ValueError('Weather array truncated: ' + key)
                buffer += chunk
                continue
            # Numeric tokens can end at a chunk boundary; get their delimiter.
            if consumed == len(buffer) or buffer[consumed:consumed+1] not in ',] \t\r\n':
                chunk = stream.read(4096)
                if chunk:
                    buffer += chunk
                    continue
            buffer = buffer[consumed:]
            yield value


def _environment_sources(carbon_path, weather_path, maximum_end):
    maximum_end = utc_time(maximum_end)
    if maximum_end > utc_time('2023-08-09T05:00:00Z'):
        raise ValueError('V2 environment envelope would exceed authorized pre-Test coverage')
    carbon_times, carbon_values = [], []
    with Path(carbon_path).open('r', encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            stamp = utc_time(pd.Timestamp(row['Datetime (UTC)'], tz='UTC'))
            if stamp > maximum_end.ceil('1h'):
                break  # Never decode September numeric observations.
            carbon_times.append(stamp)
            carbon_values.append(float(row[CI_COLUMN]))
    with Path(weather_path).open('r', encoding='utf8') as stream:
        header = stream.read(4096)
    offset_match = re.search(r'"utc_offset_seconds"\s*:\s*(-?\d+)', header)
    zone_match = re.search(r'"timezone"\s*:\s*("[^"]+")', header)
    if not offset_match:
        raise ValueError('Weather UTC offset metadata is missing')
    offset = int(offset_match.group(1))
    zone = json.loads(zone_match.group(1)) if zone_match else 'unspecified'
    weather_times = []
    for raw in _array_values(weather_path, 'time'):
        stamp = pd.Timestamp(raw)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize(timezone(timedelta(seconds=offset)))
        stamp = stamp.tz_convert('UTC')
        weather_times.append(stamp)
        if stamp >= maximum_end.ceil('1h'):
            break
    values = []
    for value in _array_values(weather_path, 'temperature_2m'):
        values.append(float(value))
        if len(values) == len(weather_times):
            break
    if len(values) != len(weather_times):
        raise ValueError('Weather authorized prefix is incomplete')
    if not np.isfinite(carbon_values).all() or not np.isfinite(values).all():
        raise ValueError('Authorized environmental observations must be finite')
    return (pd.DatetimeIndex(carbon_times), np.asarray(carbon_values),
            pd.DatetimeIndex(weather_times), np.asarray(values), offset, zone)


def _make_environment(start, end, sources, hashes):
    ci_times, ci, temp_times, temp, offset, zone = sources
    grid = pd.DatetimeIndex(interval_times(start,end))
    expected = pd.date_range(start.floor('1h'), end.ceil('1h'),freq='1h')
    if not set(expected) <= set(ci_times) or not set(expected) <= set(temp_times):
        raise ValueError('Authorized scenario lacks complete hourly environmental endpoints')
    previous = np.searchsorted(ci_times.asi8, grid.asi8, side='right')-1
    if (previous < 0).any():
        raise ValueError('No causal carbon observation is available')
    origin = temp_times.asi8[0]
    temperatures = np.interp((grid.asi8-origin)/1e9, (temp_times.asi8-origin)/1e9, temp)
    samples = tuple(EnvironmentalSample(t,float(c),float(w),float(.9*w))
                    for t,c,w in zip(grid,ci[previous],temperatures))
    return TemporalEnvironment(start,end,hashes['carbon'],hashes['weather'],samples,offset,zone)


def _make_workload(records, start, end, hashes, pressure, sla):
    scale = 5.*float(pressure)
    tasks = []
    for arrival, record in records:
        for task_index, raw in enumerate(record['matrix']):
            raw = tuple(_raw_value(v) for v in raw)
            tasks.append(TaskSpec(
                f"{hashes['workload'][:16]}/{arrival.isoformat()}/{record['source_row']}/{task_index}",
                str(raw[0]),record['source_row'],task_index,arrival,float(raw[4]),
                scale*float(raw[5])/100,scale*float(raw[6])/100,scale*float(raw[7]),
                float(raw[9]),float(sla),1,raw))
    return FrozenWorkload(start,end,hashes['workload'],scale,float(sla),2020,tuple(tasks))


def _statistics(workload, environment, capacity, pressure, sla):
    hours = (workload.end-workload.start).total_seconds()/3600
    work = [math.fsum(getattr(t,key)*t.duration/60 for t in workload.tasks)
            for key in ('cores_req','gpu_req','mem_req')]
    occupancy = max(w/(c*hours) for w,c in zip(work,capacity))
    arrival_samples = [s for s in environment.samples if s.time < workload.end]
    ci = np.asarray([s.ci_g_per_kwh for s in arrival_samples])
    temp = np.asarray([s.temperature_c for s in arrival_samples])
    slacks = [(t.sla_multiplier-1)*t.duration for t in workload.tasks]
    return dict(tasks=len(workload.tasks), base_tasks=len(workload.tasks), arrival_hours=hours,
        resource_work_hours=dict(zip(('cpu','gpu','mem'),work)), pressure_proxy=occupancy,
        capacity_pressure=occupancy, pressure_is_necessary_workload_proxy_not_feasibility_proof=True,
        synthetic=pressure != 1 or sla != 1.5,
        augmentation=dict(resource_pressure_factor=pressure,sla_multiplier=sla,
                          all_original_tasks_retained=True,arrival_order_preserved=True,
                          observed_tasks=len(workload.tasks),additional_tasks=0,dropped_tasks=0),
        slack_median_minutes=float(np.median(slacks)) if slacks else 0.,
        ci_mean=float(ci.mean()),ci_std=float(ci.std()),ci_range=float(np.ptp(ci)),
        temperature_mean_c=float(temp.mean()),temperature_min_c=float(temp.min()),temperature_max_c=float(temp.max()))


def _categorize(scenarios):
    train = [s for s in scenarios if s.role == 'train']
    fields = [('pressure_proxy','load_category',('low','medium','high')),
              ('slack_median_minutes','deadline_category',('tight','medium','loose')),
              ('ci_std','ci_category',('stable','medium','variable')),
              ('temperature_mean_c','temperature_category',('cool','medium','warm'))]
    thresholds = {}
    for key,label,names in fields:
        lo,hi = map(float,np.quantile([s.statistics[key] for s in train],[1/3,2/3]))
        thresholds[key] = dict(lower_tertile=lo,upper_tertile=hi,fit_role='train')
        for current in scenarios:
            value = current.statistics[key]
            current.statistics[label] = names[0] if value <= lo else names[1] if value <= hi else names[2]
    return thresholds


def build_pool(source_root, output, config_path=None):
    root, out = Path(source_root), Path(output)
    cfg = json.loads(Path(config_path or DEFAULT_CONFIG).read_text(encoding='utf8'))
    if cfg.get('schema') != 'joint-v2-scenario-protocol-v1' or cfg.get('test_access') is not False:
        raise ValueError('Unsupported V2 scenario configuration or sealed Test rule')
    if cfg['arrival_hours'] not in (24,48) or not 0 < cfg['safety_tail_days'] <= 30:
        raise ValueError('V2 scenarios require 24/48h arrivals and at most thirty tail days')
    if cfg['resource_multiplier'] != 5 or tuple(cfg['capacity']) != (50000,1000,80000):
        raise ValueError('V2 base task mapping or resource capacity changed')
    if any(not .5 <= float(v) <= 1.5 for v in cfg['resource_pressure_factors']):
        raise ValueError('Resource augmentation must stay in its declared light-stress range')
    if any(not 1 <= float(v) <= 3 for v in cfg['sla_multipliers']):
        raise ValueError('Synthetic SLA multipliers are outside their declared range')
    if out.exists():
        raise FileExistsError('Refusing to overwrite a frozen V2 pool')
    paths = dict(workload=root/'data/workload/alibaba_2020_dataset/result_df_full_year_2020.pkl',
        carbon=root/'data/carbon_intensity/US-CAL-CISO/2023/US-CAL-CISO_2023_hourly.csv',
        weather=root/'data/weather/US-CAL-CISO/2023.json',physical_model=root/'configs/dcs/dc_config.json')
    hashes = {name:file_hash(path) for name,path in paths.items()}
    if hashes != APPROVED_SOURCE_SHA256:
        raise ValueError('Original source bytes differ from the approved V1/E1 inputs')
    index, audit = _index_source(pd.read_pickle(paths['workload']))
    validation_path = root / cfg['validation_bundle']
    validation = load_bundle(validation_path,allowed_split='validation')
    validation_records = _window_records(index,validation.workload.start,validation.workload.end)
    reserved = _bucket_ids(validation_records)
    validation_buckets = set(reserved)
    hours = pd.Timedelta(hours=cfg['arrival_hours'])
    development = []
    for raw_start in cfg['development_starts']:
        start = utc_time(raw_start,True)
        if not utc_time('2023-06-01T00:00:00Z') <= start < utc_time('2023-07-01T00:00:00Z'):
            raise ValueError('Development environment starts must stay in June')
        records = _window_records(index,start,start+hours)
        ids = _bucket_ids(records)
        if not ids or ids & reserved:
            raise ValueError('Development original source buckets overlap held-out fragments')
        development.append((start,records))
        reserved |= ids
    first,last = map(utc_time,(cfg['train_environment_start'],cfg['train_environment_end_exclusive']))
    if first < utc_time('2023-01-01T00:00Z') or last > utc_time('2023-06-01T05:00Z'):
        raise ValueError('Train environments must precede the held-out June development range')
    months = defaultdict(list)
    for start in pd.date_range(first,last,freq='24h',inclusive='left'):
        if start+hours <= last:
            months[start.month].append(start)
    selected, selected_ids = [], set()
    # Round-robin months supplies environmental variety without counting an
    # annual repeat or any overlapping original bucket as another fragment.
    while months and len(selected) < int(cfg['max_train_base_fragments']):
        for month in list(sorted(months)):
            if not months[month]:
                del months[month]
                continue
            start = months[month].pop(0)
            records = _window_records(index,start,start+hours)
            ids = _bucket_ids(records)
            if not ids or ids & (reserved | selected_ids):
                continue
            selected.append((start,records))
            selected_ids |= ids
            if len(selected) == int(cfg['max_train_base_fragments']):
                break
    if not selected:
        raise ValueError('No disjoint real-source Train fragments are available')
    final_end = max(start+hours+pd.Timedelta(days=cfg['safety_tail_days'])
                    for start,_ in selected+development)
    if final_end > utc_time(cfg['maximum_environment_end']) or final_end > utc_time('2023-08-09T05:00Z'):
        raise ValueError('Scenario physics tails would enter unauthorized environment coverage')
    sources = _environment_sources(paths['carbon'],paths['weather'],final_end)
    dc = dict(dc_id=1,location='US-CAL-CISO',total_cores=50000,total_gpus=1000,total_mem=80000,hvac_setpoint_c=22)
    scenarios = []
    for role, groups in (('train',selected),('development',development)):
        for start,records in groups:
            end = start+hours
            env = _make_environment(start,end+pd.Timedelta(days=cfg['safety_tail_days']),sources,hashes)
            fragment = content_hash([(record['source_bucket_id'],record['fingerprint']) for _,record in records])
            factors = cfg['resource_pressure_factors'] if role == 'train' else [1.]
            slas = cfg['sla_multipliers'] if role == 'train' else [1.5]
            for factor in factors:
                for sla in slas:
                    workload = _make_workload(records,start,end,hashes,factor,sla)
                    scenario_id = f"{role}-{start.strftime('%Y%m%dT%H%M')}-r{factor:g}-s{sla:g}"
                    identity = dict(schema=SCHEMA,role=role,scenario_id=scenario_id,source_fragment_id=fragment,
                        source_fragment_sha256=fragment,source_bucket_ids=sorted(_bucket_ids(records)),
                        original_source_hashes=hashes,simulation_start=start.isoformat(),arrival_end=end.isoformat(),
                        safety_end=env.end.isoformat(),resource_pressure_factor=factor,sla_multiplier=sla,
                        seed=cfg['seed'],workload_sha256=workload.sha256,environment_sha256=env.sha256,
                        preprocessing='2023 simulation UTC year->2020 trace lookup; raw 1970 payload unchanged; duration raw[4], CPU5*raw[5]/100, GPU5*raw[6]/100, MEM5*raw[7]; synthetic factor scales demands only',
                        test_access=False)
                    stats = _statistics(workload,env,cfg['capacity'],factor,sla)
                    scenarios.append(TrainingScenario(scenario_id,role,workload,env,tuple(cfg['capacity']),
                        dict(threshold_g_per_kwh=161.075,future_ci=False),dc,paths['physical_model'],identity,stats))
    val_identity = dict(schema=SCHEMA,role='validation',scenario_id='validation-existing-july08-factor1',
        source_fragment_id=content_hash([(r['source_bucket_id'],r['fingerprint']) for _,r in validation_records]),
        source_bucket_ids=sorted(validation_buckets),original_source_hashes=hashes,
        original_e1_bundle_identity=validation.identity,resource_pressure_factor=1.,sla_multiplier=1.5,
        workload_sha256=validation.workload.sha256,environment_sha256=validation.environment.sha256,test_access=False)
    scenarios.append(TrainingScenario(val_identity['scenario_id'],'validation',validation.workload,
        validation.environment,validation.capacity,validation.parameters,dc,validation_path/'dc_config.json',
        val_identity,_statistics(validation.workload,validation.environment,validation.capacity,1.,1.5)))
    thresholds = _categorize(scenarios)
    audit.update(independent_train_base_fragments=len(selected),train_augmented_scenarios=sum(s.role=='train' for s in scenarios),
        development_fragments=len(development),validation_fragments=1,
        train_unique_original_buckets=len(selected_ids),validation_original_buckets=len(validation_buckets),
        train_development_bucket_overlap=0,train_validation_bucket_overlap=0,development_validation_bucket_overlap=0,
        selected_train_fragment_pairwise_bucket_overlap=0,
        augmentation_independence=False,actual_train_base_starts=[s.isoformat() for s,_ in selected],
        actual_development_starts=[s.isoformat() for s,_ in development],
        duplicate_scenario_fraction=1-len(selected)/sum(s.role=='train' for s in scenarios),
        duplicate_scenario_interpretation='planned augmentations of the same real fragments; never independent observations',
        environment_numeric_decode_end=final_end.ceil('1h').isoformat(),test_observations_decoded=False)
    out.mkdir(parents=True)
    write_json(out/'source_audit.json',audit)
    write_json(out/'config.json',cfg)
    entries=[]
    for current in scenarios:
        folder = out/'scenarios'/current.scenario_id
        current.save(folder)
        entries.append(dict(scenario_id=current.scenario_id,role=current.role,path=folder.relative_to(out).as_posix(),
            identity_sha256=content_hash(current.identity),manifest_sha256=file_hash(folder/'manifest.json'),statistics=current.statistics))
    manifest = dict(schema='joint-v2-scenario-pool-v1',source_root=str(root.resolve()),
        config_sha256=content_hash(cfg),source_hashes=hashes,source_audit=audit,
        source_audit_sha256=file_hash(out/'source_audit.json'),config_file_sha256=file_hash(out/'config.json'),
        quantile_thresholds=thresholds,scenarios=entries,scenarios_sha256=content_hash(entries),test_access=False,
        roles=dict(train='gradients',development='repeated checkpoint selection; no gradients',
                   validation='existing July08-10 untouched cohort, relative independent final check'),
        limitation='Source-bucket disjointness does not prove statistical independence within one seven-week trace')
    write_json(out/'manifest.json',manifest)
    return out/'manifest.json'


def load_pool(root, roles=('train',)):
    if not roles or any(role not in ROLES for role in roles):
        raise ValueError('Unsupported pool role; Test is sealed')
    path = Path(root)
    root = path.parent if path.is_file() else path
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf8'))
    if manifest.get('schema') != 'joint-v2-scenario-pool-v1' or manifest.get('test_access') is not False:
        raise ValueError('Unexpected V2 pool identity or Test rule')
    if content_hash(manifest['scenarios']) != manifest['scenarios_sha256']:
        raise ValueError('V2 pool scenario inventory hash mismatch')
    for filename,key in (('source_audit.json','source_audit_sha256'),('config.json','config_file_sha256')):
        if file_hash(root/filename) != manifest[key]:
            raise ValueError('V2 pool protocol identity mismatch')
    result=[]
    for entry in manifest['scenarios']:
        if entry['role'] not in ROLES:
            raise ValueError('Unexpected held-out role in scenario pool')
        if entry['role'] not in roles:
            continue
        folder = (root/entry['path']).resolve()
        if not folder.is_relative_to(root.resolve()):
            raise ValueError('Scenario path escapes its frozen pool')
        if file_hash(folder/'manifest.json') != entry['manifest_sha256']:
            raise ValueError('V2 pool scenario manifest hash mismatch')
        current = load_scenario(folder,allowed_role=entry['role'])
        if content_hash(current.identity) != entry['identity_sha256']:
            raise ValueError('V2 pool scenario identity mismatch')
        result.append(current)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--config')
    args=parser.parse_args(argv)
    print(build_pool(args.source_root,args.output,args.config),flush=True)


if __name__=='__main__':
    main()
