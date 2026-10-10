"""One strict configuration contract for the E1-1A baseline."""
import copy
from pathlib import Path

import yaml

from . import SEMANTICS_VERSION
from .contracts import finite, window
from .policies import make_policy

ROOT = Path(__file__).resolve().parents[2]


def validate_config(config):
    cfg = copy.deepcopy(config)
    try:
        start, end = window(cfg['start'], cfg['end'])
        cfg['start'], cfg['end'] = start.isoformat(), end.isoformat()
        policy = make_policy(cfg)
        if cfg['timestep_minutes'] != 15:
            raise ValueError('E1 uses 15-minute intervals only')
        if cfg['policy'] == 'carbon_threshold_fifo':
            cfg['policy_params']['threshold_g_per_kwh'] = policy.threshold_g_per_kwh
        if cfg['semantics_version'] != SEMANTICS_VERSION:
            raise ValueError('Unsupported execution semantics version')
        if isinstance(cfg['seed'], bool) or not isinstance(cfg['seed'], int) or cfg['seed'] < 0:
            raise ValueError('seed must be a nonnegative integer')
        dc, workload, environment = cfg['datacenter'], cfg['workload'], cfg['environment']
        for key in ('total_cores', 'total_gpus', 'total_mem'):
            dc[key] = finite(dc[key], key, positive=True)
        if dc['dc_id'] != 1 or dc['hvac_setpoint_c'] != 22:
            raise ValueError('E1-1A uses DC1 and fixed 22 C HVAC')
        for key in ('resource_multiplier', 'sla_multiplier'):
            workload[key] = finite(workload[key], key, positive=True)
        if workload['schema'] != 'alibaba_gpu_12_v1':
            raise ValueError('Unsupported workload schema')
        if not isinstance(workload['trace_year'], int):
            raise ValueError('trace_year must be an integer')
        if environment['interpolation'] != 'linear_utc' or environment['weather_transform'] != 'upstream_wet_bulb_0.9':
            raise ValueError('Unsupported environmental transform')
        if environment['noise'] is not False:
            raise ValueError('E1-1A disables environmental noise and roll')
        if environment['weather_time_basis'] != 'metadata_fixed_offset':
            raise ValueError('E1-1A weather uses its declared fixed UTC offset')
        if cfg['execution'] != {'network': 'local_zero', 'interval_load': 'left_boundary_hold', 'horizon': 'truncate'}:
            raise ValueError('Unsupported execution settings')
        for value in (workload['path'], environment['carbon_path'], environment['weather_path'], dc['dc_config_file']):
            if not isinstance(value, str) or not value:
                raise ValueError('Input paths must be nonempty strings')
    except (KeyError, TypeError) as exc:
        raise ValueError(f'Incomplete experiment configuration: {exc}') from exc
    return cfg


def load_config(path):
    return validate_config(yaml.safe_load(Path(path).read_text(encoding='utf-8')))


def resolve_path(path, root=ROOT):
    value = Path(path)
    return value if value.is_absolute() else Path(root) / value
