"""Frozen protocol checks shared by the training and comparison commands."""
import json
import math
from pathlib import Path

from . import ACCOUNTING

DEFAULT_CONFIG = Path(__file__).with_name('config.yaml')


def load_config(path=DEFAULT_CONFIG):
    # The distributed YAML uses JSON syntax to avoid adding a YAML dependency.
    config = json.loads(Path(path).read_text(encoding='utf8'))
    if config['schema'] != 'joint-task-hvac-ippo-v1' or config['accounting'] != ACCOUNTING:
        raise ValueError('Unsupported joint protocol/accounting')
    if config['test_evaluation'] is not False or config['timestep_minutes'] != 15:
        raise ValueError('Test is sealed and the physical step must remain 15 minutes')
    expected = dict(min_setpoint_c=18., max_setpoint_c=22., initial_setpoint_c=22.,
                    action_step_c=1., rack_inlet_limit_c=28.)
    if config['cooling'] != expected:
        raise ValueError('Cooling configuration differs from the audited model domain')
    if config['selection'] != dict(dmr_target=.05, max_temperature_violations=0):
        raise ValueError('Selection constraints changed')
    if config['evaluation'] != dict(endpoint='frozen_environment_end', deterministic=True):
        raise ValueError('Evaluation endpoint or determinism changed')
    for value in config['normalization'].values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Normalization scales must be positive finite')
    return config
