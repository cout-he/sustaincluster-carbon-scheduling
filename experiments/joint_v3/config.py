"""Finite frozen research protocol. JSON syntax is valid YAML."""
import copy
import json
import math
from pathlib import Path
from experiments.joint_v2.config import defaults as v2_defaults
from . import VERSION

ABLATIONS = ('full', 'no_queue_risk', 'no_carbon_features', 'no_robust', 'independent_critic', 'no_teacher')


def defaults(seed=20261017, variant='full'):
    cfg = v2_defaults()
    cfg.update(schema=VERSION, seed=int(seed), variant=variant,
        forecast=dict(horizon=16, lags=8, ridge=1.),
        risk=dict(max_tasks=256, max_slots=192, max_wait_slots=16, time_budget_ms=100., cache_size=128),
        carbon=dict(horizon=16, kappa=0. if variant == 'no_robust' else 1.,
                    max_fine_candidates=32, max_fast_candidates=32, max_slots=192),
        teacher=dict(threshold_kg=0., initial_weight=.1, decay_fraction=.5,
                     enabled=variant != 'no_teacher'),
        constraint=dict(initial_lambda=10., lambda_min=10., lambda_max=5000.,
                        eta_lambda=20., beta=.1, target_dmr=.05),
        budget=dict(total_physical_steps=65536, max_wall_seconds=21600., max_episodes=None),
        validation=dict(every_episodes=16, max_development_scenarios=2,
                        heldout_final_check=False, deterministic=True),
        pretrain=dict(cooling=dict(enabled=True, max_states=4096, horizon=4, demo_scenarios=96,
            bc_epochs=4, max_updates=256), task=dict(enabled=variant != 'no_teacher',
            max_samples=4096, demo_scenarios=4, bc_epochs=2, max_updates=64)),
        monitor=dict(tensorboard=False, max_step_history=65536))
    cfg['ppo'].update(normalize_advantage=False, learning_rate=1e-4, n_epochs=4,
                       batch_size=32, target_kl=.03, gamma=1., ent_coef=.01)
    cfg['environment']['deadline_guard_enabled'] = True
    return cfg


def load_config(path=None):
    cfg = defaults() if path is None else (copy.deepcopy(path) if isinstance(path, dict)
        else json.loads(Path(path).read_text(encoding='utf8')))
    if (cfg.get('schema') != VERSION or cfg.get('accounting') != 'joint_full_hvac_v1'
            or cfg.get('test_access') is not False or cfg['variant'] not in ABLATIONS):
        raise ValueError('V3 identity/accounting/variant/Test rule mismatch')
    if cfg['v1_protocol'] != v2_defaults()['v1_protocol']:
        raise ValueError('Frozen physical and original observation protocol changed')
    budget = cfg['budget']
    if (isinstance(budget['total_physical_steps'], bool) or not isinstance(budget['total_physical_steps'], int)
            or budget['total_physical_steps'] < 1 or budget['total_physical_steps'] > 65536
            or not math.isfinite(budget['max_wall_seconds']) or not 0 < budget['max_wall_seconds'] <= 21600):
        raise ValueError('Finite predeclared V3 budget exceeded')
    if cfg['validation']['heldout_final_check'] or not cfg['validation']['deterministic']:
        raise ValueError('July must stay unused; development decisions deterministic')
    from .reward import GroupConstraints
    GroupConstraints(cfg['constraint'])
    from experiments.joint.ppo import PPOConfig
    PPOConfig(**cfg['ppo'])
    for container, keys in ((cfg['validation'], ('every_episodes', 'max_development_scenarios')),
                           (cfg['risk'], ('max_tasks', 'max_slots', 'max_wait_slots')),
                           (cfg['carbon'], ('max_fine_candidates', 'max_slots'))):
        for key in keys:
            value = container[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError('Positive integer required: '+key)
    if cfg['forecast']['horizon'] != 16 or cfg['carbon']['horizon'] != 16:
        raise ValueError('Frozen observation/forecast horizon must be16')
    for options in cfg['pretrain'].values():
        for key in ('demo_scenarios', 'bc_epochs', 'max_updates'):
            value = options[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError('Nonnegative pretraining budget required')
    return cfg
