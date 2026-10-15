"""Guarded reuse of the existing immutable Train/Validation experiment inputs."""
import copy
import json
from pathlib import Path

from experiments.e1.rl_training_utils import load_bundle

from .config import load_config


def resolve_config(config=None):
    if not isinstance(config, dict):
        return load_config() if config is None else load_config(config)
    cfg = copy.deepcopy(config)
    expected = load_config()
    for key in ('schema', 'accounting', 'timestep_minutes', 'test_evaluation',
                'cooling', 'selection', 'evaluation'):
        if cfg.get(key) != expected[key]:
            raise ValueError('Unsupported frozen joint protocol: ' + key)
    for value in cfg['normalization'].values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < float('inf'):
            raise ValueError('Normalization scales must be positive finite')
    return cfg


def load_joint_bundle(bundle_path, split):
    if split not in ('train', 'validation'):
        raise ValueError('Only Train and Validation splits are authorized; Test is sealed')
    return load_bundle(Path(bundle_path), allowed_split=split)


def build_env(bundle, bundle_path, config, mode, max_episode_steps=None):
    # The original loader has already checked role BEFORE opening any payload.
    # Its configuration and physical bytes are hash-bound to the manifest.
    from .env import JointEnv
    from .power import JointPowerModel

    if bundle.split not in ('train', 'validation'):
        raise ValueError('Only Train and Validation inputs may create an environment')
    root = Path(bundle_path)
    cfg = resolve_config(config)
    dc = json.loads((root / 'config.json').read_text(encoding='utf8'))['datacenter']
    factory = lambda: JointPowerModel(dc, root / 'dc_config.json', cooling_config=cfg['cooling'])
    return JointEnv(bundle.workload, bundle.environment, bundle.capacity, factory,
                    bundle.parameters, cfg, mode=mode, max_episode_steps=max_episode_steps)


def make_env(bundle_path, split, config, mode, max_episode_steps=None):
    bundle = load_joint_bundle(bundle_path, split)
    return build_env(bundle, bundle_path, config, mode, max_episode_steps)


def env_factory(bundle_path, split, config, mode, max_episode_steps=None):
    bundle = load_joint_bundle(bundle_path, split)
    return lambda: build_env(bundle, bundle_path, config, mode, max_episode_steps)
