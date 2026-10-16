"""Explicit V2 training protocol, independent from the frozen V1 protocol."""
import copy
import json
import math
from pathlib import Path

from experiments.joint.config import load_config as load_v1

from . import VERSION


def defaults():
    v1 = load_v1()
    return dict(schema=VERSION, seed=20261016, accounting='joint_full_hvac_v1', test_access=False,
        v1_protocol=v1,
        environment=dict(idle_reference_setpoint_c=18., settle_after_drain=True, deadline_guard_enabled=True),
        reward=dict(v1['reward'], carbon_mode='incremental_idle18', deadline_multiplier=10.),
        ppo=dict(v1['ppo'], gamma=1., n_epochs=4, target_kl=.03, ent_coef=.01),
        pretrain=dict(
            task=dict(enabled=True, demo_scenarios=2, max_samples=2048, bc_epochs=1,
                      max_updates=4, min_wait_fraction=.05),
            cooling=dict(enabled=True, demo_scenarios=2, max_states=128, bc_epochs=1,
                         max_updates=4, ppo_steps=32, ppo_epochs=1)),
        constraint=dict(initial_lambda=10., eta=20., lambda_max=5000., target_dmr=.05,
                        update_on_incomplete=False, max_capacity_pressure_for_update=1.2),
        curriculum=dict(max_stage=2, successful_episodes_to_promote=2, target_dmr=.05),
        budget=dict(total_physical_steps=32768, max_wall_seconds=7200., max_episodes=None),
        validation=dict(every_episodes=4, max_development_scenarios=2,
                        heldout_final_check=True, deterministic=True),
        monitor=dict(tensorboard=True, max_step_history=32768))


def load_config(path=None):
    config = defaults() if path is None else (
        copy.deepcopy(path) if isinstance(path, dict) else json.loads(Path(path).read_text(encoding='utf8')))
    if config.get('schema') != VERSION or config.get('accounting') != 'joint_full_hvac_v1' or config.get('test_access') is not False:
        raise ValueError('Unsupported V2 protocol/accounting or Test access')
    if config['v1_protocol'] != load_v1():
        raise ValueError('V2 preserves the exact V1 state, normalization and physical protocol')
    env = config['environment']
    if env['idle_reference_setpoint_c'] != 18. or env['settle_after_drain'] is not True:
        raise ValueError('V2 carbon identity requires a legal fixed idle18 reference and settling')
    for name in ('total_physical_steps',):
        value = config['budget'][name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('Positive integer budget required: ' + name)
    seconds = config['budget']['max_wall_seconds']
    if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('A finite positive wall-clock budget is mandatory')
    constraint = config['constraint']
    if not (0 <= constraint['initial_lambda'] <= constraint['lambda_max'] and constraint['eta'] >= 0
            and constraint['target_dmr'] == .05 and constraint['update_on_incomplete'] is False):
        raise ValueError('Invalid bounded Train-only Deadline constraint')
    if config['validation']['deterministic'] is not True:
        raise ValueError('Model selection must use deterministic held-out evaluation')
    return config


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Write the complete frozen V2 training configuration')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(defaults(), indent=2, sort_keys=True)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
