"""Train independently initialized task/cooling PPO policies on frozen Train."""
import argparse
from collections import Counter
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess

from experiments.e1.contracts import content_hash, file_hash, write_json

from . import ACCOUNTING
from .inputs import build_env, load_joint_bundle, resolve_config


TRAIN_MODES = ('B1', 'B2', 'B3')


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(name + ' must be a positive integer')
    return value


def dependency_versions():
    versions = {'python': platform.python_version(), 'platform': platform.platform()}
    for name in ('torch', 'stable_baselines3', 'numpy', 'pandas', 'gymnasium'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'unavailable'
    return versions


def source_provenance():
    root = Path(__file__).resolve().parents[2]
    files = set()
    for folder in ('experiments/joint', 'experiments/e1', 'utils', 'envs'):
        files.update((root / folder).rglob('*.py'))
    hashes = {path.relative_to(root).as_posix(): file_hash(path) for path in sorted(files)}
    try:
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        head = None
    return dict(git_head=head, code_sha256=hashes, code_manifest_sha256=content_hash(hashes),
                dependencies=dependency_versions())


def _json_line(stream, row):
    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
    stream.flush()


def train(train_bundle, output, *, config=None, mode='B3', episodes=None, max_episode_steps=None):
    """Collect complete finite episodes; each actor updates once per macro slot."""
    from .env import TASK_FIELDS, COOLING_FIELDS
    from .ppo import IPPOAgent, SlotTransition

    if mode not in TRAIN_MODES:
        raise ValueError('Training mode must be B1, B2 or B3')
    cfg = resolve_config(config)
    bundle = load_joint_bundle(train_bundle, 'train')
    count = _positive(cfg['training']['episodes'] if episodes is None else episodes, 'episodes')
    per_update = _positive(cfg['training']['episodes_per_update'], 'episodes_per_update')
    limit = cfg['training'].get('max_episode_steps') if max_episode_steps is None else max_episode_steps
    if limit is not None:
        _positive(limit, 'max_episode_steps')
    smoke = episodes is not None or limit is not None
    out = Path(output)
    if out.exists():
        raise FileExistsError('Refusing to overwrite training output: ' + str(out))
    out.mkdir(parents=True)
    seed = int(cfg['seed']) + {'B1': 100, 'B2': 200, 'B3': 300}[mode]
    task_agent = IPPOAgent(len(TASK_FIELDS), 2, cfg['ppo'], seed + 1) if mode in ('B1', 'B3') else None
    cooling_agent = IPPOAgent(len(COOLING_FIELDS), 3, cfg['ppo'], seed + 2) if mode in ('B2', 'B3') else None
    agents = {name: agent for name, agent in (('task', task_agent), ('cooling', cooling_agent)) if agent}
    initial = {name: agent.parameter_hash() for name, agent in agents.items()}
    execution = dict(episodes=count, episodes_per_update=per_update, max_episode_steps=limit,
                     smoke=smoke, mode=mode, root_seed=cfg['seed'], mode_seed=seed,
                     agent_seeds={name: agent.seed for name, agent in agents.items()},
                     collector='whole_finite_episodes', bootstrap_value=0.)
    write_json(out / 'training_config.json', dict(protocol=cfg, execution=execution))
    provenance = dict(source_provenance(), accounting=ACCOUNTING, config_sha256=content_hash(cfg),
                      input_identity=bundle.identity, training_role='train', test_access=False)
    write_json(out / 'provenance.json', provenance)
    trajectories = {name: [] for name in agents}
    updates, episode_summaries, slots = [], [], 0
    distributions = {'task': Counter(), 'cooling': Counter()}
    reward_sum, cost_sums = 0., Counter()
    env = build_env(bundle, train_bundle, cfg, mode, limit)
    try:
        with (out / 'training_trace.jsonl').open('w', encoding='utf8') as trace:
            for episode in range(count):
                # A fixed finite workload is reused; stochastic policy streams
                # continue independently and deterministically between resets.
                env.reset(seed=seed)
                episode_reward, episode_slots = 0., 0
                while True:
                    outcome = env.step(task_agent.act if task_agent else None,
                                       cooling_agent.act if cooling_agent else None)
                    if not math.isfinite(outcome.reward):
                        raise RuntimeError('Nonfinite system reward')
                    slots += 1
                    episode_slots += 1
                    episode_reward += outcome.reward
                    reward_sum += outcome.reward
                    for name, agent, boundary, decisions in (
                        ('task', task_agent, outcome.task_boundary_observation, outcome.task_decisions),
                        ('cooling', cooling_agent, outcome.cooling_boundary_observation, outcome.cooling_decisions),
                    ):
                        if agent:
                            trajectories[name].append(SlotTransition(boundary, decisions, outcome.reward,
                                                                      outcome.done, agent.value(boundary)))
                    for name, actions in (('task', outcome.task_actions), ('cooling', outcome.cooling_actions)):
                        for action in actions:
                            distributions[name][str(action['action'])] += 1
                    for name, value in outcome.terms.items():
                        if name.endswith('_cost') and isinstance(value, (int, float)):
                            cost_sums[name] += value
                    _json_line(trace, dict(episode=episode, slot=episode_slots - 1, reward=outcome.reward,
                        done=outcome.done, terms=outcome.terms, interval=outcome.interval,
                        task_actions=outcome.task_actions, cooling_actions=outcome.cooling_actions,
                        task_decisions=len(outcome.task_decisions), cooling_decisions=len(outcome.cooling_decisions)))
                    if outcome.done:
                        break
                episode_summaries.append(dict(episode=episode, physical_slots=episode_slots,
                                              reward_sum=episode_reward, summary=env.summary()))
                if (episode + 1) % per_update == 0 or episode + 1 == count:
                    record = dict(completed_episodes=episode + 1, physical_slots=slots)
                    for name, agent in agents.items():
                        record[name] = agent.update(trajectories[name], bootstrap_value=0.)
                        trajectories[name].clear()
                    updates.append(record)
                    write_json(out / 'updates.json', updates)
                    print(json.dumps(dict(mode=mode, episode=episode + 1, episodes=count,
                                          physical_slots=slots, reward_sum=episode_reward)), flush=True)
    finally:
        env.close()
    final = {name: agent.parameter_hash() for name, agent in agents.items()}
    checkpoint_sha256 = {}
    for name, agent in agents.items():
        path = out / (name + '.pt')
        agent.save(path)
        checkpoint_sha256[path.name] = file_hash(path)
    metadata = dict(schema='joint-trained-candidate-v1', mode=mode, accounting=ACCOUNTING,
        training_role='train', train_bundle=str(Path(train_bundle).resolve()), input_identity=bundle.identity,
        config_sha256=content_hash(cfg), training_config_sha256=file_hash(out / 'training_config.json'),
        provenance_sha256=file_hash(out / 'provenance.json'), checkpoint_sha256=checkpoint_sha256,
        initial_parameter_hashes=initial, final_parameter_hashes=final,
        parameters_updated={name: initial[name] != final[name] for name in agents},
        episodes_completed=count, physical_slots=slots, smoke=smoke, execution=execution,
        reward_sum=reward_sum, cost_sums=dict(cost_sums),
        action_distributions={name: dict(values) for name, values in distributions.items()},
        task_observation_fields=list(TASK_FIELDS), cooling_observation_fields=list(COOLING_FIELDS),
        independent_initialization=True, test_access=False)
    write_json(out / 'episodes.json', episode_summaries)
    write_json(out / 'metadata.json', metadata)
    return metadata


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument('--train-bundle', required=True)
    command.add_argument('--output', required=True)
    command.add_argument('--config')
    command.add_argument('--mode', choices=TRAIN_MODES, required=True)
    command.add_argument('--episodes', type=int, help='Smoke override; never a formal training result')
    command.add_argument('--max-episode-steps', type=int, help='Smoke truncation with terminal liability')
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    train(args.train_bundle, args.output, config=args.config, mode=args.mode,
          episodes=args.episodes, max_episode_steps=args.max_episode_steps)


if __name__ == '__main__':
    main()
