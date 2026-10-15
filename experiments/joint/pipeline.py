"""One command: independent B1/B2/B3 training, then five-mode Validation."""
import argparse
from pathlib import Path

from experiments.e1.contracts import content_hash, write_json

from .evaluate import evaluate
from .inputs import load_joint_bundle, resolve_config
from .train import _positive, train


def run(train_bundle, validation_bundle, output, *, config=None, episodes=None, max_episode_steps=None):
    cfg = resolve_config(config)
    original_train = load_joint_bundle(train_bundle, 'train')
    original_validation = load_joint_bundle(validation_bundle, 'validation')
    if float(original_train.factor) != float(original_validation.factor):
        raise ValueError('Pair Train and Validation at the same frozen pressure factor')
    if original_train.identity['raw_source_sha256'] != original_validation.identity['raw_source_sha256']:
        raise ValueError('Train and Validation original source identities differ')
    if episodes is not None:
        _positive(episodes, 'episodes')
    if max_episode_steps is not None:
        _positive(max_episode_steps, 'max_episode_steps')
    out = Path(output)
    if out.exists():
        raise FileExistsError('Refusing to overwrite pipeline output: ' + str(out))
    out.mkdir(parents=True)
    smoke = episodes is not None or max_episode_steps is not None or cfg['training'].get('max_episode_steps') is not None
    write_json(out / 'pipeline_config.json', dict(protocol=cfg, config_sha256=content_hash(cfg),
        episodes_override=episodes, max_episode_steps_override=max_episode_steps, smoke=smoke,
        train_input_identity=original_train.identity, validation_input_identity=original_validation.identity,
        modes=['B0', 'B1', 'B2', 'B3', 'RiskOnly'], test_access=False))
    for mode in ('B1', 'B2', 'B3'):
        train(train_bundle, out / 'models' / mode, config=cfg, mode=mode,
              episodes=episodes, max_episode_steps=max_episode_steps)
    comparison = evaluate(validation_bundle, out / 'models', out / 'evaluation', config=cfg,
                          max_episode_steps=max_episode_steps)
    result = dict(schema='joint-pipeline-v1', output=str(out.resolve()), smoke=comparison['smoke'],
                  config_sha256=content_hash(cfg), selection=comparison['selection'], test_access=False)
    write_json(out / 'pipeline_result.json', result)
    return result


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument('--train-bundle', required=True)
    command.add_argument('--validation-bundle', required=True)
    command.add_argument('--output', required=True)
    command.add_argument('--config')
    command.add_argument('--episodes', type=int, help='Smoke override only')
    command.add_argument('--max-episode-steps', type=int, help='Smoke truncation only')
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    run(args.train_bundle, args.validation_bundle, args.output, config=args.config,
        episodes=args.episodes, max_episode_steps=args.max_episode_steps)


if __name__ == '__main__':
    main()
