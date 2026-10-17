"""One bounded real-Train integration acceptance; never counted as formal training."""
import argparse
import json
from pathlib import Path
from .config import defaults
from .checkpoint import atomic_json
from .train import run


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(argv)
    cfg = defaults()
    cfg['budget'].update(total_physical_steps=512, max_wall_seconds=300., max_episodes=2)
    cfg['pretrain']['cooling'].update(max_states=64, demo_scenarios=1, bc_epochs=1, max_updates=1)
    cfg['pretrain']['task'].update(max_samples=64, demo_scenarios=1, bc_epochs=1, max_updates=1)
    cfg['ppo']['n_epochs'] = 1
    cfg['validation']['max_development_scenarios'] = 1
    result = run(a.pool, a.output, config=cfg)
    atomic_json(Path(a.output)/'acceptance-result.json', dict(result,
        purpose='engineering_acceptance_only_not_formal_research_results'))
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
