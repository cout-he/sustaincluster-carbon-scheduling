"""One command: freeze V2 scenes, pretrain, jointly train B3, select and plot."""
import argparse
import json
from pathlib import Path
import time

from .checkpoint import atomic_json
from .config import load_config
from .scenarios import build_pool
from .train_v2 import run as train


def run(source_root,output,*,pool=None,config=None,scenario_config=None,resume=None,
        total_steps=None,wall_seconds=None,max_episodes=None,skip_pretrain=False,heldout_final=None):
    started=time.monotonic()
    out=Path(output)
    cfg=load_config(config)
    if resume is None:
        if out.exists():raise FileExistsError(out)
        out.mkdir(parents=True)
        if pool is None:
            if source_root is None:raise ValueError('--source-root is required to create a new V2 pool')
            pool=build_pool(source_root,out/'scenarios',scenario_config)
        atomic_json(out/'pipeline.json',dict(schema=cfg['schema'],pool=str(Path(pool).resolve()),
            config=cfg,source_root=str(Path(source_root).resolve()) if source_root else None,
            preparation_wall_seconds=time.monotonic()-started,
            wall_budget_scope='trainer phase; initial scenario freezing is separately timed',test_access=False))
    else:
        previous=json.loads((out/'pipeline.json').read_text(encoding='utf8'))
        if previous['config']!=cfg:raise ValueError('Pipeline resume configuration changed')
        if pool is None:pool=previous['pool']
    result=train(pool,out/'training',config=cfg,resume=resume,total_steps=total_steps,
        wall_seconds=wall_seconds,max_episodes=max_episodes,skip_pretrain=skip_pretrain,
        heldout_final=heldout_final)
    atomic_json(out/'pipeline-run-result.json',dict(training=result,
        pipeline_wall_seconds=time.monotonic()-started,
        preparation_wall_seconds=json.loads((out/'pipeline.json').read_text())['preparation_wall_seconds'],
        test_access=False))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root');parser.add_argument('--output',required=True);parser.add_argument('--pool')
    parser.add_argument('--config');parser.add_argument('--scenario-config')
    parser.add_argument('--resume',nargs='?',const='latest');parser.add_argument('--total-steps',type=int)
    parser.add_argument('--wall-seconds',type=float);parser.add_argument('--max-episodes',type=int)
    parser.add_argument('--skip-pretrain',action='store_true');parser.add_argument('--no-heldout-final',action='store_true')
    args=parser.parse_args(argv)
    run(args.source_root,args.output,pool=args.pool,config=args.config,scenario_config=args.scenario_config,
        resume=args.resume,total_steps=args.total_steps,wall_seconds=args.wall_seconds,max_episodes=args.max_episodes,
        skip_pretrain=args.skip_pretrain,heldout_final=False if args.no_heldout_final else None)


if __name__=='__main__':main()
