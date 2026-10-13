"""Registered development-Validation matrix and exact action-tape replay.

This local archival driver never reads Test or tunes a model. It emits the full
plan before launching any of the three pressure cases. All outputs are fresh.
Use the configured WSL runtime after the formal Train campaign has passed.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
NOTES = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.e1.contracts import canonical_bytes, content_hash, file_hash, write_json
from experiments.e1.temporal import experiment_sources
from experiments.e1.cars_experiment import _snapshot_sources

GROUPS = ('plain', 'state', 'shield', 'combined')
SEEDS = (55101, 55102, 55103)
BASELINES = ('edf_backfill', 'edf_carbon_threshold', 'cars_risk_only', 'task_risk_full')
STEPS = 20480
CASES = (
    ('1', 1., 'research_notes/E1/E1-5C/diagnostics-001/inputs/validation-1',
     'db65bf311e533991ef4cb2f0698c96570d2ebba0185b963c3bb3c16e656a0a18'),
    ('1.5', 1.5, 'research_notes/E1/E1-5C/diagnostics-001/inputs/validation-1.5',
     '55238b2cedba9326cc46921b2c786bff83350c2faf1e7dd69ab39083fda185ac'),
    ('2', 2., 'research_notes/E1/E1-5B/validation-input-001',
     '62658e53a8edcf1b4fe74f5b738f7f4628a8defdf271a1b8d848f23c7fc18f02'),
)
LEGACY_SHA = {
    55101: 'cd3d7b533fdfca455947ff00171d60da2c94dd114dcfc5b1ffbdeefa4a191d3e',
    55102: 'd2c0e6cd3765f7bd63814818011586de7ffc8f42d839bcb681049d62ecd024e3',
    55103: '73d2cc8bb4548d59a57501d56aa3b832c4fb7aadc1fecbe32d883ec4ee3374ac',
}
SCHEMA = 'e1-6b-development-validation-campaign-v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def stamp():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_methods():
    return list(BASELINES) + [f'{group}_{seed}' for group in GROUPS for seed in SEEDS] + [
        f'legacy_3action_{seed}' for seed in SEEDS]


def input_records():
    """Validate only the three previously frozen development-Validation bundles."""
    from experiments.e1.rl_training_utils import load_bundle
    result = []
    for label, factor, relative, manifest_sha in CASES:
        path = ROOT / relative
        require(file_hash(path / 'bundle.json') == manifest_sha, 'Registered Validation manifest changed')
        bundle = load_bundle(path, allowed_split='validation')
        require(bundle.split == 'validation' and bundle.factor == factor, 'Validation role/factor changed')
        require(bundle.parameters['threshold_g_per_kwh'] == 161.075, 'Validation CI threshold changed')
        hashes = {name: file_hash(path / name) for name in read(path / 'bundle.json')['files']}
        hashes['bundle.json'] = manifest_sha
        result.append(dict(label=label, factor=factor, path=str(path), split='validation',
            tasks=len(bundle.workload.tasks), workload_sha256=bundle.workload.sha256,
            environment_sha256=bundle.environment.sha256, capacity=list(bundle.capacity),
            threshold_g_per_kwh=161.075, files=hashes, identity=bundle.identity))
    require(len({r['environment_sha256'] for r in result}) == 1, 'Environmental inputs differ across factors')
    return result


def candidate_records(formal, source_sha):
    """Bind all 12 new models and all three original models before evaluation."""
    from experiments.e1.adaptive_evaluate import validate_checkpoint_protocol
    receipt = read(formal / 'campaign-receipt.json')
    require(receipt.get('passed') is True and receipt.get('total_actual_steps') == 12 * STEPS,
            'Formal Train gate has not passed')
    require(receipt.get('formal_test_evaluated') is False and receipt.get('validation_training') is False,
            'Formal data role guard differs')
    require(receipt['source_sha256'] == source_sha, 'Sources changed after formal Train freeze')
    models = receipt['models']
    require(len(models) == 12 and {(m['group'], m['seed']) for m in models} == {
        (g, s) for g in GROUPS for s in SEEDS}, 'Formal model matrix differs')
    audited = {(m['group'], m['seed']): m for m in models}
    result = []
    for group in GROUPS:
        for seed in SEEDS:
            folder = NOTES / f'train-{group}-{seed}-001'
            summary = read(folder / 'training_summary.json')
            config = read(folder / 'training_config.json')
            validate_checkpoint_protocol(config, summary)
            digest = file_hash(folder / 'model.zip')
            require(summary['group'] == group and summary['seed'] == seed and
                summary['num_timesteps'] == STEPS and summary['parameter_count'] == 19397 and
                summary['checkpoint_sha256'] == digest == audited[group, seed]['checkpoint_sha256'],
                'Formal checkpoint identity differs')
            require(summary['source_sha256'] == source_sha and
                summary['parameter_sha256_after'] == audited[group, seed]['final_parameter_sha256'],
                'Formal source or parameter identity differs')
            result.append(dict(kind='adaptive', name=f'{group}_{seed}', group=group, seed=seed,
                training_steps=STEPS, checkpoint=str(folder / 'model.zip'), checkpoint_sha256=digest,
                training_summary_sha256=file_hash(folder / 'training_summary.json'),
                training_config_sha256=file_hash(folder / 'training_config.json')))
    for seed in SEEDS:
        folder = ROOT / f'research_notes/E1/E1-5B/train-seed-{seed}-001'
        summary = read(folder / 'training_summary.json')
        config = read(folder / 'training_config.json')
        digest = file_hash(folder / 'model.zip')
        require(digest == LEGACY_SHA[seed] == summary['checkpoint_sha256'] and
            summary['seed'] == config['seed'] == seed and summary['num_timesteps'] == STEPS and
            config['schema'] == 'e1-5b-ppo-v0' and config['training_identity']['split'] == 'train',
            'Original three-action checkpoint differs')
        result.append(dict(kind='legacy', name=f'legacy_3action_{seed}', seed=seed,
            training_steps=STEPS, checkpoint=str(folder / 'model.zip'), checkpoint_sha256=digest,
            training_summary_sha256=file_hash(folder / 'training_summary.json'),
            training_config_sha256=file_hash(folder / 'training_config.json')))
    return result


def validate_output(folder, record, replay=False):
    """Check the complete scientific matrix; unequal work cannot pass this gate."""
    from experiments.e1.task_risk_experiment import _verify_hashes
    folder = Path(folder)
    _verify_hashes(folder, read(folder / 'output_hashes.json'))
    fairness = read(folder / 'fairness.json')
    require(fairness.get('split') == 'validation' and fairness.get('formal_test_evaluated') is False and
        fairness.get('policy_future_ci_access') is False, 'Evaluation role/online guard differs')
    require(all(fairness.get(k) is True for k in ('all_tasks_completed', 'equal_resource_hours',
        'equal_workload_carbon_comparison_valid', 'idle_included', 'development_validation')),
        'Drain/equal-work/idle fairness failed')
    require(fairness['arrived'] == record['tasks'] and
        fairness['workload_sha256'] == record['workload_sha256'] and
        fairness['environment_sha256'] == record['environment_sha256'] and
        fairness['capacity'] == record['capacity'], 'Frozen input identity differs')
    with (folder / 'common_summary.csv').open(encoding='utf8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 19 and {r['policy'] for r in rows} == set(expected_methods()),
            'Nineteen-method matrix differs')
    scientific = read(folder / 'scientific_hashes.json')
    require(set(scientific) == set(expected_methods()), 'Scientific hash matrix differs')
    for row in rows:
        require(row['split'] == 'validation' and row['phase'] == 'common' and
            float(row['factor']) == record['factor'] and
            int(row['arrived']) == int(row['completed']) == record['tasks'] and
            int(row['pending']) == int(row['running']) == int(row['deadline_right_censored']) == 0 and
            row['termination'] == 'all_completed' and row['observation_end'] == fairness['common_end'],
            'Completed work or common observation end differs')
        missed = int(row['deadline_missed_completed']) + int(row['deadline_missed_unfinished'])
        require(math.isclose(float(row['deadline_violation_rate_observed']), missed / record['tasks'],
            rel_tol=1e-12, abs_tol=1e-14), 'Deadline numerator/denominator differs')
        require(math.isclose(float(row['total_energy_kwh']),
            float(row['it_energy_kwh']) + float(row['hvac_energy_kwh']), rel_tol=1e-12),
            'Energy integral units/sum differ')
        for resource in ('cores', 'gpus', 'mem'):
            require(math.isclose(float(row[resource + '_hours']),
                float(rows[0][resource + '_hours']), rel_tol=1e-12, abs_tol=1e-8),
                'Completed resource hours differ')
    if replay:
        require(read(folder / 'replay-verification.json')['scientific_outputs_exact'] is True,
                'Scientific replay is not exact')
    return dict(factor=record['factor'],methods=19,completed_tasks=record['tasks'],
        common_end=fairness['common_end'],complete_drain=True,equal_resource_hours=True,
        idle_included=True,scientific_hashes=scientific,replay_exact=replay,
        files={name:file_hash(folder/name) for name in ('fairness.json','common_summary.csv',
            'scientific_hashes.json','output_hashes.json','source-freeze.json')})


def run_child(args):
    from experiments.e1.rl_training_utils import load_bundle
    from experiments.e1.adaptive_evaluate import evaluate_adaptive_group, load_adaptive_candidate, replay_adaptive_group
    from experiments.e1.rl_evaluate import load_ppo_candidate
    bundle = load_bundle(args.bundle, allowed_split='validation')
    if args.mode == 'replay':
        result = replay_adaptive_group(args.archive_group, args.output, bundle=bundle)
    else:
        candidates = []; external = []
        for item in read(args.candidates):
            if item['kind'] == 'adaptive':
                candidates.append(load_adaptive_candidate(item['checkpoint'],split='validation',name=item['name']))
            elif item['kind'] == 'legacy':
                external.append(load_ppo_candidate(item['checkpoint'],item,split='validation'))
            else:
                raise ValueError('Unknown candidate kind')
        result = evaluate_adaptive_group(bundle,args.output,candidates=candidates,external=external)
    print(json.dumps(result['fairness'],sort_keys=True),flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('evaluate','replay'),default='evaluate')
    parser.add_argument('--output')
    parser.add_argument('--formal',default=str(NOTES/'formal-campaign-001'))
    parser.add_argument('--archive',help='Completed evaluation campaign; required for replay')
    parser.add_argument('--workers',type=int,choices=(1,2,3),default=2)
    parser.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--bundle',help=argparse.SUPPRESS)
    parser.add_argument('--candidates',help=argparse.SUPPRESS)
    parser.add_argument('--archive-group',help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        require(args.output and args.bundle,'Missing child input/output')
        run_child(args);return
    out = Path(args.output or str(NOTES/('validation-campaign-001' if args.mode=='evaluate' else 'validation-replay-001')))
    if out.exists():
        raise FileExistsError(out)
    sources=experiment_sources();source_sha=content_hash(sources)
    cfgpath=ROOT/'experiments/e1/configs/e1_6b_ppo.json';config_sha=file_hash(cfgpath)
    formal=Path(args.formal);archive=Path(args.archive) if args.archive else None
    if args.mode=='evaluate':
        candidates=candidate_records(formal,source_sha)
        parent_binding=dict(formal_receipt_sha256=file_hash(formal/'campaign-receipt.json'))
        records=input_records()
    else:
        require(archive is not None,'Replay requires an evaluation archive')
        old=read(archive/'campaign.json');receipt=read(archive/'campaign-receipt.json')
        records=input_records()
        require(receipt.get('passed') is True and old['mode']=='evaluate' and
            old['source_sha256']==source_sha and old['inputs']==records and
            old['methods']==expected_methods(),'Replay archive, source, input or matrix differs')
        candidates=read(archive/'candidates.json')
        for record in records:
            validate_output(archive/('load-'+record['label']),record)
        parent_binding=dict(evaluation_campaign_sha256=file_hash(archive/'campaign.json'),
                            evaluation_receipt_sha256=file_hash(archive/'campaign-receipt.json'))
    jobs=[]
    for record in records:
        command=[sys.executable,str(Path(__file__).resolve()),'--child','--mode',args.mode,
            '--bundle',record['path'],'--candidates',str(out/'candidates.json'),
            '--output',str(out/('load-'+record['label']))]
        if args.mode=='replay':
            command+=['--archive-group',str(archive/('load-'+record['label']))]
        jobs.append(dict(record=record,command=command,output=str(out/('load-'+record['label'])),
            log=str(out/('load-'+record['label']+'.txt'))))
    require(experiment_sources()==sources and file_hash(cfgpath)==config_sha,'Sources changed during preflight')
    out.mkdir(parents=True,exist_ok=False)
    _snapshot_sources(out,sources)
    shutil.copyfile(cfgpath,out/'preregistered-config.json')
    write_json(out/'candidates.json',candidates)
    driver_sha=file_hash(Path(__file__))
    manifest=dict(schema=SCHEMA,created_at_utc=stamp(),mode=args.mode,inputs=records,jobs=jobs,
        methods=expected_methods(),source_sha256=source_sha,config_sha256=config_sha,
        driver_sha256=driver_sha,workers=args.workers,development_validation=True,
        formal_test_evaluated=False,validation_training=False,model_selection=False,
        total_core_results=57,**parent_binding)
    write_json(out/'campaign.json',manifest)
    write_json(out/'source-freeze.json',dict(sources=sources,source_sha256=source_sha))
    progress=(out/'progress.jsonl').open('wb');running={};completed=[];next_job=0;failed=False;reason=None
    def event(row):
        value=dict(time_utc=stamp(),**row)
        progress.write(canonical_bytes(value)+b'\n');progress.flush()
        print(json.dumps(value,sort_keys=True),flush=True)
    try:
        while running or (next_job<len(jobs) and not failed):
            if (experiment_sources()!=sources or file_hash(cfgpath)!=config_sha or
                file_hash(Path(__file__))!=driver_sha):
                failed=True;reason='source_or_protocol_changed'
                for process,_,_,_ in running.values():
                    process.terminate()
            while not failed and len(running)<args.workers and next_job<len(jobs):
                job=jobs[next_job];next_job+=1
                stream=Path(job['log']).open('x',encoding='utf8')
                process=subprocess.Popen(job['command'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                    text=True,encoding='utf8')
                running[process.pid]=(process,stream,job,time.monotonic())
                event(dict(event='started',factor=job['record']['factor'],pid=process.pid))
            for pid,(process,stream,job,began) in list(running.items()):
                if process.poll() is None:
                    if time.monotonic()-began>7200:
                        failed=True;reason='evaluation_wall_limit';process.terminate()
                    continue
                stream.close();del running[pid]
                row=dict(factor=job['record']['factor'],exit_code=process.returncode,
                    elapsed_seconds=time.monotonic()-began,output=job['output'],log=job['log'])
                if process.returncode==0:
                    try:
                        row.update(validate_output(job['output'],job['record'],args.mode=='replay'))
                    except Exception as exc:
                        failed=True;reason='scientific_artifact_gate_failed'
                        row['audit_error']=type(exc).__name__+': '+str(exc)
                else:
                    failed=True;reason=reason or 'evaluation_child_failed'
                completed.append(row);event(dict(event='finished',**row))
                write_json(out/'progress-receipt.json',dict(completed=completed,
                    running=[job for _,_,job,_ in running.values()],unstarted_jobs=jobs[next_job:],failed=failed))
            if running:
                time.sleep(2)
        if not failed:
            require(len(completed)==3 and experiment_sources()==sources and file_hash(cfgpath)==config_sha and
                    file_hash(Path(__file__))==driver_sha,
                    'Final matrix/source/config differs')
            require(input_records()==records,'Frozen Validation inputs changed during execution')
            if args.mode=='evaluate':
                require(candidate_records(formal,source_sha)==candidates,'Frozen model identity changed during execution')
        receipt=dict(schema=SCHEMA,completed_at_utc=stamp(),mode=args.mode,
            passed=not failed and len(completed)==3,failure_reason=reason,results=completed,
            unstarted_jobs=jobs[next_job:],source_sha256=source_sha,config_sha256=config_sha,
            campaign_manifest_sha256=file_hash(out/'campaign.json'),
            candidates_sha256=file_hash(out/'candidates.json'),
            formal_test_evaluated=False,validation_training=False,model_selection=False)
        write_json(out/'campaign-receipt.json',receipt)
        event(dict(event='campaign_complete',passed=receipt['passed'],core_results=sum(r.get('methods',0) for r in completed)))
        require(receipt['passed'],'Development Validation campaign failed; all evidence preserved')
        return receipt
    except Exception as exc:
        write_json(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),
            failure_reason=reason,completed=completed,unstarted_jobs=jobs[next_job:]))
        raise
    finally:
        for process,stream,_,_ in running.values():
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill();process.wait()
            stream.close()
        progress.close()


if __name__=='__main__':
    main()
