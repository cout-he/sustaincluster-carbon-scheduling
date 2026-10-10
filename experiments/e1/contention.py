"""E1-2B: fixed arrival-count pressure; unchanged temporal baselines and physics."""
import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

from .config import ROOT
from .contracts import content_hash, file_hash, write_json
from .policies import POLICY_NAMES
from .run import _fresh_run, dependencies
from .selection import TemporalEnvironment
from .stress import VERSION, stress_workload
from .temporal import experiment_sources, run_comparison, write_csv
from .workload import FrozenWorkload


def factor_folder(factor):
    return 'load_'+format(float(factor),'.2f').replace('.','p')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def verify_archive(path, current_sources=False):
    path=Path(path)
    if read_json(path/'run_status.json')['state']!='complete':
        raise ValueError('Incomplete reference archive')
    prov=read_json(path/'provenance.json')
    required={'workload.json','environment.json','dc_config.json','resolved_config.json','design.json','selection.json','comparison.csv','timeseries.csv','fairness.json','paired_task_shifts.csv'}
    required.update(f'{p}/{n}' for p in POLICY_NAMES for n in ('ledger.jsonl','events.jsonl','intervals.csv','summary.json','policy_ledger.jsonl','policy_events.jsonl','policy_intervals.jsonl','policy_summary.json','resolved_config.json','phases.json','fixed_ledger.json'))
    if set(prov['output_hashes'])!=required or content_hash(prov['sources'])!=prov['source_sha256']:
        raise ValueError('Reference manifest incomplete or corrupt')
    for name,sha in prov['output_hashes'].items():
        if file_hash(path/name)!=sha:raise ValueError('Reference output hash mismatch: '+name)
    for name,sha in prov['sources'].items():
        if file_hash(path/'source'/name)!=sha:raise ValueError('Reference source hash mismatch: '+name)
    if current_sources and (prov['sources']!=experiment_sources() or prov['dependencies']!=dependencies() or prov['python']!=sys.version):
        raise ValueError('Current source/dependency replay mismatch')
    w=FrozenWorkload.load(path/'workload.json');e=TemporalEnvironment.load(path/'environment.json')
    if w.sha256!=prov['input_hashes']['workload_snapshot'] or e.sha256!=prov['input_hashes']['environment_snapshot']:
        raise ValueError('Reference frozen input mismatch')
    return w,e


def contention_cases(ledgers, events):
    by_policy={p:{t['task_id']:t for t in rows} for p,rows in ledgers.items()}
    ids=set(by_policy['immediate_fifo'])
    if any(set(rows)!=ids for rows in by_policy.values()):raise ValueError('Paired task IDs differ')
    miss=lambda t:t['deadline_outcome'] in ('missed','missed_unfinished')
    sets={p:{key for key,t in rows.items() if miss(t)} for p,rows in by_policy.items()}
    f,d,c=(sets[p] for p in POLICY_NAMES)
    only={k for k in c if all(by_policy[p][k]['deadline_outcome']=='met' for p in POLICY_NAMES[:2])}
    counts=dict(shared_missed_all=len(f&d&c),carbon_only_missed_vs_both=len(only),
        carbon_missed_baselines_censored=len((c-f-d)-only),
        carbon_extra_vs_fifo=len(c-f),carbon_extra_vs_edf=len(c-d),
        carbon_avoided_vs_fifo=len(f-c),carbon_avoided_vs_edf=len(d-c),
        fifo_edf_discordant=len(f^d))
    grouped={key:[] for key in only}
    for event in events:
        if event['task_id'] in grouped:grouped[event['task_id']].append(event)
    cases=[]
    for key,items in grouped.items():
        deferred=sorted(x['time'] for x in items if x['event']=='policy_deferred')
        capacity=sorted(x['time'] for x in items if x['event']=='capacity_waited')
        if deferred and any(t>deferred[0] for t in capacity):
            cases.append(dict(task_id=key,first_policy_defer=deferred[0],
                first_capacity_wait_after_defer=next(t for t in capacity if t>deferred[0]),
                tasks={p:rows[key] for p,rows in by_policy.items()},carbon_events=items,
                attribution='paired association; not strict single-event causation'))
    return counts,sorted(cases,key=lambda x:(x['first_policy_defer'],x['task_id']))


def diagnose(folder):
    ledgers={p:read_jsonl(folder/p/'ledger.jsonl') for p in POLICY_NAMES}
    events=read_jsonl(folder/'carbon_threshold_fifo/policy_events.jsonl')
    counts,cases=contention_cases(ledgers,events)
    # Save only the first deterministic case; all-count diagnostics retain scope.
    case=cases[0] if cases else None
    if case:
        id=case['task_id']
        case['events_by_policy']={p:read_jsonl(folder/p/'policy_events.jsonl') for p in POLICY_NAMES}
        case['events_by_policy']={p:[e for e in rows if e['task_id']==id] for p,rows in case['events_by_policy'].items()}
        step_by_time={s['time']:s for s in read_jsonl(folder/'carbon_threshold_fifo/policy_intervals.jsonl')}
        case['competition_snapshots']=[step_by_time[x['time']] for x in case['carbon_events'] if x['event']=='capacity_waited']
        source=ledgers['carbon_threshold_fifo']
        case['competitors']={t['task_id']:t for t in source if any(t['task_id'] in snap['running_ids'] for snap in case['competition_snapshots'])}
    return dict(**counts,cases_with_defer_then_capacity=len(cases),representative=case)


def baseline_regression(reference, one):
    reference,one=Path(reference),Path(one)
    old=read_json(reference/'provenance.json')['output_hashes']
    numerical=[n for n in old if n in ('comparison.csv','timeseries.csv','paired_task_shifts.csv','fairness.json') or n.split('/')[0] in POLICY_NAMES]
    differences=[n for n in numerical if file_hash(reference/n)!=file_hash(one/n)]
    if differences:raise RuntimeError('1x historical numerical regression mismatch: '+str(differences))
    return dict(matched=True,files_checked=len(numerical),historical_provenance_untouched=True,
                rule='byte-identical scientific results; new source/provenance not relabeled as old')


def run_levels(cfg, workload, environment, model_path, output_dir, factors, seed, threshold,
               selection=None, historical_reference=None, verify_replay=None):
    factors=list(factors)
    if not factors or factors!=sorted(set(factors)) or factors[0]!=1:
        raise ValueError('Factors must be unique increasing levels beginning at1')
    # Validate all generator requests before running any policy.
    for f in factors:stress_workload(workload,f,seed)
    def action(out):
        hashes=experiment_sources()
        write_json(out/'config.json',dict(version=VERSION,factors=factors,seed=seed,threshold_g_per_kwh=threshold,
            reference_config=cfg,base_workload_sha256=workload.sha256,environment_sha256=environment.sha256,
            dc_config_sha256=file_hash(model_path),synthetic=True))
        rows,diagnostics,identities=[],{},[]
        for factor in factors:
            stressed,manifest=stress_workload(workload,factor,seed)
            folder=out/factor_folder(factor)
            print(f'Running synthetic arrival factor {factor}: {len(stressed.tasks)} tasks',flush=True)
            phases=run_comparison(cfg,stressed,environment,model_path,folder,threshold,selection)
            write_json(out/(factor_folder(factor)+'_replication.json'),manifest)
            for p in POLICY_NAMES:
                fair=read_json(folder/'fairness.json')
                for phase,s in phases[p].items():
                    child_rows=read_json(folder/p/'phases.json')
                    assert child_rows[phase]==s
                    drained=read_json(folder/p/'summary.json')['drained_at']
                    rows.append(dict(factor=float(factor),realized_factor=manifest['realized_factor'],policy=p,phase=phase,
                        drain_hours=(pd.Timestamp(drained)-workload.end).total_seconds()/3600 if drained else None,
                        termination='all_completed' if drained else 'safety_cap_incomplete',
                        equal_workload_carbon_comparison_valid=fair['equal_workload_carbon_comparison_valid'],**s))
            diagnostics[str(factor)]=diagnose(folder)
            identities.append(dict(factor=float(factor),workload_sha256=stressed.sha256,environment_sha256=environment.sha256,
                                   dc_config_sha256=file_hash(model_path),common_end=fair['common_end'],tasks=len(stressed.tasks)))
        write_csv(out/'comparison.csv',rows)
        write_json(out/'diagnostics.json',diagnostics)
        write_json(out/'input_identities.json',identities)
        regression=baseline_regression(historical_reference,out/factor_folder(1)) if historical_reference else None
        write_json(out/'baseline_regression.json',regression)
        if hashes!=experiment_sources():raise RuntimeError('Sources changed during campaign')
        names=['config.json','comparison.csv','diagnostics.json','input_identities.json','baseline_regression.json']
        names.extend(factor_folder(f)+'_replication.json' for f in factors)
        write_json(out/'provenance.json',dict(version=VERSION,sources=hashes,source_sha256=content_hash(hashes),
            command=sys.argv,python=sys.version,dependencies=dependencies(),
            output_hashes={n:file_hash(out/n) for n in names},
            child_output_manifests={factor_folder(f):read_json(out/factor_folder(f)/'provenance.json') for f in factors}))
        # Verification is part of the run lifecycle, before the complete marker.
        if verify_replay is not None:verify_replay(out)
        return rows
    return _fresh_run(output_dir,action)


def replay_levels(previous, output):
    previous=Path(previous);prov=read_json(previous/'provenance.json')
    if read_json(previous/'run_status.json')['state']!='complete' or prov['version']!=VERSION or content_hash(prov['sources'])!=prov['source_sha256'] or prov['sources']!=experiment_sources() or prov['python']!=sys.version or prov['dependencies']!=dependencies():
        raise ValueError('Campaign replay status/source/dependency mismatch')
    config=read_json(previous/'config.json')
    required={'config.json','comparison.csv','diagnostics.json','input_identities.json','baseline_regression.json'}
    children={factor_folder(f) for f in config['factors']}
    required.update(p+'_replication.json' for p in children)
    if set(prov['output_hashes'])!=required or set(prov['child_output_manifests'])!=children:
        raise ValueError('Campaign replay manifest incomplete')
    for n,sha in prov['output_hashes'].items():
        if file_hash(previous/n)!=sha:raise ValueError('Campaign replay hash mismatch: '+n)
    for f in config['factors']:
        folder=previous/factor_folder(f)
        if read_json(folder/'provenance.json')!=prov['child_output_manifests'][factor_folder(f)]:raise ValueError('Child manifest changed')
        verify_archive(folder,True)
    w,e=verify_archive(previous/factor_folder(1),True)
    # Recover selection and configuration from the sealed1x child, not external data.
    def verify(out):
        for n,sha in prov['output_hashes'].items():
            if n=='baseline_regression.json':continue  # Historical external path isn't a replay dependency.
            if file_hash(out/n)!=sha:raise RuntimeError('Campaign replay numerical mismatch: '+n)
        for f in config['factors']:
            p=factor_folder(f)
            for n,sha in prov['child_output_manifests'][p]['output_hashes'].items():
                if file_hash(out/p/n)!=sha:raise RuntimeError('Child replay mismatch: '+p+'/'+n)
    return run_levels(config['reference_config'],w,e,previous/factor_folder(1)/'dc_config.json',output,
        config['factors'],config['seed'],config['threshold_g_per_kwh'],read_json(previous/factor_folder(1)/'selection.json'),verify_replay=verify)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,help='Complete private E1-2A final archive, outside publication Git')
    parser.add_argument('--config',type=Path,default=ROOT/'experiments/e1/configs/e1_2b_stress.json')
    parser.add_argument('--replay',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.replay:
        replay_levels(args.replay,args.output);return
    if args.reference is None:parser.error('--reference is required for a new fixed-input campaign')
    rule=read_json(args.config)
    if rule['factors']!=[1,1.25,1.5,2] or rule['threshold_g_per_kwh']!=137.65 or rule['sla_multiplier']!=1.5:
        raise ValueError('This stage uses the fixed four-level E1-2B design')
    w,e=verify_archive(args.reference)
    cfg=read_json(args.reference/'resolved_config.json');design=read_json(args.reference/'design.json')
    if w.start.isoformat()!=rule['start'] or w.end.isoformat()!=rule['end'] or w.sla_multiplier!=1.5 or len(w.tasks)!=9943:
        raise ValueError('Reference is not the approved E1-2A cohort')
    expected=dict(total_cores=50000,total_gpus=1000,total_mem=80000,location='US-CAL-CISO',hvac_setpoint_c=22)
    if any(cfg['datacenter'][k]!=v for k,v in expected.items()) or not math.isclose(design['threshold_g_per_kwh'],137.65,rel_tol=0,abs_tol=1e-12):
        raise ValueError('Reference capacity/model/threshold differs from approved design')
    run_levels(cfg,w,e,args.reference/'dc_config.json',args.output,rule['factors'],rule['seed'],
               design['threshold_g_per_kwh'],read_json(args.reference/'selection.json'),args.reference)


if __name__=='__main__':main()
