"""Frozen Train-only entry point for bounded E1-7A scientific diagnostics."""
import argparse
import copy
import csv
import json
import math
from pathlib import Path
import re
import sys
import time

import pandas as pd

from .config import ROOT
from .contracts import content_hash, file_hash, write_json
from .diagnostic_selection import VERSION as SELECTION_VERSION
from .diagnostic_selection import build_selection_records, select_diagnostic_cases
from .policies import EDFBackfill
from .rl_training_utils import load_bundle
from .run import _fresh_run, _write_result, source_hashes

VERSION = 'e1-7a-diagnostics-v1'
AUDIT_VERSION = 'e1-7a-driver-audit-v1.1'
MANIFEST_SCHEMA = 'e1-7a-selection-manifest-v1'


def _same_utc_instant(left, right):
    left, right = pd.Timestamp(left), pd.Timestamp(right)
    if left.tzinfo is None or right.tzinfo is None:
        raise ValueError('Common endpoints must be timezone aware')
    return left == right


def validate_config(config):
    fixed = dict(schema=VERSION, allowed_split='train', validation_selection=False,
        test_access=False, arrival_start='2023-05-01T05:00:00+00:00',
        arrival_end='2023-05-03T05:00:00+00:00', threshold_g_per_kwh=161.075,
        sla_multiplier=1.5, resource_multiplier=5, step_minutes=15,
        capacity=[50000, 1000, 80000], stress_seed=510001, factors=[1.0, 1.5, 2.0],
        selection_version=SELECTION_VERSION,
        publication_base='80f5c84c2ae45bf86de397a4421d8cdedba7e03f')
    if any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError('Registered Train-only E1-7A protocol differs')
    if config.get('historical_root') != 'research_notes/E1/E1-5B/precheck-001':
        raise ValueError('Unregistered Train reference root')
    if config['A']['solver'] != dict(version='9.14.6206', max_seconds=120, workers=1,
                                    seed=710001, conservative_scale=1000000):
        raise ValueError('Registered bounded solver budget differs')
    if config['A']['min_tasks'] != 10 or config['A']['max_tasks'] != 40:
        raise ValueError('Registered finite subset size differs')
    if config['C']['continuation_policy'] != 'edf_backfill' or config['C']['target_pairs'] != 12:
        raise ValueError('Registered single-action intervention protocol differs')
    if config['B']['future_ci_use'] != 'offline_clairvoyant_diagnostic_only':
        raise ValueError('Future CI may only be a labelled offline diagnostic')
    hashes = config['bundle_manifest_sha256']
    if set(hashes) != {'1', '1.5', '2'} or any(not re.fullmatch('[0-9a-f]{64}', value) for value in hashes.values()):
        raise ValueError('Malformed frozen input SHA256')
    if content_hash(config) != 'b7bf0bb8129e509475cc4d962f779b7ebab5ee9b3f99f2eaed9d8b19f77ff987':
        raise ValueError('Frozen whole configuration identity differs')
    return copy.deepcopy(config)


def load_selection_manifest(path):
    document = json.loads(Path(path).read_text(encoding='utf8'))
    payload = document['payload']
    # Refuse roles before any declared input path is followed.
    if payload.get('split') != 'train' or payload.get('test_access') is not False:
        raise ValueError('Selection manifest must be Train only')
    if content_hash(payload) != document.get('sha256'):
        raise ValueError('Selection manifest hash mismatch')
    if payload.get('schema') != MANIFEST_SCHEMA:
        raise ValueError('Selection schema differs')
    cfg = validate_config(payload['config'])
    if content_hash(cfg) != payload['config_sha256']:
        raise ValueError('Registered configuration hash mismatch')
    selector = ROOT/'experiments/e1/diagnostic_selection.py'
    if payload['selector_sha256'] != file_hash(selector):
        raise ValueError('Frozen selector code changed')
    return payload


def load_cohort(config, label):
    if label not in ('1', '1.5', '2'):
        raise ValueError('Only registered Train factors permitted')
    folder = ROOT/config['historical_root']/('load-'+label)
    if file_hash(folder/'input/bundle.json') != config['bundle_manifest_sha256'][label]:
        raise ValueError('Frozen Train manifest identity differs')
    bundle = load_bundle(folder/'input', allowed_split='train')
    if bundle.factor != float(label) or bundle.stress_seed != config['stress_seed']:
        raise ValueError('Frozen Train load/seed differs')
    return bundle, folder/'evaluation'


def read_reference(reference):
    """Read only the authorized Train archive after its scientific hashes pass."""
    expected = json.loads((reference/'output_hashes.json').read_text(encoding='utf8'))
    names = ['fairness.json'] + [policy+'/'+name for policy in (
        'edf_backfill', 'edf_carbon_threshold', 'cars_risk_only') for name in (
        'ledger.jsonl', 'events.jsonl', 'intervals.csv', 'summary.json',
        'policy_ledger.jsonl', 'policy_events.jsonl', 'policy_intervals.jsonl', 'policy_summary.json')]
    actual = {}
    for name in names:
        if name not in expected or file_hash(reference/name) != expected[name]:
            raise ValueError('Train reference hash mismatch: '+name)
        actual[name] = expected[name]
    def lines(name):
        return tuple(json.loads(line) for line in (reference/name).read_text(encoding='utf8').splitlines())
    return dict(hashes=actual,
        fairness=json.loads((reference/'fairness.json').read_text(encoding='utf8')),
        tasks=lines('edf_backfill/ledger.jsonl'),
        policy_events=lines('edf_backfill/policy_events.jsonl'),
        policy_steps=lines('edf_backfill/policy_intervals.jsonl'))


def freeze_selection(config, output):
    config = validate_config(config)
    def action(out):
        cohorts, cases_a, cases_c, missing = [], [], [], []
        for factor in config['factors']:
            label = format(factor, 'g')
            bundle, reference = load_cohort(config, label)
            archive = read_reference(reference)
            records = build_selection_records(bundle.workload, bundle.capacity,
                archive['policy_events'], archive['policy_steps'], split=bundle.split)
            selected = select_diagnostic_cases(records, min_tasks=config['A']['min_tasks'],
                                               max_tasks=config['A']['max_tasks'])
            private = out/('load-'+label)
            private.mkdir()
            write_json(private/'all-candidate-features.json', records)
            specs = {spec.task_id: spec for spec in bundle.workload.tasks}
            for record in selected['A']:
                ordered = EDFBackfill().order(tuple(specs[key] for key in record['pending_ids']))
                selected_ids = [spec.task_id for spec in ordered[:config['A']['max_tasks']]]
                cases_a.append(dict(record, factor=factor, case_id='load-'+label+'-'+record['category'],
                    selected_ids=selected_ids, selected_ids_sha256=content_hash(selected_ids)))
            if factor in config['C']['factors']:
                for record in selected['C']:
                    cases_c.append(dict(record, factor=factor,
                        case_id='load-'+label+'-'+record['category']))
            missing.extend(dict(row, factor=factor) for row in selected['missing']
                           if row['experiment']=='A' or factor in config['C']['factors'])
            cohorts.append(dict(factor=factor, input_identity=bundle.identity,
                reference_path=reference.relative_to(ROOT).as_posix(),
                baseline_hashes=archive['hashes'], common_end=archive['fairness']['common_end'],
                eligible_snapshot_count=len(records['snapshots']),
                eligible_candidate_count=len(records['candidates']),
                all_features_sha256=file_hash(private/'all-candidate-features.json')))
        payload = dict(schema=MANIFEST_SCHEMA, split='train', test_access=False,
            config=config, config_sha256=content_hash(config),
            selector_sha256=file_hash(ROOT/'experiments/e1/diagnostic_selection.py'),
            cohorts=cohorts, A_cases=cases_a, C_cases=cases_c, missing_categories=missing,
            selected_before_solver_or_intervention_outcomes=True,
            selection_uses_current_past_reference_features_only=True)
        write_json(out/'E1-7A-selection-manifest.json', dict(payload=payload, sha256=content_hash(payload)))
        write_json(out/'config.json', config)
        print(json.dumps(dict(selection_sha256=content_hash(payload), A_cases=len(cases_a),
                              C_cases=len(cases_c), missing=missing)))
        return payload
    return _fresh_run(output, action)


def _csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def _source_state():
    sources = source_hashes()
    sources.update({path.relative_to(ROOT).as_posix(): file_hash(path)
                    for path in sorted((ROOT/'tests/e1').glob('*.py'))})
    return sources


def _receipt(out, sources, payload, rows, *, extra=None):
    if any(file_hash(ROOT/name) != digest for name,digest in sources.items()):
        raise RuntimeError('Registered execution source changed during stage')
    scientific = {path.relative_to(out).as_posix(): file_hash(path)
                  for path in sorted(out.rglob('*')) if path.is_file() and path.name != 'run_status.json'}
    data = dict(passed=True, schema=VERSION, audit_version=AUDIT_VERSION, split='train', test_access=False,
        selection_sha256=content_hash(payload), config_sha256=payload['config_sha256'],
        sources=sources, source_sha256=content_hash(sources), command=sys.argv,
        python=sys.version, output_hashes=scientific, rows=rows)
    if extra:
        data.update(extra)
    write_json(out/'receipt.json', data)
    return data


def run_baselines(manifest, output):
    payload = load_selection_manifest(manifest)
    from .cars import make_cars_policy
    from .diagnostic_audit import audit_fair_results, diagnostic_summary
    from .engine import LocalExecutor
    def action(out):
        sources, rows, regressions = _source_state(), [], []
        for cohort in payload['cohorts']:
            label = format(cohort['factor'],'g')
            bundle, reference = load_cohort(payload['config'], label)
            archive = read_reference(reference)
            results = {}
            for name in payload['config']['baseline_policies']:
                began=time.perf_counter()
                ex=LocalExecutor(bundle.workload,bundle.environment,bundle.capacity,bundle.power_factory(),
                    make_cars_policy(name,bundle.parameters,bundle.workload.start,bundle.capacity),drain=True)
                ex.run(stop_at_drain=True)
                result=ex.extend_idle(pd.Timestamp(cohort['common_end']))
                folder=out/('load-'+label)/name
                folder.mkdir(parents=True)
                _write_result(folder,result)
                for file in ('ledger.jsonl','events.jsonl','intervals.csv','summary.json',
                             'policy_ledger.jsonl','policy_events.jsonl','policy_intervals.jsonl','policy_summary.json'):
                    if file_hash(folder/file) != archive['hashes'][name+'/'+file]:
                        raise RuntimeError('Historic Train regression changed: '+label+'/'+name+'/'+file)
                results[name]=result
                rows.append(dict(factor=bundle.factor,policy=name,**diagnostic_summary(result,bundle.workload.end),
                                 replay_seconds=time.perf_counter()-began))
                regressions.append(dict(factor=bundle.factor,policy=name,exact_scientific_files=8,passed=True))
                print(json.dumps(dict(stage='baseline',factor=bundle.factor,policy=name,passed=True)),flush=True)
            write_json(out/('load-'+label)/'audit.json',audit_fair_results(
                results,bundle.workload,bundle.environment,bundle.capacity))
        _csv(out/'baseline-comparison.csv',rows)
        _csv(out/'historical-regression.csv',regressions)
        return _receipt(out,sources,payload,rows,extra={'exact_historical_baselines':9})
    return _fresh_run(output,action)


def run_offline(manifest, output):
    payload = load_selection_manifest(manifest)
    from .offline_schedule_cases import OfflineCase, background_from_reference, synthetic_reservation_case
    from .offline_schedule import SolverBudget, solve_deadline_case, run_conditional_edf, replay_schedule
    from .diagnostic_audit import audit_fair_results, diagnostic_summary
    from .workload import FrozenWorkload
    def action(out):
        sources, comparisons, schedules, status_rows = _source_state(), [], [], []
        cache = {}
        cases = [(None,synthetic_reservation_case(),None)]
        for selected in payload['A_cases']:
            label=format(selected['factor'],'g')
            if label not in cache:
                bundle,reference=load_cohort(payload['config'],label)
                cache[label]=(bundle,read_reference(reference))
            bundle,archive=cache[label]
            specs={spec.task_id:spec for spec in bundle.workload.tasks}
            tasks=tuple(specs[key] for key in selected['selected_ids'])
            metadata=dict(selection_case=selected['case_id'],category=selected['category'],
                          anchor=selected['time'],selection_sha256=content_hash(payload))
            for mode in ('A1','A2'):
                background=() if mode=='A1' else background_from_reference(
                    bundle.workload,archive['tasks'],selected['selected_ids'])
                start=bundle.workload.start if mode=='A1' else pd.Timestamp(selected['time'])
                case=OfflineCase(selected['case_id']+'-'+mode,mode,tasks,start,
                    pd.Timestamp(archive['fairness']['common_end']),bundle.capacity,start,background,
                    bundle.workload.sha256,metadata)
                cases.append((bundle,case,selected['factor']))
        for bundle,case,factor in cases:
            folder=out/case.case_id
            folder.mkdir()
            write_json(folder/'case.json',case.to_dict())
            began=time.perf_counter()
            if bundle is None:
                edf=run_conditional_edf(case)
                solution=solve_deadline_case(case,SolverBudget())
                actual=None
            else:
                w=(FrozenWorkload(bundle.workload.start,bundle.workload.end,bundle.workload.source_sha256,
                    bundle.workload.resource_multiplier,bundle.workload.sla_multiplier,
                    bundle.workload.trace_year,case.tasks) if case.mode=='A1' else bundle.workload)
                input_identity=w.sha256,bundle.environment.sha256
                edf=run_conditional_edf(case,workload=w,environment=bundle.environment,power_model=bundle.power_factory())
                solution=solve_deadline_case(case,SolverBudget())
                actual={}
                # Reproduce both registered schedules under the persistent true model.
                for name,sol in (('conditional_edf',edf),('oracle',solution)):
                    if sol.status not in ('EDF_REFERENCE','OPTIMAL','FEASIBLE'):
                        continue
                    result=replay_schedule(case,dict(sol.starts),w,bundle.environment,bundle.power_factory())
                    sub=folder/name
                    sub.mkdir()
                    _write_result(sub,result)
                    actual[name]=result
                if (w.sha256,bundle.environment.sha256)!=input_identity:
                    raise RuntimeError('Offline schedule mutated complete frozen input')
                write_json(folder/'physical-audit.json',audit_fair_results(actual,w,bundle.environment,bundle.capacity))
            write_json(folder/'edf-solution.json',edf.to_dict())
            write_json(folder/'oracle-solution.json',solution.to_dict())
            status_rows.append(dict(case_id=case.case_id,factor=factor,mode=case.mode,
                status=solution.status,objective=solution.miss_count,best_bound=solution.best_bound,
                solver_seconds=solution.wall_seconds,total_case_seconds=time.perf_counter()-began,
                solver_version=solution.solver_version,case_sha256=case.sha256,
                integer_modes='/'.join(value['mode'] for value in solution.precision)))
            comparisons.append(dict(case_id=case.case_id,factor=factor,mode=case.mode,task_count=len(case.tasks),
                background_tasks=len(case.background),edf_misses=edf.miss_count,oracle_misses=solution.miss_count,
                status=solution.status,best_bound=solution.best_bound,
                edf_mean_jct_minutes=edf.verification['mean_jct_minutes'],
                oracle_mean_jct_minutes=solution.verification['mean_jct_minutes'] if solution.verification else None,
                edf_max_wait_minutes=edf.verification['maximum_wait_minutes'],
                oracle_max_wait_minutes=solution.verification['maximum_wait_minutes'] if solution.verification else None,
                unfinished_edf=0,unfinished_oracle=0 if solution.verification else None,
                physical_carbon_edf=actual['conditional_edf'].summary['carbon_kg'] if actual else None,
                physical_carbon_oracle=actual['oracle'].summary['carbon_kg'] if actual and 'oracle' in actual else None,
                optimality_scope=solution.optimality_scope))
            for name,sol in (('conditional_edf',edf),('oracle',solution)):
                if sol.verification:
                    schedules.extend(dict(row,case_id=case.case_id,method=name,factor=factor,mode=case.mode)
                                     for row in sol.verification['tasks'])
            print(json.dumps(status_rows[-1]),flush=True)
        _csv(out/'A-edf-vs-oracle.csv',comparisons)
        _csv(out/'A-solver-status.csv',status_rows)
        _csv(out/'A-schedules.csv',schedules)
        _csv(out/'A-cases.csv',[{key:row[key] for key in ('case_id','factor','mode','task_count','background_tasks')}
                               for row in comparisons])
        return _receipt(out,sources,payload,comparisons)
    return _fresh_run(output,action)


def run_carbon(manifest, output, *, baseline_run=None):
    payload=load_selection_manifest(manifest)
    from .carbon_opportunity import (energy_decomposition,ci_temporal_statistics,
                                    calibrate_it_proxy,execution_window_opportunities)
    def action(out):
        sources,rows,bin_rows,activity_rows,opportunity_summary=_source_state(),[],[],[],[]
        baseline=Path(baseline_run) if baseline_run is not None else None
        if baseline is None:
            raise ValueError('Real verified baseline replay directory is required')
        baseline_receipt=json.loads((baseline/'receipt.json').read_text(encoding='utf8'))
        if not baseline_receipt['passed'] or baseline_receipt['selection_sha256']!=content_hash(payload):
            raise ValueError('Baseline replay is not bound to this frozen selection')
        for name,digest in baseline_receipt['output_hashes'].items():
            if file_hash(baseline/name)!=digest:
                raise ValueError('Baseline scientific replay was changed')
        train1=pd.read_csv(baseline/'load-1/edf_backfill/intervals.csv').to_dict('records')
        proxy=calibrate_it_proxy(train1,role='train')
        write_json(out/'IT-proxy-calibration.json',proxy)
        for cohort in payload['cohorts']:
            label=format(cohort['factor'],'g')
            bundle,reference=load_cohort(payload['config'],label)
            archive=read_reference(reference)
            for policy in payload['config']['baseline_policies']:
                steps=pd.read_csv(baseline/('load-'+label)/policy/'intervals.csv').to_dict('records')
                decomposition=energy_decomposition(steps)
                rows.append(dict(factor=bundle.factor,policy=policy,**decomposition['totals'],
                                 **decomposition['ci_means']))
                bin_rows.extend(dict(row,factor=bundle.factor,policy=policy) for row in decomposition['ci_bins'])
                activity_rows.extend(dict(value,factor=bundle.factor,policy=policy,activity=key)
                                     for key,value in decomposition['activity_groups'].items())
            temporal=ci_temporal_statistics(bundle.environment,bundle.workload.start,bundle.workload.end)
            private=out/('load-'+label)
            private.mkdir()
            write_json(private/'ci-statistics.json',temporal)
            _csv(private/'ci-hourly.csv',temporal['hour_rows'])
            _csv(private/'ci-quarter-hour.csv',temporal['sample_rows'])
            opportunity=execution_window_opportunities(bundle.workload,archive['policy_events'],
                                                       bundle.environment,proxy,delta_minutes=15)
            _csv(private/'task-carbon-opportunities.csv',opportunity['rows'])
            write_json(private/'opportunity-rejections.json',dict(
                coverage=opportunity['coverage_rejections'],eligibility=opportunity['eligibility_rejections']))
            eligible=opportunity['rows']
            high=[row for row in eligible if row['high_ci_at_candidate']]
            nonnegative=[row for row in eligible if row['it_proxy_nonnegative']]
            opportunity_summary.append(dict(factor=bundle.factor,eligible=len(eligible),high_current_ci=len(high),
                negative_predicted_it_count=len(eligible)-len(nonnegative),
                high_ci_with_gain=sum(row['local_it_proxy_gain_kg']>1e-9 for row in high),
                high_ci_without_gain=sum(row['local_it_proxy_gain_kg']<=1e-9 for row in high),
                same_start_ci_fraction=sum(row['start_ci_now_g_per_kwh']==row['start_ci_defer_g_per_kwh']
                                          for row in eligible)/len(eligible) if eligible else None,
                positive_proxy_gain=sum(row['local_it_proxy_gain_kg']>1e-9 for row in eligible),
                negative_proxy_gain=sum(row['local_it_proxy_gain_kg']< -1e-9 for row in eligible),
                zero_proxy_gain=sum(abs(row['local_it_proxy_gain_kg'])<=1e-9 for row in eligible),
                mean_duration_minutes=math.fsum(spec.duration for spec in bundle.workload.tasks)/len(bundle.workload.tasks),
                mean_grid_occupied_minutes=math.fsum(
                    ((pd.Timedelta(minutes=spec.duration).value+pd.Timedelta(minutes=15).value-1)
                     //pd.Timedelta(minutes=15).value)*15 for spec in bundle.workload.tasks)/len(bundle.workload.tasks),
                offline_clairvoyant_diagnostic=True,is_cluster_carbon_result=False))
        _csv(out/'B-energy-decomposition.csv',rows)
        _csv(out/'B-ci-bin-energy-carbon.csv',bin_rows)
        _csv(out/'B-active-idle-energy-carbon.csv',activity_rows)
        _csv(out/'B-carbon-opportunity.csv',opportunity_summary)
        return _receipt(out,sources,payload,rows,extra={'proxy':proxy,'opportunity_summary':opportunity_summary})
    return _fresh_run(output,action)


def run_counterfactual(manifest, output, *, case_id=None):
    payload=load_selection_manifest(manifest)
    from .counterfactual import InterventionTarget,run_paired_case
    from .counterfactual_audit import audit_pair
    from .diagnostic_audit import core_scientific_hash,full_scientific_hash,diagnostic_summary
    def action(out):
        sources,rows,features,impact_summary,cache=_source_state(),[],[],[],{}
        cases=[row for row in payload['C_cases'] if case_id is None or row['case_id']==case_id]
        if not cases:
            raise ValueError('Requested frozen counterfactual case is absent')
        for selected in cases:
            label=format(selected['factor'],'g')
            if label not in cache:
                bundle,reference=load_cohort(payload['config'],label)
                cache[label]=(bundle,read_reference(reference))
            bundle,archive=cache[label]
            target=InterventionTarget(selected['case_id'],selected['time'],selected['task_id'],
                                      selected['candidate_occurrence'])
            folder=out/target.case_id
            folder.mkdir()
            write_json(folder/'target.json',target.to_dict())
            began=time.perf_counter()
            pair=run_paired_case(bundle.workload,bundle.environment,bundle.capacity,bundle.power_factory,
                target,bundle.parameters,minimum_common_end=archive['fairness']['common_end'])
            audit=audit_pair(pair,bundle.workload,bundle.environment,bundle.capacity)
            write_json(folder/'audit.json',audit)
            write_json(folder/'prefixes.json',pair.prefixes)
            write_json(folder/'online-features.json',pair.features)
            write_json(folder/'interventions.json',pair.interventions)
            summaries={}
            for name,result in pair.results.items():
                sub=folder/name
                sub.mkdir()
                _write_result(sub,result)
                if name=='force_execute' and _same_utc_instant(pair.common_end,archive['fairness']['common_end']):
                    for file in ('ledger.jsonl','events.jsonl','intervals.csv','summary.json','policy_ledger.jsonl'):
                        if file_hash(sub/file)!=archive['hashes']['edf_backfill/'+file]:
                            raise RuntimeError('Execute control differs from frozen original EDF: '+file)
                summaries[name]=diagnostic_summary(result,bundle.workload.end)
            _csv(folder/'task-impacts.csv',pair.task_impacts)
            row=dict(case_id=target.case_id,factor=bundle.factor,category=selected['category'],
                intervention_time=target.time.isoformat(),task_id=target.task_id,
                prefix_equal=True,common_end=pair.common_end.isoformat(),
                historical_execute_control_files_verified=(5 if _same_utc_instant(
                    pair.common_end,archive['fairness']['common_end']) else 0),
                delta_misses=summaries['force_defer_once']['misses']-summaries['force_execute']['misses'],
                delta_carbon=summaries['force_defer_once']['carbon_kg']-summaries['force_execute']['carbon_kg'],
                pair_seconds=time.perf_counter()-began,
                execute_core_sha256=core_scientific_hash(pair.results['force_execute']),
                defer_core_sha256=core_scientific_hash(pair.results['force_defer_once']),
                execute_full_sha256=full_scientific_hash(pair.results['force_execute']),
                defer_full_sha256=full_scientific_hash(pair.results['force_defer_once']))
            for name,summary in summaries.items():
                row.update({name+'_'+key:value for key,value in summary.items()})
            rows.append(row)
            # Keep local large known-ID lists out of the small risk table.
            scalar={key:value for key,value in pair.features.items() if isinstance(value,(int,float,str,bool)) or value is None}
            features.append(dict(scalar,case_id=target.case_id,factor=bundle.factor,
                delta_misses=row['delta_misses'],delta_carbon=row['delta_carbon']))
            for partition in ('candidate','known_other','post_intervention_arrival'):
                selected_impacts=[item for item in pair.task_impacts if item['partition']==partition]
                impact_summary.append(dict(case_id=target.case_id,partition=partition,
                    total_tasks=len(selected_impacts),changed_tasks=sum(item['changed'] for item in selected_impacts),
                    starts_later=sum(item['delta_start_minutes']>0 for item in selected_impacts),
                    starts_earlier=sum(item['delta_start_minutes']<0 for item in selected_impacts),
                    met_to_missed=sum(item['execute_deadline_outcome']=='met' and item['defer_deadline_outcome']=='missed'
                                      for item in selected_impacts),
                    missed_to_met=sum(item['execute_deadline_outcome']=='missed' and item['defer_deadline_outcome']=='met'
                                      for item in selected_impacts)))
            print(json.dumps({key:row[key] for key in ('case_id','delta_misses','delta_carbon','pair_seconds')}),flush=True)
        _csv(out/'C-paired-outcomes.csv',rows)
        _csv(out/'C-risk-vs-outcome.csv',features)
        _csv(out/'C-task-impact-summary.csv',impact_summary)
        write_json(out/'C-case-manifest.json',dict(cases=cases,selection_sha256=content_hash(payload)))
        return _receipt(out,sources,payload,rows,extra={'single_intervention_only':True,'all_prefixes_equal':True})
    return _fresh_run(output,action)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    freeze = sub.add_parser('freeze')
    freeze.add_argument('--config', required=True)
    freeze.add_argument('--output', required=True)
    for name in ('baseline','offline','carbon','counterfactual'):
        stage=sub.add_parser(name)
        stage.add_argument('--manifest',required=True)
        stage.add_argument('--output',required=True)
        if name=='carbon':
            stage.add_argument('--baseline-run',required=True)
        if name=='counterfactual':
            stage.add_argument('--case-id')
    args = parser.parse_args(argv)
    if args.command == 'freeze':
        freeze_selection(json.loads(Path(args.config).read_text(encoding='utf8')), args.output)
    elif args.command=='baseline':
        run_baselines(args.manifest,args.output)
    elif args.command=='offline':
        run_offline(args.manifest,args.output)
    elif args.command=='carbon':
        run_carbon(args.manifest,args.output,baseline_run=args.baseline_run)
    elif args.command=='counterfactual':
        run_counterfactual(args.manifest,args.output,case_id=args.case_id)


if __name__ == '__main__':
    main()
