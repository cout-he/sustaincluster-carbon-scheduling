"""E1-5C offline Train/Validation diagnostics; existing policies remain unchanged."""
import argparse
import csv
import json
from pathlib import Path
import time

from .config import ROOT, resolve_path
from .contracts import content_hash, file_hash, write_json

VERSION = 'e1-5c-sla-pressure-diagnostics-v1'
POLICIES = ('edf_backfill', 'edf_carbon_threshold', 'cars_risk_only')
# Preserve the exact E1-5B protocol; the new diagnostic configuration adds no
# scheduler, reward, physics or training changes.
CONFIG_SHA = 'd9bc61383558d211aff2d7aa466b7396d12db8bda327b6cb8b519e20b6019e28'


def require_development_split(split):
    if split not in ('train', 'validation'):
        raise ValueError('Test is sealed: diagnostics permit only train/validation')


def _relative(root, name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Unsafe archive path')
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('Archive path escapes root')
    return resolved


def audit_historical_group(group, workload_sha256, environment_sha256):
    """Verify immutable historical artifacts without running or rewriting them."""
    group = Path(group)
    status = json.loads((group / 'run_status.json').read_text(encoding='utf8'))
    if status['state'] != 'complete':
        raise ValueError('Historical group is not complete')
    manifest = json.loads((group / 'output_hashes.json').read_text(encoding='utf8'))
    for name, digest in manifest.items():
        if file_hash(_relative(group, name)) != digest:
            raise ValueError('Historical artifact hash mismatch: ' + name)
    fairness = json.loads((group / 'fairness.json').read_text(encoding='utf8'))
    if (fairness['workload_sha256'], fairness['environment_sha256']) != (
            workload_sha256, environment_sha256):
        raise ValueError('Historical frozen input identity mismatch')
    required = {'fairness.json','comparison.csv','common_summary.csv'}
    required.update(f'{name}/{file}' for name in POLICIES for file in (
        'ledger.jsonl','intervals.csv','policy_intervals.jsonl','summary.json','phases.json'))
    if not required <= set(manifest):
        raise ValueError('Historical hash inventory is incomplete')
    if fairness.get('formal_test_evaluated') is not False:
        raise ValueError('Historical group Test status is invalid')
    return dict(artifact_manifest_sha256=file_hash(group / 'output_hashes.json'),
        files_verified=len(manifest), fairness=fairness)


def classify_pressure(reference):
    """Operational diagnostic bands; near-target does not establish 5% support."""
    rate = reference['deadline_violation_rate_observed']
    if rate is None:
        return 'unclassified'
    if rate <= .05:
        return 'low' if reference['capacity_wait_intervals'] == 0 else 'medium_supported_5pct'
    return 'medium_near_target_not_supported' if rate <= .10 else 'high'


def _jsonlines(path):
    with Path(path).open(encoding='utf8') as stream:
        return tuple(json.loads(line) for line in stream if line.strip())


def _csv(path):
    import pandas as pd
    return tuple(pd.read_csv(path).to_dict(orient='records'))


def _analysis(bundle, group, out):
    from .workload_pressure import summarize, align_edf
    from .sla_feasibility import analyze_sla_feasibility
    from .temporal import write_csv
    out.mkdir()
    pressure = summarize(bundle.workload, bundle.capacity, history_minutes=120)
    folder = group / 'edf_backfill'
    aligned = align_edf(pressure, _csv(folder / 'intervals.csv'),
        _jsonlines(folder / 'ledger.jsonl'), _jsonlines(folder / 'policy_intervals.jsonl'))
    write_json(out / 'pressure-summary.json', pressure.summary)
    write_csv(out / 'arrival-pressure.csv', pressure.arrivals)
    write_csv(out / 'task-characteristics.csv', pressure.task_details)
    write_csv(out / 'edf-pressure-timeline.csv', aligned['series'])
    write_json(out / 'edf-pressure-audit.json', {key:value for key,value in aligned.items() if key != 'series'})
    feasibility = {}
    for mode in ('raw_deadline', 'occupied_deadline'):
        began = time.perf_counter()
        result = analyze_sla_feasibility(bundle.workload, bundle.capacity,
            interval_mode=mode, top_witnesses=5)
        result['analysis_seconds'] = time.perf_counter() - began
        write_json(out / (mode + '-feasibility.json'), result)
        feasibility[mode] = result
    return dict(pressure=pressure.summary, edf_alignment={key:value for key,value in aligned.items() if key != 'series'},
        feasibility=feasibility)


def run_diagnostics(config, output_dir, *, history_root=None):
    """Freeze all inputs first, reuse four histories, run only two new groups."""
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError('Refusing to overwrite an E1-5C experiment')
    if (config.get('schema') != VERSION or config.get('test_evaluation') is not False
            or config.get('new_training') is not False):
        raise ValueError('Diagnostic scope/schema or Test/training prohibition differs')
    if (tuple(config['policies']) != POLICIES or config['new_validation_factors'] != [1., 1.5]
            or config['validation_stress_seed'] != 520001 or config['history_minutes'] != 120
            or config['sla_target'] != .05 or config['near_target_upper_dmr'] != .10):
        raise ValueError('Predeclared E1-5C diagnostic settings differ')
    for entry in config['historical_groups']:
        require_development_split(entry['split'])
    if [(item['split'],item['factor']) for item in config['historical_groups']] != [
            ('train',1.),('train',1.5),('train',2.),('validation',2.)]:
        raise ValueError('Historical cohort inventory differs')
    history = Path(history_root) if history_root is not None else resolve_path(config['history_root'])
    cfg_path = resolve_path(config['e1_5b_config'])
    if file_hash(cfg_path) != CONFIG_SHA or config['e1_5b_config_sha256'] != CONFIG_SHA:
        raise ValueError('E1-5B immutable configuration changed')
    original_cfg = json.loads(cfg_path.read_text(encoding='utf8'))
    from .rl_training_utils import load_bundle, freeze_bundle
    from .temporal import experiment_sources, write_csv
    from .rl_evaluate import evaluate_group
    sources = experiment_sources()
    cohorts = []
    for entry in config['historical_groups']:
        bundle_path = _relative(history, entry['bundle'])
        bundle = load_bundle(bundle_path, allowed_split=entry['split'])
        if bundle.factor != entry['factor']:
            raise ValueError('Historical load factor differs')
        if bundle.stress_seed != (510001 if entry['split']=='train' else 520001):
            raise ValueError('Historical stress seed differs')
        group = _relative(history, entry['group'])
        audit = audit_historical_group(group, bundle.workload.sha256, bundle.environment.sha256)
        cohorts.append(dict(split=entry['split'], factor=entry['factor'], bundle_path=bundle_path,
            bundle=bundle, group=group, origin='E1-5B_history_read_only', historical_audit=audit))
    out.mkdir(parents=True)
    write_json(out / 'run_status.json', dict(state='freezing_inputs', formal_test_evaluated=False))
    write_json(out / 'resolved_config.json', config)
    write_json(out / 'source-freeze.json', dict(sources=sources, canonical_source_sha256=content_hash(sources),
        config_sha256=content_hash(config), prior_ppo_config_sha256=file_hash(cfg_path)))
    try:
        for factor in config['new_validation_factors']:
            path = out / 'inputs' / f'validation-{factor:g}'
            freeze_bundle(original_cfg, 'validation', factor, 520001, path)
            bundle = load_bundle(path, allowed_split='validation')
            cohorts.append(dict(split='validation', factor=factor, bundle_path=path,
                bundle=bundle, group=out / 'exploratory_baselines' / f'validation-{factor:g}',
                origin='E1-5C_exploratory_diagnostic', historical_audit=None))
        write_json(out / 'input-manifest.json', dict(cohorts=[dict(split=c['split'],factor=c['factor'],
            origin=c['origin'],identity=c['bundle'].identity,
            input_path=str(c['bundle_path']), group_path=str(c['group']),
            historical_audit=c['historical_audit']) for c in cohorts],
            all_inputs_frozen_before_new_baselines=True, model_selection_changed=False,
            test_evaluation=False, training=False))
        write_json(out / 'run_status.json', dict(state='inputs_frozen_before_execution', formal_test_evaluated=False))
        for cohort in cohorts:
            if cohort['origin'] == 'E1-5C_exploratory_diagnostic':
                evaluate_group(cohort['bundle'], cohort['group'], max_deadline_violation_rate=.05)
                print(json.dumps(dict(state='new_baseline_complete', split=cohort['split'], factor=cohort['factor'])), flush=True)
        rows = []; analyses = []
        for cohort in sorted(cohorts, key=lambda c:(c['split'], c['factor'])):
            label = f"{cohort['split']}-{cohort['factor']:g}"
            group = cohort['group']; bundle = cohort['bundle']
            # Verify new outputs using the same sealed artifact contract as history.
            audit = audit_historical_group(group, bundle.workload.sha256, bundle.environment.sha256)
            baseline_rows = [row for row in _csv(group / 'common_summary.csv') if row['policy'] in POLICIES]
            if {row['policy'] for row in baseline_rows} != set(POLICIES) or len(baseline_rows) != 3:
                raise ValueError('Incomplete baseline comparison')
            reference = next(row for row in baseline_rows if row['policy'] == 'edf_backfill')
            tier = classify_pressure(reference)
            for row in baseline_rows:
                rows.append(dict(row, origin=cohort['origin'], diagnostic_tier=tier,
                    workload_sha256=bundle.workload.sha256, environment_sha256=bundle.environment.sha256,
                    common_end=audit['fairness']['common_end']))
            analysis = _analysis(bundle, group, out / label)
            analyses.append(dict(split=cohort['split'],factor=cohort['factor'],origin=cohort['origin'],
                diagnostic_tier=tier,workload_sha256=bundle.workload.sha256,
                environment_sha256=bundle.environment.sha256,**analysis))
            print(json.dumps(dict(state='analysis_complete',cohort=label,tier=tier)), flush=True)
        write_csv(out / 'baseline-comparison.csv', rows)
        write_json(out / 'analysis-summary.json', analyses)
        if experiment_sources() != sources:
            raise RuntimeError('Source changed during diagnostics; preserve this run and use a new ID')
        inventory = {p.relative_to(out).as_posix(): file_hash(p) for p in sorted(out.rglob('*'))
            if p.is_file() and p.name != 'run_status.json'}
        write_json(out / 'output-hashes.json', inventory)
        write_json(out / 'run_status.json', dict(state='complete',groups=len(cohorts),baseline_rows=len(rows),
            historical_groups_reused=4,new_exploratory_groups=2,formal_test_evaluated=False,
            new_training=False,e1_5b_model_selection_changed=False))
        return dict(baseline_rows=rows,analyses=analyses)
    except Exception as error:
        write_json(out / 'run_status.json', dict(state='failed',error_type=type(error).__name__,
            message=str(error),formal_test_evaluated=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--history-root',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding='utf8'))
    run_diagnostics(config,args.output,history_root=args.history_root)


if __name__ == '__main__':
    main()
