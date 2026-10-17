"""Aggregate completed real Development evaluations; never fabricate absent runs."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

from experiments.e1.contracts import content_hash, file_hash
from experiments.joint_v2.checkpoint import atomic_json

SEEDS = (20261017, 20261018, 20261019)
SUMS = ('arrived', 'completed', 'deadline_misses', 'unfinished', 'deadline_right_censored',
        'carbon_kg', 'incremental_carbon_kg', 'it_carbon_kg', 'hvac_carbon_kg',
        'total_energy_kwh', 'it_energy_kwh', 'hvac_energy_kwh', 'temperature_violations',
        'actively_deferred_unique_tasks', 'active_deferral_decisions', 'capacity_wait_intervals')
WEIGHTED = {'waiting_mean_minutes': 'arrived', 'jct_completed_mean_minutes': 'completed'}


def _csv(path, rows, fields=None):
    names = fields or sorted({key for row in rows for key in row})
    with Path(path).open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def analyze(root, output=None, *, figures=True):
    root = Path(root).resolve()
    out = Path(output).resolve() if output else root/'analysis'
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = root/'protocol.json'
    protocol = None
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding='utf8'))
        unsigned = dict(protocol)
        digest = unsigned.pop('protocol_sha256', None)
        if digest != content_hash(unsigned) or protocol.get('test_access') is not False:
            raise ValueError('Analysis protocol identity differs')
    progress = []
    for path in sorted((root/'jobs').glob('*/training/run-result.json')):
        result = json.loads(path.read_text(encoding='utf8'))
        target = protocol['budget']['total_physical_steps'] if protocol else 65536
        actual = result.get('physical_steps')
        progress.append(dict(seed=result.get('seed'), variant=result.get('variant'),
            actual_physical_steps=actual, target_physical_steps=target,
            target_reached=actual >= target if _finite(actual) else False,
            completed_episodes=result.get('completed_episodes'), elapsed_seconds=result.get('elapsed_seconds'),
            stop_reason=result.get('last_run', {}).get('stop_reason'),
            source_run_result=str(path.relative_to(root))))
    _csv(out/'training-progress.csv', progress, None if progress else
         ['variant', 'seed', 'actual_physical_steps', 'target_physical_steps', 'target_reached'])
    raw, sources, seen = [], [], {}
    for path in sorted((root/'jobs').glob('*/evaluation*/results.json')):
        payload = json.loads(path.read_text(encoding='utf8'))
        if not payload.get('complete'):
            continue
        if payload.get('test_access') is not False:
            raise ValueError('Analysis accepts only sealed-Test Development evaluations')
        if protocol and payload['results']:
            first = payload['results'][0]
            methods = protocol['evaluation']['full_methods'] if first['variant'] == 'full' else ['B5']
            expected = {(s, m) for s in protocol['evaluation']['scenario_ids'] for m in methods}
            if (len(payload['results']) != len(expected)
                    or {(r['scenario_id'], r['comparison_method']) for r in payload['results']} != expected
                    or any((r['seed'], r['variant']) != (first['seed'], first['variant']) for r in payload['results'])):
                raise ValueError('Completed result is missing frozen Development comparisons')
        retained = 0
        for result in payload['results']:
            if result.get('scenario_role') != 'development':
                raise ValueError('Analysis cannot consume Validation/Test results')
            key = tuple(result[k] for k in ('variant', 'seed', 'comparison_method', 'scenario_id'))
            if result['seed'] not in SEEDS:
                raise ValueError('Unexpected seed in the frozen evaluation')
            if key in seen:
                if seen[key] != result:
                    raise ValueError('Conflicting completed evaluation for one seed/scenario/method')
                continue
            seen[key] = result
            raw.append(dict(result, source_evaluation=str(path.relative_to(root))))
            retained += 1
        if retained:
            sources.append(dict(path=str(path.relative_to(root)), sha256=file_hash(path)))
    _csv(out/'source-data.csv', raw, None if raw else ['variant', 'seed', 'comparison_method', 'scenario_id'])
    grouped = defaultdict(list)
    for row in raw:
        grouped[(row['variant'], row['seed'], row['comparison_method'])].append(row)
    per_seed = []
    for (variant, seed, method), rows in sorted(grouped.items()):
        item = dict(variant=variant, seed=seed, comparison_method=method, scenario_count=len(rows),
                    scenario_ids=';'.join(sorted(r['scenario_id'] for r in rows)))
        for metric in SUMS:
            values = [row.get(metric) for row in rows]
            item[metric] = math.fsum(values) if all(map(_finite, values)) else None
        arrived, missed = item['arrived'], item['deadline_misses']
        item['dmr'] = missed/arrived if _finite(arrived) and arrived > 0 and _finite(missed) else (0. if arrived == 0 else None)
        item['completion_fraction'] = item['completed']/arrived if _finite(arrived) and arrived > 0 and _finite(item['completed']) else None
        item['max_rack_inlet_c'] = (max(row['max_rack_inlet_c'] for row in rows)
            if all(_finite(row.get('max_rack_inlet_c')) for row in rows) else None)
        for metric, weight in WEIGHTED.items():
            valid = all(_finite(row.get(metric)) and _finite(row.get(weight)) for row in rows)
            denominator = math.fsum(row[weight] for row in rows) if valid else 0.
            item[metric] = math.fsum(row[metric]*row[weight] for row in rows)/denominator if denominator else None
        per_seed.append(item)
    _csv(out/'per-seed.csv', per_seed, None if per_seed else ['variant', 'seed', 'comparison_method'])
    groups = defaultdict(list)
    for row in per_seed:
        groups[(row['variant'], row['comparison_method'])].append(row)
    aggregate = []
    for (variant, method), rows in sorted(groups.items()):
        item = dict(variant=variant, comparison_method=method, n=len(rows),
                    seeds=';'.join(str(r['seed']) for r in rows),
                    three_seed_complete=({r['seed'] for r in rows} == set(SEEDS)
                                         and len({r['scenario_ids'] for r in rows}) == 1))
        for metric in (*SUMS, 'dmr', 'completion_fraction', 'max_rack_inlet_c', *WEIGHTED):
            values = [r.get(metric) for r in rows]
            valid = all(map(_finite, values))
            item[metric+'_mean'] = statistics.mean(values) if valid else None
            item[metric+'_std'] = statistics.stdev(values) if valid and item['three_seed_complete'] else None
        aggregate.append(item)
    _csv(out/'aggregate.csv', aggregate, None if aggregate else ['variant', 'comparison_method', 'n'])
    report = dict(schema='rac-mappo-actual-analysis-v1', evaluation_files=len(sources),
        scenario_rows=len(raw), per_seed_rows=len(per_seed), aggregate_rows=len(aggregate), sources=sources,
        training_runs=len(progress), protocol_sha256=file_hash(protocol_path) if protocol else None,
        expected_seeds=list(SEEDS), standard_deviation_rule='sample std only after all three distinct frozen seeds complete',
        missing_values='blank; never imputed', role='development', test_access=False, figures=[])
    if figures and aggregate:
        report['figures'] = _figures(out, aggregate)
    atomic_json(out/'analysis.json', report)
    return report


def _figures(out, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    rows = [r for r in rows if r['variant'] == 'full'] + [r for r in rows if r['variant'] != 'full']
    labels = [f"{r['comparison_method']}" if r['variant'] == 'full' else r['variant'] for r in rows]
    paths = []
    for name, metric, ylabel, factor in [('carbon-comparison', 'carbon_kg', 'Fixed-window carbon (kg)', 1.),
                                      ('incremental-carbon', 'incremental_carbon_kg', 'Incremental carbon (kg)', 1.),
                                      ('deadline-miss-rate', 'dmr', 'Deadline miss rate (%)', 100.)]:
        chosen = [(label, r) for label, r in zip(labels, rows) if _finite(r[metric+'_mean'])]
        if not chosen:
            continue
        fig, ax = plt.subplots(figsize=(max(3.5, len(chosen)*.55), 2.7), layout='constrained')
        means = [r[metric+'_mean']*factor for _, r in chosen]
        ax.bar(range(len(chosen)), means, color='#3B6B8A', width=.65)
        for index, (_, row) in enumerate(chosen):
            deviation = row[metric+'_std']
            if _finite(deviation):
                ax.errorbar(index, means[index], yerr=deviation*factor, color='#252525', capsize=2, linewidth=.8)
        if metric == 'dmr':
            ax.axhline(5., color='#B6413C', linestyle='--', linewidth=.8, label='5% target')
            ax.legend(frameon=False, fontsize=7)
        ax.set_xticks(range(len(chosen)), [f"{label}\n(n={row['n']})" for label, row in chosen], rotation=35, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        for suffix in ('png', 'pdf', 'svg'):
            path = out/(name+'.'+suffix)
            fig.savefig(path, dpi=300); paths.append(path.name)
        plt.close(fig)
    chosen = [(label, r) for label, r in zip(labels, rows)
              if _finite(r['it_carbon_kg_mean']) and _finite(r['hvac_carbon_kg_mean'])]
    if chosen:
        fig, ax = plt.subplots(figsize=(max(3.5, len(chosen)*.55), 2.7), layout='constrained')
        it, hvac = [r['it_carbon_kg_mean'] for _, r in chosen], [r['hvac_carbon_kg_mean'] for _, r in chosen]
        ax.bar(range(len(chosen)), it, color='#3B6B8A', label='IT', width=.65)
        ax.bar(range(len(chosen)), hvac, bottom=it, color='#B98A50', label='HVAC', width=.65)
        ax.set_xticks(range(len(chosen)), [f"{label}\n(n={row['n']})" for label, row in chosen], rotation=35, ha='right')
        ax.set_ylabel('Fixed-window carbon (kg)'); ax.legend(frameon=False, fontsize=7)
        for suffix in ('png', 'pdf', 'svg'):
            path = out/('it-hvac-carbon.'+suffix)
            fig.savefig(path, dpi=300); paths.append(path.name)
        plt.close(fig)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True); parser.add_argument('--output')
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.root, args.output, figures=not args.no_figures)), flush=True)


if __name__ == '__main__':
    main()
