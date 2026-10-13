"""Supplementary paired 2x2 contrasts from completed E1-6B summaries only.

No simulation, training, coefficient selection or access to Test inputs.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math

FACTORS = (1.0, 1.5, 2.0)
SEEDS = (55101, 55102, 55103)
GROUPS = ('plain', 'state', 'shield', 'combined')
CONTRASTS = {
    'state_without_filter': {'state': 1, 'plain': -1},
    'state_with_filter': {'combined': 1, 'shield': -1},
    'filter_without_state': {'shield': 1, 'plain': -1},
    'filter_with_state': {'combined': 1, 'state': -1},
    'state_filter_interaction': {'combined': 1, 'state': -1, 'shield': -1, 'plain': 1},
}
METRICS = {
    'misses': ('misses', 1),
    'dmr_percentage_points': ('deadline_violation_rate_observed', 100),
    'carbon_kg': ('carbon_kg', 1),
    'total_energy_kwh': ('total_energy_kwh', 1),
    'jct_minutes': ('mean_jct_minutes_completed', 1),
    'wait_minutes': ('mean_wait_minutes_started', 1),
    'pending_mean': ('pending_mean', 1),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contrast(rows, coefficients, key):
    values = [float(rows[group][key]) * coefficient for group, coefficient in coefficients.items()]
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Nonfinite paired effect input')
    return math.fsum(values)


def self_test():
    example = {'plain': {'x': 10}, 'state': {'x': 8}, 'shield': {'x': 11}, 'combined': {'x': 7}}
    expected = (-2, -4, 1, -1, -2)
    actual = tuple(contrast(example, coefficients, 'x') for coefficients in CONTRASTS.values())
    if actual != expected:
        raise AssertionError((actual, expected))
    # Zero interaction if two module effects add exactly.
    example['combined']['x'] = 9
    if contrast(example, CONTRASTS['state_filter_interaction'], 'x') != 0:
        raise AssertionError('Additive interaction should be zero')
    print(json.dumps({'arithmetic_checks': 6, 'passed': True}))


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    digest = sha(source)
    with source.open(encoding='utf-8', newline='') as stream:
        all_rows = list(csv.DictReader(stream))
    if len(all_rows) != 57:
        raise ValueError('Completed three-factor nineteen-method analysis required')
    rows = [row for row in all_rows if row['group'] in GROUPS]
    indexed = {(float(row['factor']), int(float(row['seed'])), row['group']): row for row in rows}
    expected = {(factor, seed, group) for factor in FACTORS for seed in SEEDS for group in GROUPS}
    if len(rows) != 36 or set(indexed) != expected:
        raise ValueError('Incomplete/duplicate registered group-seed-factor matrix')
    for row in rows:
        if row['split'] != 'validation' or row['phase'] != 'common':
            raise ValueError('Only development Validation common results permitted')
        if (row['termination'] != 'all_completed' or float(row['arrived']) != float(row['completed'])
                or any(float(row[name]) != 0 for name in ('pending', 'running', 'deadline_right_censored', 'deadline_missed_unfinished'))):
            raise ValueError('Incomplete Drain or censored workloads cannot enter this contrast')
    effects = []
    for factor in FACTORS:
        common = {indexed[factor, seed, group]['observation_end'] for seed in SEEDS for group in GROUPS}
        if len(common) != 1:
            raise ValueError('Different accounting endpoints within a pressure factor')
        for seed in SEEDS:
            selected = {group: indexed[factor, seed, group] for group in GROUPS}
            for name, coefficients in CONTRASTS.items():
                values = {f'delta_{metric}': scale * contrast(selected, coefficients, key)
                          for metric, (key, scale) in METRICS.items()}
                effects.append({'factor': factor, 'seed': seed, 'contrast': name,
                                'common_end': next(iter(common)), **values})
    summaries = []
    for factor in FACTORS:
        for name in CONTRASTS:
            selected = [row for row in effects if row['factor'] == factor and row['contrast'] == name]
            summary = {'factor': factor, 'contrast': name, 'training_seeds': 3, 'workload_realizations': 1}
            for metric in METRICS:
                values = [row[f'delta_{metric}'] for row in selected]
                summary.update({f'delta_{metric}_mean': math.fsum(values) / 3,
                                f'delta_{metric}_min': min(values), f'delta_{metric}_max': max(values)})
            summaries.append(summary)
    if sha(source) != digest:
        raise AssertionError('Read-only analysis source changed')
    output.mkdir(parents=True)
    write_csv(output / 'factorial-effects.csv', effects)
    write_csv(output / 'factorial-effects-summary.csv', summaries)
    receipt = {
        'schema': 'e1-6b-paired-factorial-effects-v1',
        'source_file': source.name, 'source_sha256': digest,
        'input_unchanged': True, 'paired_rows': len(effects), 'summary_rows': len(summaries),
        'contrasts': CONTRASTS, 'seeds': SEEDS, 'factors': FACTORS,
        'interpretation': 'paired training-seed contrasts within one frozen workload per load; no event-level causal or significance claim',
        'range': 'minimum/maximum over three training seeds, not confidence interval',
        'test_read': False, 'script_sha256': sha(Path(__file__)),
        'output_sha256': {name: sha(output / name) for name in ('factorial-effects.csv', 'factorial-effects-summary.csv')},
    }
    (output / 'manifest.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'paired_rows': len(effects), 'summary_rows': len(summaries), 'source_unchanged': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.source is None or args.output is None:
        parser.error('--source and --output required')
    else:
        main(args.source, args.output)
