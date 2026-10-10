"""One fixed-threshold, three-policy comparison on a verified FIFO archive."""
import argparse
import csv
import json
from pathlib import Path

from .contracts import file_hash, write_json
from .policies import CarbonThresholdFIFO, POLICY_NAMES
from .run import (_fresh_run, LEGACY_FILES, run_frozen_experiment, shared_config,
                  verified_archive)


def compare_experiments(previous_dir, output_dir, threshold_g_per_kwh):
    # Validate the literal before reading any carbon samples. No fitting function.
    threshold = CarbonThresholdFIFO(threshold_g_per_kwh).threshold_g_per_kwh
    previous = Path(previous_dir)
    def action(out):
        cfg, _, _, parent = verified_archive(previous, strict_source=False)
        if cfg['policy'] != 'immediate_fifo':
            raise ValueError('Comparison requires an Immediate/FIFO reference archive')
        rows, manifests = [], []
        for policy in POLICY_NAMES:
            policy_cfg = json.loads(json.dumps(cfg))
            # The FIFO reference must preserve even explicitly empty params:
            # resolved_config.json belongs to the eight-file byte regression.
            if policy != 'immediate_fifo':
                policy_cfg.pop('policy_params', None)
            policy_cfg['policy'] = policy
            if policy == 'carbon_threshold_fifo':
                policy_cfg['policy_params'] = {'threshold_g_per_kwh': threshold}
            result = run_frozen_experiment(policy_cfg, previous, out / policy)
            rows.append(dict(policy=policy, **result.summary,
                             **{k: v for k, v in result.policy_summary.items() if k != 'policy'}))
            manifests.append(json.loads((out / policy / 'provenance.json').read_text(encoding='utf-8')))
        identical = all(m['input_hashes'] == parent['input_hashes'] for m in manifests)
        identical_sources = len({m['source_sha256'] for m in manifests}) == 1
        identical_configs = all(shared_config(json.loads((out / p / 'resolved_config.json').read_text(
            encoding='utf-8'))) == shared_config(cfg) for p in POLICY_NAMES)
        checks = {name: file_hash(out / 'immediate_fifo' / name) == parent['output_hashes'][name]
                  for name in LEGACY_FILES}
        if not (identical and identical_sources and identical_configs and all(checks.values())):
            raise RuntimeError('Fairness or Immediate/FIFO historical regression failed')
        with (out / 'comparison.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        write_json(out / 'fairness.json', dict(
            frozen_from=str(previous.resolve()), identical_frozen_inputs=identical,
            input_hashes=parent['input_hashes'], identical_current_sources=identical_sources,
            identical_shared_config=identical_configs, threshold_g_per_kwh=threshold,
            threshold_selection='fixed_literal_before_run_no_test_window_fit',
            window_carbon_is_not_completion_normalized=True,
            completion_counts_differ=len({r['completed'] for r in rows}) > 1,
            fifo_legacy_regression=dict(passed=all(checks.values()), files=checks)))
        return rows
    return _fresh_run(output_dir, action)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-from', type=Path, required=True)
    parser.add_argument('--threshold-g-per-kwh', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = compare_experiments(args.frozen_from, args.output, args.threshold_g_per_kwh)
    print(json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False))
    print(f'Artifacts: {args.output.resolve()}')


if __name__ == '__main__':
    main()
