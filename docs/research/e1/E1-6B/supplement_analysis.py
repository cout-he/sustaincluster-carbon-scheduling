"""Read-only supplemental analysis of completed E1-6B scientific outputs.

Contract, declared before any figures: describe the 2x2 state/filter interaction
and actual PPO optimization/episode behavior without presupposing benefit or
convergence. Figure 1 is a quantitative 2x2 grid: entropy, explained variance,
approximate KL, clip fraction; preserve every registered seed/update. Figure 2
has separate Train1.5/2 episode-return facets; preserve every completed episode.
These training diagnostics are not scientific deadline/carbon evaluation.

Python/matplotlib only; 7.2-inch figures, PNG300dpi, editable PDF/SVG, minimum
6.5pt glyphs. All individual seed curves remain visible; no confidence bands,
synthetic formal values, smoothing, exclusions, Test access or parameter tuning.
Sources/config/tests remain frozen. The only writable destination is a fresh
explicit output directory. Actual figure inspection is required after running.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

GROUPS = ('plain', 'state', 'shield', 'combined')
SEEDS = (55101, 55102, 55103)
FACTORS = (1., 1.5, 2.)
PAIR_KEYS = {(g, s) for g in GROUPS for s in SEEDS}
COLORS = {'plain': '#6e7781', 'state': '#4c78a8',
          'shield': '#c58e4e', 'combined': '#9372b2'}
SEED_STYLES = {55101: '-', 55102: '--', 55103: ':'}
CONTRASTS = {
    'state_with_filter': dict(combined=1, shield=-1),
    'filter_with_state': dict(combined=1, state=-1),
    'state_filter_interaction': dict(combined=1, state=-1, shield=-1, plain=1),
}
METRICS = {
    'delta_misses': ('misses', 1),
    'delta_dmr_percentage_points': ('deadline_violation_rate_observed', 100),
    'delta_carbon_kg': ('carbon_kg', 1),
    'delta_jct_minutes': ('mean_jct_minutes_completed', 1),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def read_csv(path):
    with Path(path).open(encoding='utf8', newline='') as stream:
        return list(csv.DictReader(stream))


def number(value):
    if value in (None, '', 'None'):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Nonfinite analysis value')
    return result


def integer(value):
    result = number(value)
    if result is None or int(result) != result:
        raise ValueError('Expected an integer count')
    return int(result)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                   allow_nan=False) + '\n', encoding='utf8')


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def contrasts(by_group):
    return {name: {
        metric: scale * math.fsum(coefficient * number(by_group[group][source])
                                 for group, coefficient in terms.items())
        for metric, (source, scale) in METRICS.items()}
        for name, terms in CONTRASTS.items()}


def _validate_tables(analysis, hashes):
    manifest = read_json(analysis / 'manifest.json')
    hashes[str(analysis / 'manifest.json')] = digest(analysis / 'manifest.json')
    require(manifest.get('schema') == 'e1-6b-adaptive-analysis-v1' and
            manifest.get('development_validation') is True and
            manifest.get('no_test_access') is True and
            manifest.get('all_equal_workload_carbon_comparisons_valid') is True,
            'Complete development Validation analysis required; Test sealed')
    require(manifest.get('training_seeds') == list(SEEDS) and
            manifest.get('group_count') == 4, 'Registered seed/group protocol differs')
    tables = {}
    for name in ('all-methods.csv', 'training-summary.csv', 'training-updates.csv'):
        path = analysis / name
        require(manifest['output_hashes'].get(name) == digest(path),
                'Analysis table hash mismatch: ' + name)
        hashes[str(path)] = digest(path)
        tables[name] = read_csv(path)
    trained = tables['training-summary.csv']
    require(len(trained) == 12 and {(r['group'], integer(r['seed'])) for r in trained} == PAIR_KEYS,
            'Exactly 12 registered models required')
    require(all(integer(r['num_timesteps']) == 20480 and
                integer(r['parameter_count']) == 19397 for r in trained),
            'Formal training budget/network mismatch')
    updates = tables['training-updates.csv']
    require(len(updates) == 960, 'All 12x80 PPO updates required')
    for group, seed in sorted(PAIR_KEYS):
        rows = [r for r in updates if (r['group'], integer(r['seed'])) == (group, seed)]
        require(len(rows) == 80, 'Missing/duplicate registered update curve')
        require([integer(r['environment_steps']) for r in rows] == list(range(256, 20481, 256)) and
                [integer(r['train/n_updates']) for r in rows] == list(range(10, 801, 10)),
                'Actual PPO update sequence differs')
        for r in rows:
            for metric in ('train/entropy_loss', 'train/explained_variance',
                           'train/approx_kl', 'train/clip_fraction'):
                require(number(r.get(metric)) is not None, 'Missing/nonfinite PPO diagnostic: ' + metric)
    methods = tables['all-methods.csv']
    require(len(methods) == 57, 'All three nineteen-method scenario tables required')
    require(all(r['split'] == 'validation' and r['phase'] == 'common' for r in methods),
            'Only saved common-window development Validation rows permitted')
    adaptive = [r for r in methods if r['group'] in GROUPS]
    require(len(adaptive) == 36 and all(number(r['factor']) in FACTORS for r in adaptive),
            'Exactly 36 registered adaptive Validation rows required')
    for factor in FACTORS:
        rows = [r for r in adaptive if number(r['factor']) == factor]
        require(len(rows) == 12 and {(r['group'], integer(r['seed'])) for r in rows} == PAIR_KEYS,
                'Incomplete adaptive Validation matrix')
        require(len({r['observation_end'] for r in rows}) == 1 and
                len({integer(r['arrived']) for r in rows}) == 1,
                'Unequal public observation boundary or workload')
        require(all(integer(r['completed']) == integer(r['arrived']) and
                    integer(r['pending']) == integer(r['running']) == 0 and
                    integer(r['deadline_right_censored']) == 0 and
                    r['termination'] == 'all_completed' for r in rows),
                'Incomplete Drain/censored tasks cannot be compared')
        for row in rows:
            require(math.isclose(number(row['deadline_violation_rate_observed']),
                                 integer(row['misses']) / integer(row['arrived']), abs_tol=1e-14),
                    'Deadline count/rate mismatch')
    return trained, updates, adaptive


def _episodes(training_dirs, training_rows, hashes):
    expected = {(r['group'], integer(r['seed'])): r for r in training_rows}
    seen = set(); merged = []; partial = []
    for folder in map(Path, training_dirs):
        summary_path = folder / 'training_summary.json'
        summary = read_json(summary_path)
        key = (summary['group'], integer(summary['seed']))
        require(key in PAIR_KEYS and key not in seen, 'Duplicate/unknown training directory')
        seen.add(key)
        require(summary['checkpoint_sha256'] == expected[key]['checkpoint_sha256'] and
                summary['source_sha256'] == expected[key]['source_sha256'] and
                summary['num_timesteps'] == 20480 and summary['parameter_count'] == 19397 and
                [r.get('split') for r in summary['training_identities']] == ['train', 'train'] and
                [r.get('factor') for r in summary['training_identities']] == [1.5, 2.],
                'Original Train identity/checkpoint/protocol mismatch; Test sealed')
        hashes[str(summary_path)] = digest(summary_path)
        episodes_path = folder / 'episodes.csv'
        require(summary['files'].get('episodes.csv') == digest(episodes_path),
                'Original episode artifact hash mismatch')
        hashes[str(episodes_path)] = digest(episodes_path)
        rows = read_csv(episodes_path)
        require(len(rows) == summary['completed_episodes'], 'Original completed-episode count differs')
        previous = 0
        for row in rows:
            end = integer(row['ending_step']); factor = number(row['factor'])
            require(previous < end <= 20480 and integer(row['length']) > 0 and
                    row['terminated'].lower() == 'true' and row['truncated'].lower() == 'false' and
                    factor in (1.5, 2.), 'Invalid completed Train episode')
            previous = end
            for metric in ('episode_return', 'carbon_kg', 'wait_task_minutes', 'new_deadline_misses',
                           'carbon_cost', 'wait_cost', 'deadline_cost', 'terminal_cost'):
                require(number(row.get(metric)) is not None, 'Missing episode metric: ' + metric)
            merged.append(dict(group=key[0], seed=key[1], **row))
        partial.append(dict(group=key[0], seed=key[1],
            partial_episode_steps=summary['current_partial_episode_steps'],
            budget_ends_mid_episode=summary['budget_ends_mid_episode'],
            reason='partial budget-ending episodes excluded; never silently treated as completed'))
    require(seen == PAIR_KEYS, 'All 12 original training directories required')
    return merged, partial


def _quarter_summary(episodes):
    result = []
    for group, seed in sorted(PAIR_KEYS):
        for factor in (1.5, 2.):
            rows = sorted((r for r in episodes if r['group'] == group and
                           integer(r['seed']) == seed and number(r['factor']) == factor),
                          key=lambda r: integer(r['ending_step']))
            require(rows, 'Missing completed episode pressure stratum')
            count = max(1, math.ceil(len(rows) / 4))
            row = dict(group=group, seed=seed, factor=factor, completed_episodes=len(rows),
                       quarter_rule='ceil(n_completed_same_scene/4), at least one',
                       first_quarter_n=count, last_quarter_n=count,
                       quarters_overlap=count * 2 > len(rows),
                       interpretation='descriptive training behavior; not a convergence or performance certificate')
            for metric in ('episode_return', 'carbon_kg', 'wait_task_minutes', 'new_deadline_misses', 'length'):
                means = {}
                for label, selected in (('first', rows[:count]), ('last', rows[-count:])):
                    values = [number(r[metric]) for r in selected]
                    means[label] = math.fsum(values) / len(values)
                    row[label + '_' + metric + '_mean'] = means[label]
                    row[label + '_' + metric + '_min'] = min(values)
                    row[label + '_' + metric + '_max'] = max(values)
                row['last_minus_first_' + metric] = means['last'] - means['first']
            result.append(row)
    return result


def _action_rows(adaptive, training):
    result = []
    for row in adaptive:
        actions = [integer(row[f'action_{a}_count']) for a in range(4)]
        total = sum(actions)
        checked, eligible, veto, deferred = [integer(row[k]) for k in
                                           ('candidate_checks', 'solo_eligible', 'filter_veto', 'deferred')]
        require(total > 0 and 0 <= veto <= eligible <= checked and deferred + veto == eligible,
                'Candidate/filter/action accounting mismatch')
        require(row['group'] in ('shield', 'combined') or veto == 0,
                'Unfiltered group unexpectedly vetoed tasks')
        require(row['policy_defer_intervals'] != '' and
                deferred == integer(row['policy_defer_intervals']), 'Actual defer counts differ')
        item = dict(split='development_validation', factor=number(row['factor']),
                    group=row['group'], seed=integer(row['seed']), action_intervals=total,
                    candidate_checks=checked, solo_eligible=eligible, filter_veto=veto,
                    real_policy_deferrals=deferred,
                    filter_veto_per_solo_eligible=veto / eligible if eligible else None,
                    deferral_per_solo_eligible=deferred / eligible if eligible else None,
                    deferral_per_candidate_check=deferred / checked if checked else None,
                    capacity_wait_intervals=integer(row['capacity_wait_intervals']),
                    scope='actions stop at own Drain; capacity-failed candidates excluded from filter denominator')
        for action in range(4):
            item[f'action_{action}_count'] = actions[action]
            item[f'action_{action}_fraction'] = actions[action] / total
        result.append(item)
    training_usage = []
    for row in training:
        counts = [integer(row[f'action_{a}_count']) for a in range(4)]
        require(sum(counts) == 20480, 'Training action budget mismatch')
        item = dict(split='train', group=row['group'], seed=integer(row['seed']), environment_steps=20480)
        for action, count in enumerate(counts):
            item[f'action_{action}_count'] = count
            item[f'action_{action}_fraction'] = count / 20480
        training_usage.append(item)
    return result, training_usage


def _plot(updates, episodes, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 7,
        'axes.labelsize': 7, 'axes.titlesize': 8, 'xtick.labelsize': 6.5,
        'ytick.labelsize': 6.5, 'legend.fontsize': 6.5, 'legend.frameon': False,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'e1-6b-supplement',
        'axes.spines.top': False, 'axes.spines.right': False})
    directory = output / 'figures'; directory.mkdir()
    limits = {}; line_counts = {}
    groups = [Line2D([0], [0], color=COLORS[g], label=g.capitalize()) for g in GROUPS]
    seeds = [Line2D([0], [0], color='#777777', ls=SEED_STYLES[s], label=str(s)) for s in SEEDS]
    def save(figure, stem):
        figure.savefig(directory / (stem + '.png'), dpi=300)
        figure.savefig(directory / (stem + '.pdf'), metadata={'CreationDate': None, 'ModDate': None})
        figure.savefig(directory / (stem + '.svg'), metadata={'Date': None})
        plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    panels = [('train/entropy_loss', 'Policy entropy (nats)', -1, 'entropy'),
              ('train/explained_variance', 'Value explained variance', 1, 'explained-variance'),
              ('train/approx_kl', 'Approximate KL', 1, 'approx-kl'),
              ('train/clip_fraction', 'PPO clip fraction', 1, 'clip-fraction')]
    for axis, (field, label, sign, name) in zip(axes.flat, panels):
        values = []
        for group, seed in sorted(PAIR_KEYS):
            rows = [r for r in updates if (r['group'], integer(r['seed'])) == (group, seed)]
            x = [integer(r['environment_steps']) for r in rows]
            y = [sign * number(r[field]) for r in rows]; values.extend(y)
            axis.plot(x, y, color=COLORS[group], ls=SEED_STYLES[seed], lw=.7, alpha=.75)
        low, high = min(values + [0]), max(values + [0])
        padding = max((high - low) * .08, 1e-6)
        lower = low - padding if low < 0 else 0
        axis.set(xlim=(0, 20480), ylim=(lower, high + padding), ylabel=label,
                 xlabel='Environment steps', xticks=(0, 5120, 10240, 15360, 20480))
        limits[name] = list(axis.get_ylim()); line_counts[name] = 12
    figure.legend(handles=groups + seeds, loc='lower center', bbox_to_anchor=(.53, .1), ncol=7)
    figure.text(.5, .02, 'All 12 models × 80 saved updates; no smoothing or confidence interval.\n'
                 'Finite optimization diagnostics do not certify convergence or SLA/carbon improvement.',
                 ha='center', fontsize=6.5)
    figure.subplots_adjust(left=.11, right=.94, top=.97, bottom=.25, hspace=.45, wspace=.28)
    save(figure, 'training-optimization-diagnostics')
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), sharey=True)
    all_returns = [number(r['episode_return']) for r in episodes]
    low, high = min(all_returns + [0]), max(all_returns + [0]); padding = max((high - low) * .06, .001)
    for axis, factor in zip(axes, (1.5, 2.)):
        for group, seed in sorted(PAIR_KEYS):
            rows = sorted((r for r in episodes if r['group'] == group and
                           integer(r['seed']) == seed and number(r['factor']) == factor),
                          key=lambda r: integer(r['ending_step']))
            axis.plot([integer(r['ending_step']) for r in rows],
                      [number(r['episode_return']) for r in rows], color=COLORS[group],
                      ls=SEED_STYLES[seed], lw=.7, alpha=.75)
        axis.set(xlim=(0, 20480), ylim=(low - padding, high + padding), xlabel='Environment steps',
                 title=f'Train {factor:g}×', xticks=(0, 5120, 10240, 15360, 20480))
        line_counts[f'episode-return-{factor:g}'] = 12
    axes[0].set_ylabel('Completed-episode return')
    figure.legend(handles=groups + seeds, loc='lower center', bbox_to_anchor=(.53, .12), ncol=7)
    figure.text(.5, .025, 'All completed Train episodes, separated by load. Partial budget-ending episodes excluded.\n'
                 'Returns use the registered Reward; they do not replace common-window scientific metrics.',
                 ha='center', fontsize=6.5)
    figure.subplots_adjust(left=.1, right=.94, top=.91, bottom=.34, wspace=.13)
    limits['episode-return'] = list(axes[0].get_ylim()); save(figure, 'training-episode-returns')
    return dict(axis_limits=limits, curves_per_panel=line_counts,
                observations_plotted=dict(updates=len(updates), completed_episodes=len(episodes)),
                exclusions='only unfinished budget-ending episodes; saved separately',
                uncertainty='individual registered training-seed lines; no CI or significance claim',
                preflight_warning_rationale='PNG300dpi is the registered preview; editable PDF/SVG are the vector delivery. TIFF was not requested.',
                actual_visual_qa_still_required=True)


def analyze(analysis_dir, training_dirs, output_dir, *, render=True):
    analysis = Path(analysis_dir); output = Path(output_dir)
    require(not output.exists(), 'Fresh output directory required; never overwrite')
    hashes = {}; training, updates, adaptive = _validate_tables(analysis, hashes)
    episodes, partial = _episodes(training_dirs, training, hashes)
    paired = []
    for factor in FACTORS:
        for seed in SEEDS:
            by_group = {r['group']: r for r in adaptive if number(r['factor']) == factor and integer(r['seed']) == seed}
            for name, metrics in contrasts(by_group).items():
                paired.append(dict(factor=factor, seed=seed, contrast=name,
                    coefficients=json.dumps(CONTRASTS[name], sort_keys=True), **metrics,
                    interpretation='matched training seed; one frozen workload; no single-event causal claim'))
    aggregate = []
    for factor in FACTORS:
        for name in CONTRASTS:
            rows = [r for r in paired if r['factor'] == factor and r['contrast'] == name]
            row = dict(factor=factor, contrast=name, training_seeds=3, workload_realizations=1)
            for metric in METRICS:
                values = [r[metric] for r in rows]
                row[metric + '_mean'] = math.fsum(values) / len(values)
                row[metric + '_min'] = min(values); row[metric + '_max'] = max(values)
            aggregate.append(row)
    action_rows, training_actions = _action_rows(adaptive, training)
    quarters = _quarter_summary(episodes)
    require(all(digest(p) == h for p, h in hashes.items()), 'Read-only input changed during analysis')
    output.mkdir(parents=True, exist_ok=False)
    contract = dict(backend='python', archetype='quantitative grid',
        conclusion='Describe measured factorial interaction and training behavior; no assumed benefit or convergence',
        panels='entropy, explained variance, approximate KL, clip fraction; separate Train1.5/2 episode returns',
        size_inches=[7.2, 5.2], episode_size_inches=[7.2, 3.7], minimum_glyph_points=6.5,
        exports=['PNG300dpi', 'editable PDF', 'editable SVG'], smoothing=False,
        all_12_models_all_80_updates=True, all_completed_episodes=True,
        test_access=False, uncertainty='individual training-seed curves, never confidence intervals')
    write_json(output / 'figure-contract.json', contract)
    for name, rows in [('conditional-and-interaction-effects.csv', paired),
                       ('conditional-and-interaction-summary.csv', aggregate),
                       ('validation-actions-and-filters.csv', action_rows),
                       ('training-action-usage.csv', training_actions),
                       ('training-completed-episodes.csv', episodes),
                       ('convergence-summary.csv', quarters),
                       ('training-partial-episodes.csv', partial)]:
        write_csv(output / name, rows)
    # Preserve raw saved update values, not only the columns selected for plots.
    (output / 'training-updates-source.csv').write_bytes((analysis / 'training-updates.csv').read_bytes())
    audit = _plot(updates, episodes, output) if render else dict(rendered=False)
    write_json(output / 'figure-audit.json', audit)
    require(all(digest(p) == h for p, h in hashes.items()), 'Read-only input changed after plotting')
    manifest = dict(schema='e1-6b-supplemental-analysis-v1', input_hashes=hashes,
        source_sha256=digest(__file__), inputs_unchanged=True, test_access=False,
        development_validation_only=True, training_models=12, update_rows=960,
        completed_episode_rows=len(episodes), paired_rows=27, convergence_rows=24,
        no_convergence_certificate=True, legacy_action_space_effect_not_isolated=True,
        ratio_denominator='can-fit solo-eligible candidate opportunities; excludes capacity failures',
        output_hashes={p.relative_to(output).as_posix(): digest(p) for p in output.rglob('*') if p.is_file()})
    write_json(output / 'manifest.json', manifest)
    return manifest


def self_check():
    # Pure arithmetic/schema checks only; no invented formal CSV or figures.
    rows = {g: dict(misses=v, deadline_violation_rate_observed=v / 100,
                   carbon_kg=v * 2, mean_jct_minutes_completed=v * 3)
            for g, v in zip(GROUPS, (10, 20, 30, 50))}
    checked = contrasts(rows)
    require(checked['state_with_filter']['delta_misses'] == 20, 'Conditional-state arithmetic')
    require(checked['filter_with_state']['delta_misses'] == 30, 'Conditional-filter arithmetic')
    require(checked['state_filter_interaction']['delta_misses'] == 10, 'Factorial-interaction arithmetic')
    require(math.isclose(checked['state_filter_interaction']['delta_dmr_percentage_points'], 10),
            'DMR percentage-point conversion')
    require(len(PAIR_KEYS) == 12 and len(CONTRASTS) * len(FACTORS) * len(SEEDS) == 27,
            'Registered matrix count')
    print(json.dumps(dict(self_check=True, scientific_data_generated=False, figures_generated=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-dir')
    parser.add_argument('--training-dir', nargs='+')
    parser.add_argument('--output')
    parser.add_argument('--no-render', action='store_true')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check(); return
    if not all((args.analysis_dir, args.training_dir, args.output)):
        parser.error('--analysis-dir, --training-dir (12 paths), and --output are required')
    result = analyze(args.analysis_dir, args.training_dir, args.output, render=not args.no_render)
    print(json.dumps({k: result[k] for k in ('training_models', 'update_rows', 'paired_rows',
                                            'convergence_rows', 'completed_episode_rows')}))


if __name__ == '__main__':
    main()
