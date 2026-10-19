"""A1 aggregate-only scientific figures; never open raw/scenario payloads.

Figure contract (quantitative grids, Python/matplotlib exclusively):
1. Capacity changes expose heterogeneous competition and Deadline risk.
   Panels ask about pooled waiting, worst-fragment misses, resource utilization,
   and the time/location of competition, respectively.
2. Matched FIFO-to-threshold differences expose CI-proxy/Deadline trade-offs.
   The hero panel retains every source-fragment pair; subordinate panels show
   consumed legal waiting capacity and risk among initially flexible tasks.
Exports: 7.2-inch width, editable PDF/SVG, 600-dpi PNG, all text >=6 pt.
Statistics: descriptive pooled counts/quantiles and all source-fragment points;
no confidence intervals, significance tests, independent-replicate claims, or
physical kWh/kg-carbon interpretation. Legacy scales remain in source CSVs;
figures deliberately compare only the three Observed x1 capacity candidates.

No simulated examples or incomplete runs are accepted. metadata.complete must
be true before any figure is rendered. Input hashes and panel selections are
written alongside the exports for reproducibility.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


CANDIDATES = ('OL', 'OM', 'OH')
PRESSURES = (.75, 1., 1.25)
SLAS = (1.5, 2.)
POLICIES = ('fifo', 'edf', 'carbon_threshold_fifo')
MAPPING = 'observed_usage_x1_pressure_v1'
PRESSURE_COLORS = ('#777777', '#3874A4', '#C46C3C')
CANDIDATE_COLORS = ('#3874A4', '#A66D3F', '#78669A')
RESOURCE_COLORS = ('#3874A4', '#A66D3F', '#64865D')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def number(row, field):
    value = float(row[field])
    if not np.isfinite(value):
        raise ValueError(f'Non-finite {field}: {row.get("scenario_id", row.get("candidate_id"))}')
    return value


def select(rows, **criteria):
    """Explicit semantic selections, never a sample for visual convenience."""
    result = []
    for row in rows:
        if all((np.isclose(number(row, key), wanted, rtol=0, atol=1e-12)
                if isinstance(wanted, (float, int)) else row[key] == wanted)
               for key, wanted in criteria.items()):
            result.append(row)
    return result


def exactly_one(rows, **criteria):
    result = select(rows, **criteria)
    if len(result) != 1:
        raise ValueError(f'Expected exactly one row for {criteria}; found {len(result)}')
    return result[0]


def observed(rows):
    result = [r for r in rows if r['candidate_id'] in CANDIDATES]
    if any(r['mapping_version'] != MAPPING for r in result):
        raise ValueError('Observed candidate has an unexpected mapping')
    return result


def footer(fig, text):
    # Reserve a separate footer band; explicit text must not collide with the
    # bottom axes' labels or the heatmap's rotated date labels.
    fig.get_layout_engine().set(rect=(0., .065, 1., .935))
    fig.text(.5, .012, text, fontsize=6, ha='center', va='bottom')


def panel_title(ax, letter, title):
    ax.set_title(f'{letter}  {title}', fontsize=8, fontweight='bold', loc='left', pad=7)


def capacity_grid(groups, scenarios, utilization, panel_counts):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.5), layout='constrained',
                             gridspec_kw={'height_ratios': [1., 1.25]})
    x = np.arange(len(CANDIDATES))
    ax = axes[0, 0]
    for pressure, color in zip(PRESSURES, PRESSURE_COLORS):
        for policy, style, marker in (('fifo', '-', 'o'), ('edf', '--', 's')):
            rows = [exactly_one(groups, candidate_id=c, resource_pressure_factor=pressure,
                                sla_multiplier=1.5, policy=policy) for c in CANDIDATES]
            ax.plot(x, [100*number(r, 'capacity_waited_fraction') for r in rows],
                    color=color, ls=style, marker=marker, ms=3.5, lw=1,
                    label=f'Pressure {pressure:g}' if policy == 'fifo' else '_nolegend_')
    ax.set(xticks=x, xticklabels=CANDIDATES, ylabel='Tasks with capacity waiting (%)',
           xlabel='Observed-use capacity candidate')
    panel_title(ax, 'a', 'Capacity waiting; SLA 1.5')
    handles = [Line2D([], [], color=c, lw=1, label=f'Pressure {p:g}')
               for p, c in zip(PRESSURES, PRESSURE_COLORS)]
    handles += [Line2D([], [], color='#333333', ls='-', marker='o', ms=3,
                      label='FIFO'),
                Line2D([], [], color='#333333', ls='--', marker='s', ms=3,
                       label='EDF')]
    ax.legend(handles=handles, fontsize=6, frameon=False, loc='upper left')
    panel_counts['capacity_wait'] = dict(candidate_rows=18, sla=1.5,
        policies=['fifo', 'edf'], pressures=list(PRESSURES), summary='pooled task fraction')

    ax = axes[0, 1]
    positions = [(c, p) for c in CANDIDATES for p in PRESSURES]
    for sla, color, offset in ((1.5, '#3874A4', -.10), (2., '#A66D3F', .10)):
        for i, (candidate, pressure) in enumerate(positions):
            row = exactly_one(groups, candidate_id=candidate, resource_pressure_factor=pressure,
                              sla_multiplier=sla, policy='edf')
            ax.plot([i+offset, i+offset], [100*number(row, 'dmr'),
                    100*number(row, 'worst_fragment_dmr')], color=color, lw=.65, alpha=.55)
            ax.scatter(i+offset, 100*number(row, 'dmr'), s=15, c=color, marker='o', zorder=3)
            ax.scatter(i+offset, 100*number(row, 'worst_fragment_dmr'), s=19,
                       facecolors='none', edgecolors=color, marker='^', lw=.8, zorder=3)
    ax.axhline(5, color='#555555', ls=':', lw=.8)
    ax.set(xticks=np.arange(9), xticklabels=[f'{c}\n{p:g}' for c, p in positions],
           xlabel='Candidate / resource pressure', ylabel='Deadline miss ratio (%)')
    panel_title(ax, 'b', 'EDF: pooled and worst-fragment DMR')
    handles = [Line2D([], [], color=c, marker='o', ls='', ms=3, label=f'SLA {s:g}')
               for s, c in ((1.5, '#3874A4'), (2., '#A66D3F'))]
    handles += [Line2D([], [], color='#333333', marker='o', ls='', ms=3, label='Pooled'),
                Line2D([], [], color='#333333', marker='^', markerfacecolor='none',
                       ls='', ms=4, label='Worst fragment'),
                Line2D([], [], color='#555555', ls=':', label='5% reference')]
    ax.legend(handles=handles, fontsize=6, frameon=False, loc='upper left')
    panel_counts['deadline'] = dict(candidate_rows=18, policy='edf', slas=list(SLAS),
        pressures=list(PRESSURES), summaries=['pooled task fraction', 'maximum of 16 fragment DMRs'])

    ax = axes[1, 0]
    for resource, color in zip(('cpu', 'gpu', 'mem'), RESOURCE_COLORS):
        rows = [exactly_one(utilization, candidate_id=c, resource_pressure_factor=1.,
                            sla_multiplier=1.5, policy='edf', resource=resource)
                for c in CANDIDATES]
        if any(int(float(r['slots'])) != 1536 for r in rows):
            raise ValueError('Expected 16 original 24-hour fragments, each with 96 arrival slots')
        ax.plot(x, [100*number(r, 'utilization_mean') for r in rows], color=color,
                marker='o', ms=4, lw=1, label=resource.upper())
        ax.plot(x, [100*number(r, 'utilization_p99') for r in rows], color=color,
                marker='s', ms=3.5, ls='--', lw=1)
    ax.set(xticks=x, xticklabels=CANDIDATES, ylabel='Logical capacity utilization (%)',
           xlabel='Candidate (1,536 original arrival slots)', ylim=(0, 105))
    panel_title(ax, 'c', 'Mean (solid), P99 (dashed)\nEDF, pressure 1, SLA 1.5')
    ax.legend(frameon=False, fontsize=6, loc='upper left')
    panel_counts['utilization'] = dict(rows=9, policy='edf', pressure=1., sla=1.5,
        resources=['cpu', 'gpu', 'mem'], slots_per_row=1536, quantile='numpy linear, pooled slots')

    ax = axes[1, 1]
    rows = select(scenarios, policy='edf')
    dates = sorted({r['arrival_start'] for r in rows})
    if len(dates) != 16:
        raise ValueError('Competition heatmap requires all 16 distinct source fragments')
    conditions = [(c, p, s) for c in CANDIDATES for p in PRESSURES for s in SLAS]
    values = np.array([[number(exactly_one(rows, candidate_id=c,
        resource_pressure_factor=p, sla_multiplier=s, arrival_start=date),
        'capacity_competition_minutes') for date in dates] for c, p, s in conditions])
    if np.any(values < 0):
        raise ValueError('Competition duration must be nonnegative')
    image = ax.imshow(values, cmap='Blues', aspect='auto', interpolation='nearest',
                      vmin=0, vmax=max(15., float(np.max(values))))
    ax.set(xticks=np.arange(16), xticklabels=[d[5:10] for d in dates],
           yticks=np.arange(18), yticklabels=[f'{c} {p:g}/{s:g}' for c, p, s in conditions],
           xlabel='Source fragment start (UTC, month-day)', ylabel='Candidate pressure / SLA')
    ax.tick_params(axis='x', labelrotation=90)
    for text in ax.get_xticklabels():
        text.set_rotation_mode('anchor')
    for boundary in (5.5, 11.5):
        ax.axhline(boundary, color='white', lw=.8)
    panel_title(ax, 'd', 'EDF competition duration in each fragment')
    colorbar = fig.colorbar(image, ax=ax, fraction=.046, pad=.035)
    colorbar.set_label('Minutes with capacity waiting', fontsize=6)
    colorbar.ax.tick_params(labelsize=6)
    panel_counts['competition'] = dict(rows=288, policy='edf', source_fragments=16,
        candidates=list(CANDIDATES), pressures=list(PRESSURES), slas=list(SLAS),
        metric='union of 15-minute intervals with at least one capacity-waited task')
    footer(fig, 'Train only; all 16 source fragments retained per condition. Descriptive statistics; no independent-replicate inference.')
    return fig


def matched_pairs(scenarios):
    pairs = []
    for row in select(scenarios, policy='fifo'):
        other = exactly_one(scenarios, candidate_id=row['candidate_id'],
            resource_pressure_factor=number(row, 'resource_pressure_factor'),
            sla_multiplier=number(row, 'sla_multiplier'), arrival_start=row['arrival_start'],
            policy='carbon_threshold_fifo')
        for field in ('mapped_workload_sha256', 'original_environment_sha256', 'arrived'):
            if row[field] != other[field]:
                raise ValueError('Unmatched FIFO-to-threshold pair: '+field)
        baseline = number(row, 'gpu_ci_exposure_proxy')
        if baseline <= 0:
            raise ValueError('Positive FIFO GPU-exposure denominator required')
        pairs.append(dict(candidate_id=row['candidate_id'],
            pressure=number(row, 'resource_pressure_factor'), sla=number(row, 'sla_multiplier'),
            fragment=row['arrival_start'],
            dmr_change_pp=100*(number(other, 'dmr')-number(row, 'dmr')),
            gpu_proxy_reduction_percent=100*(1-number(other, 'gpu_ci_exposure_proxy')/baseline)))
    if len(pairs) != 288:
        raise ValueError(f'Expected 288 complete Observed fragment pairs, found {len(pairs)}')
    return pairs


def decision_grid(groups, scenarios, panel_counts):
    pairs = matched_pairs(scenarios)
    fig = plt.figure(figsize=(7.2, 7.0), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=[1.12, 1.])
    ax = fig.add_subplot(grid[0, :])
    sizes = dict(zip(PRESSURES, (13., 23., 35.)))
    for candidate, color in zip(CANDIDATES, CANDIDATE_COLORS):
        for sla, marker in ((1.5, 'o'), (2., '^')):
            for pressure in PRESSURES:
                values = [p for p in pairs if p['candidate_id']==candidate and
                          p['sla']==sla and p['pressure']==pressure]
                if len(values) != 16:
                    raise ValueError('Every plotted condition must retain all 16 source-fragment pairs')
                ax.scatter([v['dmr_change_pp'] for v in values],
                           [v['gpu_proxy_reduction_percent'] for v in values],
                           color=color, marker=marker, s=sizes[pressure], alpha=.63,
                           edgecolors='none')
    ax.axvline(0, color='#555555', ls=':', lw=.8)
    ax.axhline(0, color='#555555', ls=':', lw=.8)
    ax.set(xlabel='Threshold minus FIFO DMR (percentage points)',
           ylabel='GPU CI-exposure proxy reduction (%)')
    panel_title(ax, 'a', 'Matched threshold / FIFO trade-off: all 288 fragment pairs')
    handles = [Line2D([], [], color=c, marker='o', ls='', label=cid, ms=4)
               for cid, c in zip(CANDIDATES, CANDIDATE_COLORS)]
    handles += [Line2D([], [], color='#555555', marker=m, ls='', ms=4,
                       label=f'SLA {s:g}') for s, m in ((1.5, 'o'), (2., '^'))]
    handles += [Line2D([], [], color='#555555', marker='o', ls='', ms=np.sqrt(sizes[p]),
                       label=f'Pressure {p:g}') for p in PRESSURES]
    ax.legend(handles=handles, fontsize=6, frameon=False, loc='upper left',
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    panel_counts['paired_tradeoff'] = dict(pairs=len(pairs),
        candidates=list(CANDIDATES), pressures=list(PRESSURES), slas=list(SLAS),
        source_fragments_per_condition=16,
        matching=['candidate', 'pressure', 'SLA', 'arrival_start', 'workload_sha', 'environment_sha'],
        formula='100*(1-threshold_gpu_ci_exposure_proxy/fifo_gpu_ci_exposure_proxy)',
        proxy='observed GPU-equivalent * original-duration hours * CI g/kWh; NOT physical emissions')

    positions = [(c, s) for c in CANDIDATES for s in SLAS]
    x = np.arange(6)
    ax = fig.add_subplot(grid[1, 0])
    keys = ('negative', '0', '1', '2', '3', 'ge4')
    labels = ('Late', '0', '1', '2', '3', '4+')
    colors = ('#A85F53', '#CACACA', '#BDD2E1', '#8DB3CE', '#5C93B7', '#2F6D98')
    for policy, offset in (('fifo', -.18), ('carbon_threshold_fifo', .18)):
        rows = [exactly_one(groups, candidate_id=c, resource_pressure_factor=1.,
                            sla_multiplier=s, policy=policy) for c, s in positions]
        bottom = np.zeros(6)
        for key, color in zip(keys, colors):
            values = np.array([100*number(r, f'observed_remaining_legal_wait_slots_{key}_tasks')/
                               number(r, 'arrived') for r in rows])
            ax.bar(x+offset, values, bottom=bottom, width=.33, color=color, linewidth=0)
            bottom += values
        if not np.allclose(bottom, 100, atol=1e-8, rtol=0):
            raise ValueError('Remaining-Slack categories must retain every task')
    ax.set(xticks=x, xticklabels=[f'{c}\n{s:g}' for c, s in positions], ylim=(0, 100),
           xlabel='Candidate / SLA; each pair: FIFO, threshold', ylabel='Tasks (%)')
    ax.set_title('b  Legal slots left at start; pressure 1', fontsize=8,
                 fontweight='bold', loc='left', pad=25)
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(colors, labels)],
              fontsize=6, frameon=False, ncol=6, handlelength=.9,
              columnspacing=.8, loc='lower left', bbox_to_anchor=(0, 1.01), borderaxespad=0)
    panel_counts['remaining_slack'] = dict(candidate_rows=12, pressure=1.,
        slas=list(SLAS), policies=['fifo', 'carbon_threshold_fifo'],
        categories=list(keys), denominator='all arrived tasks',
        metric='floor((original initial Slack - observed waiting)/15), negative separately')

    ax = fig.add_subplot(grid[1, 1])
    for policy, offset, color in (('fifo', -.15, '#3874A4'),
                                   ('carbon_threshold_fifo', .15, '#A66D3F')):
        rows = [exactly_one(groups, candidate_id=c, resource_pressure_factor=1.,
                            sla_multiplier=s, policy=policy) for c, s in positions]
        denominators = np.array([number(r, 'long_slack_tasks') for r in rows])
        if np.any(denominators <= 0):
            raise ValueError('Long-Slack conditional percentages require positive denominators')
        waiting = np.array([number(r, 'long_slack_capacity_waited_tasks') for r in rows])
        missed = np.array([number(r, 'long_slack_all_missed_tasks') for r in rows])
        ax.bar(x+offset, 100*waiting/denominators, width=.27, color=color, alpha=.8)
        ax.scatter(x+offset, 100*missed/denominators, color=color, marker='D',
                   edgecolors='black', linewidths=.45, s=16, zorder=3)
    ax.set(xticks=x, xticklabels=[f'{c}\n{s:g}' for c, s in positions],
           xlabel='Candidate / SLA', ylabel='Among tasks with initial Slack >=60 min (%)')
    panel_title(ax, 'c', 'Flexible tasks still face competition; pressure 1')
    handles = [Patch(color='#3874A4', label='FIFO'), Patch(color='#A66D3F', label='Threshold'),
               Patch(color='#999999', label='Bar: capacity waited'),
               Line2D([], [], color='#555555', marker='D', ls='', ms=4,
                      markeredgecolor='black', label='Diamond: deadline missed')]
    ax.legend(handles=handles, fontsize=6, frameon=False, loc='upper left')
    panel_counts['long_slack_risk'] = dict(candidate_rows=12, pressure=1., slas=list(SLAS),
        policies=['fifo', 'carbon_threshold_fifo'],
        denominator='tasks with original initial Slack >=60 minutes',
        metrics=['any capacity wait', 'any deadline miss'])
    footer(fig, 'All source-fragment pairs retained; coincident points overlap. Original Train CI and complete task sets.\n'
           'CI exposure is a scheduling proxy; no IT/HVAC model or physical carbon claim.')
    return fig


def save_figure(fig, output, stem):
    pdf, svg, png = (output/f'{stem}.{extension}' for extension in ('pdf', 'svg', 'png'))
    fig.savefig(output/f'{stem}.pdf', dpi=600, facecolor='white',
                metadata={'Creator': 'A1 aggregate diagnostic', 'CreationDate': None, 'ModDate': None})
    fig.savefig(output/f'{stem}.svg', dpi=600, facecolor='white',
                metadata={'Creator': 'A1 aggregate diagnostic', 'Date': None})
    fig.savefig(output/f'{stem}.png', dpi=600, facecolor='white')
    text = svg.read_bytes().decode('utf-8')
    svg.write_bytes(('\n'.join(line.rstrip() for line in text.splitlines())+'\n').encode('utf-8'))
    plt.close(fig)
    return [pdf, svg, png]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='docs/research/object-validity')
    args = parser.parse_args(argv)
    root = Path(args.root)
    data, output = root/'data/a1/rules', root/'figures/a1'
    metadata_path = data/'metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata.get('complete') is not True:
        raise ValueError('Refuse to render incomplete A1 experiments')
    if (metadata.get('allowed_roles') != ['train'] or
            metadata.get('physics_model_called') is not False or
            metadata.get('trained_models') is not False):
        raise ValueError('A1 figures require Train-only, no-physics, no-training provenance')
    if metadata.get('completed_runs') != metadata.get('planned_runs'):
        raise ValueError('Metadata planned/completed run counts differ')
    names = ('candidate-summary.csv', 'scenario-summary.csv', 'utilization-pooled.csv', 'task-types.csv')
    paths = [metadata_path]+[data/name for name in names]
    tables = {name: read_csv(data/name) for name in names}
    groups, scenarios, utilization = [observed(tables[name]) for name in names[:3]]
    if len(groups) != 54 or len(scenarios) != 864 or len(utilization) != 162:
        raise ValueError('Observed candidates must retain the complete 3x3x2x3 comparison grid')
    if any(number(r, 'completed_fraction') != 1 for r in scenarios):
        raise ValueError('No unreported unfinished task is allowed in the complete comparison')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 8, 'legend.fontsize': 6,
        'xtick.labelsize': 6, 'ytick.labelsize': 6, 'pdf.fonttype': 42,
        'svg.fonttype': 'none', 'svg.hashsalt': 'a1-object-model-v1',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .7, 'lines.linewidth': 1, 'figure.constrained_layout.h_pad': .08,
        'figure.constrained_layout.w_pad': .06})
    output.mkdir(parents=True, exist_ok=True)
    panels = {}
    exports = save_figure(capacity_grid(groups, scenarios, utilization, panels),
                          output, 'A1-capacity-competition')
    exports += save_figure(decision_grid(groups, scenarios, panels),
                           output, 'A1-ci-deadline-conflict')
    provenance = dict(schema='a1-aggregate-figure-provenance-v1',
        source_sha=metadata['source_sha'], base_source_sha=metadata['base_source_sha'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        backend='Python matplotlib', dpi=600, font_sizes_pt=[6, 7, 8],
        export_contract='Editable PDF/SVG and requested 600-dpi PNG preview; TIFF submission raster not requested.',
        inputs={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in exports},
        input_row_counts={k:len(v) for k,v in tables.items()},
        observed_row_counts=dict(candidate_summary=len(groups), scenario_summary=len(scenarios),
                                 utilization=len(utilization)),
        exclusion_rule='Legacy L/M/H retained in source tables, omitted from figures to avoid cross-mapping absolute comparison',
        panels=panels,
        statistics='Descriptive only; pooled original tasks or equal 15-minute slots as specified. '
                   'Sixteen source fragments are not independent biological/statistical replicates; no error bars/tests.',
        physical_interpretation='No kWh or kgCO2; CI proxy reductions cannot establish physical carbon savings.',
        image_integrity='Quantitative matplotlib marks only; no raster image manipulation, fitted curves, or sampling.',
        visual_qa_status='Requires panel-by-panel inspection of final exports; automated checks alone are insufficient')
    (output/'figure-provenance.json').write_bytes((json.dumps(provenance, indent=2)+'\n').encode('utf-8'))
    print(json.dumps(dict(exports=[str(p) for p in exports], panels=list(panels)), indent=2))


if __name__ == '__main__':
    main()
