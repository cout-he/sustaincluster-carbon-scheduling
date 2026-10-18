"""A0 aggregate-only figures. No raw/scenario payload is opened by this module.

Descriptive statistics only: scenario augmentations are not independent samples.
Inputs are the explicitly exported CSVs alongside the report. No fitted curves,
error bars, invented data, significance tests, or physical-model corrections.
"""
from pathlib import Path
import argparse
import csv
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='docs/research/object-validity')
    args = parser.parse_args()
    root = Path(args.root)
    data = root / 'data'
    out = root / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 7,
        'axes.labelsize': 7, 'axes.titlesize': 8, 'legend.fontsize': 6,
        'xtick.labelsize': 6, 'ytick.labelsize': 6, 'pdf.fonttype': 42,
        'svg.fonttype': 'none', 'axes.spines.top': False,
        'axes.spines.right': False, 'savefig.facecolor': 'white'})
    colors = ['#2166AC', '#D6604D', '#4D9221']
    inputs = []

    def read(name):
        path = data / name
        inputs.append(path)
        return rows(path)

    def save(fig, name):
        fig.savefig(out / f'{name}.pdf', dpi=600)
        fig.savefig(out / f'{name}.svg', dpi=600)
        svg = out / f'{name}.svg'
        svg.write_bytes(('\n'.join(line.rstrip() for line in
                        svg.read_text(encoding='utf8').splitlines())+'\n').encode('utf8'))
        fig.savefig(out / f'{name}.png', dpi=600)
        plt.close(fig)

    utilization = read('resources/utilization-pooled.csv')
    pressure = read('resources/pressure-policy-summary.csv')
    opportunity = read('carbon/opportunity_groups.csv')
    pairs = read('carbon/rule_pairs.csv')
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.1), layout='constrained')
    ax = axes[0, 0]
    for resource, color in zip(('cpu', 'gpu', 'mem'), colors):
        selection = [r for r in utilization if r['policy'] == 'edf' and
                     r['sla_multiplier'] == '1.5' and r['resource'] == resource]
        x = [float(r['resource_pressure_factor']) for r in selection]
        ax.plot(x, [100*float(r['utilization_mean']) for r in selection],
                marker='o', color=color, label=resource.upper())
        ax.plot(x, [100*float(r['utilization_p99']) for r in selection],
                marker='s', ls='--', color=color, label='_nolegend_')
    ax.set(xlabel='Resource pressure factor', ylabel='Capacity utilization (%)',
           title='a  Mean (solid) and P99 (dashed)', ylim=(0, 105))
    ax.legend(loc='upper left', frameon=False)
    ax = axes[0, 1]
    for policy, color in zip(('edf', 'fifo'), colors):
        selection = [r for r in pressure if r['policy'] == policy and r['sla_multiplier'] == '1.5']
        x = [float(r['resource_pressure_factor']) for r in selection]
        ax.plot(x, [100*float(r['capacity_waited_tasks_fraction']) for r in selection],
                marker='o', color=color, label=f'{policy.upper()} capacity wait')
        ax.plot(x, [100*float(r['deadline_misses_fraction']) for r in selection],
                marker='s', ls='--', color=color, label=f'{policy.upper()} deadline miss')
    ax.set(xlabel='Resource pressure factor', ylabel='Tasks (%)',
           title='b  Real capacity waiting; no active defer', ylim=(0, 5.5))
    ax.legend(frameon=False)
    ax = axes[1, 0]
    for idx, sla in enumerate(('1.5', '2.0')):
        r = next(r for r in pressure if r['policy'] == 'edf' and
                 r['resource_pressure_factor'] == '1.0' and r['sla_multiplier'] == sla)
        vals = [100*float(r[f'legal_wait_slots_{key}_fraction']) for key in ('0', '1', '2', '3', 'ge4')]
        ax.bar(np.arange(5)+(idx-.5)*.36, vals, width=.36, color=colors[idx], label=f'SLA {sla}')
    ax.set(xticks=np.arange(5), xticklabels=['0', '1', '2', '3', '4+'],
           xlabel='Legal waiting slots (15 min each)', ylabel='Tasks (%)',
           title='c  Deadline flexibility is heterogeneous')
    ax.legend(frameon=False)
    ax = axes[1, 1]
    ideal = [r for r in opportunity if r['group'] in ('sla=1.5', 'sla=2.0')]
    gpu = [float(r['gpu_resource_weighted_ci_integral_reduction_percent']) for r in ideal]
    ax.bar([0, 1], gpu, color=colors[:2], width=.55)
    for x, y in enumerate(gpu):
        ax.text(x, y+.12, f'{y:.2f}%', ha='center', fontsize=7)
    ax.set(xticks=[0, 1], xticklabels=['SLA 1.5', 'SLA 2.0'], ylim=(0, 8),
           ylabel='GPU-weighted CI integral reduction (%)',
           title='d  Ideal opportunity: no capacity or power model')
    save(fig, 'A0-resource-and-flexibility')

    probe = read('physics/physical-probes.csv')
    comparison = read('physics/setpoint-comparisons.csv')
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.1), layout='constrained')
    ax = axes[0, 0]
    for u, color in zip((0, .5, 1), colors):
        selection = [r for r in probe if r['family'] == 'factorial' and
                     float(r['ambient_dry_bulb_c']) == 25 and
                     all(float(r[f'{key}_utilization']) == u for key in ('cpu', 'gpu', 'mem'))]
        selection.sort(key=lambda r: float(r['setpoint_c']))
        ax.plot([float(r['setpoint_c']) for r in selection],
                [float(r['system_kw'])/1000 for r in selection],
                marker='o', color=color, label=f'CPU/GPU/MEM = {u:g}')
    ax.set(xlabel='Cooling setpoint (C)', ylabel='System electric power (MW)',
           title='a  Inherited equations; ambient 25 C')
    ax.legend(frameon=False)
    ax = axes[0, 1]
    for ambient, color in zip((5, 25, 40), colors):
        selection = [r for r in comparison if r['family'] == 'diagonal' and
                     float(r['ambient_dry_bulb_c']) == ambient]
        selection.sort(key=lambda r: float(r['cpu_utilization']))
        ax.plot([float(r['cpu_utilization']) for r in selection],
                [float(r['delta_22_minus_18_kw']) for r in selection],
                color=color, marker='.', label=f'Ambient {ambient} C')
    ax.axhline(0, color='#777777', lw=.6)
    ax.set(xlabel='Equal CPU/GPU/MEM utilization', ylabel='Power at 22 C minus 18 C (kW)',
           title='b  Crossings depend on suspect CT sizing')
    ax.legend(frameon=False)
    ax = axes[1, 0]
    fields = [('cpu_kw', 'CPU'), ('gpu_kw', 'GPU'), ('it_kw', 'IT'),
              ('ct_fan_kw', 'CT fan'), ('compressor_kw', 'Comp.'), ('system_kw', 'Total')]
    for idx, sp in enumerate((18, 22)):
        r = next(r for r in probe if r['family'] == 'factorial' and
                 float(r['ambient_dry_bulb_c']) == 25 and float(r['setpoint_c']) == sp and
                 all(float(r[f'{key}_utilization']) == 1 for key in ('cpu', 'gpu', 'mem')))
        ax.bar(np.arange(len(fields))+(idx-.5)*.36,
               [float(r[field])/1000 for field, _ in fields],
               width=.36, color=colors[idx], label=f'{sp} C')
    ax.set(xticks=np.arange(len(fields)), xticklabels=[label for _, label in fields],
           ylabel='Electric power (MW)', title='c  Full load at 25 C: CT fan drives reversal')
    ax.legend(frameon=False)
    ax = axes[1, 1]
    primary = [r for r in pairs if r['baseline_policy'] == 'immediate_fifo']
    for sla, color in zip((1.5, 2), colors):
        selection = [r for r in primary if float(r['sla']) == sla]
        ax.scatter([100*(float(r['dmr_threshold'])-float(r['dmr_immediate'])) for r in selection],
                   [float(r['incremental_carbon_reduction_percent']) for r in selection],
                   color=color, s=18, alpha=.85, label=f'SLA {sla:g}')
    ax.axvline(0, color='#777777', lw=.6)
    ax.set(xlabel='Threshold minus FIFO deadline misses (pp)',
           ylabel='Incremental carbon reduction (%)',
           title='d  Rule outcomes conditional on flawed power')
    ax.legend(frameon=False)
    save(fig, 'A0-physics-and-rule-boundaries')
    provenance = {'source_sha': 'bf8a35ef647f19b5ef10ce1b99f723370b7a0732',
        'method': 'Descriptive aggregate-only figures; no raw/scenario payload read; no inferential error bars.',
        'inputs': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        'backend': 'matplotlib', 'dpi': 600, 'font_sizes_pt': [6, 7, 8]}
    (out/'figure-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
