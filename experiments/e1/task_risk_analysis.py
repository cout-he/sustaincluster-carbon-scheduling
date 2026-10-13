"""Matched E1-6A diagnostics and source-data figures; no strategy tuning."""
import math
import argparse
import csv
import json
from pathlib import Path
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

VERSION = 'e1-6a-matched-analysis-v1'
LABELS = {
    'edf_backfill': 'EDF', 'edf_carbon_threshold': 'EDF + Carbon',
    'cars_risk_only': 'RiskOnly', 'task_risk_full': 'Task Full',
    'task_risk_no_history': 'No History', 'task_risk_no_urgent': 'No Urgent',
    'task_risk_no_utilization': 'No Utilization', 'task_risk_max_required': 'Required Max',
}
SHORT = {'task_risk_full':'Full', 'task_risk_no_history':'NoH',
         'task_risk_no_urgent':'NoQ', 'task_risk_no_utilization':'NoU',
         'task_risk_max_required':'Max'}
FIGURE_CONTRACT = dict(
    conclusion='Matched development cohorts expose carbon-SLA and task-budget tradeoffs; no safety guarantee.',
    archetype='quantitative grid', backend='python/matplotlib', minimum_font_pt=6,
    exports=['PNG 300 dpi', 'PDF editable TrueType text', 'SVG editable text'],
    evidence=['all eight matched policies: deadline miss rate', 'carbon relative to paired EDF',
              'high-pressure Pareto associations', 'first-eligible task buffer distribution',
              'first-eligible buffer by resource-demand dominance'],
    uncertainty='One deterministic frozen trace per cohort; no inferential error bars.',
    exclusions='No task exclusions; five new policies alone have task-level budget diagnostics.',
    validity='Validation is development data; Test is sealed; differing cohorts are never carbon-paired.')


def _distribution(values):
    return dict(n=len(values), mean_minutes=math.fsum(values)/len(values) if values else None,
                minimum_minutes=min(values) if values else None,
                maximum_minutes=max(values) if values else None,
                median_minutes=float(np.median(values)) if values else None,
                p90_minutes=float(np.quantile(values, .9)) if values else None,
                histogram_minutes=dict(sorted(Counter(values).items())))


def budget_statistics(diagnostics):
    """Capacity failures are absent here; never encode absent budgets as zero."""
    diagnostics = list(diagnostics)
    eligible = [d for d in diagnostics if d.get('eligible') and d.get('budget') is not None]
    first = {}
    for row in eligible:
        first.setdefault(row['task_id'], row)
    all_dist = _distribution([d['budget']['buffer_minutes'] for d in eligible])
    first_dist = _distribution([d['budget']['buffer_minutes'] for d in first.values()])
    resources = defaultdict(list)
    for row in first.values():
        resources[row['budget']['dominant_resource']].append(row['budget']['buffer_minutes'])
    return dict(candidate_checks=len(diagnostics), eligible_checks=len(eligible),
                first_eligible_tasks=len(first), all_buffer_mean_minutes=all_dist['mean_minutes'],
                first_buffer_mean_minutes=first_dist['mean_minutes'], all_distribution=all_dist,
                first_distribution=first_dist, first_resource={r: _distribution(v) for r, v in resources.items()},
                defer_fraction_eligible=sum(d['reason'] == 'high_ci_defer' for d in eligible)/len(eligible)
                    if eligible else None,
                reason_counts=dict(Counter(d['reason'] for d in diagnostics)))


def pareto_flags(rows):
    """Descriptive within-cohort nondominance; no significance or optimality claim."""
    return [not any(other['split'] == row['split'] and other['factor'] == row['factor']
                    and other['carbon_kg'] <= row['carbon_kg'] and other['dmr'] <= row['dmr']
                    and (other['carbon_kg'] < row['carbon_kg'] or other['dmr'] < row['dmr'])
                    for other in rows) for row in rows]


def paired_tasks(reference, candidate):
    """Trajectory association only, not an intervention's causal attribution."""
    ref = {r['task_id']: r for r in reference}
    can = {r['task_id']: r for r in candidate}
    if len(ref) != len(reference) or len(can) != len(candidate) or set(ref) != set(can):
        raise ValueError('Paired task identity mismatch')
    output = []
    for task_id in sorted(ref):
        a, b = ref[task_id], can[task_id]
        miss_a, miss_b = a['deadline_outcome'].startswith('missed'), b['deadline_outcome'].startswith('missed')
        shift = ((pd.Timestamp(b['start_time'])-pd.Timestamp(a['start_time'])).total_seconds()/60
                 if a['start_time'] and b['start_time'] else None)
        output.append(dict(task_id=task_id, reference_start=a['start_time'], candidate_start=b['start_time'],
                           shift_minutes=shift, additional_miss=miss_b and not miss_a,
                           avoided_miss=miss_a and not miss_b, shared_miss=miss_a and miss_b))
    return output


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _lines(path):
    with Path(path).open(encoding='utf8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _csv(path):
    with Path(path).open(encoding='utf8', newline='') as stream:
        return list(csv.DictReader(stream))


def _number(value):
    if value is None or value == '':
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('Nonfinite analysis input')
    return number


def _miss(row):
    return row['deadline_outcome'].startswith('missed')


def _identity(rows, policy):
    ids = [row['task_id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate task identity for ' + policy)
    return {row['task_id']: row for row in rows}


def _checked_groups(campaign):
    """Validate every registered group before creating any analysis directory."""
    from .contracts import content_hash
    from .task_risk_experiment import _campaign_config, _completed_group, GROUP_POLICIES, PARAMETERS
    campaign = Path(campaign)
    frozen = _read(campaign/'campaign-freeze.json')
    config = _campaign_config(frozen['config'])
    if (frozen['config_sha256'] != content_hash(config)
            or frozen['source_sha256'] != content_hash(frozen['sources'])
            or frozen.get('formal_test_evaluated') is not False):
        raise ValueError('Campaign source/config identity mismatch')
    groups = []
    for entry in config['cohorts']:
        folder = campaign/f"{entry['split']}-{entry['factor']:g}"
        fairness = _completed_group(folder)  # Includes complete-status and all scientific hashes.
        source = _read(folder/'source-freeze.json')
        if (fairness['split'] != entry['split'] or fairness['factor'] != entry['factor']
                or fairness['policies'] != list(GROUP_POLICIES) or fairness['parameters'] != PARAMETERS
                or not fairness['identical_frozen_inputs'] or not fairness['idle_included']
                or fairness.get('policy_future_ci_access') is not False
                or fairness.get('formal_test_evaluated') is not False
                or source['sources'] != frozen['sources']
                or source['source_sha256'] != frozen['source_sha256']
                or not _read(folder/'baseline-regression.json')['matched']):
            raise ValueError('Group fairness/source/regression guard failed')
        if 'workload_sha256' in entry and fairness['workload_sha256'] != entry['workload_sha256']:
            raise ValueError('Frozen workload identity mismatch')
        if 'environment_sha256' in entry and fairness['environment_sha256'] != entry['environment_sha256']:
            raise ValueError('Frozen environment identity mismatch')
        common = _csv(folder/'common_summary.csv')
        if len(common) != len(GROUP_POLICIES) or {row['policy'] for row in common} != set(GROUP_POLICIES):
            raise ValueError('Eight common-window policy rows required')
        if any(row['split'] != entry['split'] or float(row['factor']) != entry['factor']
               or row['observation_end'] != fairness['common_end'] for row in common):
            raise ValueError('Common observation endpoint mismatch')
        groups.append((entry, folder, fairness, common))
    return frozen, groups


def _summarize_group(entry, folder, fairness, common):
    from .task_risk_experiment import GROUP_POLICIES
    from .task_risk import POLICY_NAMES
    split, factor = entry['split'], entry['factor']
    cohort = dict(split=split, factor=factor)
    runtime = {row['policy']:row for row in _csv(folder/'runtime.csv')}
    common_by_policy = {row['policy']:row for row in common}
    ledgers = {}; waits = {}; first_diagnostics = {}; first_budgets = {}
    summaries = []; phase_rows = []; budget_rows = []; histogram_rows = []
    resource_rows = []; resource_bins = []; reason_rows = []
    for name in GROUP_POLICIES:
        sub = folder/name
        ledger = tuple(_lines(sub/'ledger.jsonl'))
        ledgers[name] = _identity(ledger, name)
        waits[name] = _identity(tuple(_lines(sub/'policy_ledger.jsonl')), name)
        if set(ledgers[name]) != set(waits[name]) or len(ledger) != fairness['arrived']:
            raise ValueError('Task/wait ledger identity mismatch')
        source = common_by_policy[name]
        numeric = ('arrived','completed','pending','running','deadline_missed_completed',
            'deadline_missed_unfinished','deadline_right_censored','it_energy_kwh','hvac_energy_kwh',
            'total_energy_kwh','carbon_kg','policy_defer_intervals','capacity_wait_intervals',
            'pending_mean','pending_max','mean_wait_minutes_started','mean_jct_minutes_completed',
            'cores_fraction_mean','cores_fraction_max','gpus_fraction_mean','gpus_fraction_max',
            'mem_fraction_mean','mem_fraction_max','cores_hours','gpus_hours','mem_hours','drain_hours')
        row = dict(cohort, policy=name, **{key:_number(source.get(key)) for key in numeric})
        for key in ('arrived','completed','pending','running','deadline_missed_completed',
                    'deadline_missed_unfinished','deadline_right_censored'):
            row[key] = int(row[key])
        row['misses'] = row['deadline_missed_completed'] + row['deadline_missed_unfinished']
        row['dmr'] = row['misses']/row['arrived'] if row['arrived'] else None
        if (row['arrived'] != len(ledger)
                or row['completed']+row['pending']+row['running'] != row['arrived']
                or row['misses'] != sum(_miss(task) for task in ledger)
                or not math.isclose(row['total_energy_kwh'], row['it_energy_kwh']+row['hvac_energy_kwh'],
                                    rel_tol=1e-12, abs_tol=1e-9)
                or not math.isclose(row['dmr'], float(source['deadline_violation_rate_observed']),
                                    rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError('Task/deadline/energy summary guard failed')
        if fairness['all_tasks_completed'] and (row['completed'] != row['arrived']
            or row['pending'] or row['running'] or row['deadline_right_censored']):
            raise ValueError('Complete-Drain assertion contradicts task ledger')
        row.update(observation_end=source['observation_end'],
            own_observation_end=source['own_observation_end'], termination=source['termination'],
            workload_sha256=fairness['workload_sha256'], environment_sha256=fairness['environment_sha256'],
            equal_workload_carbon_comparison_valid=fairness['equal_workload_carbon_comparison_valid'],
            meets_5_percent=row['dmr'] <= .05 and row['deadline_right_censored'] == 0,
            run_seconds=float(runtime[name]['run_seconds']),
            decision_seconds=float(runtime[name]['decision_seconds']),
            order_seconds=float(runtime[name]['order_seconds']))
        row['total_policy_seconds'] = math.fsum(float(runtime[name].get(key, 0)) for key in
            ('order_seconds','decision_seconds','step_state_seconds','candidate_state_seconds',
             'start_notification_seconds','defer_notification_seconds'))
        summaries.append(row)
        phases = _read(sub/'phases.json')
        if set(phases) != {'fixed','own','common'}:
            raise ValueError('Three phase summaries required')
        for phase, values in phases.items():
            phase_rows.append(dict(cohort, policy=name, phase=phase, **{
                key:value for key,value in values.items() if key not in ('split','factor','policy','phase')}))
        for key in ('arrived','completed','pending','running','carbon_kg','total_energy_kwh'):
            if not math.isclose(float(phases['common'][key]), row[key], rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError('CSV and common-phase summary disagree')
        diagnostics = list(_lines(sub/'diagnostics.jsonl'))
        first_all = {}; first_eligible = {}
        for diag in diagnostics:
            if diag['task_id'] not in ledgers[name]:
                raise ValueError('Unknown diagnostic task')
            first_all.setdefault(diag['task_id'], diag)
            if diag.get('eligible') and diag.get('budget') is not None:
                first_eligible.setdefault(diag['task_id'], diag)
        first_diagnostics[name], first_budgets[name] = first_all, first_eligible
        reasons = Counter(event['reason'] for event in _lines(sub/'policy_events.jsonl'))
        reason_rows.extend(dict(cohort, policy=name, reason=reason, count=count)
                           for reason,count in sorted(reasons.items()))
        if name not in POLICY_NAMES:
            continue
        stat = budget_statistics(diagnostics)
        budget_rows.append(dict(cohort, policy=name,
            candidate_checks=stat['candidate_checks'], capacity_wait_checks=row['capacity_wait_intervals'],
            eligible_checks=stat['eligible_checks'], first_eligible_tasks=stat['first_eligible_tasks'],
            all_buffer_mean_minutes=stat['all_buffer_mean_minutes'],
            first_buffer_mean_minutes=stat['first_buffer_mean_minutes'],
            first_buffer_median_minutes=stat['first_distribution']['median_minutes'],
            first_buffer_p90_minutes=stat['first_distribution']['p90_minutes'],
            first_buffer_min_minutes=stat['first_distribution']['minimum_minutes'],
            first_buffer_max_minutes=stat['first_distribution']['maximum_minutes'],
            defer_fraction_eligible=stat['defer_fraction_eligible'],
            guards_eligible=sum(d['reason']=='task_risk_guard' for d in diagnostics),
            total_arrived=row['arrived']))
        for scope, distribution in (('all_eligible',stat['all_distribution']),
                                     ('first_eligible',stat['first_distribution'])):
            for steps in range(11):
                count = distribution['histogram_minutes'].get(steps*15, 0)
                histogram_rows.append(dict(cohort, policy=name, scope=scope, buffer_steps=steps,
                    buffer_minutes=steps*15, count=count, denominator=distribution['n'],
                    fraction=count/distribution['n'] if distribution['n'] else None))
        for resource in ('cpu','gpu','mem','tie','zero'):
            distribution = stat['first_resource'].get(resource, _distribution([]))
            resource_rows.append(dict(cohort, policy=name, dominant_resource=resource,
                n=distribution['n'], mean_minutes=distribution['mean_minutes'],
                median_minutes=distribution['median_minutes'], p90_minutes=distribution['p90_minutes'],
                minimum_minutes=distribution['minimum_minutes'], maximum_minutes=distribution['maximum_minutes']))
            for steps in range(11):
                count = distribution['histogram_minutes'].get(steps*15, 0)
                resource_bins.append(dict(cohort, policy=name, dominant_resource=resource,
                    buffer_steps=steps, count=count, denominator=distribution['n'],
                    fraction=count/distribution['n'] if distribution['n'] else None))
        del diagnostics
    baseline_ids = set(ledgers['edf_backfill'])
    if any(set(ledger) != baseline_ids for ledger in ledgers.values()):
        raise ValueError('Policies have different finite task sets')
    features = ('arrival_time','duration','cores_req','gpu_req','mem_req','sla_deadline')
    for name in GROUP_POLICIES:
        if any(tuple(ledgers[name][key][f] for f in features) !=
               tuple(ledgers['edf_backfill'][key][f] for f in features) for key in baseline_ids):
            raise ValueError('Frozen task features changed between strategies')
    edf_carbon = next(row['carbon_kg'] for row in summaries if row['policy']=='edf_backfill')
    for row in summaries:
        row['carbon_reduction_percent_vs_edf'] = ((edf_carbon-row['carbon_kg'])/edf_carbon*100
            if edf_carbon > 0 and row['equal_workload_carbon_comparison_valid'] else None)
    paired = []
    stored_pairs = {(row['policy'],row['reference']):row for row in _csv(folder/'paired-miss-summary.csv')}
    for name in GROUP_POLICIES:
        for reference in ('edf_backfill','cars_risk_only'):
            pairs = paired_tasks(list(ledgers[reference].values()), list(ledgers[name].values()))
            pair = dict(cohort, policy=name, reference=reference,
                extra_misses=sum(p['additional_miss'] for p in pairs),
                avoided_misses=sum(p['avoided_miss'] for p in pairs),
                both_missed=sum(p['shared_miss'] for p in pairs),
                started_differently=sum(p['shift_minutes'] not in (0,None) for p in pairs),
                attribution='matched trajectory association, not single-event causality')
            if any(int(stored_pairs[name,reference][key]) != pair[key] for key in
                   ('extra_misses','avoided_misses','both_missed','started_differently')):
                raise ValueError('Paired miss summary disagrees with task ledgers')
            paired.append(pair)
    selection = []; details = []
    risk, full = ledgers['cars_risk_only'], ledgers['task_risk_full']
    rules = {
        'added_miss':lambda key:_miss(full[key]) and not _miss(risk[key]),
        'avoided_miss':lambda key:_miss(risk[key]) and not _miss(full[key]),
        'changedstart_sameoutcome':lambda key:full[key]['start_time'] != risk[key]['start_time']
            and full[key]['deadline_outcome'] == risk[key]['deadline_outcome'],
    }
    for case, rule in rules.items():
        task_id = next((key for key in sorted(baseline_ids) if rule(key)), None)
        selection.append(dict(cohort, case=case, found=task_id is not None, task_id=task_id,
                              selection_rule='first lexical stable_id in preregistered category'))
        if task_id is None:
            continue
        for name in GROUP_POLICIES:
            task_row = ledgers[name][task_id]; wait = waits[name][task_id]
            first = first_diagnostics[name].get(task_id, {})
            first_budget = first_budgets[name].get(task_id, {})
            value = first_budget.get('budget') or {}
            details.append(dict(cohort, case=case, task_id=task_id, policy=name,
                **{key:task_row.get(key) for key in features+('start_time','finish_time','state','deadline_outcome')},
                capacity_wait_intervals=wait['capacity_wait_intervals'],
                policy_defer_intervals=wait['policy_defer_intervals'],
                wait_minutes=wait['observed_pending_minutes'],
                jct_minutes=(pd.Timestamp(task_row['finish_time'])-pd.Timestamp(task_row['arrival_time'])).total_seconds()/60
                    if task_row['finish_time'] else None,
                first_decision_time=first.get('time'), first_decision_reason=first.get('reason'),
                legacy_history_buffer_steps=first.get('buffer_steps') if name=='cars_risk_only' else None,
                first_eligible_time=first_budget.get('time'), first_buffer_minutes=value.get('buffer_minutes'),
                first_solo_slack_minutes=value.get('solo_slack_minutes'),
                first_safe_slack_minutes=value.get('safe_slack_minutes'),
                first_dominant_resource=value.get('dominant_resource'),
                first_budget_components=json.dumps(value, sort_keys=True) if value else None))
    return dict(summary=summaries, phases=phase_rows, budgets=budget_rows, buffers=histogram_rows,
                resources=resource_rows, resource_bins=resource_bins, reasons=reason_rows,
                paired=paired, selection=selection, cases=details)


def _plots(output, tables):
    """Paper figures use every registered method/cohort; no synthetic error bars."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .task_risk_experiment import GROUP_POLICIES
    from .task_risk import POLICY_NAMES
    colors = ('#444444','#b17c45','#507d52','#285a91','#5a7fa6','#849dbc','#7d6195','#ad7898')
    markers = ('o','s','^','D','v','P','X','h')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':7,'axes.labelsize':7,'axes.titlesize':8,'xtick.labelsize':6,
        'ytick.labelsize':6,'legend.fontsize':6,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.7})
    summary = pd.DataFrame(tables['summary'])
    def save(fig, stem):
        fig.savefig(output/(stem+'.png'), dpi=300, bbox_inches='tight')
        fig.savefig(output/(stem+'.pdf'), bbox_inches='tight',
                    metadata={'CreationDate':None,'ModDate':None})
        fig.savefig(output/(stem+'.svg'), bbox_inches='tight', metadata={'Date':None})
        plt.close(fig)
    def caption(fig):
        fig.text(.5,.01,'One deterministic frozen cohort per point; Validation is development data.',
                 ha='center',fontsize=6)
    for stem, metric, ylabel in (
            ('deadline-miss-rate','dmr','Deadline miss rate (%)'),
            ('carbon-relative-edf','carbon_reduction_percent_vs_edf','Carbon reduction vs paired EDF (%)')):
        fig, axes = plt.subplots(1,2,figsize=(7.2,3.3),sharey=True)
        for ax,split in zip(axes,('train','validation')):
            for name,color,marker in zip(GROUP_POLICIES,colors,markers):
                rows = summary[(summary['split']==split)&(summary['policy']==name)].sort_values('factor')
                values = rows[metric].astype(float)*(100 if metric=='dmr' else 1)
                ax.plot(rows['factor'],values, color=color,marker=marker,ms=3,lw=1,label=LABELS[name])
            ax.set(title=split.title(),xlabel='Synthetic workload multiplier',xticks=[1,1.5,2])
            ax.grid(axis='y',alpha=.15,lw=.5)
            if metric=='dmr':
                ax.axhline(5,color='#999999',lw=.8,ls='--')
                ax.set_ylim(bottom=0)
            else:
                ax.axhline(0,color='#555555',lw=.8)
        if metric!='dmr':
            values = summary[metric].dropna().astype(float)
            extent = max(1e-3, float(values.abs().max()))*1.15
            axes[0].set_ylim(-extent,extent)
        axes[0].set_ylabel(ylabel)
        handles,labels = axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,.055))
        caption(fig);fig.subplots_adjust(bottom=.31,wspace=.14)
        save(fig,stem)
    fig, axes = plt.subplots(1,3,figsize=(7.2,3.35))
    for ax,(split,factor) in zip(axes,(('train',2.),('validation',1.5),('validation',2.))):
        for name,color,marker in zip(GROUP_POLICIES,colors,markers):
            row = summary[(summary['split']==split)&(summary['factor']==factor)&(summary['policy']==name)].iloc[0]
            ax.scatter(row['dmr']*100,row['carbon_reduction_percent_vs_edf'],s=24,
                       color=color,marker=marker,label=LABELS[name])
        ax.set(title=f'{split.title()} {factor:g}x',xlabel='Deadline miss rate (%)')
        ax.set_xlim(left=0);ax.axhline(0,color='#555555',lw=.7);ax.grid(alpha=.15,lw=.5)
        extent=max(.001,summary[(summary['split']==split)&(summary['factor']==factor)]
                   ['carbon_reduction_percent_vs_edf'].abs().max())*1.15
        ax.set_ylim(-extent,extent)
    axes[0].set_ylabel('Carbon reduction vs paired EDF (%)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,.045))
    caption(fig);fig.subplots_adjust(bottom=.30,wspace=.35)
    save(fig,'high-pressure-pareto')
    hist = pd.DataFrame(tables['buffers'])
    combinations=[(split,factor,name) for split in ('train','validation') for factor in (1.,1.5,2.)
                  for name in POLICY_NAMES]
    matrix=[];labels=[]
    for split,factor,name in combinations:
        rows=hist[(hist['split']==split)&(hist['factor']==factor)&(hist['policy']==name)
                  &(hist['scope']=='first_eligible')].sort_values('buffer_steps')
        matrix.append(rows['fraction'].astype(float).to_numpy()*100)
        labels.append(f"{'T' if split=='train' else 'V'} {factor:g}x {SHORT[name]} (n={int(rows.iloc[0]['denominator'])})")
    fig,ax=plt.subplots(figsize=(7.2,8.5))
    image=ax.imshow(np.array(matrix),vmin=0,vmax=100,cmap='Blues',aspect='auto')
    ax.set(yticks=range(len(labels)),yticklabels=labels,xticks=range(11),
           xticklabels=range(11),xlabel='First-eligible buffer (15-minute steps)',
           title='Task-level buffer distribution; no budget encoded for early exits')
    colorbar=fig.colorbar(image,ax=ax,pad=.025,fraction=.04)
    colorbar.set_label('First-eligible tasks (%)')
    caption(fig);fig.subplots_adjust(left=.33,bottom=.07)
    save(fig,'first-eligible-buffer-distribution')
    resource=pd.DataFrame(tables['resources'])
    fig,axes=plt.subplots(1,5,figsize=(7.2,3.35),sharey=True)
    categories=('cpu','gpu','mem','tie','zero')
    resource_colors=('#4477aa','#cc9944','#66aa77','#9977aa','#888888')
    for ax,name in zip(axes,POLICY_NAMES):
        for category,color in zip(categories,resource_colors):
            values=[]
            for split,factor in (('train',1.),('train',1.5),('train',2.),
                                 ('validation',1.),('validation',1.5),('validation',2.)):
                row=resource[(resource['split']==split)&(resource['factor']==factor)
                    &(resource['policy']==name)&(resource['dominant_resource']==category)].iloc[0]
                values.append(row['mean_minutes'])
            ax.plot(range(6),values,color=color,marker='o',ms=2,lw=.8,label=category.upper())
        ax.set(title=SHORT[name],xticks=range(6),xticklabels=('T1','T1.5','T2','V1','V1.5','V2'))
        ax.tick_params(axis='x',labelrotation=45);ax.set_ylim(0,150)
        ax.grid(axis='y',alpha=.15,lw=.5)
    axes[0].set_ylabel('Mean first-eligible buffer (min)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=5,bbox_to_anchor=(.5,.06))
    caption(fig);fig.subplots_adjust(bottom=.32,wspace=.18)
    save(fig,'buffer-by-demand-category')


def analyze_campaign(campaign, output, plots=True):
    """Read-only six-cohort audit; new outputs only, with paired fairness guards."""
    from .contracts import content_hash, file_hash, write_json
    from .run import _fresh_run
    from .temporal import experiment_sources, write_csv
    campaign, output = Path(campaign), Path(output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite E1-6A analysis')
    frozen, groups = _checked_groups(campaign)
    sources = experiment_sources()
    def action(out):
        tables = defaultdict(list)
        for entry,folder,fairness,common in groups:
            group = _summarize_group(entry,folder,fairness,common)
            for key,values in group.items():
                tables[key].extend(values)
        for row,flag in zip(tables['summary'],pareto_flags(tables['summary'])):
            row['within_cohort_pareto_nondominated'] = flag
        artifacts = {'summary.csv':'summary','phase-comparison.csv':'phases',
            'budget-statistics.csv':'budgets','buffer-histogram.csv':'buffers',
            'budget-by-resource.csv':'resources','buffer-by-resource-histogram.csv':'resource_bins',
            'reason-counts.csv':'reasons','paired-miss-summary.csv':'paired',
            'representative-case-selection.csv':'selection','representative-tasks.csv':'cases',
            'figure-source-summary.csv':'summary','figure-source-buffer.csv':'buffers',
            'figure-source-resource.csv':'resources'}
        for filename,key in artifacts.items():
            if tables[key]:
                write_csv(out/filename,tables[key])
            else:
                (out/filename).write_text('no_cases\n',encoding='utf8')
        write_json(out/'figure-contract.json',FIGURE_CONTRACT)
        provenance=dict(version=VERSION, command=sys.argv, python=sys.version,
            campaign_sha256=file_hash(campaign/'campaign-freeze.json'),
            campaign_config_sha256=frozen['config_sha256'], experiment_source_sha256=frozen['source_sha256'],
            analysis_sources=sources, analysis_source_sha256=content_hash(sources),
            groups=[dict(split=entry['split'],factor=entry['factor'],
                scientific_manifest_sha256=file_hash(folder/'scientific-hashes.json'),
                runtime_sha256=file_hash(folder/'runtime.csv')) for entry,folder,_,_ in groups],
            formal_test_evaluated=False, new_training=False, validation_is_development=True,
            single_deterministic_run_per_cohort=True, rows=len(tables['summary']),
            representative_selection='first stable_id: Full/RiskOnly added miss, avoided miss, changed start/same outcome',
            attribution='Paired associations are not strict single-event causal effects.',
            budget_population='Only can-fit, high-CI, at-least-one-grid-step slack candidates; early exits have null B.',
            no_history='a_h=0; common cold=2 retained; numerical history removed, cold protection retained.')
        write_json(out/'analysis-provenance.json',provenance)
        captions = ('# E1-6A figure interpretation\n\n'
            'All figures use one deterministic frozen cohort per split and multiplier; no error bars or significance claims. '
            'Validation is development data, and Test remains sealed.\n\n'
            '1. Deadline miss rate: all eight strategies, with the preregistered 5% reference line.\n'
            '2. Carbon relative to paired EDF: derived 100*(EDF-carbon)/EDF-carbon; equal work and common idle-inclusive endpoint required.\n'
            '3. High-pressure Pareto: within-cohort carbon reduction against DMR; association, not optimality.\n'
            '4. Buffer heatmap: percentages among each first-eligible task population, n stated per row; omitted budgets are not zero.\n'
            '5. Demand category: normalized-demand dominant CPU/GPU/MEM, tie or zero; missing categories remain missing.\n\n'
            'Source CSV mappings: figures 1-3 use figure-source-summary.csv; figure 4 uses figure-source-buffer.csv; '
            'figure 5 uses figure-source-resource.csv. No task rows are selected away from cohort statistics. '
            'Representative cases separately follow the fixed first-stable-ID rule in all three outcome categories.\n')
        (out/'figure-captions.md').write_text(captions,encoding='utf8')
        if plots:
            _plots(out,tables)
        if experiment_sources() != sources:
            raise RuntimeError('Analysis source changed during run')
        hashes={path.relative_to(out).as_posix():file_hash(path) for path in sorted(out.rglob('*'))
                if path.is_file() and path.name not in ('analysis-hashes.json','run_status.json')}
        write_json(out/'analysis-hashes.json',hashes)
        return provenance
    return _fresh_run(output,action)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args(argv)
    result=analyze_campaign(args.campaign,args.output,plots=not args.no_plots)
    print(json.dumps(dict(version=VERSION,rows=result['rows'],formal_test_evaluated=False)))


if __name__=='__main__':
    main()
