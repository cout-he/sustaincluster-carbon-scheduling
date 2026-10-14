"""Read-only, hash-checked publication figures for registered Train diagnostics.

Display selection is descriptive and does not change the registered cases.
No workload, forecasting, solver or online policy is imported here.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .contracts import file_hash

VERSION = 'e1-7a-figures-v1'
STYLE = {'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8,
         'axes.titlesize': 9, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
         'legend.fontsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
         'svg.fonttype': 'none', 'axes.spines.top': False,
         'axes.spines.right': False, 'savefig.dpi': 300}
COLORS = {'edf_backfill': '#0072B2', 'edf_carbon_threshold': '#D55E00',
          'cars_risk_only': '#009E73', 'force_execute': '#0072B2',
          'force_defer_once': '#D55E00', 'oracle': '#D55E00',
          'conditional_edf': '#0072B2'}


def choose_a_display(rows):
    valid = [r for r in rows if r['mode'] == 'A2' and r.get('factor') is not None
             and not pd.isna(r.get('factor'))]
    if not valid:
        raise ValueError('No real A2 case to display')
    return sorted(valid, key=lambda r: (-(r['edf_misses']-r['oracle_misses']), r['case_id']))[0]['case_id']


def choose_c_displays(rows):
    chosen, absent = [], []
    extra = sorted([r for r in rows if r['delta_misses'] > 0],
                   key=lambda r: (-r['delta_misses'], r['case_id']))
    neutral = sorted([r for r in rows if r['delta_misses'] <= 0], key=lambda r: r['case_id'])
    tradeoff = sorted([r for r in rows if r['delta_misses'] > 0 and r['delta_carbon'] < -1e-9],
                      key=lambda r: r['case_id'])
    for name, group in [('extra_miss', extra), ('nonincrease', neutral), ('carbon_sla_tradeoff', tradeoff)]:
        if not group:
            absent.append(name)
        elif group[0]['case_id'] not in chosen:
            chosen.append(group[0]['case_id'])
    return chosen, absent


def verify_campaign(path):
    path = Path(path).resolve()
    receipt = json.loads((path/'receipt.json').read_text(encoding='utf8'))
    if receipt.get('split') != 'train' or receipt.get('test_access') is not False:
        raise ValueError('Train-only receipt required before following any data path')
    status = json.loads((path/'run_status.json').read_text(encoding='utf8'))
    if receipt.get('passed') is not True or status.get('state', status.get('status')) != 'complete':
        raise ValueError('Campaign must be passed and complete')
    for name, digest in receipt['output_hashes'].items():
        target = (path/name).resolve()
        if not target.is_relative_to(path):
            raise ValueError('Output path escapes campaign')
        if file_hash(target) != digest:
            raise ValueError('Campaign output hash differs: '+name)
    return receipt


def _csv(path, rows):
    with Path(path).open('x', encoding='utf8', newline='') as stream:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        writer = csv.DictWriter(stream, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def _save(fig, output, name, caption, **extra):
    output = Path(output)
    paths = [output/(name+'.'+suffix) for suffix in ('png', 'pdf', 'svg')]
    if any(p.exists() for p in paths):
        raise FileExistsError('Figure output already exists')
    fig.align_ylabels()
    for path in paths:
        fig.savefig(path, bbox_inches='tight', facecolor='white')
    axes = [dict(xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim())) for ax in fig.axes]
    plt.close(fig)
    return dict(name=name, caption=caption, files={p.name: file_hash(p) for p in paths},
                source_csv=name+'.csv', axes=axes, **extra)


def _time_axes(axes):
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M', tz='UTC'))
        ax.grid(alpha=.18)


def cumulative_carbon_curve(frame):
    """A completed interval's integral belongs to its right endpoint."""
    return pd.to_datetime(frame.interval_end, utc=True), frame.carbon_kg.cumsum()


def _ledger(folder):
    return {r['task_id']: r for r in (json.loads(x) for x in
            (folder/'ledger.jsonl').read_text(encoding='utf8').splitlines())}


def plot_risk_scatter(records, output):
    output = Path(output)
    data = [dict(index=i+1, **r) for i,r in enumerate(sorted(records, key=lambda x:x['case_id']))]
    _csv(output/'C-risk-scatter.csv', data)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), layout='constrained')
        for ax, key, label in zip(axes, ['delta_misses', 'delta_carbon'],
                                 ['Extra deadline misses (defer − execute)', 'Carbon difference (kg CO₂e)']):
            for factor, color, marker in [(1.5, '#0072B2', 'o'), (2, '#D55E00', '^')]:
                group = [r for r in data if r['factor'] == factor]
                ax.scatter([r['score'] for r in group], [r[key] for r in group],
                           c=color, marker=marker, s=25, label=f'{factor:g}×')
                for r in group:
                    ax.annotate(str(r['index']), (r['score'], r[key]),
                                xytext=(3,6+13*((r['index']-1)%4)),
                                textcoords='offset points', fontsize=7)
            ax.axhline(0, color='0.45', linewidth=.7)
            ax.set_xlabel('Continuous frozen v1 risk score z')
            ax.set_ylabel(label)
            ax.set_xlim(left=0)
            ax.legend(frameon=False)
        return _save(fig, output, 'C-risk-scatter',
            'All registered single-defer pairs. Index maps to CSV; descriptive only, no fit or significance claim.', n=len(data))


def _plot_a(a_dir, case_id, output):
    folder = a_dir/case_id
    case = json.loads((folder/'case.json').read_text(encoding='utf8'))
    tasks = case['tasks']
    ids = [t['task_id'] for t in tasks]
    ledgers = {n: _ledger(folder/n) for n in ('conditional_edf', 'oracle')}
    anchor = pd.Timestamp(case['metadata']['anchor'])
    releases = [pd.Timestamp(ledgers[n][key]['completion_observed_at'])
                for n in ledgers for key in ids]
    lo = anchor-pd.Timedelta(minutes=30)
    hi = max(releases)+pd.Timedelta(minutes=30)
    source = []
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.8), sharex=True, layout='constrained')
        for name in ledgers:
            frame = pd.read_csv(folder/name/'intervals.csv')
            times = pd.to_datetime(frame.interval_start, utc=True)
            for ax,key,label,cap in zip(axes,['used_cores','used_gpus'], ['CPU (cores)', 'GPU (units)'],[50000,1000]):
                ax.step(times,frame[key],where='post',color=COLORS[name],label=name)
                ax.axhline(cap,color='0.4',linestyle=':',linewidth=.7)
                ax.set_ylim(0,cap*1.03)
                ax.set_ylabel(label)
            source.extend(dict(branch=name,**row) for row in frame.to_dict('records'))
        axes[0].legend(frameon=False,ncol=2)
        axes[0].set_title(case_id+' · fixed background + candidates')
        axes[-1].set_xlim(lo,hi)
        axes[-1].set_xlabel('UTC')
        _time_axes(axes)
        _csv(output/'A-resource-timeline.csv', source)
        resource = _save(fig,output,'A-resource-timeline',
            'Conditional A2 comparison; fixed background is included. Display selected by largest verified local miss reduction.', case_id=case_id)
        fig,ax=plt.subplots(figsize=(7.2,max(4, len(ids)*.18)),layout='constrained')
        data=[]
        for i,key in enumerate(ids):
            for name,offset in [('conditional_edf',-.16),('oracle',.16)]:
                row=ledgers[name][key]
                start,finish,release,deadline=(pd.Timestamp(row[k]) for k in
                    ['start_time','finish_time','completion_observed_at','sla_deadline'])
                y=i+offset
                ax.plot([start,finish],[y,y],color=COLORS[name],linewidth=2,
                        label=name if i==0 else None)
                ax.plot([finish,release],[y,y],color=COLORS[name],linewidth=1,linestyle=':')
                ax.plot(deadline,y,marker='|',color='0.2',markersize=5)
                data.append(dict(index=i+1,branch=name,task_id=key,start=start.isoformat(),
                                 finish=finish.isoformat(),release=release.isoformat(),deadline=deadline.isoformat(),
                                 deadline_outcome=row['deadline_outcome']))
        ax.set_yticks(range(len(ids)),[str(i+1) for i in range(len(ids))])
        ax.set_ylabel('Registered candidate index (ID in CSV)')
        ax.set_xlabel('UTC · solid: execution · dotted: release rounding · tick: deadline')
        ax.legend(frameon=False,ncol=2)
        _time_axes([ax])
        _csv(output/'A-task-timeline.csv',data)
        timeline=_save(fig,output,'A-task-timeline',
            'All selected A2 candidates; raw finish determines SLA, grid release determines occupancy. Minimum-miss objective has no JCT/carbon secondary objective.',case_id=case_id)
    return [resource,timeline]


def _plot_b(b_dir, output):
    energy=pd.read_csv(b_dir/'B-energy-decomposition.csv')
    ci=pd.read_csv(b_dir/'load-1/ci-quarter-hour.csv')
    for factor in ('1.5','2'):
        other=pd.read_csv(b_dir/('load-'+factor)/'ci-quarter-hour.csv')
        pd.testing.assert_frame_equal(ci,other,check_exact=True)
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
        labels=[]
        for i,row in enumerate(energy.to_dict('records')):
            labels.append(f"{row['factor']:g}×\n"+{'edf_backfill':'EDF','edf_carbon_threshold':'Carbon','cars_risk_only':'Risk'}[row['policy']])
            for ax,keys,scale in [(axes[0],['it_energy_kwh','hvac_energy_kwh'],1000),
                                  (axes[1],['it_carbon_kg','hvac_carbon_kg'],1000)]:
                bottom=0
                for key,color,label in zip(keys,['#0072B2','#E69F00'],['IT','HVAC']):
                    val=row[key]/scale
                    ax.bar(i,val,bottom=bottom,color=color,label=label if i==0 else None)
                    bottom+=val
        for ax in axes:
            ax.set_xticks(range(len(labels)),labels)
            ax.set_ylim(bottom=0)
            ax.legend(frameon=False)
        axes[0].set_ylabel('Energy (MWh)')
        axes[1].set_ylabel('Carbon (t CO₂e)')
        _csv(output/'B-physical-decomposition.csv',energy.to_dict('records'))
        decomposition=_save(fig,output,'B-physical-decomposition',
            'Actual persistent IT/HVAC physical replay; identical 57-hour public accounting window and all tasks completed. One deterministic load per level.')
        fig,axes=plt.subplots(1,2,figsize=(7.2,2.9),gridspec_kw={'width_ratios':[3,1]},layout='constrained')
        ax=axes[0]
        ax.step(pd.to_datetime(ci.time,utc=True),ci.ci_g_per_kwh,where='post',color='#0072B2',label='Observed Train CI')
        ax.axhline(161.075,color='#D55E00',linestyle='--',label='Fixed threshold 161.075')
        ax.set_ylabel('CI (g CO₂e/kWh)')
        ax.set_ylim(bottom=0)
        ax.set_xlabel('48-hour arrival window · UTC')
        ax.set_xlim(pd.to_datetime(ci.time.iloc[0],utc=True),pd.to_datetime(ci.time.iloc[-1],utc=True)+pd.Timedelta(minutes=15))
        ax.legend(frameon=False)
        _time_axes([ax])
        stats=json.loads((b_dir/'load-1/ci-statistics.json').read_text(encoding='utf8'))
        runs=[]
        for group,color in [('high','#D55E00'),('low','#0072B2')]:
            values=stats[group+'_run_lengths_minutes']
            positions=np.arange(len(runs),len(runs)+len(values))
            axes[1].bar(positions,values,color=color,label=group)
            runs.extend(dict(index=int(i),ci_group=group,run_minutes=value) for i,value in zip(positions,values))
        axes[1].set_xticks([])
        axes[1].set_ylabel('Continuous run (min)')
        axes[1].set_ylim(bottom=0)
        axes[1].legend(frameon=False)
        _csv(output/'B-ci-runs.csv',runs)
        _csv(output/'B-ci-time.csv',ci.to_dict('records'))
        carbon=_save(fig,output,'B-ci-time',
            'All 48 complete observed hours have four identical CI values; no synthetic forecasts or repeated tail.')
    return [decomposition,carbon]


def _plot_c(c_dir,case_id,output):
    folder=c_dir/case_id
    target=json.loads((folder/'target.json').read_text(encoding='utf8'))
    task_id=target['task_id']
    point=pd.Timestamp(target['time'])
    frames={n:pd.read_csv(folder/n/'intervals.csv') for n in ('force_execute','force_defer_once')}
    for frame in frames.values():
        frame['time']=pd.to_datetime(frame.interval_start,utc=True)
    ledgers={n:_ledger(folder/n)[task_id] for n in frames}
    lo=point-pd.Timedelta(minutes=60)
    hi=max(point+pd.Timedelta(minutes=360),max(pd.Timestamp(r['completion_observed_at']) for r in ledgers.values())+pd.Timedelta(minutes=60))
    hi=min(hi,pd.Timestamp(frames['force_execute'].iloc[-1].interval_end))
    name='C-'+case_id
    source=[]
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(5,1,figsize=(7.2,7.8),sharex=True,layout='constrained')
        for branch,frame in frames.items():
            color=COLORS[branch]
            if branch=='force_execute':
                axes[0].step(frame.time,frame.ci_g_per_kwh,where='post',color='0.3')
                axes[0].axhline(161.075,color='0.5',linestyle=':')
            axes[1].step(frame.time,frame.gpus_fraction,where='post',color=color,label=branch+' GPU')
            axes[1].step(frame.time,frame.cores_fraction,where='post',color=color,linestyle='--',label=branch+' CPU')
            axes[2].step(frame.time,frame.pending,where='post',color=color)
            row=ledgers[branch]
            y=0 if branch=='force_execute' else 1
            start,finish,release,deadline=(pd.Timestamp(row[k]) for k in
                  ['start_time','finish_time','completion_observed_at','sla_deadline'])
            axes[4].plot([start,finish],[y,y],color=color,linewidth=3)
            axes[4].plot([finish,release],[y,y],color=color,linewidth=1,linestyle=':')
            axes[4].plot(deadline,y,marker='|',color='black',markersize=8)
            source.extend(dict(branch=branch,**r) for r in frame.drop(columns='time').to_dict('records'))
        end_times,execute_cumulative=cumulative_carbon_curve(frames['force_execute'])
        _,defer_cumulative=cumulative_carbon_curve(frames['force_defer_once'])
        delta=defer_cumulative-execute_cumulative
        axes[3].step(end_times,delta,where='post',color='#CC79A7')
        axes[3].axhline(0,color='0.5',linewidth=.7)
        for ax,label in zip(axes,['CI (g/kWh)','Resource utilization','Pending (tasks)','Cumulative Δcarbon (kg)','Target task']):
            ax.set_ylabel(label)
            ax.axvline(point,color='0.5',linestyle='--',linewidth=.7)
        axes[0].set_ylim(bottom=0)
        axes[1].set_ylim(0,1.03)
        axes[2].set_ylim(bottom=0)
        axes[4].set_yticks([0,1],['Execute','Defer once'])
        axes[4].set_ylim(-.5,1.5)
        axes[0].set_title(case_id+' · one action intervention, then EDF')
        axes[1].legend(frameon=False,ncol=2)
        axes[-1].set_xlabel('UTC · vertical dashed: intervention · task tick: deadline')
        axes[-1].set_xlim(lo,hi)
        _time_axes(axes)
        _csv(output/(name+'.csv'),source)
        return _save(fig,output,name,
            'Same physical and scheduler prefix; only specified action differs, followed by EDF. Full common-window data retained in CSV; this crop is for display.',case_id=case_id,
            target_ledger=ledgers,final_delta_carbon_kg=float(delta.iloc[-1]))


def plot_campaigns(a_dir,b_dir,c_dir,output_dir,selection_manifest=None):
    paths=[Path(p).resolve() for p in (a_dir,b_dir,c_dir)]
    receipts=[verify_campaign(p) for p in paths]
    identities={r['selection_sha256'] for r in receipts}
    if len(identities)!=1:
        raise ValueError('Campaign selection identities differ')
    if selection_manifest is not None:
        from .diagnostic_experiment import load_selection_manifest
        from .contracts import content_hash
        if content_hash(load_selection_manifest(selection_manifest)) not in identities:
            raise ValueError('Figure selection identity differs')
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=False)
    a_rows=pd.read_csv(paths[0]/'A-edf-vs-oracle.csv').to_dict('records')
    c_rows=pd.read_csv(paths[2]/'C-paired-outcomes.csv').to_dict('records')
    risk=pd.read_csv(paths[2]/'C-risk-vs-outcome.csv').to_dict('records')
    a_id=choose_a_display(a_rows)
    c_ids,absent=choose_c_displays(c_rows)
    figures=_plot_a(paths[0],a_id,output)+_plot_b(paths[1],output)
    figures.extend(_plot_c(paths[2],key,output) for key in c_ids)
    figures.append(plot_risk_scatter(risk,output))
    source_hashes={str(path/'receipt.json'):file_hash(path/'receipt.json') for path in paths}
    manifest=dict(schema=VERSION,split='train',test_access=False,
        selection_sha256=next(iter(identities)),sources=source_hashes,figures=figures,
        selected_a=a_id,selected_c=c_ids,absent_c_classes=absent,style=STYLE,
        display_selection_only=True,source_sha256=file_hash(Path(__file__)),
        files={p.name:file_hash(p) for p in sorted(output.iterdir())})
    with (output/'figures-manifest.json').open('x',encoding='utf8') as stream:
        json.dump(manifest,stream,indent=2,allow_nan=False)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('a','b','c','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--manifest')
    args=parser.parse_args()
    result=plot_campaigns(args.a,args.b,args.c,args.output,args.manifest)
    print(json.dumps(dict(figures=len(result['figures']),selected_a=result['selected_a'],
                         selected_c=result['selected_c'],absent_c_classes=result['absent_c_classes'])))


if __name__=='__main__':
    main()
