"""Descriptive E1-2B figures; all fixed factors, no single-seed significance claims."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from .contention import factor_folder, read_json
from .contracts import file_hash, write_json
from .policies import POLICY_NAMES

LABELS=('FIFO + Backfill','EDF + Backfill','Carbon-Threshold/FIFO')
COLORS=('#616b75','#426c9a','#be744a')
STYLES=('-','--','-')
MARKERS=('o','s','^')


def timeline_execution(task, observation_end):
    endpoint=pd.Timestamp(observation_end)
    predicted=pd.Timestamp(task['finish_time'])
    censored=task['state']!='completed'
    return (min(predicted,endpoint) if censored else predicted),censored


def plot_stress(experiment, output):
    experiment,output=Path(experiment),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
        'axes.titlesize':9,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,
        'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    all_rows=pd.read_csv(experiment/'comparison.csv')
    data=all_rows[all_rows.phase=='common'].copy()
    if data.duplicated(['factor','policy']).any() or set(data.policy)!=set(POLICY_NAMES):
        raise ValueError('Incomplete/duplicate policy plotting rows')
    factors=sorted(data.factor.unique())
    if len(data)!=3*len(factors):raise ValueError('Missing policy/factor row')
    def save(fig,name,frame):
        fig.tight_layout()
        fig.savefig(output/(name+'.pdf'))
        fig.savefig(output/(name+'.png'),dpi=600)
        frame.to_csv(output/(name+'.csv'),index=False)
        plt.close(fig)
    for name,metric,label in (
        ('01_carbon','carbon_kg','Common-window carbon emissions (kgCO2e)'),
        ('02_deadline','deadline_violation_rate_observed','Observed deadline miss rate (%)'),
        ('03_jct','mean_jct_minutes_completed','Mean JCT of completed tasks (min)')):
        fig,ax=plt.subplots(figsize=(180/25.4,75/25.4))
        for p,text,color,style,marker in zip(POLICY_NAMES,LABELS,COLORS,STYLES,MARKERS):
            d=data[data.policy==p].sort_values('factor')
            values=d[metric]*(100 if 'rate' in metric else 1)
            ax.plot(d.factor,values,label=text,color=color,linestyle=style,marker=marker,markersize=4)
        ax.set(xlabel='Synthetic arrival-count multiplier',ylabel=label,xticks=factors)
        ax.set_ylim(bottom=0)
        if metric=='deadline_violation_rate_observed':ax.set_ylim(top=max(1,data[metric].max()*115))
        ax.legend(loc='upper left',ncol=3,bbox_to_anchor=(0,1.2))
        ax.grid(axis='y',alpha=.2)
        note='Matched workload within each factor; one deterministic seed'
        if not data.equal_workload_carbon_comparison_valid.all():
            note+='; incomplete levels: carbon totals not equal-work benefits'
        fig.text(.5,.005,note,ha='center',fontsize=7)
        save(fig,name,data)
    selected=max(factors)
    folder=experiment/factor_folder(selected)
    series=pd.read_csv(folder/'timeseries.csv')
    series['time']=pd.to_datetime(series.interval_start,utc=True)
    design=read_json(folder/'design.json')
    arrival_end=pd.Timestamp(design['arrival_end'])
    fig,axes=plt.subplots(3,1,figsize=(180/25.4,135/25.4),sharex=True)
    ci=series[series.policy==POLICY_NAMES[0]]
    axes[0].plot(ci.time,ci.ci_g_per_kwh,color='#616b75',label='Current observed CI')
    axes[0].axhline(design['threshold_g_per_kwh'],color='#be744a',linestyle='--',label='Fixed threshold')
    axes[0].set_ylabel('CI (gCO2e/kWh)');axes[0].legend(ncol=2,loc='upper left',fontsize=7)
    for p,text,color,style in zip(POLICY_NAMES,LABELS,COLORS,STYLES):
        d=series[series.policy==p]
        axes[1].plot(d.time,d.running,color=color,linestyle=style,label=text)
        axes[2].plot(d.time,d.pending,color=color,linestyle=style)
    axes[1].set_ylabel('Running tasks');axes[2].set_ylabel('Pending tasks')
    axes[1].legend(ncol=3,loc='upper left',fontsize=7)
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.axvline(arrival_end,color='black',linestyle=':',linewidth=.9)
        ax.axvspan(arrival_end,series.time.max()+pd.Timedelta(minutes=15),color='#e8eaed',alpha=.5,zorder=0)
        ax.grid(axis='y',alpha=.2)
    axes[0].set_title(f'Highest preregistered load: {selected:g}x; shaded region is drain')
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4,maxticks=7))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M',tz='UTC'))
    axes[-1].set_xlabel('UTC')
    save(fig,'04_contention',series.drop(columns='time'))
    diagnostics=read_json(experiment/'diagnostics.json')
    choices=[(float(f),x['representative']) for f,x in diagnostics.items() if x.get('representative')]
    timeline=False
    if choices:
        factor,case=min(choices,key=lambda x:x[0])
        observation_end=read_json(experiment/factor_folder(factor)/'fairness.json')['common_end']
        origin=pd.Timestamp(case['tasks'][POLICY_NAMES[0]]['arrival_time'])
        minutes=lambda t:(pd.Timestamp(t)-origin).total_seconds()/60
        fig,ax=plt.subplots(figsize=(180/25.4,80/25.4))
        timeline_rows=[]
        for i,(p,color) in enumerate(zip(POLICY_NAMES,COLORS)):
            task=case['tasks'][p]
            observed_end,censored=timeline_execution(task,observation_end) if task['start_time'] else (None,task['state']!='completed')
            if task['start_time'] and task['finish_time']:
                start,end=minutes(task['start_time']),minutes(observed_end)
                ax.broken_barh([(start,end-start)],(i-.22,.44),facecolors=color)
                ax.plot([0,start],[i,i],color=color,linestyle=':',linewidth=1)
                if censored:ax.scatter(end,i,marker='>',color=color,s=24,label='Execution censored' if i==2 else None)
            for event in case['events_by_policy'][p]:
                if event['event'] in ('policy_deferred','capacity_waited'):
                    ax.scatter(minutes(event['time']),i+.30,marker='v' if event['event']=='policy_deferred' else 'x',
                               color=color,s=15)
            timeline_rows.append(dict(factor=factor,policy=p,task_id=case['task_id'],state=task['state'],observation_end=observation_end,
                observed_execution_end=observed_end.isoformat() if observed_end is not None else None,execution_censored=censored,**{k:task[k] for k in (
                'arrival_time','duration','cores_req','gpu_req','mem_req','sla_deadline','start_time','finish_time','deadline_outcome')}))
        ax.axvline(minutes(case['tasks'][POLICY_NAMES[0]]['sla_deadline']),color='black',linestyle='--',label='Synthetic deadline')
        ax.set(yticks=[0,1,2],yticklabels=LABELS,xlabel='Minutes since task arrival',title=f'Paired task trajectory at {factor:g}x synthetic load')
        ax.set_xlim(left=0);ax.set_ylim(-.6,2.9);ax.legend(loc='upper left',fontsize=7)
        fig.text(.5,.01,'Bars: execution; dotted line: waiting; triangle: policy defer; x: capacity wait. Association only.',ha='center',fontsize=7)
        save(fig,'05_task_timeline',pd.DataFrame(timeline_rows));timeline=True
    info=dict(summary_rows=len(data),time_series_rows=len(series),zero_y_axes=True,inferential_statistics=False,
        synthetic=True,seed_count=1,representative_factor=selected,task_timeline=timeline,
        source_comparison_sha256=file_hash(experiment/'comparison.csv'),
        outputs={p.name:file_hash(p) for p in sorted(output.iterdir())})
    write_json(output/'figure_manifest.json',info)
    return info


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();plot_stress(args.experiment,args.output)


if __name__=='__main__':main()
