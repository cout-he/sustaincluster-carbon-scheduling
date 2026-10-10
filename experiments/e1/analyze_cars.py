"""Descriptive CARS ablations and reproducible PNG/PDF figures; no significance tests."""
import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from .cars import CARS_POLICY_NAMES
from .contracts import file_hash, write_json
from .temporal import write_csv

LABELS=('FIFO','EDF','Carbon/FIFO','EDF+Carbon','KnownOnly','RiskOnly','Full')
COLORS=('#777777','#222222','#c48753','#789cb1','#8b83a9','#69a79e','#3b6393')
MARKERS=('o','s','^','v','D','P','X')
CASE_ID='3ba8a8e0067288f8/2023-07-01T10:15:00+00:00/17475/189~e1-2b-20261010-copy1'


def summarize_rows(factor,rows):
    common={r['policy']:r for r in rows if r['phase']=='common'}
    if set(common)!=set(CARS_POLICY_NAMES): raise ValueError('Seven common-window rows required')
    fifo=float(common['immediate_fifo']['carbon_kg']);edf=float(common['edf_backfill']['carbon_kg'])
    result=[]
    for policy in CARS_POLICY_NAMES:
        row=dict(common[policy]);n=int(row['arrived'])
        misses=int(row['deadline_missed_completed'])+int(row['deadline_missed_unfinished'])
        row.update(factor=factor,deadline_misses=misses,deadline_miss_rate=misses/n if n else None,
            carbon_reduction_vs_fifo_pct=(1-float(row['carbon_kg'])/fifo)*100,
            carbon_reduction_vs_edf_pct=(1-float(row['carbon_kg'])/edf)*100,
            equal_work_completed=int(row['completed'])==n)
        result.append(row)
    return result


def timeline_record(policy,row,endpoint):
    start=pd.Timestamp(row['start_time']) if row['start_time'] else None
    finish=pd.Timestamp(row['finish_time']) if row['finish_time'] else None
    complete=row['state']=='completed'
    return dict(policy=policy,task_id=row['task_id'],arrival_time=row['arrival_time'],
        sla_deadline=row['sla_deadline'],start_time=row['start_time'],predicted_finish=row['finish_time'],
        observed_execution_end=(finish if complete else min(finish,endpoint)).isoformat()
             if start is not None and finish is not None else None,
        observation_end=endpoint.isoformat(),state=row['state'],deadline_outcome=row['deadline_outcome'],
        censored=not complete)


def _jsonl(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def analyze(campaign,output):
    """Select no favorable samples: use every declared level and the prior E1-2B case."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    levels=sorted(Path(campaign).glob('load_*'))
    if not levels: raise ValueError('No load archives')
    summary=[];mechanisms={};cases=[];events=[];runtime=[]
    for folder in levels:
        factor=float(folder.name.removeprefix('load_').replace('p','.'))
        fairness=json.loads((folder/'fairness.json').read_text())
        with (folder/'comparison.csv').open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
        summary.extend(summarize_rows(factor,rows))
        missed={};reasons={}
        for policy in CARS_POLICY_NAMES:
            path=folder/policy
            tasks=_jsonl(path/'ledger.jsonl')
            ids=set()
            for row in tasks:
                if row['deadline_outcome'] in ('missed','missed_unfinished'):ids.add(row['task_id'])
                if factor==1.5 and row['task_id']==CASE_ID:
                    item=timeline_record(policy,row,pd.Timestamp(fairness['common_end']))
                    item.update(duration=row['duration'],cpu=row['cores_req'],gpu=row['gpu_req'],mem=row['mem_req'])
                    cases.append(item)
            missed[policy]=ids
            for event in _jsonl(path/'policy_events.jsonl'):
                if factor==1.5 and event['task_id']==CASE_ID:
                    events.append(dict(policy=policy,time=event['time'],event=event['event'],reason=event['reason'],
                                       ci_g_per_kwh=event['ci_g_per_kwh']))
            diag=path/'diagnostics.jsonl'
            if diag.exists():
                counts={};shared_checks=0;new_risk_checks=0;unknown_checks=0
                for d in _jsonl(diag):
                    counts[d['reason']]=counts.get(d['reason'],0)+1
                    p=d.get('planner')
                    if p:
                        shared_checks+=bool(set(p['missed_now']) & set(p['missed_defer']))
                        new_risk_checks+=p['status']=='risk';unknown_checks+=p['status']=='uncertain'
                reasons[policy]=dict(reason_counts=counts,shared_miss_checks=shared_checks,
                                    additional_miss_checks=new_risk_checks,uncertain_checks=unknown_checks)
        mechanisms[str(factor)]=dict(misses={p:len(v) for p,v in missed.items()},
            all_seven_missed_count=len(set.intersection(*missed.values())),
            full_missed_edf_carbon_met=len(missed['cars_full']-missed['edf_carbon_threshold']),
            edf_carbon_missed_full_met=len(missed['edf_carbon_threshold']-missed['cars_full']),
            known_missed_edf_carbon_met=len(missed['cars_known_only']-missed['edf_carbon_threshold']),
            edf_carbon_missed_known_met=len(missed['edf_carbon_threshold']-missed['cars_known_only']),
            reasons=reasons)
        runtime_path=folder/'runtime.json'
        if runtime_path.exists():runtime.append(dict(factor=factor,measurements=json.loads(runtime_path.read_text())))
    write_csv(output/'E1-3-ablation.csv',summary)
    write_json(output/'mechanisms.json',mechanisms)
    if cases:write_csv(output/'representative_task.csv',cases)
    if events:write_csv(output/'representative_events.csv',events)
    write_json(output/'runtime_summary.json',runtime)
    plot(summary,cases,events,output)
    write_json(output/'analysis_hashes.json',{p.name:file_hash(p) for p in output.iterdir() if p.is_file()})
    return summary


def plot(rows,cases,events,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
        'font.size':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
        'axes.spines.right':False,'legend.frameon':False,'axes.linewidth':.7})
    width=180/25.4
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=300)
        fig.savefig(out/(name+'.pdf'))
        plt.close(fig)
    # Editable programmatic source; diagram's branches are not quantitative observations.
    fig,ax=plt.subplots(figsize=(width,3.6));ax.set(xlim=(0,10),ylim=(0,10));ax.axis('off')
    nodes=[(5,9.3,'Public executor: candidate fits CPU/GPU/MEM?'),
           (5,7.8,'Current CI > fixed threshold AND one-step slack?'),
           (5,6.2,'Risk enabled: retain history buffer (0 / 1 / 2 steps)'),
           (5,4.6,'Known enabled: compare NOW vs next-step EDF plans'),
           (5,3.0,'Any extra missed ID or unresolved outcome?'),
           (5,1.3,'Defer one step; re-evaluate next boundary'),
           (9,.8,'Execute'),(.85,8.3,'Capacity\nwait')]
    for x,y,label in nodes:
        w=6.4 if x==5 else 1.65
        ax.add_patch(FancyBboxPatch((x-w/2,y-.37),w,.74,boxstyle='round,pad=.08',
                                  facecolor='#edf1f4',edgecolor='#6d7e8b',linewidth=.7))
        ax.text(x,y,label,ha='center',va='center',fontsize=7)
    for upper,lower in zip(nodes[:5],nodes[1:6]):
        ax.annotate('',xy=(5,lower[1]+.42),xytext=(5,upper[1]-.42),arrowprops=dict(arrowstyle='->',lw=.8))
    for y,label in [(7.8,'No'),(6.2,'Unsafe'),(3.0,'Yes')]:
        ax.annotate('',xy=(9,1.2),xytext=(8.3,y),arrowprops=dict(arrowstyle='->',lw=.7,
                     connectionstyle='angle,angleA=0,angleB=90,rad=3'))
        ax.text(8.4,y+.14,label,fontsize=7)
    ax.annotate('',xy=(.85,8.75),xytext=(1.72,9.3),arrowprops=dict(arrowstyle='->',lw=.8))
    ax.text(.85,9.35,'No',fontsize=7,ha='center')
    ax.text(5,.1,'Known check is bounded; uncertainty executes. No future arrivals or CI.',ha='center',fontsize=7)
    fig.subplots_adjust(left=.025,right=.975,bottom=.05,top=.99);save(fig,'fig1_cars_flow')
    write_csv(out/'fig1_flow_nodes.csv',[dict(x=x,y=y,label=label) for x,y,label in nodes])
    factors=sorted({float(r['factor']) for r in rows})
    for metric,label,name in [('deadline_miss_rate','Deadline miss rate (%)','fig2_deadline'),
                              ('carbon_kg','Common-window carbon (kgCO2e)','fig3_carbon')]:
        fig,ax=plt.subplots(figsize=(width,3.5))
        for p,l,c,m in zip(CARS_POLICY_NAMES,LABELS,COLORS,MARKERS):
            data=sorted((r for r in rows if r['policy']==p),key=lambda r:float(r['factor']))
            values=[float(r[metric])*(100 if metric=='deadline_miss_rate' else 1) for r in data]
            ax.plot([float(r['factor']) for r in data],values,color=c,marker=m,label=l,lw=1,ms=4)
        ax.set(xlabel='Synthetic arrival load multiplier',ylabel=label,xticks=factors,ylim=(0,None))
        ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,1.2),fontsize=7)
        ax.grid(axis='y',alpha=.16);fig.subplots_adjust(left=.12,right=.98,bottom=.15,top=.8)
        save(fig,name)
    fig,axes=plt.subplots(2,2,figsize=(width,5.6))
    for ax,factor in zip(axes.flat,factors):
        data={r['policy']:r for r in rows if float(r['factor'])==factor}
        for j,(p,l,c,m) in enumerate(zip(CARS_POLICY_NAMES,LABELS,COLORS,MARKERS)):
            r=data[p];ax.scatter(float(r['carbon_kg']),float(r['deadline_miss_rate'])*100,color=c,marker=m,s=22,label=l)
        ax.set(title=f'{factor:g}× load',xlabel='Carbon (kgCO2e)',ylabel='Miss rate (%)',ylim=(0,None))
        # Scatter's carbon axis focuses on the observed range; zero-origin DMR
        # and the separate zero-origin total-carbon plot preserve magnitude context.
        ax.ticklabel_format(axis='x',style='plain',useOffset=False)
        ax.grid(alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4,fontsize=7)
    fig.subplots_adjust(left=.10,right=.98,bottom=.10,top=.85,hspace=.55,wspace=.32)
    save(fig,'fig4_tradeoff')
    if cases:
        fig,ax=plt.subplots(figsize=(width,3.7));base=pd.Timestamp(cases[0]['arrival_time'])
        minutes=lambda t:(pd.Timestamp(t)-base).total_seconds()/60
        for j,(p,l,c) in enumerate(zip(CARS_POLICY_NAMES,LABELS,COLORS)):
            r=next(r for r in cases if r['policy']==p)
            start=minutes(r['start_time']) if r['start_time'] else None
            end=minutes(r['observed_execution_end']) if r['observed_execution_end'] else None
            if start is not None:
                ax.plot([0,start],[j,j],color=c,lw=1,ls=':')
                ax.plot([start,end],[j,j],color=c,lw=5,solid_capstyle='butt')
            for e in events:
                if e['policy']!=p:continue
                t=minutes(e['time'])
                if e['event']=='policy_deferred':ax.scatter(t,j,marker='o',s=18,facecolors='white',edgecolors=c,zorder=3)
                elif e['event']=='capacity_waited':ax.scatter(t,j,marker='^',s=22,color='#c48753',zorder=3)
                elif e['event']=='started':
                    ax.text(t,j+.22,e['reason'].replace('_',' '),fontsize=7,ha='left')
            if r['censored'] and end is not None:ax.scatter(end,j,marker='|',s=40,color='black')
        deadline=minutes(cases[0]['sla_deadline']);ax.axvline(deadline,color='#555555',ls='--',lw=1,label='Deadline')
        ax.set(yticks=range(7),yticklabels=LABELS,xlabel='Minutes since arrival (2023-07-01 10:15 UTC)',
               xlim=(-4,max(minutes(r['observed_execution_end']) for r in cases if r['observed_execution_end'])+8),ylim=(-.6,6.8))
        ax.invert_yaxis();ax.legend(loc='lower right',fontsize=7)
        fig.text(.20,.02,'Open circle: policy defer; triangle: capacity wait; bar: observed execution.',fontsize=7)
        fig.subplots_adjust(left=.20,right=.98,bottom=.17,top=.97);save(fig,'fig5_task_timeline')
    write_json(out/'figure_qa.json',dict(backend='python',width_mm=180,pdf_font_floor_pt=7,
        all_methods=7,all_declared_levels=factors,vertical_axes_start_zero=True,
        scatter_carbon_axis='observed horizontal range; total-carbon magnitude shown separately from zero',
        inference='single fixed window and seed; descriptive values, no CI or significance',
        timeline_clips_unfinished=True,source_data='E1-3-ablation.csv; representative_task.csv; representative_events.csv'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();rows=analyze(args.campaign,args.output)
    print(json.dumps(dict(comparisons=len(rows),output=str(args.output)),indent=2))


if __name__=='__main__':main()
