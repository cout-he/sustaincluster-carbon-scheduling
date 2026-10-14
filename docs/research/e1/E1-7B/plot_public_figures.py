"""Redraw all six CGD V1 figures from public small CSV, without private inputs."""
import argparse,json
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

NAMES={'edf_backfill':'EDF','edf_carbon_threshold':'EDF+Carbon','cars_risk_only':'RiskOnly','m0':'M0',
       'm1':'M1 (IT)','m2':'M2 (Total)','m2_persistence':'M2 Persistence','m2_moderate':'M2 Moderate'}
COLORS=dict(zip(NAMES,('#555555','#a5a5a5','#bba8ce','#81739e','#759fbd','#b77d75','#c3a989','#719c91')))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tables',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();t=a.tables;o=a.output;o.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':8,'axes.labelsize':8,'axes.titlesize':9,
        'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'legend.frameon':False,
        'axes.spines.right':False,'axes.spines.top':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    def save(fig,name):
        fig.savefig(o/(name+'.png'),dpi=300,bbox_inches='tight',facecolor='white')
        fig.savefig(o/(name+'.pdf'),bbox_inches='tight',facecolor='white')
        fig.savefig(o/(name+'.svg'),bbox_inches='tight',facecolor='white')
        plt.close(fig)
    nodes=pd.read_csv(t/'architecture.csv');fig,ax=plt.subplots(figsize=(7.2,3));ax.set(xlim=(0,3),ylim=(0,2));ax.axis('off')
    xy=[(.05,1.1),(1.05,1.1),(2.05,1.1),(.05,.1),(1.05,.1),(2.05,.1)]
    for row,(x,y) in zip(nodes.itertuples(),xy):
        label=row.description.replace('tasks/resources/','tasks/resources\n').replace(' + ',' +\n').replace('; ','\n').replace(' / ','\n')
        ax.add_patch(FancyBboxPatch((x,y),.9,.65,boxstyle='round,pad=.015',fc='#edf1f4',ec='#7692a6'))
        ax.text(x+.45,y+.46,row.node,ha='center',weight='bold');ax.text(x+.45,y+.23,label,ha='center',va='center',fontsize=7)
    for start,end in ((0,1),(1,2),(2,3),(3,4),(4,5)):
        sx,sy=xy[start];ex,ey=xy[end]
        ax.annotate('',xy=(ex+.45,ey+.68) if start==2 else(ex,ey+.32),xytext=(sx+.45,sy) if start==2 else(sx+.9,sy+.32),arrowprops={'arrowstyle':'->','color':'#666666'})
    ax.set_title('Candidate V1: causal prediction controls eligibility; real physics supplies reward');save(fig,'figure1-architecture')
    f=pd.read_csv(t/'forecast-errors.csv');f=f[f.factor==1];fig,axes=plt.subplots(1,2,figsize=(7.2,2.5))
    for method,color in (('persistence','#888888'),('seasonal','#759fbd')):
        g=f[f.method==method]
        for ax,metric in zip(axes,('mae','rmse')):
            ax.plot(g.horizon_steps/4,g[metric],'-o',label=method,color=color);ax.set_ylim(bottom=0)
            ax.set_xlabel('Forecast horizon (h)');ax.set_ylabel(metric.upper()+' (gCO2e/kWh)')
    axes[0].legend();axes[1].set_title('Same Train CI; loads are not replicates');fig.tight_layout();save(fig,'figure2-ci-errors')
    p=pd.read_csv(t/'registered-predictions.csv');x=np.arange(len(p));fig,axes=plt.subplots(1,2,figsize=(7.2,2.9))
    axes[0].plot(x,p.gain_it_kg,'o-',label='Predicted IT',color=COLORS['m1'])
    axes[0].plot(x,p.gain_total_kg,'s-',label='Predicted IT+HVAC',color=COLORS['m2'])
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,1.19),ncol=2)
    axes[0].set_xlabel('All 12 preselected cases');axes[0].set_ylabel('Carbon gain (kgCO2e)')
    axes[1].scatter(p.gain_total_kg,p.actual_gain_total_kg,s=22,color=COLORS['m2'])
    axes[1].set_xlabel('Predicted ideal target gain (kgCO2e)');axes[1].set_ylabel('Actual full-cohort gain (kgCO2e)')
    for ax in axes:ax.axhline(0,color='#777777',lw=.7)
    axes[1].axvline(0,color='#777777',lw=.7);fig.tight_layout();save(fig,'figure3-gain-vs-total')
    c=pd.read_csv(t/'comparison.csv');fig,axes=plt.subplots(2,2,figsize=(7.2,5.5))
    for i,split in enumerate(('train','validation')):
        for name in NAMES:
            g=c[(c.split==split)&(c.policy==name)].sort_values('factor');ls=':' if name=='m0' else '-'
            axes[i,0].plot(g.factor,g.carbon_kg/1000,ls+'o',ms=3,color=COLORS[name],label=NAMES[name])
            axes[i,1].plot(g.factor,g.deadline_violation_rate_observed*100,ls+'o',ms=3,color=COLORS[name])
        for ax in axes[i]:ax.set_ylim(bottom=0);ax.set_xlabel('Synthetic load multiplier');ax.set_xticks([1,1.5,2]);ax.set_title('Train' if i==0 else 'Development Validation')
        axes[i,0].set_ylabel('Common-window carbon (tCO2e)');axes[i,1].set_ylabel('Deadline miss rate (%)');axes[i,1].axhline(5,ls=':',color='#777777',lw=.8)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=4,bbox_to_anchor=(.5,-.02))
    fig.tight_layout(rect=(0,.08,1,1));save(fig,'figure4-fixed-policy-tradeoff')
    timeline=pd.read_csv(t/'representative-task.csv');origin=pd.Timestamp(timeline.iloc[0].arrival)
    fig,ax=plt.subplots(figsize=(7.2,2.2))
    for y,r in enumerate(timeline.itertuples()):
        start=(pd.Timestamp(r.start)-origin).total_seconds()/60;finish=(pd.Timestamp(r.finish)-origin).total_seconds()/60
        ax.broken_barh([(0,start)],(y-.2,.4),facecolors='#cccccc');ax.broken_barh([(start,finish-start)],(y-.2,.4),facecolors=COLORS[r.policy.lower()])
    deadline=(pd.Timestamp(timeline.iloc[0].deadline)-origin).total_seconds()/60;ax.axvline(deadline,ls=':',color='#555555',label='Deadline')
    ax.set_yticks([0,1],['M0','M2']);ax.set_xlabel('Minutes since arrival');ax.set_xlim(left=0)
    ax.set_title('First differing task by arrival/ID (Train 1.5x); different policy trajectories');ax.legend();fig.tight_layout();save(fig,'figure5-task-timeline')
    r=pd.read_csv(t/'runtime.csv');fig,axes=plt.subplots(1,2,figsize=(7.2,2.7))
    for split,marker in (('train','o'),('validation','s')):
        for name in ('m1','m2','m2_persistence','m2_moderate'):
            g=r[(r.split==split)&(r.policy==name)].sort_values('factor')
            axes[0].plot(g.factor,g.mean_ms,marker+'-',ms=3,color=COLORS[name],label=NAMES[name]+' '+split)
            axes[1].plot(g.factor,g.max_step_calls,marker+'-',ms=3,color=COLORS[name])
    axes[0].set_ylabel('Mean estimator time (ms/candidate)');axes[1].set_ylabel('Maximum estimator calls / step')
    for ax in axes:ax.set_ylim(bottom=0);ax.set_xlabel('Synthetic load multiplier');ax.set_xticks([1,1.5,2])
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,bbox_to_anchor=(.5,-.18));fig.tight_layout();save(fig,'figure6-overhead')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    document=dict(script_sha256=digest(Path(__file__)),source_csv_sha256={p.name:digest(p) for p in t.glob('*.csv')},
        outputs={p.name:digest(p) for p in o.iterdir() if p.is_file()},backend='python',dpi=300,font_floor_pt=7,
        data='all registered observations; identical CI copies plotted once, M0 equals RiskOnly',
        statistics='one deterministic run per scenario; no independent seeds/CI replicates or significance inference')
    (o/'figure-manifest.json').write_text(json.dumps(document,indent=2)+'\n')

if __name__=='__main__':main()
