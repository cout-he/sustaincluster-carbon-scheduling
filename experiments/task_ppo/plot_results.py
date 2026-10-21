"""Python-only figures from saved pilot CSVs; never execute a policy.

Figure contract: separate optimization behavior from paired service/carbon
outcomes, quantitative grid, all32 training episodes and all8 heldout conditions.
Single seed and4 correlated source windows: no confidence intervals or p-values.
183mm report figure; editable vectors and600dpi PNG, minimum7pt glyphs.
"""
import argparse
import csv
from pathlib import Path


def read(path):
    with path.open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))


def generate(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);training=read(root/'training-metrics.csv');evaluation=read(root/'evaluation-summary.csv')
    if len(training)!=32 or len(evaluation)!=40:
        raise ValueError('Do not present a partial/failed run as this complete pilot figure')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7,'axes.labelsize':7,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,
        'legend.fontsize':7,'svg.fonttype':'none','pdf.fonttype':42,
        'axes.spines.right':False,'axes.spines.top':False,'legend.frameon':False})
    fig,axes=plt.subplots(2,2,figsize=(7.205,5.04),layout='constrained')
    import json
    for cap,color,marker in [('OM','#436B8E','o'),('OH','#9C6D62','s')]:
        rows=[r for r in training if r['capacity']==cap];x=[int(r['episode']) for r in rows]
        axes[0,0].plot(x,[float(r['reward_sum']) for r in rows],marker=marker,color=color,lw=.8,ms=3,label=cap)
        rates=[]
        for row in rows:
            counts=json.loads(row['binary_action_counts']);rates.append(counts[1]/sum(counts))
        axes[0,1].plot(x,rates,marker=marker,color=color,lw=.8,ms=3,label=cap)
    axes[0,0].set(xlabel='Training episode',ylabel='Actual episode reward',title='a  Returns vary across workloads and capacities')
    axes[0,0].legend(loc='lower left')
    axes[0,1].set(xlabel='Training episode',ylabel='Requested defer / task decisions',ylim=(0,1),
                  title='b  Original binary proposals (before DDF)')
    fragments=sorted({r['fragment'] for r in evaluation})
    keys=[(f,c) for c in ['OM','OH'] for f in fragments];x=list(range(len(keys)))
    for method,color,marker,label in [('TaskPPO-final','#436B8E','o','Task-PPO final'),
                                    ('MacroPPO-history','#999999','s','Macro-PPO history')]:
        dc=[];dm=[]
        for fragment,cap in keys:
            rows={r['method']:r for r in evaluation if r['fragment']==fragment and r['capacity']==cap}
            dc.append(float(rows[method]['carbon_kg'])-float(rows['CarbonRule-DDF-CG']['carbon_kg']))
            dm.append(100*(float(rows[method]['dmr'])-float(rows['CarbonRule-DDF-CG']['dmr'])))
        axes[1,0].plot(x,dc,marker=marker,color=color,lw=.8,ms=3,label=label)
        axes[1,1].plot(x,dm,marker=marker,color=color,lw=.8,ms=3,label=label)
    labels=[f'{f[10:12]}/{f[12:14]}\n{c}' for f,c in keys]
    for axis in axes[1]:
        axis.axhline(0,color='#777777',lw=.7);axis.axvline(3.5,color='#BBBBBB',lw=.7,ls=':')
        axis.set_xticks(x,labels);axis.set_xlabel('Held-out Train window and capacity')
    axes[1,0].set(ylabel='Total carbon minus Rule+DDF (kg)',title='c  Paired carbon differences')
    axes[1,1].set(ylabel='DMR minus Rule+DDF (percentage points)',title='d  Paired Deadline outcomes')
    axes[1,0].legend(loc='best')
    svg=root/'task-ppo-pilot.svg'
    fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',
                   encoding='utf-8',newline='\n')
    fig.savefig(root/'task-ppo-pilot.pdf')
    fig.savefig(root/'task-ppo-pilot.png',dpi=600)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='docs/research/task-ppo-v1/results')
    generate(parser.parse_args().input)
