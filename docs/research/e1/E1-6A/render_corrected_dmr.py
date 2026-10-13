"""Presentation-only DMR axis correction; preserves sealed simulation outputs.

Run from either local notes or published docs; reads summary CSV, no simulator.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

POLICIES=('edf_backfill','edf_carbon_threshold','cars_risk_only','task_risk_full',
          'task_risk_no_history','task_risk_no_urgent','task_risk_no_utilization','task_risk_max_required')
LABELS=('EDF','EDF+Carbon','RiskOnly','Task risk Full','No history','No urgent queue',
        'No utilization','Max required resource')
COLORS=('#444444','#b17c45','#507d52','#285a91','#5a7fa6','#849dbc','#7d6195','#ad7898')
MARKERS=('o','s','^','D','v','P','X','h')

def dmr_limit(rows):
    values=[float(row['dmr'])*100 for row in rows]
    if not values or any(not math.isfinite(v) or v<0 or v>100 for v in values):
        raise ValueError('DMR must be finite and between zero and one')
    return max(5,max(values))*1.10

def build_figure(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':7,'axes.labelsize':7,'axes.titlesize':8,'xtick.labelsize':6,
        'ytick.labelsize':6,'legend.fontsize':6,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.7})
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.3),sharey=True)
    for ax,split in zip(axes,('train','validation')):
        for name,label,color,marker in zip(POLICIES,LABELS,COLORS,MARKERS):
            points=sorted((row for row in rows if row['split']==split and row['policy']==name),
                          key=lambda r:float(r['factor']))
            ax.plot([float(r['factor']) for r in points],[100*float(r['dmr']) for r in points],
                    color=color,marker=marker,ms=3,lw=1,label=label)
        ax.set(title=split.title(),xlabel='Synthetic workload multiplier',xticks=[1,1.5,2])
        ax.grid(axis='y',alpha=.15,lw=.5)
        ax.axhline(5,color='#999999',lw=.8,ls='--')
    # Set common limits only after BOTH panels have their complete data.
    axes[0].set_ylim(0,dmr_limit(rows))
    axes[0].set_ylabel('Deadline miss rate (%)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,.055))
    fig.text(.5,.01,'One deterministic frozen cohort per point; Validation is development data.',
             ha='center',fontsize=6)
    fig.subplots_adjust(bottom=.31,wspace=.14)
    for ax in axes:
        lower,upper=ax.get_ylim()
        for line in ax.lines:
            if any(not lower<=float(v)<=upper for v in line.get_ydata()):
                raise RuntimeError('A plotted point lies outside the visible axis')
    return fig,axes

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--companions',type=Path,help='Optional sealed original analysis figures/CSV')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Refusing to overwrite corrected figures')
    with args.summary.open(encoding='utf8',newline='') as stream:rows=list(csv.DictReader(stream))
    identity={(r['split'],float(r['factor']),r['policy']) for r in rows}
    expected={(s,f,p) for s in ('train','validation') for f in (1.,1.5,2.) for p in POLICIES}
    if identity!=expected or len(rows)!=48:raise ValueError('Exactly six matched eight-policy cohorts required')
    args.output.mkdir(parents=True)
    figure,axes=build_figure(rows)
    stem=args.output/'deadline-miss-rate'
    figure.savefig(stem.with_suffix('.png'),dpi=300,bbox_inches='tight')
    figure.savefig(stem.with_suffix('.pdf'),bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
    figure.savefig(stem.with_suffix('.svg'),bbox_inches='tight',metadata={'Date':None})
    if args.companions:
        for name in ('carbon-relative-edf','high-pressure-pareto','first-eligible-buffer-distribution','buffer-by-demand-category'):
            for suffix in ('.png','.pdf','.svg'):
                shutil.copyfile(args.companions/(name+suffix),args.output/(name+suffix))
        for name in ('figure-source-buffer.csv','figure-source-resource.csv'):
            shutil.copyfile(args.companions/name,args.output/name)
    shutil.copyfile(args.summary,args.output/'figure-source-summary.csv')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    receipt=dict(presentation_only=True,simulation_outputs_modified=False,
        source_summary_sha256=sha(args.summary),renderer_sha256=sha(Path(__file__)),
        rows=48,all_plotted_points_visible=True,y_limits=list(axes[0].get_ylim()),
        fix='Shared upper bound from all cohorts, set after plotting both panels.',
        outputs={p.name:sha(p) for p in sorted(args.output.iterdir()) if p.is_file()})
    (args.output/'correction-provenance.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print(json.dumps(receipt))

if __name__=='__main__':main()
