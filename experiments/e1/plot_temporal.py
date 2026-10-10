"""Four finite-cohort scientific plots, all intervals, common accounting endpoint."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .contracts import file_hash, write_json
from .policies import POLICY_NAMES
from .run import _fresh_run

STYLE={
    'immediate_fifo':('Immediate/FIFO','#566573','-'),
    'edf_backfill':('EDF with Backfilling','#0072B2','--'),
    'carbon_threshold_fifo':('Carbon-Threshold/FIFO','#D55E00','-'),
}


def series_xy(rows,key,end,terminal=None):
    x=rows['hours_from_start'].to_numpy();y=rows[key].to_numpy()
    if key=='cumulative_carbon_kg':
        return np.r_[x,end],np.r_[0,y]
    return np.r_[x,end],np.r_[y,y[-1] if terminal is None else terminal]


def coincidence_note(data):
    a,b=data['immediate_fifo'],data['edf_backfill']
    return 'FIFO and EDF coincide' if all(np.array_equal(x,y) for x,y in zip(a,b)) else ''


def make_figures(previous,output):
    previous=Path(previous)
    def action(out):
        if json.loads((previous/'run_status.json').read_text())['state']!='complete':
            raise ValueError('Figures require a completed experiment')
        prov=json.loads((previous/'provenance.json').read_text())
        for name in ('timeseries.csv','fairness.json'):
            if file_hash(previous/name)!=prov['output_hashes'][name]:
                raise ValueError('Figure source hash mismatch')
        frame=pd.read_csv(previous/'timeseries.csv')
        info=json.loads((previous/'fairness.json').read_text())
        time=pd.to_datetime(frame['interval_start'],utc=True);start=time.min()
        frame['hours_from_start']=(time-start).dt.total_seconds()/3600
        stop=(pd.Timestamp(info['arrival_end'])-start).total_seconds()/3600
        end=(pd.Timestamp(info['common_end'])-start).total_seconds()/3600
        plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
            'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
            'legend.fontsize':7,'pdf.fonttype':42,'svg.fonttype':'none',
            'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8})
        panels=(('01_carbon_intensity','ci_g_per_kwh','Carbon intensity (gCO2e/kWh)',False),
                ('02_running_tasks','running','Running tasks',False),
                ('03_pending_queue','pending','Pending tasks',False),
                ('04_cumulative_carbon','cumulative_carbon_kg','Cumulative carbon (kgCO2e)',True))
        for filename,key,ylabel,cumulative in panels:
            fig,ax=plt.subplots(figsize=(178/25.4,78/25.4))
            fig.subplots_adjust(left=.13,right=.98,bottom=.28,top=.85)
            ax.axvspan(stop,end,color='#EAECEE',alpha=.65,zorder=0)
            ax.axvline(stop,color='#555555',linewidth=.8,linestyle=':',label='Arrivals stop')
            ax.set_title(f'{start:%Y-%m-%d %H:%M} UTC; fixed arrivals then drain',fontsize=8,pad=12)
            ax.text(.98,1.02,'Drain / common observation',transform=ax.transAxes,ha='right',fontsize=7)
            data={}
            for policy in POLICY_NAMES:
                rows=frame[frame['policy']==policy]
                path=previous/policy/'phases.json'
                if file_hash(path)!=prov['output_hashes'][policy+'/phases.json']:
                    raise ValueError('Figure terminal-state source hash mismatch')
                terminal=json.loads(path.read_text())['common'].get(key)
                data[policy]=series_xy(rows,key,end,terminal if key in ('pending','running') else None)
            if key=='ci_g_per_kwh':
                x,y=data['immediate_fifo'];ax.step(x,y,where='post',color='#333333',linewidth=1.1,label='Current observed CI')
                ax.axhline(info['threshold_g_per_kwh'],color='#D55E00',linestyle='--',linewidth=1,
                           label=f"Historical threshold: {info['threshold_g_per_kwh']:.2f}")
                source=frame[frame['policy']=='immediate_fifo'][['interval_start','interval_end','hours_from_start',key]].copy()
                source['threshold_g_per_kwh']=info['threshold_g_per_kwh']
            else:
                for policy in POLICY_NAMES:
                    label,color,linestyle=STYLE[policy];x,y=data[policy]
                    ax.step(x,y,where='post',color=color,linestyle=linestyle,linewidth=1.1,label=label)
                note=coincidence_note(data)
                if note:
                    fig.text(.98,.02,note,ha='right',fontsize=7)
                source=frame[['policy','interval_start','interval_end','hours_from_start',key,'phase']].copy()
            source['plot_hours_from_start']=source['hours_from_start']+(.25 if cumulative else 0)
            source.to_csv(out/(filename+'.csv'),index=False)
            ax.set(xlim=(0,end),ylim=(0,None),xlabel='Hours from start (UTC)',ylabel=ylabel)
            ax.legend(loc='upper center',bbox_to_anchor=(.5,-.19),ncol=2,columnspacing=1.5)
            # All observations plotted: no thinning, no smoothing, no uncertainty from one run.
            fig.savefig(out/(filename+'.png'),dpi=600)
            fig.savefig(out/(filename+'.pdf'))
            plt.close(fig)
        write_json(out/'figure_contract.json',dict(backend='python',archetype='quantitative time-series',
            source=str(previous.resolve()),source_sha256=prov['source_sha256'],
            input_csv_sha256=file_hash(previous/'timeseries.csv'),all_rows_used=len(frame),excluded_rows=0,
            dimensions_mm=[178,78],font_min_pt=7,png_dpi=600,pdf_editable_text=True,
            uncertainty='Single deterministic finite cohort; no replicates, CI bands or significance tests',
            cumulative_plot='Interval-end cumulative sums; includes common idle tail',
            task_plot_endpoint='Completion-only common boundary; zero tasks when fully drained',
            captions={name:dict(y=label,source_data=name+'.csv') for name,_,label,_ in panels}))
    return _fresh_run(output,action)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();make_figures(args.experiment,args.output)


if __name__=='__main__': main()
