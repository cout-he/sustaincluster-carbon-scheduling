"""Produce concise diagnostics and six scientific figures from frozen outputs."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from experiments.e1.contracts import file_hash,write_json

NAMES={'edf_backfill':'EDF','edf_carbon_threshold':'EDF+Carbon','cars_risk_only':'RiskOnly',
       'm0':'M0','m1':'M1 (IT)','m2':'M2 (Total)','m2_persistence':'M2 Persistence','m2_moderate':'M2 Moderate'}
COLORS={'edf_backfill':'#555555','edf_carbon_threshold':'#a5a5a5','cars_risk_only':'#bba8ce',
        'm0':'#81739e','m1':'#759fbd','m2':'#b77d75','m2_persistence':'#c3a989','m2_moderate':'#719c91'}
STYLE={'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,
       'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'legend.frameon':False,
       'axes.spines.right':False,'axes.spines.top':False,'pdf.fonttype':42,'svg.fonttype':'none',
       'savefig.facecolor':'white'}

def lines(path):return [json.loads(r) for r in path.read_text().splitlines()]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).parent)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();root=a.root;out=a.output
    out.mkdir(parents=True,exist_ok=False);tables=out/'tables';tables.mkdir();figures=out/'figures';figures.mkdir()
    comparisons=pd.concat([pd.read_csv(root/s/'comparison.csv') for s in ('train-002','validation-001')],ignore_index=True)
    comparisons.to_csv(tables/'comparison.csv',index=False,lineterminator='\n')
    forecast=pd.read_csv(root/'calibration-003/forecast-summary.csv');forecast.to_csv(tables/'forecast-errors.csv',index=False)
    pairs=pd.read_csv(root/'paired-analysis-002/registered-predictions.csv');pairs.to_csv(tables/'registered-predictions.csv',index=False)
    allprofile=[];gains=[];decisions=[];starts=[]
    for split,folder in (('train','train-002'),('validation','validation-001')):
        for factor in ('1','1.5','2'):
            base=root/folder/('load-'+factor)
            m1=lines(base/'m1/diagnostics.jsonl');m2=lines(base/'m2/diagnostics.jsonl')
            lookup={(r['time'],r['task_id']):r for r in m1}
            matched=0;changed=0;predicted=[]
            for r in m2:
                if r['status']=='ok':
                    predicted.append(r);gains.append(dict(split=split,factor=float(factor),gain_it=r['gain_it_kg'],gain_hvac=r['gain_hvac_kg'],gain_total=r['gain_total_kg']))
                old=lookup.get((r['time'],r['task_id']))
                if old is not None:matched+=1;changed+=old['defer']!=r['defer']
            decisions.append(dict(split=split,factor=float(factor),same_id_time_events=matched,decision_changed=changed,
                caveat='different policy trajectories; matched ID/time does not ensure identical background',
                m2_valid=len(predicted),it_total_sign_changes=sum((r['gain_it_kg']>0)!=(r['gain_total_kg']>0) for r in predicted),
                positive_gains=sum(r['gain_total_kg']>0 for r in predicted),negative_gains=sum(r['gain_total_kg']<0 for r in predicted),
                zero_gains=sum(r['gain_total_kg']==0 for r in predicted)))
            for policy in ('m1','m2','m2_persistence','m2_moderate'):
                rr=lines(base/policy/'profiling.jsonl')
                for r in rr:allprofile.append(dict(split=split,factor=float(factor),policy=policy,**r))
                diag=lines(base/policy/'diagnostics.jsonl');ledger={r['task_id']:r for r in lines(base/policy/'ledger.jsonl')}
                deferred={}
                for r in diag:
                    if r['defer']:deferred.setdefault(r['task_id'],r['time'])
                offsets=[(pd.Timestamp(ledger[k]['start_time'])-(pd.Timestamp(t)+pd.Timedelta(minutes=15))).total_seconds()/60 for k,t in deferred.items()]
                starts.append(dict(split=split,factor=float(factor),policy=policy,deferred_tasks=len(offsets),
                    actual_start_after_first_defer_plus15=sum(x>0 for x in offsets),
                    mean_extra_minutes=float(np.mean(offsets)) if offsets else 0.,max_extra_minutes=max(offsets,default=0),
                    caveat='later re-deferrals and capacity waits both contribute, not a pure prediction error'))
    gainframe=pd.DataFrame(gains)
    gain_summary=[]
    for key,g in gainframe.groupby(['split','factor']):
        row=dict(split=key[0],factor=key[1],samples=len(g))
        for col in ('gain_it','gain_hvac','gain_total'):
            for quantile in (.0,.25,.5,.75,.95,1.):row[col+'_p'+str(int(quantile*100))]=float(g[col].quantile(quantile))
        gain_summary.append(row)
    pd.DataFrame(gain_summary).to_csv(tables/'gain-distribution.csv',index=False)
    pd.DataFrame(decisions).to_csv(tables/'decision-changes.csv',index=False)
    pd.DataFrame(starts).to_csv(tables/'start-deviation.csv',index=False)
    profile=pd.DataFrame(allprofile);runtime=[]
    for key,g in profile.groupby(['split','factor','policy']):
        nonzero=g[g.elapsed_seconds>0];counts=nonzero.groupby('time').size()
        runtime.append(dict(split=key[0],factor=key[1],policy=key[2],calls=len(nonzero),
            mean_ms=float(nonzero.elapsed_seconds.mean()*1000),p95_ms=float(nonzero.elapsed_seconds.quantile(.95)*1000),
            max_ms=float(nonzero.elapsed_seconds.max()*1000),max_step_calls=int(counts.max()),
            max_step_estimator_seconds=float(nonzero.groupby('time').elapsed_seconds.sum().max())))
    runtime=pd.DataFrame(runtime);runtime.to_csv(tables/'runtime.csv',index=False)
    quality=[]
    for split,folder in (('train','train-002'),('validation','validation-001')):
        for factor in ('1','1.5','2'):
            for policy in NAMES:
                q=json.loads((root/folder/('load-'+factor)/policy/'surrogate-errors.json').read_text())
                quality.append(dict(split=split,factor=float(factor),policy=policy,it_mae_kw=q['it']['mae'],
                    it_rmse_kw=q['it']['rmse'],hvac_mae_kw=q['hvac']['mae'],hvac_rmse_kw=q['hvac']['rmse'],outside_fit_intervals=q['outside_fit']))
    pd.DataFrame(quality).to_csv(tables/'power-model-quality.csv',index=False)
    from experiments.e1.carbon_gain_experiment import CONFIG,load_registered_bundle
    from experiments.e1.online_ci_forecast import extract_prehistory,evaluate_forecasts
    from experiments.e1.config import ROOT
    cfg=json.loads(CONFIG.read_text());bundle=load_registered_bundle(cfg,'validation','1')
    prior=extract_prehistory(ROOT/'data/carbon_intensity/US-CAL-CISO/2023/US-CAL-CISO_2023_hourly.csv',bundle.workload.start,
        expected_sha256=bundle.environment.carbon_source_sha256)
    score=evaluate_forecasts([dict(time=s.time,ci=s.ci_g_per_kwh) for s in bundle.environment.samples if s.time<bundle.workload.end],prior)
    pd.DataFrame(score['summary']).to_csv(tables/'forecast-development-validation.csv',index=False)
    nonzero=pairs[pairs.actual_gain_total_kg!=0]
    pairstats=dict(n=12,nonzero_actual=len(nonzero),it_sign_agreement=int((np.sign(nonzero.gain_it_kg)==np.sign(nonzero.actual_gain_total_kg)).sum()),
        total_sign_agreement=int((np.sign(nonzero.gain_total_kg)==np.sign(nonzero.actual_gain_total_kg)).sum()),
        it_mae=float(pairs.it_error_kg.abs().mean()),total_mae=float(pairs.total_error_kg.abs().mean()),
        positive_total_but_actual_negative=int(((pairs.gain_total_kg>0)&(pairs.actual_gain_total_kg<0)).sum()),
        negative_total_but_actual_positive=int(((pairs.gain_total_kg<0)&(pairs.actual_gain_total_kg>0)).sum()),
        caveat='all 12 preselected examples; conditional full-cohort labels differ from ideal-target prediction scope; no general accuracy estimate')
    write_json(out/'paired-statistics.json',pairstats)
    nodes=[('Known state','Current tasks/resources/CI/weather'),('Forecast','Causal CI + weather persistence'),
        ('Gain','Train IT + HVAC; NOW vs +15min'),('Guard','RiskOnly buffer; no SLA guarantee'),
        ('PPO','0 immediate / 1 moderate / 2 positive'),('Executor','Original LocalExecutor + real IT/HVAC')]
    pd.DataFrame(nodes,columns=['node','description']).to_csv(tables/'architecture.csv',index=False)
    def save(fig,name):
        for fmt in ('png','pdf','svg'):fig.savefig(figures/(name+'.'+fmt),dpi=300,bbox_inches='tight')
        plt.close(fig)
    with plt.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(7.2,3));ax.set_xlim(0,3);ax.set_ylim(0,2);ax.axis('off')
        xy=[(.05,1.1),(1.05,1.1),(2.05,1.1),(.05,.1),(1.05,.1),(2.05,.1)]
        for (name,label),(x,y) in zip(nodes,xy):
            ax.add_patch(FancyBboxPatch((x,y),.9,.65,boxstyle='round,pad=0.015',fc='#edf1f4',ec='#7692a6'))
            ax.text(x+.45,y+.46,name,ha='center',weight='bold')
            ax.text(x+.45,y+.23,label.replace(' + ',' +\n').replace('; ','\n').replace(' / ','\n'),ha='center',va='center',fontsize=7)
        for start,end in ((0,1),(1,2),(2,3),(3,4),(4,5)):
            sx,sy=xy[start];ex,ey=xy[end]
            ax.annotate('',xy=(ex+.45,ey+.68) if start==2 else (ex,ey+.32),xytext=(sx+.45,sy) if start==2 else(sx+.9,sy+.32),arrowprops={'arrowstyle':'->','color':'#666666'})
        ax.set_title('Candidate V1: causal prediction controls eligibility; real physics supplies reward')
        save(fig,'figure1-architecture')
        fig,axes=plt.subplots(1,2,figsize=(7.2,2.5));f=forecast[forecast.factor==1]
        for method,color in (('persistence','#888888'),('seasonal','#759fbd')):
            g=f[f.method==method]
            for ax,metric in zip(axes,('mae','rmse')):ax.plot(g.horizon_steps/4,g[metric],'-o',label=method,color=color);ax.set_ylim(bottom=0);ax.set_xlabel('Forecast horizon (h)');ax.set_ylabel(metric.upper()+' (gCO2e/kWh)')
        axes[0].legend();axes[1].set_title('Same Train CI; loads are not replicates');fig.tight_layout();save(fig,'figure2-ci-errors')
        fig,axes=plt.subplots(1,2,figsize=(7.2,2.7));x=np.arange(len(pairs))
        axes[0].plot(x,pairs.gain_it_kg,'o-',label='Predicted IT',color='#759fbd')
        axes[0].plot(x,pairs.gain_total_kg,'s-',label='Predicted IT+HVAC',color='#b77d75')
        axes[1].scatter(pairs.gain_total_kg,pairs.actual_gain_total_kg,color='#b77d75',s=22)
        for ax in axes:ax.axhline(0,color='#777777',lw=.7);ax.set_ylabel('Carbon gain (kgCO2e)')
        axes[0].set_xlabel('All 12 preselected cases');axes[0].legend();axes[1].axvline(0,color='#777777',lw=.7)
        axes[1].set_xlabel('Predicted ideal target gain (kgCO2e)');axes[1].set_ylabel('Actual full-cohort gain (kgCO2e)')
        fig.tight_layout();save(fig,'figure3-gain-vs-total')
        fig,axes=plt.subplots(2,2,figsize=(7.2,5.5))
        for row,split in enumerate(('train','validation')):
            for name in NAMES:
                g=comparisons[(comparisons.split==split)&(comparisons.policy==name)].sort_values('factor')
                axes[row,0].plot(g.factor,g.carbon_kg/1000,'o-',ms=3,color=COLORS[name],label=NAMES[name])
                axes[row,1].plot(g.factor,g.deadline_violation_rate_observed*100,'o-',ms=3,color=COLORS[name])
            for ax in axes[row]:ax.set_ylim(bottom=0);ax.set_xlabel('Synthetic load multiplier');ax.set_xticks([1,1.5,2]);ax.set_title('Train' if split=='train' else 'Development Validation')
            axes[row,0].set_ylabel('Common-window carbon (tCO2e)');axes[row,1].set_ylabel('Deadline miss rate (%)')
            axes[row,1].axhline(5,ls=':',color='#777777',lw=.8)
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,-.02));fig.tight_layout(rect=(0,.08,1,1));save(fig,'figure4-fixed-policy-tradeoff')
        base=root/'train-002/load-1.5';a0={r['task_id']:r for r in lines(base/'m0/ledger.jsonl')};a2={r['task_id']:r for r in lines(base/'m2/ledger.jsonl')}
        different=[a0[k] for k in a0 if a0[k]['start_time']!=a2[k]['start_time']]
        first=sorted(different,key=lambda r:(r['arrival_time'],r['task_id']))[0];key=first['task_id'];other=a2[key]
        alias=hashlib.sha256(key.encode()).hexdigest()[:16];timeline=[]
        for name,r in (('M0',first),('M2',other)):
            timeline.append(dict(policy=name,task_alias=alias,arrival=r['arrival_time'],start=r['start_time'],finish=r['finish_time'],deadline=r['sla_deadline'],outcome=r['deadline_outcome'],duration=r['duration'],cpu=r['cores_req'],gpu=r['gpu_req'],mem=r['mem_req']))
        pd.DataFrame(timeline).to_csv(tables/'representative-task.csv',index=False)
        diag=[]
        for name in ('m0','m2'):
            for r in lines(base/name/'diagnostics.jsonl'):
                if r['task_id']==key:diag.append(dict(policy=name,time=r['time'],ci=r['ci_g_per_kwh'],defer=r['defer'],reason=r['reason'],gain_it=r['gain_it_kg'],gain_total=r['gain_total_kg'],buffer=r['buffer_steps']))
        pd.DataFrame(diag).to_csv(tables/'representative-decisions.csv',index=False)
        fig,ax=plt.subplots(figsize=(7.2,2.2));origin=pd.Timestamp(first['arrival_time'])
        for y,r in enumerate(timeline):
            start=(pd.Timestamp(r['start'])-origin).total_seconds()/60;finish=(pd.Timestamp(r['finish'])-origin).total_seconds()/60
            ax.broken_barh([(0,start)],(y-.2,.4),facecolors='#cccccc');ax.broken_barh([(start,finish-start)],(y-.2,.4),facecolors=COLORS[r['policy'].lower()])
        deadline=(pd.Timestamp(first['sla_deadline'])-origin).total_seconds()/60;ax.axvline(deadline,ls=':',color='#555555',label='Deadline')
        ax.set_yticks([0,1],['M0','M2']);ax.set_xlabel('Minutes since arrival');ax.set_xlim(left=0);ax.set_title('First differing task by arrival/ID (Train 1.5x); different policy trajectories');ax.legend();fig.tight_layout();save(fig,'figure5-task-timeline')
        fig,axes=plt.subplots(1,2,figsize=(7.2,2.7))
        for split,marker in (('train','o'),('validation','s')):
            for name in ('m1','m2','m2_persistence','m2_moderate'):
                g=runtime[(runtime.split==split)&(runtime.policy==name)].sort_values('factor')
                axes[0].plot(g.factor,g.mean_ms,marker+'-',ms=3,color=COLORS[name],label=NAMES[name]+' '+split)
                axes[1].plot(g.factor,g.max_step_calls,marker+'-',ms=3,color=COLORS[name])
        axes[0].set_ylabel('Mean estimator time (ms/candidate)');axes[1].set_ylabel('Maximum estimator calls / step')
        for ax in axes:ax.set_ylim(bottom=0);ax.set_xlabel('Synthetic load multiplier');ax.set_xticks([1,1.5,2])
        fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,bbox_to_anchor=(.5,-.18));fig.tight_layout();save(fig,'figure6-overhead')
    manifest={p.relative_to(out).as_posix():file_hash(p) for p in out.rglob('*') if p.is_file()}
    write_json(out/'output-hashes.json',manifest);write_json(out/'analysis-summary.json',dict(paired=pairstats,figures=6,comparison_rows=len(comparisons),test_access=False))
    print(json.dumps(dict(figures=6,paired=pairstats,rows=len(comparisons))))

if __name__=='__main__':main()
