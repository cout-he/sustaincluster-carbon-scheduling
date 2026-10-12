"""Derive auditable PPO-v0 Train/Validation diagnostics; never access Test data."""
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

for ancestor in Path(__file__).resolve().parents:
    if (ancestor/'experiments/e1/rl_env.py').is_file():
        sys.path.insert(0,str(ancestor))
        break
else:
    raise RuntimeError('Place the analysis driver inside the research/publication repository')
from experiments.e1.rl_env import OBSERVATION_FIELDS

SEEDS=(55101,55102,55103)
COLORS=('#4477AA','#EE6677','#228833')
ACTION_COLORS=('#999999','#66CCEE','#AA3377')
ACTION_NAMES=('EDF','EDF+Carbon','RiskOnly')
BASELINE_NAMES={'edf_backfill':'EDF','edf_carbon_threshold':'EDF+Carbon','cars_risk_only':'RiskOnly'}
COMPONENTS=('carbon_cost','wait_cost','deadline_cost','terminal_cost')
EXPECTED_BOUNDARIES={'ci_lags_valid_fraction','history_effective_fraction','history_cold',
    'risk_buffer_fraction','arrivals_open','utc_day_sin','utc_day_cos',
    'utc_weekday_sin','utc_weekday_cos','cpu_utilization','gpu_utilization','mem_utilization'}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,payload):
    Path(path).write_text(json.dumps(payload,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def csv(path,rows):pd.DataFrame(rows).to_csv(path,index=False)


def source_rows(folder):
    with gzip.open(folder/'steps.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:yield json.loads(line)


def conditions(ci,rho,cold,urgent,pending,threshold):
    return [('all','all'),('ci','high' if ci>threshold else 'low_or_equal'),
        ('pressure','cold' if cold else ('<0.5' if rho<.5 else '0.5-1' if rho<1 else '>=1')),
        ('urgency','urgent' if urgent else 'none'),('pending','nonempty' if pending else 'empty')]


def conditional_rows(rows,seed,phase,threshold):
    counts=defaultdict(Counter)
    for row in rows:
        if phase=='train':
            state=row['online_state'];rhos=[r for r in state['history_rho_resources'] if r is not None]
            rho=max(rhos) if rhos else 0.
            ci=state['ci_g_per_kwh'];cold=state['history_cold'];urgent=state['urgent_tasks'];pending=state['pending_tasks']
        else:
            rho=row['history_rho']
            rho=0. if pd.isna(rho) else float(rho)
            ci=row['ci_g_per_kwh'];cold=bool(row['history_cold']);urgent=row['urgent_tasks'];pending=row['pending_tasks']
        for scope in ('all_steps','pending_nonempty'):
            if scope=='pending_nonempty' and not pending:continue
            for axis,level in conditions(ci,rho,cold,urgent,pending,threshold):
                counts[(scope,axis,level)][int(row['action'])]+=1
    result=[]
    for scope in ('all_steps','pending_nonempty'):
        for axis,levels in (('all',('all',)),('ci',('low_or_equal','high')),
            ('pressure',('cold','<0.5','0.5-1','>=1')),('urgency',('none','urgent')),
            ('pending',('empty','nonempty'))):
            for level in levels:counts[(scope,axis,level)]
    for (scope,axis,level),values in sorted(counts.items()):
        n=sum(values.values())
        for action in range(3):
            result.append(dict(seed=seed,phase=phase,scope=scope,condition=axis,level=level,
                action=action,action_name=ACTION_NAMES[action],count=values[action],denominator=n,
                fraction=values[action]/n if n else None))
    return result


def feature_rows(observations,seed,phase):
    values=np.asarray(observations,dtype=float)
    if values.ndim!=2 or values.shape[1]!=40:raise ValueError('Unexpected actor observation schema')
    if not np.isfinite(values).all():raise ValueError('Nonfinite actor observations')
    result=[]
    for i,name in enumerate(OBSERVATION_FIELDS):
        vector=values[:,i]
        expected=name in EXPECTED_BOUNDARIES
        result.append(dict(seed=seed,phase=phase,feature=name,n=len(vector),
            minimum=float(vector.min()),maximum=float(vector.max()),mean=float(vector.mean()),
            std=float(vector.std()),nonfinite_count=0,abs_ge_099_fraction=float((np.abs(vector)>=.99).mean()),
            boundary_interpretation='expected flag/phase/cyclic or physical-utilization boundary' if expected else 'continuous normalized feature; inspect high fraction',
            constant=bool(np.ptp(vector)==0)))
    return result


def correlation_rows(observations,seed,phase):
    """Descriptive redundancy warning only; never fit or remove actor features."""
    values=np.asarray(observations,dtype=float)
    correlation=np.corrcoef(values,rowvar=False)
    result=[]
    for first in range(40):
        for second in range(first+1,40):
            value=float(correlation[first,second])
            if math.isfinite(value) and abs(value)>=.995:
                result.append(dict(seed=seed,phase=phase,feature_1=OBSERVATION_FIELDS[first],
                    feature_2=OBSERVATION_FIELDS[second],pearson_r=value,
                    absolute_r=abs(value),n=len(values),criterion='upper triangle; |Pearson r| >= 0.995',
                    interpretation='descriptive temporal association; no feature removal, causal or significance claim'))
    return result


def export(fig,destination):
    fig.savefig(str(destination)+'.png',dpi=300,bbox_inches='tight',facecolor='white')
    fig.savefig(str(destination)+'.pdf',bbox_inches='tight',facecolor='white')
    fig.savefig(str(destination)+'.svg',bbox_inches='tight',facecolor='white')
    plt.close(fig)


def training_figure(episodes,out):
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.45),layout='constrained')
    for seed,color in zip(SEEDS,COLORS):
        frame=episodes[episodes.seed==seed]
        axes[0].plot(frame.ending_step,frame.episode_return,color=color,label=str(seed),linewidth=.9)
        axes[1].plot(frame.ending_step,frame.carbon_cost,color=color,linewidth=.9)
        axes[2].plot(frame.ending_step,frame.new_deadline_misses,color=color,linewidth=.9)
    axes[0].set(ylabel='Complete episode return',ylim=(min(episodes.episode_return)*1.06,0))
    axes[1].set(ylabel='Carbon reward cost',ylim=(0,max(episodes.carbon_cost)*1.08))
    axes[2].set(ylabel='First deadline misses',ylim=(0,max(episodes.new_deadline_misses)*1.08))
    for letter,ax in zip('abc',axes):
        ax.set_xlabel('Environment steps');ax.set_xlim(0,20480)
        ax.text(-.12,1.02,letter,transform=ax.transAxes,weight='bold',fontsize=9)
    axes[0].legend(title='RL seed',fontsize=6.5,title_fontsize=7)
    export(fig,out/'training-curves')


def reward_figure(episodes,out):
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.35),layout='constrained')
    names=('Carbon','Wait','Deadline','Terminal')
    for ax,seed in zip(axes,SEEDS):
        data=episodes[episodes.seed==seed]
        for key,label,color in zip(COMPONENTS,names,('#4477AA','#CCBB44','#AA3377','#BBBBBB')):
            ax.plot(data.ending_step,data[key],label=label,color=color,linewidth=.9)
        ax.set(title=str(seed),xlabel='Environment steps',ylabel='Weighted reward cost',ylim=(0,None),xlim=(0,20480))
    axes[-1].legend(fontsize=6.5)
    export(fig,out/'reward-components')


def action_figure(actions,out):
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.45),layout='constrained',sharey=True)
    for ax,phase in zip(axes,('train','validation')):
        bottom=np.zeros(3)
        for action,name,color in zip(range(3),ACTION_NAMES,ACTION_COLORS):
            values=[float(actions[(actions.phase==phase)&(actions.seed==seed)&(actions.action==action)].fraction.iloc[0]) for seed in SEEDS]
            ax.bar(range(3),values,bottom=bottom,color=color,label=name,width=.6);bottom+=values
        ax.set(title='Training sampling' if phase=='train' else 'Validation deterministic',
               xticks=range(3),xticklabels=[str(s) for s in SEEDS],xlabel='RL seed',ylabel='Action fraction',ylim=(0,1))
    axes[-1].legend(loc='upper center',bbox_to_anchor=(.5,1.28),ncol=3,fontsize=6.5)
    export(fig,out/'action-distribution')


def conditional_figure(frame,out):
    fig,axes=plt.subplots(3,3,figsize=(7.1,5.3),layout='constrained',sharey=True)
    levels=(('ci',('low_or_equal','high'),('Low','High'),'CI'),
            ('pressure',('cold','<0.5','0.5-1','>=1'),('Cold','<0.5','0.5–1','≥1'),'Pressure'),
            ('urgency',('none','urgent'),('None','Urgent'),'Urgency'))
    for row,seed in enumerate(SEEDS):
        for col,(axis,possible,short,title) in enumerate(levels):
            ax=axes[row,col]
            subset=frame[(frame.phase=='validation')&(frame.scope=='pending_nonempty')&
                         (frame.condition==axis)&(frame.seed==seed)]
            xs=[];labels=[];bottom=[];values={a:[] for a in range(3)}
            for level,label in zip(possible,short):
                data=subset[subset.level==level]
                n=int(data.denominator.iloc[0]) if not data.empty else 0
                xs.append(len(xs));labels.append(label+'\nn='+str(n));bottom.append(0.)
                for a in range(3):values[a].append(float(data[data.action==a].fraction.iloc[0]) if n else np.nan)
                if not n:ax.text(xs[-1],.48,'No states',ha='center',fontsize=6.5)
            bottom=np.asarray(bottom,dtype=float)
            for a,color in zip(range(3),ACTION_COLORS):
                ax.bar(xs,values[a],bottom=bottom,color=color,width=.6,label=ACTION_NAMES[a]);bottom+=values[a]
            ax.set(xticks=xs,xticklabels=labels,ylim=(0,1))
            if row==0:ax.set_title(title)
            if col==0:ax.set_ylabel(str(seed)+'\nAction fraction')
            ax.tick_params(axis='x',labelsize=6.5)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.04),ncol=3,fontsize=7)
    export(fig,out/'conditioned-actions')


def tradeoff_figure(frame,out,threshold):
    fig,(ax,table_ax)=plt.subplots(1,2,figsize=(7.1,3.35),layout='constrained',
                                   gridspec_kw={'width_ratios':[1.15,1.05]})
    entries=[]
    table_rows=[]
    for _,row in frame.iterrows():
        name=BASELINE_NAMES[row.policy] if row.policy in BASELINE_NAMES else 'PPO '+str(int(row.seed))
        if row.policy in BASELINE_NAMES:color='#777777';marker={'edf_backfill':'s','edf_carbon_threshold':'^','cars_risk_only':'D'}[row.policy]
        else:color=COLORS[SEEDS.index(int(row.seed))];marker='o'
        x=float(row.carbon_kg);y=100*float(row.deadline_violation_rate_observed)
        ax.scatter(x,y,c=color,marker=marker,s=32,label=name,linewidths=.7)
        entries.append((x,y,name))
        table_rows.append([name,str(int(row.deadline_missed_completed)),f'{y:.2f}',f'{x:,.1f}'])
    ax.axhline(threshold*100,color='#999999',linestyle='--',linewidth=.8,label='Fixed 5% eligibility')
    ax.set(xlabel='Common-window carbon emissions (kg CO2e)',ylabel='Deadline miss rate (%)',
           xlim=(0,max(frame.carbon_kg)*1.08),ylim=(0,max(float(frame.deadline_violation_rate_observed.max())*100,threshold*100)*1.14))
    ax.legend(fontsize=6.5,loc='upper left')
    table_ax.axis('off')
    table=table_ax.table(cellText=table_rows,colLabels=['Policy','Misses','DMR (%)','Carbon (kg)'],
                        loc='center',cellLoc='right',colWidths=[.39,.18,.21,.29])
    table.auto_set_font_size(False);table.set_fontsize(6.5);table.scale(1,1.55)
    for (row,col),cell in table.get_celld().items():
        cell.set_linewidth(.35)
        if col==0:cell.get_text().set_ha('left')
        if row==0:cell.set_facecolor('#EEEEEE');cell.get_text().set_weight('bold')
    table_ax.set_title('All six policies; common window',fontsize=8)
    duplicate={}
    for x,y,name in entries:duplicate.setdefault((x,y),[]).append(name)
    for (x,y),names in duplicate.items():
        if len(names)>1:ax.annotate('Overlap: '+', '.join(names),(x,y),xytext=(-4,-16),
            textcoords='offset points',ha='right',fontsize=6.0)
    export(fig,out/'carbon-DMR')


def analyze(root,output):
    root=Path(root).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Refusing to overwrite analysis outputs')
    # The canonical configuration is copied into every candidate before learn().
    folders={seed:root/f'train-seed-{seed}-001' for seed in SEEDS}
    required=[root/'validation-001'/'common_summary.csv',root/'validation-001'/'fairness.json',root/'validation-001'/'selection.json']
    for folder in folders.values():required.extend(folder/name for name in ('training_summary.json','training_config.json','episodes.json','updates.json','steps.jsonl.gz'))
    if any(not path.is_file() for path in required):raise ValueError('All three candidates and complete Validation evidence are required')
    config=read(folders[SEEDS[0]]/'training_config.json')
    threshold=float(config['threshold_g_per_kwh'])
    if threshold!=161.075 or config['test_evaluation'] is not False:raise ValueError('Frozen threshold or Test guard differs')
    if tuple(config['rl_seeds'])!=SEEDS:raise ValueError('Declared RL seeds differ')
    fairness=read(root/'validation-001'/'fairness.json')
    if fairness['split']!='validation' or fairness['formal_test_evaluated'] is not False:raise ValueError('Only unevaluated-Test Validation evidence allowed')
    validation=pd.read_csv(root/'validation-001'/'common_summary.csv')
    if len(validation)!=6:raise ValueError('Require three baseline and all three PPO candidates')
    for name,expected in read(root/'validation-001'/'output_hashes.json').items():
        if sha(root/'validation-001'/name)!=expected:raise ValueError('Validation artifact changed: '+name)
    training=[];episodes=[];updates=[];scales=[];actions=[];conditioned=[];features=[];correlations=[];checks=[]
    for seed,folder in folders.items():
        summary=read(folder/'training_summary.json');candidate_cfg=read(folder/'training_config.json')
        if summary['seed']!=seed or summary['num_timesteps']!=20480:raise ValueError('Candidate seed or budget changed')
        if any(candidate_cfg[key]!=config[key] for key in ('ppo','reward','normalization','training_input','training_identity','threshold_g_per_kwh')):
            raise ValueError('Cross-seed training protocol differs')
        for name,expected in summary['files'].items():
            if sha(folder/name)!=expected:raise ValueError('Training artifact changed: '+name)
        entries=list(source_rows(folder));observations=[r['observation'] for r in entries]
        if len(entries)!=summary['num_timesteps']:raise ValueError('Actor trace length mismatch')
        if any(r['truncated'] for r in entries):raise ValueError('Formal training contained censored episodes')
        seen=set()
        for row in entries:
            ids=row['reward_terms']['new_deadline_miss_ids']
            if seen.intersection(ids) or len(ids)!=len(set(ids)):raise ValueError('Repeated deadline penalty within an episode')
            seen.update(ids)
            if row['terminated']:seen.clear()
        episode_records=read(folder/'episodes.json')
        endings=[r['step'] for r in entries if r['terminated']]
        if len(endings)!=summary['completed_episodes'] or len(episode_records)!=len(endings):
            raise ValueError('Completed episode count differs from actual transition boundaries')
        previous=0
        for episode,ending in zip(episode_records,endings):
            if episode['ending_step']!=ending or episode['length']!=ending-previous:
                raise ValueError('Complete episode length/boundary mismatch')
            previous=ending
            if not episode['terminated'] or episode['truncated']:raise ValueError('Complete-episode curve contains censored episode')
            if episode['new_deadline_misses']!=episode['summary']['deadline_missed_completed']:
                raise ValueError('Completed ledger misses differ from reward events')
            episodes.append(dict(seed=seed,**{key:value for key,value in episode.items() if key!='summary'}))
        if len(entries)-previous!=summary['current_partial_episode_steps']:
            raise ValueError('Sample-budget tail length differs from actual transitions')
        if summary['current_partial_episode_steps']:
            partial=summary['partial_episode'];tail=entries[previous:]
            if partial['length']!=len(tail) or any(r['terminated'] or r['truncated'] for r in tail):
                raise ValueError('Budget tail incorrectly marked as a complete/terminated episode')
            if not math.isclose(partial['episode_return'],math.fsum(r['reward'] for r in tail),abs_tol=1e-10):
                raise ValueError('Budget tail return differs from actual rewards')
        updates.extend(dict(seed=seed,**row) for row in read(folder/'updates.json'))
        total_cost=math.fsum(math.fsum(r['reward_terms'][key] for r in entries) for key in COMPONENTS)
        raw_definitions={'carbon_cost':('carbon_kg','kgCO2e','carbon_normalized'),
            'wait_cost':('wait_task_minutes','task-minutes','wait_normalized'),
            'deadline_cost':('new_deadline_misses','tasks','deadline_normalized'),
            'terminal_cost':('terminal_liability_normalized','dimensionless combined backlog liability','terminal_liability_normalized')}
        for key in COMPONENTS:
            values=np.asarray([r['reward_terms'][key] for r in entries],dtype=float)
            raw_key,raw_unit,normalized_key=raw_definitions[key]
            raw_values=np.asarray([r['reward_terms'][raw_key] for r in entries],dtype=float)
            normalized_values=np.asarray([r['reward_terms'][normalized_key] for r in entries],dtype=float)
            scales.append(dict(seed=seed,component=key,sum=float(math.fsum(values)),
                mean_per_step=float(values.mean()),minimum_per_step=float(values.min()),
                median_per_step=float(np.quantile(values,.5)),p95_per_step=float(np.quantile(values,.95)),
                maximum_per_step=float(values.max()),fraction_total_weighted_cost=float(math.fsum(values)/total_cost),
                positive_step_fraction=float((values>0).mean()),units='dimensionless weighted reward cost',
                raw_quantity=raw_key,raw_units=raw_unit,raw_sum=float(math.fsum(raw_values)),
                raw_mean=float(raw_values.mean()),raw_p95=float(np.quantile(raw_values,.95)),
                normalized_mean=float(normalized_values.mean()),normalized_p95=float(np.quantile(normalized_values,.95))))
        training.append(dict(seed=seed,num_timesteps=summary['num_timesteps'],
            elapsed_seconds=summary['elapsed_seconds'],peak_rss_kib=summary['peak_rss_kib'],
            completed_episodes=summary['completed_episodes'],partial_episode_steps=summary['current_partial_episode_steps'],
            checkpoint_sha256=summary['checkpoint_sha256'],parameter_sha256_before=summary['parameter_sha256_before'],
            parameter_sha256_after=summary['parameter_sha256_after'],
            parameter_updated=summary['parameter_sha256_before']!=summary['parameter_sha256_after'],
            deterministic_roundtrip=summary['deterministic_roundtrip'],**summary['reward_components']))
        counts=Counter(r['action'] for r in entries)
        for action in range(3):actions.append(dict(seed=seed,phase='train',action=action,action_name=ACTION_NAMES[action],
                count=counts[action],denominator=len(entries),fraction=counts[action]/len(entries)))
        conditioned.extend(conditional_rows(entries,seed,'train',threshold));features.extend(feature_rows(observations,seed,'train'))
        correlations.extend(correlation_rows(observations,seed,'train'))
        candidate_rows=validation[validation.seed==seed]
        if len(candidate_rows)!=1:raise ValueError('Missing or duplicated candidate Validation record')
        candidate_row=candidate_rows.iloc[0]
        if (candidate_row.checkpoint_sha256!=summary['checkpoint_sha256']
            or int(candidate_row.training_steps)!=summary['num_timesteps']
            or candidate_row.training_config_sha256!=summary['config_sha256']):
            raise ValueError('Validation checkpoint/budget/config differs from corresponding Train candidate')
        name=str(candidate_rows.iloc[0].policy)
        actor=pd.read_csv(root/'validation-001'/name/'actions.csv')
        actor_rows=actor.to_dict(orient='records')
        values=actor[[f'obs_{name}' for name in OBSERVATION_FIELDS]].to_numpy(dtype=float)
        features.extend(feature_rows(values,seed,'validation'))
        correlations.extend(correlation_rows(values,seed,'validation'))
        conditioned.extend(conditional_rows(actor_rows,seed,'validation',threshold))
        counts=Counter(int(a) for a in actor.action)
        for action in range(3):actions.append(dict(seed=seed,phase='validation',action=action,action_name=ACTION_NAMES[action],
            count=counts[action],denominator=len(actor),fraction=counts[action]/len(actor)))
        checks.append(dict(seed=seed,training_log_sha256=sha(folder/'steps.jsonl.gz'),
            checkpoint_sha256=sha(folder/'model.zip'),validation_actions_sha256=sha(root/'validation-001'/name/'actions.csv'),
            no_training_censoring=True,no_duplicate_miss_penalties=True))
    output.mkdir(parents=True,exist_ok=False)
    tables={'training-summary':training,'training-episodes':episodes,'training-updates':updates,
            'reward-scales':scales,'action-distribution':actions,'conditioned-actions':conditioned,'observation-audit':features}
    for name,rows in tables.items():csv(output/(name+'.csv'),rows)
    csv(output/'observation-correlations.csv',correlations)
    columns=['policy','seed','factor','stress_seed','arrived','completed','pending','running',
        'deadline_missed_completed','deadline_missed_unfinished','deadline_right_censored',
        'deadline_violation_rate_observed','carbon_kg','it_energy_kwh','hvac_energy_kwh','total_energy_kwh',
        'mean_jct_minutes_completed','mean_wait_minutes_started','pending_mean','pending_max',
        'policy_defer_intervals','capacity_wait_intervals','cores_fraction_mean','gpus_fraction_mean','mem_fraction_mean',
        'drain_hours','own_observation_end','observation_end','run_seconds','inference_seconds',
        'action_0_count','action_1_count','action_2_count','checkpoint_sha256']
    compact=validation[columns]
    compact.to_csv(output/'validation-summary.csv',index=False)
    selection=read(root/'validation-001'/'selection.json')
    write(output/'selection.json',selection)
    write(output/'audit.json',dict(train_seeds=list(SEEDS),validation_fairness=fairness,
        verified_training_artifacts=checks,partial_episodes_excluded_from_complete_episode_curves=True,
        exclusion='sample-budget tails are retained in original summaries; no seed or complete episode removed',
        confidence_intervals='none; three initialization seeds on one deterministic workload/window',
        formal_test_evaluated=False,source_script_sha256=sha(Path(__file__))))
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7.5,'axes.labelsize':7.5,
        'axes.titlesize':8,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':7,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'pdf.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white'})
    training_figure(pd.DataFrame(episodes),output);reward_figure(pd.DataFrame(episodes),output)
    action_figure(pd.DataFrame(actions),output);conditional_figure(pd.DataFrame(conditioned),output)
    tradeoff_figure(validation,output,config['selection']['maximum_deadline_miss_rate'])
    write(output/'output-hashes.json',{path.name:sha(path) for path in sorted(output.iterdir()) if path.is_file()})
    return dict(output=str(output),training=training,selection=selection,validation=compact.to_dict(orient='records'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=analyze(args.root,args.output)
    print(json.dumps({'output':result['output'],'selection':result['selection']},indent=2))


if __name__=='__main__':main()
