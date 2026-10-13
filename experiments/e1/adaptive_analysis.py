"""Complete development-Validation ablations; no tuning or Test access.

Three training seeds share one frozen workload per factor. Their min/max is
training variability, not a confidence interval or independent arrival samples.
"""
import argparse
import csv
import json
import math
from pathlib import Path

from .contracts import file_hash, write_json

SCHEMA='e1-6b-adaptive-analysis-v1'
GROUPS=('plain','state','shield','combined')
SEEDS=(55101,55102,55103)
BASELINES=('edf_backfill','edf_carbon_threshold','cars_risk_only','task_risk_full')
FACTORS=(1.,1.5,2.)
METRICS=('misses','deadline_violation_rate_observed','carbon_kg','total_energy_kwh',
         'it_energy_kwh','hvac_energy_kwh','mean_jct_minutes_completed',
         'mean_wait_minutes_started','pending_mean','pending_max','capacity_wait_intervals',
         'policy_defer_intervals','filter_veto_rate','extra_misses_vs_edf','carbon_reduction_vs_edf_percent')
COLORS=('#6e7781','#4c78a8','#c58e4e','#9372b2')


def _json(path):return json.loads(path.read_text(encoding='utf8'))


def _csv(path):
    with path.open(encoding='utf8',newline='') as stream:return list(csv.DictReader(stream))


def _write_csv(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)


def _number(value):
    if value in (None,'','None'):return None
    value=float(value)
    if not math.isfinite(value):raise ValueError('Nonfinite summary value')
    return value


def _mean(values):return math.fsum(values)/len(values)


def _aggregates(rows,metrics):
    result={}
    for metric in metrics:
        values=[row[metric] for row in rows if row.get(metric) is not None]
        result[metric+'_n_defined']=len(values)
        # No partial-seed mean: missing opportunity denominators remain explicit.
        complete=len(values)==len(rows)
        result[metric+'_mean']=_mean(values) if complete else None
        result[metric+'_min']=min(values) if complete else None
        result[metric+'_max']=max(values) if complete else None
    return result


def _training(paths,hashes):
    summaries={};rows=[];updates=[];episodes=[];configs=[]
    for folder in map(Path,paths):
        path=folder/'training_summary.json';summary=_json(path);hashes[str(path)]=file_hash(path)
        key=(summary['group'],summary['seed'])
        if key in summaries or key[0] not in GROUPS or key[1] not in SEEDS:
            raise ValueError('Missing/duplicate/unregistered training group or seed')
        if (summary.get('schema')!='e1-6b-adaptive-ppo-v1' or summary.get('num_timesteps')!=20480
            or summary.get('observation_size')!=83 or summary.get('action_size')!=4
            or summary.get('parameter_count')!=19397 or not summary.get('deterministic_roundtrip')
            or summary.get('optimizer_state_entries',0)<=0
            or summary.get('parameter_sha256_before')==summary.get('parameter_sha256_after')
            or [i.get('split') for i in summary.get('training_identities',[])]!=['train','train']
            or [i.get('factor') for i in summary.get('training_identities',[])]!=[1.5,2.]):
            raise ValueError('Invalid formal training protocol or update evidence')
        for name in ('training_config.json','updates.csv','episodes.csv'):
            path=folder/name;digest=file_hash(path);hashes[str(path)]=digest
            if summary.get('files',{}).get(name)!=digest:raise ValueError('Training small-file hash mismatch')
        config=_json(folder/'training_config.json')
        configs.append({k:config.get(k) for k in ('ppo','policy_kwargs','reward','normalization',
                                                 'risk_parameters','threshold_g_per_kwh')})
        sums=summary['scene_steps']
        if sum(sums.values())!=20480 or sum(summary['action_counts'])!=20480:
            raise ValueError('Actual formal scene/action step budget differs')
        row={k:summary.get(k) for k in ('group','seed','num_timesteps','parameter_count',
            'parameter_sha256_before','parameter_sha256_after','checkpoint_sha256','source_sha256',
            'elapsed_seconds','completed_episodes','current_partial_episode_steps','budget_ends_mid_episode',
            'optimizer_state_entries','deterministic_roundtrip','peak_rss_kib')}
        row.update({f'action_{i}_count':v for i,v in enumerate(summary['action_counts'])})
        row.update({f'scene_{factor}_steps':sums.get(str(factor),0) for factor in (1.5,2.)})
        for factor in (1.5,2.):row[f'scene_{factor}_step_fraction']=row[f'scene_{factor}_steps']/20480
        rows.append(row);summaries[key]=summary
        update_rows=_csv(folder/'updates.csv')
        if len(update_rows)!=80 or float(update_rows[-1]['train/n_updates'])!=800:
            raise ValueError('Incomplete frozen PPO update sequence')
        for update in update_rows:
            converted={k:_number(v) for k,v in update.items()}
            updates.append(dict(group=key[0],seed=key[1],**converted))
        episode_rows=_csv(folder/'episodes.csv')
        if len(episode_rows)!=summary['completed_episodes']:
            raise ValueError('Completed episode count differs')
        if any(float(r['factor']) not in (1.5,2.) or r['truncated'].lower()!='false' for r in episode_rows):
            raise ValueError('Unexpected training scene or environment truncation')
        metrics=('length','episode_return','carbon_kg','wait_task_minutes','new_deadline_misses',
                 'carbon_cost','wait_cost','deadline_cost','terminal_cost')
        for factor in (1.5,2.):
            selected=[{k:_number(r[k]) for k in metrics} for r in episode_rows if float(r['factor'])==factor]
            episodes.append(dict(group=key[0],seed=key[1],factor=factor,completed_episodes=len(selected),
                **(_aggregates(selected,metrics) if selected else {k+'_n_defined':0 for k in metrics})))
    expected={(g,s) for g in GROUPS for s in SEEDS}
    if set(summaries)!=expected:raise ValueError('Incomplete formal group/seed matrix')
    if any(c!=configs[0] for c in configs):raise ValueError('Shared PPO/Reward/network config differs')
    if len({s['source_sha256'] for s in summaries.values()})!=1:
        raise ValueError('Formal models used different frozen sources')
    if len({json.dumps(s['training_identities'],sort_keys=True) for s in summaries.values()})!=1:
        raise ValueError('Formal models used different frozen Train inputs')
    for seed in SEEDS:
        if len({summaries[g,seed]['parameter_sha256_before'] for g in GROUPS})!=1:
            raise ValueError('Same-seed initial network parameters differ')
    return summaries,rows,updates,episodes


def analyze_campaign(group_dirs,training_dirs,output_dir,*,render=True):
    """Reject incomplete/unequal work before emitting reductions or plots."""
    output=Path(output_dir)
    if output.exists():raise FileExistsError(output)
    hashes={};training,train_rows,updates,episodes=_training(training_dirs,hashes)
    methods=[];fairness_rows=[];seen_factors=set();legacy_seeds=None
    numeric=set(METRICS)|{'factor','seed','arrived','completed','pending','running','deadline_missed_completed',
        'deadline_missed_unfinished','deadline_right_censored','cores_hours','gpus_hours','mem_hours',
        'candidate_checks','solo_eligible','filter_veto','deferred','run_seconds','inference_seconds',
        'drain_hours','training_steps','training_elapsed_seconds'}
    for folder in map(Path,group_dirs):
        manifest=_json(folder/'output_hashes.json')
        for name in ('common_summary.csv','fairness.json','scientific_hashes.json'):
            path=folder/name;digest=file_hash(path);hashes[str(path)]=digest
            if manifest.get(name)!=digest:raise ValueError('Evaluation small-file hash mismatch')
        hashes[str(folder/'output_hashes.json')]=file_hash(folder/'output_hashes.json')
        fairness=_json(folder/'fairness.json');raw=_csv(folder/'common_summary.csv')
        if fairness.get('split')!='validation' or fairness.get('formal_test_evaluated'):
            raise ValueError('Only development Validation analysis permitted; Test sealed')
        if (not all(fairness.get(k) is True for k in ('all_tasks_completed','equal_resource_hours',
                'equal_workload_carbon_comparison_valid','idle_included','development_validation'))
                or fairness.get('policy_future_ci_access') is not False):
            raise ValueError('Complete Drain/equal-work/idle/current-information fairness required')
        if len(raw)!=19 or len({r['policy'] for r in raw})!=19:
            raise ValueError('Incomplete/duplicate nineteen-method matrix')
        factor=_number(raw[0]['factor'])
        if factor not in FACTORS or factor in seen_factors:raise ValueError('Missing/duplicate pressure factors')
        seen_factors.add(factor)
        rows=[{k:(_number(v) if k in numeric else v) for k,v in r.items()} for r in raw]
        adaptive=[r for r in rows if r['group'] in GROUPS]
        if {(r['group'],r['seed']) for r in adaptive}!={(g,s) for g in GROUPS for s in SEEDS} or len(adaptive)!=12:
            raise ValueError('Incomplete/duplicate adaptive group/seed matrix')
        heuristic=[r for r in rows if r['group']=='heuristic']
        external=[r for r in rows if r['group']=='legacy_3action']
        if {r['policy'] for r in heuristic}!=set(BASELINES) or len(heuristic)!=4 or len(external)!=3:
            raise ValueError('External baseline set differs')
        current_legacy={r['seed'] for r in external}
        if len(current_legacy)!=3 or (legacy_seeds is not None and current_legacy!=legacy_seeds):
            raise ValueError('Legacy seed set differs')
        legacy_seeds=current_legacy
        if set(_json(folder/'scientific_hashes.json'))!={r['policy'] for r in rows}:
            raise ValueError('Scientific result hash set differs')
        for row in rows:
            if (row['factor']!=factor or row['split']!='validation' or row['phase']!='common'
                or row['arrived']!=fairness['arrived'] or row['completed']!=fairness['arrived']
                or row['pending']!=0 or row['running']!=0 or row['deadline_right_censored']!=0
                or row['deadline_missed_unfinished']!=0 or row['termination']!='all_completed'
                or row['observation_end']!=fairness['common_end']):
                raise ValueError('Incomplete/unequal task or observation boundary')
            row['misses']=int(row['deadline_missed_completed']+row['deadline_missed_unfinished'])
            if not math.isclose(row['deadline_violation_rate_observed'],row['misses']/row['arrived'],abs_tol=1e-14):
                raise ValueError('Deadline count/rate differs')
            if not math.isclose(row['total_energy_kwh'],row['it_energy_kwh']+row['hvac_energy_kwh'],rel_tol=1e-12):
                raise ValueError('IT/HVAC total energy differs')
            if row['group'] in GROUPS:
                trained=training[row['group'],int(row['seed'])]
                if row['checkpoint_sha256']!=trained['checkpoint_sha256'] or row['training_steps']!=20480:
                    raise ValueError('Evaluation checkpoint/training budget differs')
            row['filter_veto_rate']=row['filter_veto']/row['solo_eligible'] if row['solo_eligible'] else None
            row['meets_5_percent']=row['deadline_violation_rate_observed']<=.05
        for resource in ('cores','gpus','mem'):
            if any(not math.isclose(r[resource+'_hours'],rows[0][resource+'_hours'],rel_tol=1e-12,abs_tol=1e-8) for r in rows):
                raise ValueError('Unequal completed resource hours')
        edf=next(r for r in heuristic if r['policy']=='edf_backfill')
        for row in rows:
            row['extra_misses_vs_edf']=row['misses']-edf['misses']
            row['carbon_reduction_vs_edf_percent']=100*(edf['carbon_kg']-row['carbon_kg'])/edf['carbon_kg']
        methods.extend(rows);fairness_rows.append(fairness)
    if seen_factors!=set(FACTORS):raise ValueError('All three Validation pressure factors required')
    if len({f['environment_sha256'] for f in fairness_rows})!=1 or len({tuple(f['capacity']) for f in fairness_rows})!=1:
        raise ValueError('Validation environment/capacity differs between factors')
    paired=[];summaries=[]
    for factor in FACTORS:
        for group in GROUPS:
            subset=[r for r in methods if r['factor']==factor and r['group']==group]
            summaries.append(dict(factor=factor,group=group,training_seeds=3,workload_realizations=1,
                                  **_aggregates(subset,METRICS)))
            if group=='plain':continue
            for seed in SEEDS:
                new=next(r for r in subset if r['seed']==seed)
                plain=next(r for r in methods if r['factor']==factor and r['group']=='plain' and r['seed']==seed)
                paired.append(dict(factor=factor,group=group,seed=seed,reference='plain',
                    delta_misses=new['misses']-plain['misses'],
                    delta_dmr_percentage_points=100*(new['deadline_violation_rate_observed']-plain['deadline_violation_rate_observed']),
                    delta_carbon_kg=new['carbon_kg']-plain['carbon_kg'],
                    delta_energy_kwh=new['total_energy_kwh']-plain['total_energy_kwh'],
                    delta_jct_minutes=new['mean_jct_minutes_completed']-plain['mean_jct_minutes_completed'],
                    delta_wait_minutes=new['mean_wait_minutes_started']-plain['mean_wait_minutes_started'],
                    delta_pending_mean=new['pending_mean']-plain['pending_mean']))
    if any(file_hash(Path(path))!=digest for path,digest in hashes.items()):
        raise ValueError('Read-only input changed during analysis')
    output.mkdir(parents=True)
    tables={'all-methods.csv':methods,'paired-effects.csv':paired,'group-summary.csv':summaries,
            'training-summary.csv':train_rows,'training-updates.csv':updates,'training-episodes-summary.csv':episodes}
    for name,rows in tables.items():_write_csv(output/name,rows)
    axes=plot_results(methods,paired,output/'figures') if render else {}
    manifest=dict(schema=SCHEMA,input_hashes=hashes,inputs_unchanged=True,
        all_equal_workload_carbon_comparisons_valid=True,development_validation=True,
        no_test_access=True,training_seeds=list(SEEDS),group_count=4,
        uncertainty='min-max over PPO training seeds; one frozen workload per factor; no significance claim',
        figure_axis_limits=axes,analysis_source_sha256=file_hash(Path(__file__)),
        output_hashes={p.relative_to(output).as_posix():file_hash(p) for p in output.rglob('*') if p.is_file()})
    write_json(output/'manifest.json',manifest)
    return dict(methods=methods,paired=paired,groups=summaries,training=train_rows,updates=updates,episodes=episodes,manifest=manifest)


def plot_results(methods,paired,output_dir):
    """Absolute axes start0; signed paired axes preserve negative effects."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    directory=Path(output_dir)
    if directory.exists():raise FileExistsError(directory)
    directory.mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.,'axes.labelsize':7.,'axes.titlesize':8.,
        'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.5,
        'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'e1-6b-adaptive',
        'axes.spines.top':False,'axes.spines.right':False})
    labels=('Plain','State','Shield','Combined')
    baseline_labels=('EDF','EDF+Carbon','RiskOnly','Task Full')
    baseline_colors=('#1f1f1f','#8d9f79','#588c8c','#ae8a99')
    styles=('-', '--', '-.', ':')
    axis_limits={}
    def save(fig,stem):
        fig.savefig(directory/(stem+'.png'),dpi=300)
        fig.savefig(directory/(stem+'.pdf'),metadata={'CreationDate':None,'ModDate':None})
        fig.savefig(directory/(stem+'.svg'),metadata={'Date':None})
        plt.close(fig)
    def dots(ax,values,x,color):
        ax.scatter([x-.07,x,x+.07],values,s=16,color=color,zorder=4)
        ax.vlines(x,min(values),max(values),color=color,lw=1.)
        ax.hlines(_mean(values),x-.15,x+.15,color=color,lw=1.6,zorder=5)
    for metric,stem,ylabel,scale in [('deadline_violation_rate_observed','deadline-miss-rate','Deadline miss rate (%)',100.),
                                    ('carbon_kg','carbon-emissions','Common-window carbon (kgCO₂e)',1.)]:
        values=[r[metric]*scale for r in methods]
        upper=max(values+([5.] if scale==100 else [0.]))*1.12
        if upper==0:upper=1.
        fig,axes=plt.subplots(1,3,figsize=(7.2,3.4),sharey=True)
        for ax,factor in zip(axes,FACTORS):
            subset=[r for r in methods if r['factor']==factor]
            for index,group in enumerate(GROUPS):
                selected=sorted((r for r in subset if r['group']==group),key=lambda r:r['seed'])
                dots(ax,[r[metric]*scale for r in selected],index,COLORS[index])
            for name,color,style in zip(BASELINES,baseline_colors,styles):
                value=next(r[metric]*scale for r in subset if r['policy']==name)
                ax.axhline(value,color=color,lw=.9,ls=style,zorder=1)
            legacy=[r[metric]*scale for r in subset if r['group']=='legacy_3action']
            ax.axhspan(min(legacy),max(legacy),color='#a7a7a7',alpha=.15,zorder=0)
            ax.axhline(_mean(legacy),color='#777777',lw=1.,ls=(0,(5,2)),zorder=1)
            if scale==100:ax.axhline(5.,color='#999999',lw=.7,ls=(0,(1,3)))
            ax.set(xticks=range(4),xticklabels=labels,title=f'{factor:g}× development Validation',ylim=(0,upper),xlim=(-.35,3.35))
        axes[0].set_ylabel(ylabel)
        handles=[Line2D([0],[0],color=c,ls=s,lw=1.,label=l) for c,s,l in zip(baseline_colors,styles,baseline_labels)]
        handles.append(Line2D([0],[0],color='#777777',ls=(0,(5,2)),label='Legacy 3-action PPO mean/range'))
        if scale==100:handles.append(Line2D([0],[0],color='#999999',ls=(0,(1,3)),label='5% operational target'))
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.1),ncol=3)
        fig.text(.5,.02,'Dots: 3 training seeds, same workload. Horizontal mark: mean; vertical span: min–max.\nTraining variability only; no independent-workload or significance claim.',ha='center',fontsize=6.5)
        fig.subplots_adjust(left=.1,right=.99,top=.9,bottom=.31,wspace=.12)
        axis_limits[stem]=list(axes[0].get_ylim());save(fig,stem)
    fig,axes=plt.subplots(2,3,figsize=(7.2,4.8),sharey='row')
    for row,(metric,ylabel,key) in enumerate([('delta_dmr_percentage_points','DMR − Plain (percentage points)','paired-dmr'),
                                            ('delta_carbon_kg','Carbon − Plain (kgCO₂e)','paired-carbon')]):
        values=[r[metric] for r in paired];low=min(values+[0.]);high=max(values+[0.]);padding=max((high-low)*.12,.01)
        for ax,factor in zip(axes[row],FACTORS):
            for index,group in enumerate(GROUPS[1:]):
                selected=sorted((r for r in paired if r['factor']==factor and r['group']==group),key=lambda r:r['seed'])
                dots(ax,[r[metric] for r in selected],index,COLORS[index+1])
            ax.axhline(0,color='#777777',lw=.8,ls='--')
            ax.set(xticks=range(3),xticklabels=labels[1:],ylim=(low-padding,high+padding),xlim=(-.3,2.3))
            if row==0:ax.set_title(f'{factor:g}× development Validation')
        axes[row,0].set_ylabel(ylabel);axis_limits[key]=list(axes[row,0].get_ylim())
    fig.text(.5,.025,'Paired by training seed; State/Shield/Combined minus Plain.\nDots: all 3 seeds; mark: mean; span: min–max. Negative means lower metric, not joint dominance.',ha='center',fontsize=6.5)
    fig.subplots_adjust(left=.11,right=.99,top=.93,bottom=.16,wspace=.12,hspace=.38)
    save(fig,'paired-effects')
    write_json(directory/'axis-audit.json',axis_limits)
    return axis_limits


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groups',nargs='+',required=True)
    parser.add_argument('--training',nargs='+',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=analyze_campaign(args.groups,args.training,args.output)
    print(json.dumps(dict(methods=len(result['methods']),paired=len(result['paired']),models=len(result['training']))))


if __name__=='__main__':main()
