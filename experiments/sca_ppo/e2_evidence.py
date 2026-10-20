"""Postprocess the sole completed pilot; no simulator, data loader or training.

Figure contract: show actual updates separately from performance improvement.
Quantitative grid, one seed, all32 episodes, all4 paired held-out Train windows.
Python-only rendering,183mm width, editable SVG/PDF plus PNG, minimum7pt text.
No confidence intervals: these are correlated development windows, not seeds.
"""
import argparse
from collections import defaultdict
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

ROOT=Path(__file__).resolve().parents[2]
METHODS=('EDF-CG','CarbonRule-DDF-CG','PPO-initial','PPO-final')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8',newline='') as stream:
        return list(csv.DictReader(stream))


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',
                          encoding='utf-8',newline='\n')


def number(row,key):return float(row[key])


def audit_saved(source):
    """Independent arithmetic over saved accounts, without dispatch replay."""
    errors=[];count=ppo_count=0;max_reward_error=max_incremental_error=0.
    for path in sorted(source.rglob('result.json')):
        result=json.loads(path.read_text());summary=result['summary'];steps=result['steps']
        count+=1
        def require(condition,message):
            if not condition:errors.append(str(path.relative_to(source))+': '+message)
        audit=json.loads(path.with_name('verification.json').read_text())
        require(not audit['errors'],'saved original verifier errors')
        require(len(steps)==288,'physical horizon not288')
        if path.parts[-3]=='backgrounds':continue
        require(not summary['unfinished'] and not summary['deadline_censored'],'incomplete responsibility')
        require(not summary['reference_guard_failure_count'],'CG failure')
        require(all(row['pending']==row['running']==0 and row['supply_requested_c']==20
                    for row in steps if row['minute']>=3960),'invalid idle tail')
        records=[json.loads(line) for line in path.with_name('decisions.jsonl').read_text().splitlines()]
        require(len(records)==288,'decision horizon not288')
        starts=Counter(identity for row in records for identity in row['actual_started'])
        policy=Counter(identity for row in records for identity in row['effective_policy_wait'])
        capacity=Counter(identity for row in records for identity in row['capacity_wait'])
        events=[identity for row in records for identity in row['new_irreversible_miss_ids']]
        misses={row['task_id'] for row in result['tasks'] if row['deadline_missed']}
        require(len(events)==len(set(events)) and set(events)==misses,'per-ID miss event mismatch')
        for row in result['tasks']:
            identity=row['task_id']
            require(starts[identity]==1,'task start count not1')
            require(policy[identity]*15==row['policy_wait_minutes'] and
                    capacity[identity]*15==row['capacity_wait_minutes'],'per-ID wait mismatch')
        if 'reward' not in records[0]:continue
        ppo_count+=1;terms=[];increments=[]
        for actual,record in zip(steps,records):
            require(record['row']==actual,'PPO row differs from actual account')
            incremental=actual['carbon_kg']-record['idle_carbon_kg']
            reward=-incremental/100.-.01*len(record['new_irreversible_miss_ids'])
            require(abs(reward-record['reward'])<1e-12,'per-step reward mismatch')
            require(record['policy_sampled'] or (record['selected_action']==0 and record['log_prob'] is None),
                    'forced slot pretends to sample')
            increments.append(incremental);terms.append(record['reward'])
        max_reward_error=max(max_reward_error,abs(math.fsum(terms)-summary['reward_sum']),
            abs(summary['reward_sum']+math.fsum(increments)/100.+.01*len(misses)))
        max_incremental_error=max(max_incremental_error,
            abs(math.fsum(increments)-summary['common_incremental_carbon_kg']))
    return dict(saved_accounts=count,workload_accounts=count-16,ppo_accounts=ppo_count,
        all_saved_verifiers_pass=not errors,errors=errors,
        maximum_reward_conservation_error=max_reward_error,
        maximum_incremental_sum_error_kg=max_incremental_error,
        scope='Saved arithmetic/ID/wait/horizon audit; actual previews asserted during original execution, no resimulation.')


def summarize(rows):
    count=sum(int(row['arrived']) for row in rows)
    result=dict(episodes=len(rows),task_exposures=count)
    for field in ('completed','unfinished','deadline_missed','deadline_censored','attempts','approved','rejected',
                  'reference_guard_failure_count'):
        result[field]=sum(int(row[field]) for row in rows)
    for field in ('carbon_kg','common_idle_carbon_kg','common_incremental_carbon_kg','it_kwh','hvac_kwh',
                  'total_kwh','capacity_wait_minutes','policy_wait_minutes','reward_sum'):
        result[field]=sum(number(row,field) for row in rows)
    result['task_weighted_dmr']=result['deadline_missed']/count
    result['task_weighted_jct_minutes']=sum(number(row,'mean_jct_minutes')*int(row['completed']) for row in rows)/result['completed']
    result['verification_pass_count']=sum(row['verification_pass']=='True' for row in rows)
    result['valid_idle_tail_count']=sum(row['actual_common_idle_tail_valid']=='True' for row in rows)
    result['maximum_actual_release_minute']=max(number(row,'actual_last_release_minute') for row in rows)
    result['minimum_actual_release_minute']=min(number(row,'actual_last_release_minute') for row in rows)
    result['selected_action_counts']=[sum(json.loads(row['selected_action_counts'])[i]
        for row in rows if row.get('selected_action_counts')) for i in range(7)]
    reasons=defaultdict(int)
    for row in rows:
        for key,value in json.loads(row['rejection_reason_counts']).items():reasons[key]+=value
    result['rejection_reason_counts']=dict(reasons)
    return result


def generate(source,output):
    source=Path(source).resolve();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    execution=json.loads((source/'execution.json').read_text())
    if execution['status']!='completed':raise ValueError('Do not present a failed pilot as completed')
    training=read_csv(source/'training-metrics.csv');evaluation=read_csv(source/'evaluation-summary.csv')
    ledger_audit=audit_saved(source)
    dump(output/'ledger-audit.json',ledger_audit)
    assert not ledger_audit['errors']
    assert len(training)==32 and len(evaluation)==16
    assert execution['training_steps']==9216 and execution['optimizer_updates']==32
    for row in training+evaluation:
        assert row['verification_pass']=='True' and row['actual_common_idle_tail_valid']=='True'
        assert int(row['unfinished'])==int(row['deadline_censored'])==int(row['reference_guard_failure_count'])==0
    for name in ('training-metrics.csv','evaluation-summary.csv'):
        shutil.copyfile(source/name,output/name)
    aggregates={method:summarize([row for row in evaluation if row['method']==method]) for method in METHODS}
    pairs=[]
    for fragment in sorted(set(row['fragment'] for row in evaluation)):
        rows={row['method']:row for row in evaluation if row['fragment']==fragment}
        for method in METHODS[1:]:
            for baseline in ('EDF-CG','CarbonRule-DDF-CG'):
                if method==baseline:continue
                a,b=rows[method],rows[baseline]
                dt=number(a,'inlet_final_c')-number(b,'inlet_final_c')
                pairs.append(dict(fragment=fragment,method=method,baseline=baseline,
                    delta_carbon_kg=number(a,'carbon_kg')-number(b,'carbon_kg'),
                    delta_incremental_carbon_kg=number(a,'common_incremental_carbon_kg')-number(b,'common_incremental_carbon_kg'),
                    delta_misses=int(a['deadline_missed'])-int(b['deadline_missed']),
                    delta_dmr=number(a,'dmr')-number(b,'dmr'),
                    delta_jct_minutes=number(a,'mean_jct_minutes')-number(b,'mean_jct_minutes'),
                    delta_it_kwh=number(a,'it_kwh')-number(b,'it_kwh'),
                    delta_final_temperature_k=dt,
                    delta_thermal_storage_kwh=number(a,'thermal_capacity_kj_k')*dt/3600))
    with (output/'paired-differences.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    repeated=defaultdict(list)
    for row in training:repeated[row['fragment']].append(row)
    matched=[dict(fragment=fragment,visits=len(rows),first_episode=int(rows[0]['episode']),
        last_episode=int(rows[-1]['episode']),first_reward=number(rows[0],'reward_sum'),
        last_reward=number(rows[-1],'reward_sum'),delta_reward=number(rows[-1],'reward_sum')-number(rows[0],'reward_sum'))
        for fragment,rows in sorted(repeated.items())]
    summary=dict(schema='sca-ppo-e2-pilot-evidence-v1',source_sha=execution['source_sha'],
        protocol_sha256=execution['protocol_sha256'],data_admission_sha256=execution['data_admission_sha256'],
        execution={key:value for key,value in execution.items() if key not in ('evaluations','source_sha256','checkpoints')},
        evaluation=aggregates,training=summarize(training),
        paired_max_abs_temperature_k=max(abs(row['delta_final_temperature_k']) for row in pairs),
        paired_max_abs_thermal_storage_kwh=max(abs(row['delta_thermal_storage_kwh']) for row in pairs),
        paired_max_abs_it_kwh=max(abs(row['delta_it_kwh']) for row in pairs),
        repeated_training_first_last=matched,
        repeated_training_mean_delta_reward=statistics.mean(row['delta_reward'] for row in matched),
        training_update_statistics={field:dict(first=number(training[0],field),last=number(training[-1],field),
            minimum=min(number(row,field) for row in training),maximum=max(number(row,field) for row in training),
            first8_mean=statistics.mean(number(row,field) for row in training[:8]),
            last8_mean=statistics.mean(number(row,field) for row in training[-8:]))
            for field in ('policy_loss','value_loss','entropy','approx_kl','clip_fraction','actor_parameter_delta_l2',
                          'backbone_parameter_delta_l2','parameter_delta_l2','reward_conservation_error')},
        limitations=['One seed and32 stochastic training episodes; no confidence intervals or significance claims.',
            '4 held-out Train windows from2 correlated time components; previously used development data.',
            'Repeated-window return differences are descriptive, not a controlled improvement estimate.',
            'No tasks or failed cases excluded. Raw traces and checkpoints remain local.'])
    dump(output/'aggregate.json',summary)
    paths=sorted(path for path in source.rglob('*') if path.is_file())
    dump(output/'local-evidence-manifest.json',dict(root='research_notes/sca-ppo/e2-pilot-001',
        source_sha=execution['source_sha'],files=[dict(path=str(path.relative_to(source)).replace('\\','/'),
            bytes=path.stat().st_size,sha256=sha(path)) for path in paths]))
    plot(training,evaluation,output)
    print(json.dumps(dict(training_episodes=len(training),evaluation_episodes=len(evaluation),
        delta_final_rule_carbon_kg=aggregates['PPO-final']['carbon_kg']-aggregates['CarbonRule-DDF-CG']['carbon_kg'],
        output=str(output)),indent=2))


def plot(training,evaluation,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7,
        'legend.fontsize':7,'svg.fonttype':'none','pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    fig,axes=plt.subplots(2,2,figsize=(7.205,4.882),layout='constrained')
    episode=[int(row['episode']) for row in training]
    axes[0,0].plot(episode,[number(row,'reward_sum') for row in training],'.-',color='#436B8E',lw=.8,ms=3)
    axes[0,0].set(xlabel='Training episode',ylabel='Actual episode reward',title='a  Returns vary with workload')
    axes[0,1].plot(episode,[number(row,'value_loss') for row in training],'.-',color='#8B647F',lw=.8,ms=3)
    axes[0,1].set(xlabel='Training episode',ylabel='Value loss (half MSE)',title='b  Value loss does not steadily fall')
    firsts={}
    deltas=[]
    for row in training:
        firsts.setdefault(row['fragment'],number(row,'reward_sum'))
        deltas.append(number(row,'reward_sum')-firsts[row['fragment']])
    axes[1,0].axhline(0,color='#888888',lw=.7)
    axes[1,0].scatter(episode,deltas,color='#436B8E',s=10)
    axes[1,0].set(xlabel='Training episode',ylabel='Return minus first visit, same window',
                  title='c  Repeated-window returns (descriptive)')
    fragments=sorted(set(row['fragment'] for row in evaluation));x=list(range(len(fragments)))
    colors={'PPO-initial':'#999999','PPO-final':'#436B8E'}
    for method,marker in [('PPO-initial','o'),('PPO-final','s')]:
        changes=[]
        for fragment in fragments:
            group={row['method']:row for row in evaluation if row['fragment']==fragment}
            changes.append(number(group[method],'carbon_kg')-number(group['CarbonRule-DDF-CG'],'carbon_kg'))
        axes[1,1].plot(x,changes,marker=marker,lw=.8,ms=3,color=colors[method],label=method)
    axes[1,1].axhline(0,color='#888888',lw=.7)
    axes[1,1].set(xticks=x,xticklabels=[fragment[10:14][:2]+'/'+fragment[12:14] for fragment in fragments],
        xlabel='Held-out Train window (month/day)',ylabel='Total carbon minus Rule+DDF (kg)',
        title='d  Final policy does not beat rule carbon')
    axes[1,1].legend(loc='upper right')
    fig.savefig(output/'E2-pilot-learning.svg')
    fig.savefig(output/'E2-pilot-learning.pdf')
    fig.savefig(output/'E2-pilot-learning.png',dpi=600)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default=str(ROOT/'research_notes/sca-ppo/e2-pilot-001'))
    parser.add_argument('--output',default=str(ROOT/'docs/research/sca-ppo/e2-results'))
    args=parser.parse_args();generate(args.input,args.output)
