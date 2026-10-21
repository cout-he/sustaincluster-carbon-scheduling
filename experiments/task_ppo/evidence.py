"""Read-only Task-PPO pilot evidence; never dispatch, train or select a model.

All expected counts are reported alongside actual counts, including partial or
failed executions. Optional frozen-network feature probes use saved final-policy
contexts only and are not schedule re-evaluations or performance estimates.
"""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

ROOT=Path(__file__).resolve().parents[2]
METHODS=('EDF-CG','CarbonRule-DDF-CG','MacroPPO-history','TaskPPO-initial','TaskPPO-final')
TASK_FEATURES=('cpu_fraction','gpu_fraction','mem_fraction','duration','age','deadline_remaining',
               'personal_slack','dynamic_power','reference_remaining','grid_duration','arrival_fraction','deadline_fraction')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',
                          encoding='utf-8',newline='\n')


def read_csv(path):
    path=Path(path)
    if not path.exists():return []
    with path.open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))


def number(row,key,default=None):
    value=row.get(key)
    if value in (None,''):return default
    result=float(value)
    return result if math.isfinite(result) else default


def truth(value):return value is True or str(value).lower()=='true'


def summarize(rows):
    exposures=sum(int(row.get('arrived') or 0) for row in rows)
    out=dict(episodes=len(rows),task_exposures=exposures)
    for field in ('completed','unfinished','deadline_missed','deadline_censored','attempts','approved','rejected',
                  'reference_guard_failure_count','task_decisions','sampled_slots','forced_slots',
                  'mixed_binary_slots','internal_candidate_checks','internal_calendar_plans'):
        out[field]=sum(int(number(row,field,0)) for row in rows)
    for field in ('carbon_kg','common_idle_carbon_kg','common_incremental_carbon_kg','it_kwh','hvac_kwh',
                  'total_kwh','capacity_wait_minutes','policy_wait_minutes','reward_sum'):
        values=[number(row,field) for row in rows]
        out[field]=math.fsum(value for value in values if value is not None)
        if any(value is None for value in values):out.setdefault('missing_common_metrics',[]).append(field)
    out['task_weighted_dmr']=out['deadline_missed']/exposures if exposures else None
    means=[(number(row,'mean_jct_minutes'),int(row.get('completed') or 0)) for row in rows]
    out['task_weighted_jct_minutes']=(math.fsum(mean*count for mean,count in means if mean is not None)/out['completed']
        if out['completed'] and all(mean is not None or not count for mean,count in means) else None)
    out['verification_pass_count']=sum(truth(row.get('verification_pass')) for row in rows)
    out['valid_idle_tail_count']=sum(truth(row.get('actual_common_idle_tail_valid')) for row in rows)
    out['failure_latched_count']=sum(truth(row.get('failure_latched')) for row in rows)
    releases=[number(row,'actual_last_release_minute') for row in rows]
    releases=[value for value in releases if value is not None]
    out['maximum_actual_release_minute']=max(releases,default=None)
    out['minimum_actual_release_minute']=min(releases,default=None)
    out['binary_action_counts']=[sum(json.loads(row['binary_action_counts'])[i]
        for row in rows if row.get('binary_action_counts')) for i in range(2)]
    out['macro_action_counts']=[sum(json.loads(row['selected_action_counts'])[i]
        for row in rows if row.get('selected_action_counts')) for i in range(7)]
    reasons=Counter()
    for row in rows:
        if row.get('rejection_reason_counts'):reasons.update(json.loads(row['rejection_reason_counts']))
    out['rejection_reason_counts']=dict(reasons)
    out['ddf_rejection_fraction']=out['rejected']/out['attempts'] if out['attempts'] else None
    out['binary_defer_fraction']=out['binary_action_counts'][1]/sum(out['binary_action_counts']) if sum(out['binary_action_counts']) else None
    out['mixed_binary_slot_fraction']=out['mixed_binary_slots']/out['sampled_slots'] if out['sampled_slots'] else None
    out['actual_policy_wait_events']=out['policy_wait_minutes']/15
    return out


def audit_saved(source):
    """Arithmetic and original-ID audit over saved real execution only."""
    errors=[];warnings=[];counts=Counter();maxima=defaultdict(float)
    for path in sorted(source.rglob('result.json')):
        relative=str(path.relative_to(source)).replace('\\','/')
        result=json.loads(path.read_text(encoding='utf-8'));summary=result['summary'];steps=result['steps']
        kind=path.parent.parent.name;counts['saved_accounts']+=1;counts['actual_saved_physical_steps']+=len(steps)
        counts['background_accounts' if kind=='backgrounds' else 'workload_accounts']+=1
        def require(condition,message):
            if not condition:errors.append(relative+': '+message)
        verifier=path.with_name('verification.json')
        if verifier.exists():
            audit=json.loads(verifier.read_text(encoding='utf-8'))
            require(not audit['errors'],'saved V2 verifier errors')
            counts['saved_verifiers']+=1
        else:warnings.append(relative+': verification not yet saved')
        require(len(steps)==288,'saved account does not contain288 physical slots')
        require([row['minute'] for row in steps]==list(range(0,len(steps)*15,15)),'physical step grid mismatch')
        for actual in steps:
            recomputed=actual['total_kw']*.25*actual['ci_g_per_kwh']/1000
            maxima['per_step_carbon_error_kg']=max(maxima['per_step_carbon_error_kg'],abs(actual['carbon_kg']-recomputed))
            require(abs(actual['carbon_kg']-recomputed)<1e-9,'physical carbon integral mismatch')
        maxima['episode_carbon_sum_error_kg']=max(maxima['episode_carbon_sum_error_kg'],
            abs(math.fsum(row['carbon_kg'] for row in steps)-summary['carbon_kg']))
        if kind=='backgrounds':continue
        require(not summary['unfinished'] and not summary['deadline_censored'],'incomplete responsibility retained')
        require(not summary['reference_guard_failure_count'],'CG failure retained')
        tail=[row for row in steps if row['minute']>=3960]
        require(len(tail)==24 and all(row['pending']==row['running']==0 and row['supply_requested_c']==20
                    for row in tail),'invalid common idle tail')
        decision_path=path.with_name('decisions.jsonl')
        if not decision_path.exists():warnings.append(relative+': decisions not saved');continue
        records=[json.loads(line) for line in decision_path.read_text(encoding='utf-8').splitlines()]
        require(len(records)==len(steps),'one physical record required per actual15min step')
        starts=Counter(identity for record in records for identity in record['actual_started'])
        policy=Counter(identity for record in records for identity in record['effective_policy_wait'])
        capacity=Counter(identity for record in records for identity in record['capacity_wait'])
        events=[identity for record in records for identity in record['new_irreversible_miss_ids']]
        require(len(events)==len(set(events)),'irreversible miss ID penalized twice')
        misses={row['task_id'] for row in result['tasks'] if row['deadline_missed']}
        if not summary['unfinished']:
            require(set(events)==misses,'irreversible miss IDs differ from official raw-finish result')
        for row in result['tasks']:
            identity=row['task_id'];expected_started=int(row.get('start_minute') is not None)
            require(starts[identity]==expected_started,'per-ID actual start count mismatch')
            require(policy[identity]*15==row['policy_wait_minutes'] and capacity[identity]*15==row['capacity_wait_minutes'],
                    'per-ID policy/capacity wait mismatch')
        for actual,record in zip(steps,records):
            starts_now=set(record['actual_started']);policy_now=set(record['effective_policy_wait']);capacity_now=set(record['capacity_wait'])
            require(not(starts_now&policy_now or starts_now&capacity_now or policy_now&capacity_now),'actual start/wait categories overlap')
            require(len(starts_now)==actual['started'] and len(policy_now)==actual['policy_wait_tasks']
                    and len(capacity_now)==actual['capacity_wait_tasks'],'actual per-slot event count mismatch')
        if not records or 'reward' not in records[0]:continue
        counts['reward_accounts']+=1;terms=[];increments=[]
        for actual,record in zip(steps,records):
            require(record['row']==actual,'policy record differs from actual V2 row')
            incremental=actual['carbon_kg']-record['idle_carbon_kg']
            expected=-incremental/100.-.01*len(record['new_irreversible_miss_ids'])
            maxima['step_reward_error']=max(maxima['step_reward_error'],abs(expected-record['reward']))
            require(abs(expected-record['reward'])<1e-12,'one-step actual reward mismatch')
            terms.append(record['reward']);increments.append(incremental)
            if 'choices' not in record:
                require(record['policy_sampled'] or (record['selected_action']==0 and record['log_prob'] is None),
                        'macro forced slot has fake sampled probability')
                continue
            counts['task_joint_physical_slots']+=1
            choices=record['choices'];counts['task_binary_decisions']+=len(choices)
            require(record['value_state'][27:]==[0.]*21,'critic query depends on current task/prefix')
            require(len({choice['task_id'] for choice in choices})==len(choices),'candidate consumed twice in same slot')
            require(record['proposed']==[choice['task_id'] for choice in choices if choice['action']==1],
                    'executed original proposal differs from binary action path')
            approved=set(record['approved']);require(approved<=set(record['proposed']),'approval outside original proposal')
            require(set(record['effective_policy_wait'])<=approved,'voluntary wait outside approved defer set')
            expected_log=math.fsum(choice['log_prob'] for choice in choices) if choices else None
            if choices:
                counts['task_joint_sampled_slots']+=1
                error=abs(expected_log-record['joint_log_prob'])
                maxima['joint_log_probability_sum_error']=max(maxima['joint_log_probability_sum_error'],error)
                require(error<1e-10,'joint log-probability is not sum of original conditional logs')
                keys=[(choice['state'][38],choice['state'][37],choice['task_id']) for choice in choices]
                require(keys==sorted(keys),'frozen candidate sequence not in EDF order')
            else:require(record['joint_log_prob'] is None,'empty/forced slot pretends to sample joint proposal')
            prefix=[];rejected=0;global_values=record['value_state'][:27]
            for index,choice in enumerate(choices):
                state=choice['state'];action=choice['action'];n=len(choices)
                require(len(state)==48 and state[-1]==1 and state[:27]==global_values,'actor query/global snapshot mismatch')
                require(type(action) is int and action in (0,1),'proposal not original binary action')
                p=choice['defer_probability'];require(0<=p<=1,'invalid conditional probability')
                selected=p if action else 1-p
                require(selected>0 and abs(math.log(selected)-choice['log_prob'])<1e-9,'conditional log probability mismatch')
                expected_prefix=[n/1000,index/n,len(prefix)/n,
                    *(math.fsum(previous['state'][27+axis] for previous in prefix) for axis in range(3)),
                    math.fsum(previous['state'][28]*previous['state'][36] for previous in prefix),rejected/n]
                require(all(abs(a-b)<1e-10 for a,b in zip(state[39:47],expected_prefix)),
                        'teacher-forced approved prefix differs from final original-path approval')
                if choice['task_id'] in approved:prefix.append(choice)
                elif action:rejected+=1
        if not summary['unfinished']:
            error=abs(math.fsum(terms)+math.fsum(increments)/100.+.01*len(misses))
            maxima['reward_conservation_error']=max(maxima['reward_conservation_error'],error)
            require(error<1e-10,'complete-episode actual reward conservation mismatch')
        maxima['reward_summary_sum_error']=max(maxima['reward_summary_sum_error'],abs(math.fsum(terms)-summary['reward_sum']))
        maxima['incremental_summary_sum_error_kg']=max(maxima['incremental_summary_sum_error_kg'],
            abs(math.fsum(increments)-summary['common_incremental_carbon_kg']))
    partial=[]
    for path in sorted(source.rglob('error.json')):
        error=json.loads(path.read_text(encoding='utf-8'))
        partial.append(dict(path=str(path.relative_to(source)).replace('\\','/'),error=error.get('error'),
            minute=error.get('minute'),actual_steps=len(error.get('past_steps',[])),
            partial_task_decisions=len(error.get('partial_task_decisions',error.get('current_choices',[])))))
    orphan_decisions=[]
    for path in sorted(source.rglob('decisions.jsonl')):
        if path.with_name('result.json').exists():continue
        records=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        orphan_decisions.append(dict(path=str(path.relative_to(source)).replace('\\','/'),
            saved_physical_records=len(records),last_saved_minute=records[-1].get('minute') if records else None,
            saved_records_with_actual_row=sum('row' in record for record in records),
            full_window_account_available=False))
        counts['partial_saved_decision_records']+=len(records)
        warnings.append(str(path.relative_to(source))+': retained incomplete account, not a72h evaluation')
    return dict(schema='task-ppo-v1-saved-ledger-audit',counts=dict(counts),maximum_errors=dict(maxima),
        errors=errors,warnings=warnings,partial_error_accounts=partial,passed=not errors,
        partial_decision_accounts=orphan_decisions,
        all_observed_accounts_have_verified_full_window=not errors and not warnings and not partial and not orphan_decisions,
        scope='Only saved arithmetic, IDs and teacher-forced path; original V2 saved verifiers plus runtime preview assertions. No dispatch replay.')


def pearson(left,right):
    if len(left)<2:return None
    a=statistics.fmean(left);b=statistics.fmean(right)
    numerator=math.fsum((x-a)*(y-b) for x,y in zip(left,right))
    denominator=math.sqrt(math.fsum((x-a)**2 for x in left)*math.fsum((y-b)**2 for y in right))
    return numerator/denominator if denominator else None


def task_diagnostics(source):
    """Feature-controlled network queries over saved final-policy contexts."""
    checkpoint=source/'checkpoints/final.pt';initial_path=source/'checkpoints/initial.pt'
    if not checkpoint.exists() or not initial_path.exists():
        return dict(available=False,reason='Final/initial checkpoint unavailable; no completed feature probe fabricated.')
    import torch
    from .joint_ppo import JointPPOTrainer
    torch.set_num_threads(1)
    initial=JointPPOTrainer.load(initial_path).model.eval();final=JointPPOTrainer.load(checkpoint).model.eval()
    groups={}
    for cap in ('OM','OH'):
        ranges_initial=[];ranges_final=[];raw_spreads=[];controlled_initial=[];controlled_final=[]
        centered_initial=[];centered_final=[];centered_tasks=[]
        task_features=[];slot_count=multi_count=raw_mixed=controlled_initial_mixed=controlled_final_mixed=0
        actions=Counter();raw_probabilities=[]
        for path in sorted((source/'evaluations').glob(f'*__{cap}/TaskPPO-final/decisions.jsonl')):
            for line in path.open(encoding='utf-8'):
                record=json.loads(line);choices=record['choices']
                if not choices:continue
                slot_count+=1;actions.update(choice['action'] for choice in choices)
                raw=[choice['defer_probability'] for choice in choices];raw_probabilities.extend(raw)
                raw_mixed+=len({choice['action'] for choice in choices})>1
                first=choices[0]['state']
                # Hold global and prefix equal to the first candidate context;
                # only observed task attributes vary. This is not a schedule.
                contexts=[[*first[:27],*choice['state'][27:39],*first[39:47],1.] for choice in choices]
                with torch.no_grad():
                    initial_logits,_=initial(contexts);final_logits,_=final(contexts)
                    p0=initial_logits.softmax(-1)[:,1].tolist();p1=final_logits.softmax(-1)[:,1].tolist()
                controlled_initial.extend(p0);controlled_final.extend(p1)
                task_features.extend(choice['state'][27:39] for choice in choices)
                if len(choices)>1:
                    multi_count+=1;raw_spreads.append(max(raw)-min(raw))
                    ranges_initial.append(max(p0)-min(p0));ranges_final.append(max(p1)-min(p1))
                    controlled_initial_mixed+=min(p0)<.5<=max(p0)
                    controlled_final_mixed+=min(p1)<.5<=max(p1)
                    mean0=statistics.fmean(p0);mean1=statistics.fmean(p1)
                    attributes=[choice['state'][27:39] for choice in choices]
                    feature_means=[statistics.fmean(row[i] for row in attributes) for i in range(12)]
                    centered_initial.extend(value-mean0 for value in p0)
                    centered_final.extend(value-mean1 for value in p1)
                    centered_tasks.extend([value-feature_means[i] for i,value in enumerate(row)] for row in attributes)
        groups[cap]=dict(sampled_physical_slots=slot_count,multi_candidate_slots=multi_count,
            candidate_exposures=len(raw_probabilities),raw_binary_action_counts=[actions[0],actions[1]],
            raw_mixed_binary_slots=raw_mixed,raw_mixed_slot_fraction=raw_mixed/slot_count if slot_count else None,
            raw_probability_min=min(raw_probabilities,default=None),raw_probability_max=max(raw_probabilities,default=None),
            raw_probability_mean=statistics.fmean(raw_probabilities) if raw_probabilities else None,
            raw_within_slot_spread_mean=statistics.fmean(raw_spreads) if raw_spreads else None,
            same_context_initial_range_mean=statistics.fmean(ranges_initial) if ranges_initial else None,
            same_context_final_range_mean=statistics.fmean(ranges_final) if ranges_final else None,
            same_context_initial_range_max=max(ranges_initial,default=None),same_context_final_range_max=max(ranges_final,default=None),
            same_context_initial_probability_mean=statistics.fmean(controlled_initial) if controlled_initial else None,
            same_context_final_probability_mean=statistics.fmean(controlled_final) if controlled_final else None,
            same_context_mean_abs_probability_change_from_initial=(statistics.fmean(abs(a-b) for a,b in zip(controlled_initial,controlled_final))
                if controlled_initial else None),
            same_context_initial_mixed_argmax_slots=controlled_initial_mixed,
            same_context_final_mixed_argmax_slots=controlled_final_mixed,
            same_context_initial_range_over_1e_4_slots=sum(value>1e-4 for value in ranges_initial),
            same_context_final_range_over_1e_4_slots=sum(value>1e-4 for value in ranges_final),
            task_feature_pearson_initial={name:pearson([row[i] for row in task_features],controlled_initial)
                for i,name in enumerate(TASK_FEATURES)},
            task_feature_pearson_final={name:pearson([row[i] for row in task_features],controlled_final)
                for i,name in enumerate(TASK_FEATURES)},
            within_slot_centered_task_feature_pearson_initial={name:pearson([row[i] for row in centered_tasks],centered_initial)
                for i,name in enumerate(TASK_FEATURES)},
            within_slot_centered_task_feature_pearson_final={name:pearson([row[i] for row in centered_tasks],centered_final)
                for i,name in enumerate(TASK_FEATURES)})
    return dict(available=True,initial_checkpoint_sha256=sha(initial_path),final_checkpoint_sha256=sha(checkpoint),
        by_capacity=groups,interpretation=['Original recorded final-policy action mixtures are actual deterministic evaluation behavior.',
            'Feature probes freeze each slot first-candidate global27/prefix8 and substitute only real saved task12 attributes.',
            'Initial random networks also respond to task attributes; nonzero probability ranges do not demonstrate learned beneficial discrimination.',
            'Pearson associations pool correlated recorded task exposures and are descriptive, not independent tests, causality or scheduling benefit.',
            'No simulator step, reward calculation, optimization, model selection or future information is used by this probe.'])


def execution_snapshot(source):
    for name in ('execution.json','interruption.json','interrupted.json'):
        path=source/name
        if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    return dict(status='not-finalized',note='No finalized execution or interruption record exists.')


def generate(source,output,interrupted_source=None):
    source=Path(source).resolve();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if not source.is_relative_to((ROOT/'research_notes/task-ppo-v1').resolve()):
        raise ValueError('Evidence input must be saved local Task-PPO output, never raw/sealed scenario data')
    execution=execution_snapshot(source)
    training=read_csv(source/'training-metrics.csv');evaluation=read_csv(source/'evaluation-summary.csv')
    for name in ('training-metrics.csv','evaluation-summary.csv'):
        if (source/name).exists():shutil.copyfile(source/name,output/name)
        else:(output/name).write_text('',encoding='utf-8')
    audit=audit_saved(source);dump(output/'ledger-audit.json',audit)
    grouped={cap:{method:summarize([row for row in evaluation if row['capacity']==cap and row['method']==method])
                 for method in METHODS} for cap in ('OM','OH')}
    pairs=[];terminal_groups=[]
    for cap in ('OM','OH'):
        for fragment in sorted(set(row['fragment'] for row in evaluation if row['capacity']==cap)):
            rows={row['method']:row for row in evaluation if row['capacity']==cap and row['fragment']==fragment}
            temperatures=[number(row,'inlet_final_c') for row in rows.values()]
            energies=[number(row,'it_kwh') for row in rows.values()]
            terminal_groups.append(dict(fragment=fragment,capacity=cap,methods_present=list(rows),complete_five_method_group=len(rows)==5,
                max_temperature_spread_k=max(temperatures)-min(temperatures),max_it_energy_spread_kwh=max(energies)-min(energies)))
            for method,baseline in [(method,baseline) for method in METHODS[1:] for baseline in METHODS[:2] if method!=baseline]+[
                    ('TaskPPO-final','MacroPPO-history'),('TaskPPO-final','TaskPPO-initial')]:
                if method not in rows or baseline not in rows:continue
                a,b=rows[method],rows[baseline];delta_temperature=number(a,'inlet_final_c')-number(b,'inlet_final_c')
                pairs.append(dict(fragment=fragment,capacity=cap,method=method,baseline=baseline,
                    delta_carbon_kg=number(a,'carbon_kg')-number(b,'carbon_kg'),
                    delta_incremental_carbon_kg=number(a,'common_incremental_carbon_kg')-number(b,'common_incremental_carbon_kg'),
                    delta_misses=int(a['deadline_missed'])-int(b['deadline_missed']),delta_dmr=number(a,'dmr')-number(b,'dmr'),
                    delta_jct_minutes=number(a,'mean_jct_minutes')-number(b,'mean_jct_minutes'),
                    delta_policy_wait_minutes=number(a,'policy_wait_minutes')-number(b,'policy_wait_minutes'),
                    delta_capacity_wait_minutes=number(a,'capacity_wait_minutes')-number(b,'capacity_wait_minutes'),
                    delta_it_kwh=number(a,'it_kwh')-number(b,'it_kwh'),delta_final_temperature_k=delta_temperature,
                    delta_thermal_storage_kwh=number(a,'thermal_capacity_kj_k')*delta_temperature/3600.))
    with (output/'paired-differences.csv').open('w',encoding='utf-8',newline='') as stream:
        if pairs:
            writer=csv.DictWriter(stream,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    metrics={}
    for field in ('policy_loss','value_loss','entropy','approx_kl','clip_fraction','actor_parameter_delta_l2',
                  'backbone_parameter_delta_l2','parameter_delta_l2','joint_logratio_min','joint_logratio_max'):
        values=[number(row,field) for row in training if number(row,field) is not None]
        if values:metrics[field]=dict(first=values[0],last=values[-1],minimum=min(values),maximum=max(values),mean=statistics.fmean(values))
    interruption=None;interruption_paths=[]
    if interrupted_source is not None:
        interrupted_source=Path(interrupted_source).resolve()
        if interrupted_source.exists() and interrupted_source!=source:
            if not interrupted_source.is_relative_to((ROOT/'research_notes/task-ppo-v1').resolve()):
                raise ValueError('Engineering interruption input outside saved local Task-PPO outputs')
            interrupted_audit=audit_saved(interrupted_source)
            interruption=dict(root=str(interrupted_source.relative_to(ROOT)).replace('\\','/'),
                execution=execution_snapshot(interrupted_source),audit=interrupted_audit,
                included_in_training_or_comparison_averages=False,
                interpretation='Engineering interruption retained separately; no completed main-pilot episode or performance sample fabricated.')
            interruption_paths=sorted(path for path in interrupted_source.rglob('*') if path.is_file())
    summary=dict(schema='task-ppo-v1-read-only-evidence',execution=execution,
        expected_if_completed=dict(training_episodes=32,evaluation_episodes=40,background_episodes=32,saved_accounts=104,physical_steps=29952),
        actual=dict(training_csv_rows=len(training),evaluation_csv_rows=len(evaluation),**audit['counts']),
        training_by_capacity={cap:summarize([row for row in training if row['capacity']==cap]) for cap in ('OM','OH')},
        training_pooled=summarize(training),evaluation_by_capacity=grouped,
        evaluation_pooled={method:summarize([row for row in evaluation if row['method']==method]) for method in METHODS},
        terminal_groups=terminal_groups,training_update_statistics=metrics,
        task_discrimination=task_diagnostics(source),
        engineering_interruption=interruption,
        limitations=['All actual rows are retained, including failed or partial execution evidence; expected counts never replace observed counts.',
            'Same Train-internal holdout across capacities; capacity variants and repeated episodes are not independent source samples.',
            'MacroPPO-history was trained on OM only with different seed/state/entropy. It is historical context, not a single-variable action-space ablation.',
            'One seed, correlated development Train windows, no confidence intervals or claims of independent generalization.'])
    dump(output/'aggregate.json',summary)
    paths=sorted(path for path in source.rglob('*') if path.is_file())
    dump(output/'local-evidence-manifest.json',dict(root=str(source.relative_to(ROOT)).replace('\\','/'),source_sha=execution.get('source_sha'),
        files=[dict(path=str(path.relative_to(source)).replace('\\','/'),bytes=path.stat().st_size,sha256=sha(path)) for path in paths],
        engineering_interruption_files=[dict(path=str(path.relative_to(interrupted_source)).replace('\\','/'),
            bytes=path.stat().st_size,sha256=sha(path)) for path in interruption_paths]))
    print(json.dumps(dict(status=execution.get('status'),training_rows=len(training),evaluation_rows=len(evaluation),
        saved_accounts=audit['counts'].get('saved_accounts',0),ledger_errors=len(audit['errors']),output=str(output)),indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default=str(ROOT/'research_notes/task-ppo-v1/pilot-002'))
    parser.add_argument('--output',default=str(ROOT/'docs/research/task-ppo-v1/results'))
    parser.add_argument('--interrupted-input',default=str(ROOT/'research_notes/task-ppo-v1/pilot-001'))
    args=parser.parse_args();generate(args.input,args.output,args.interrupted_input)
