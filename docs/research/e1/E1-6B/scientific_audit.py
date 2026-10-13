"""Independent offline ledger/resource/integral audit of the 57 E1-6B results.

No policy/model execution, fitting, tuning, or Test input. A completed campaign
is mandatory. This checks accounting and execution evidence, not model truth.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
from pathlib import Path
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
NOTES=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from experiments.e1.contracts import content_hash, file_hash, write_json

STEP=pd.Timedelta(minutes=15)
RESOURCES=(('cores','cores_req'),('gpus','gpu_req'),('mem','mem_req'))
BASELINES=('edf_backfill','edf_carbon_threshold','cars_risk_only','task_risk_full')
CORE_FILES=('ledger.jsonl','events.jsonl','policy_ledger.jsonl','policy_events.jsonl')
READ_FILES=CORE_FILES+('policy_intervals.jsonl','intervals.csv','summary.json','policy_summary.json')


def require(value,message):
    if not value:raise ValueError(message)


def near(actual,expected,message,absolute=1e-9):
    require(math.isfinite(float(actual)) and math.isclose(float(actual),float(expected),
        rel_tol=1e-12,abs_tol=absolute),message+f': {actual!r} != {expected!r}')


@lru_cache(maxsize=65536)
def stamp(value):
    result=pd.Timestamp(value)
    require(result.tzinfo is not None,'UTC time must be explicit')
    return result.tz_convert('UTC')


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def jsonl(path):
    with Path(path).open(encoding='utf8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def csv_rows(path):
    with Path(path).open(encoding='utf8',newline='') as stream:return list(csv.DictReader(stream))


def lists(value):return json.loads(value) if isinstance(value,str) else list(value)


def write_csv(path,rows):
    with Path(path).open('x',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader();writer.writerows(rows)


def audit_method(specs,samples,capacity,arrival_start,arrival_end,common_end,folder,common_row):
    """Reconstruct occupancy from immutable tasks plus observed starts/releases.

    No executor/metric function is used to derive the results below. Nominal
    completion decides SLA; ceil-to-grid completion decides physical release.
    """
    folder=Path(folder);arrival_start=stamp(arrival_start);arrival_end=stamp(arrival_end);common_end=stamp(common_end)
    ledger=jsonl(folder/'ledger.jsonl');events=jsonl(folder/'events.jsonl')
    pe=jsonl(folder/'policy_events.jsonl');pi=jsonl(folder/'policy_intervals.jsonl')
    pl=jsonl(folder/'policy_ledger.jsonl');intervals=csv_rows(folder/'intervals.csv');summary=read(folder/'summary.json')
    require(len(ledger)==len(specs) and len({r['task_id'] for r in ledger})==len(specs), 'Unique task ownership failed')
    rows={r['task_id']:r for r in ledger}
    require(set(rows)==set(specs),'Task ID set differs from frozen cohort')
    occupancy=defaultdict(list);pre_pending=defaultdict(list);post_pending=defaultdict(list)
    arrivals=defaultdict(list);starts=defaultdict(list);releases=defaultdict(list);misses=set();deadline_boundary={}
    expected_events=Counter();hours=[];jct=[];wait=[];order_keys={}
    times=list(pd.date_range(arrival_start,common_end,freq=STEP,inclusive='left'))
    for task_id,spec in specs.items():
        row=rows[task_id]
        require({key:row[key] for key in spec}==spec,'Immutable task feature changed: '+task_id)
        arrival=stamp(row['arrival_time']);start=stamp(row['start_time']);finish=stamp(row['finish_time'])
        completion=stamp(row['completion_observed_at']);deadline=arrival+pd.Timedelta(minutes=row['duration']*row['sla_multiplier'])
        require(arrival_start<=arrival<arrival_end and arrival==arrival.floor('15min'), 'Arrival outside finite grid cohort')
        require(start>=arrival and start==start.floor('15min'),'Invalid task start')
        require(finish==start+pd.Timedelta(minutes=row['duration']),'Task duration/nominal finish changed')
        require(completion==finish.ceil('15min') and completion<=common_end,'Physical completion/release grid differs')
        require(stamp(row['sla_deadline'])==deadline,'Synthetic deadline changed')
        order_keys[task_id]=(deadline,arrival,task_id)
        met=finish<=deadline
        require(row['state']=='completed' and row['sla_met'] is met and
            row['deadline_outcome']==('met' if met else 'missed'),'Final deadline outcome differs')
        require(row['dc_id']==row['dest_dc_id']==1,'Local DC assignment differs')
        for key in ('transmission_delay_seconds','transmission_energy_kwh','transmission_carbon_kg'):
            near(row[key],0.,'Local transmission must be zero',absolute=0.)
        arrivals[arrival].append(task_id);starts[start].append(task_id);releases[completion].append(task_id)
        for kind,time in (('arrived',arrival),('enqueued',arrival),('started',start),('completed_resources_released',completion)):
            expected_events[time.isoformat(),task_id,kind]+=1
        ai=(arrival-arrival_start)//STEP;si=(start-arrival_start)//STEP;ri=(completion-arrival_start)//STEP
        for index in range(si,ri):occupancy[times[index]].append(task_id)
        for index in range(ai,si+1):pre_pending[times[index]].append(task_id)
        for index in range(ai,si):post_pending[times[index]].append(task_id)
        duration_hours=(completion-start).total_seconds()/3600
        hours.append(tuple(row[demand]*duration_hours for _,demand in RESOURCES))
        jct.append((finish-arrival).total_seconds()/60);wait.append((start-arrival).total_seconds()/60)
        if not met:
            misses.add(task_id);deadline_boundary[task_id]=deadline.ceil('15min')
    actual_events=Counter((e['time'],e['task_id'],e['event']) for e in events)
    require(actual_events==expected_events,'Arrival/enqueue/start/complete event conservation failed')
    require(all(stamp(e['time'])<arrival_end for e in events if e['event']=='arrived'),'New arrivals during Drain')
    require(all(stamp(events[i]['time'])<=stamp(events[i+1]['time']) for i in range(len(events)-1)),
            'Event times are not monotonic')
    # Arrival records precede release, then enqueue/start; all releases precede allocation.
    event_boundaries=defaultdict(list)
    for event in events:event_boundaries[event['time']].append(event['event'])
    ranks={'arrived':0,'completed_resources_released':1,'enqueued':2,'started':3}
    for kinds in event_boundaries.values():
        require([ranks[k] for k in kinds]==sorted(ranks[k] for k in kinds),'Boundary execution event order differs')
    require(len(intervals)==len(pi)==len(times),'Interval/policy coverage differs')
    policy_by_time=defaultdict(list)
    for row in pe:policy_by_time[stamp(row['time'])].append(row)
    defer_counts=Counter();cap_counts=Counter();sums=defaultdict(list);peak=[0.,0.,0.];max_resource_error=0.
    for index,(time,interval,detail) in enumerate(zip(times,intervals,pi)):
        require(stamp(interval['interval_start'])==time and stamp(interval['interval_end'])==time+STEP and
            stamp(detail['time'])==time,'Interval boundary/grid gap')
        near(interval['interval_hours'],.25,'Energy integration must span 15 minutes',absolute=0.)
        sample=samples[time]
        for key in ('ci_g_per_kwh','temperature_c','wet_bulb_c'):
            near(interval[key],sample[key],'Frozen environmental sample differs',absolute=1e-12)
        running=set(occupancy[time]);before=set(pre_pending[time]);pending=set(post_pending[time])
        require(len(detail['running_ids'])==len(set(detail['running_ids'])) and set(detail['running_ids'])==running,
            'Running queue differs from ledger')
        require(len(detail['pending_end_ids'])==len(set(detail['pending_end_ids'])) and set(detail['pending_end_ids'])==pending,
            'Pending queue differs from ledger')
        require(len(detail['decision_pending_ids'])==len(before) and set(detail['decision_pending_ids'])==before,
            'Decision did not include the whole pending queue')
        ordered=sorted(before,key=order_keys.__getitem__)
        require(detail['ordered_waiting_ids']==ordered,'EDF whole-queue order/tie-break differs')
        decisions=policy_by_time[time]
        require(len(decisions)==len(ordered) and [r['task_id'] for r in decisions]==ordered and
            [r['decision_rank'] for r in decisions]==list(range(len(ordered))),'Candidate decision sequence differs')
        old_running=running-set(starts[time]);allocated=list(old_running);by_kind=defaultdict(list)
        for decision in decisions:
            task_id=decision['task_id'];kind=decision['event'];can_fit=all(
                math.fsum([rows[k][demand] for k in allocated]+[rows[task_id][demand]])<=limit
                for (_,demand),limit in zip(RESOURCES,capacity))
            require(decision['can_fit_at_decision'] is can_fit,'Capacity feasibility flag differs')
            for (resource,demand),limit in zip(RESOURCES,capacity):
                near(decision['available_before'][resource],limit-math.fsum(rows[k][demand] for k in allocated),
                    'Candidate available resources differ')
            require(kind in ('started','capacity_waited','policy_deferred'),'Unexpected decision event')
            require((kind=='capacity_waited') is (not can_fit),'Capacity-first waiting attribution differs')
            if kind=='started':allocated.append(task_id)
            elif kind=='capacity_waited':cap_counts[task_id]+=1
            else:defer_counts[task_id]+=1
            by_kind[kind].append(task_id)
        for field,kind in (('started_ids','started'),('policy_deferred_ids','policy_deferred'),
                           ('capacity_waiting_ids','capacity_waited')):
            require(detail[field]==by_kind[kind],'Recorded real decision set differs')
        require(set(detail['started_ids'])==set(starts[time]),'Started task IDs differ from ledger')
        require(set(allocated)==running,'Within-step allocation resource update differs')
        require(int(interval['pending'])==len(pending) and int(interval['running'])==len(running),
            'Interval queue counts differ')
        for dimension,((resource,demand),limit) in enumerate(zip(RESOURCES,capacity)):
            used=math.fsum(rows[k][demand] for k in running)
            require(used<=limit,'CPU/GPU/MEM oversubscription')
            error=abs(float(interval['used_'+resource])-used);max_resource_error=max(max_resource_error,error)
            near(interval['used_'+resource],used,'Ledger/interval resource occupancy differs')
            near(interval['available_'+resource],limit-used,'Resource conservation differs')
            near(interval[resource+'_fraction'],used/limit,'Resource utilization units differ',absolute=1e-12)
            peak[dimension]=max(peak[dimension],used/limit)
        near(interval['total_power_kw'],float(interval['it_power_kw'])+float(interval['hvac_power_kw']),
            'Total IT+HVAC power differs')
        for component in ('it','hvac','total'):
            energy=float(interval[component+'_power_kw'])*.25
            near(interval[component+'_energy_kwh'],energy,'Power-to-energy integration differs')
            sums[component+'_energy_kwh'].append(energy)
        carbon=float(interval['total_power_kw'])*.25*sample['ci_g_per_kwh']/1000
        near(interval['carbon_kg'],carbon,'g/kWh-to-kg carbon integration differs')
        sums['carbon_kg'].append(carbon)
        near(interval['transmission_energy_kwh'],0.,'Transmission energy must remain zero',absolute=0.)
    policy_ledger={r['task_id']:r for r in pl}
    require(len(pl)==len(specs) and set(policy_ledger)==set(specs),'Policy task ledger identity differs')
    for task_id,row in rows.items():
        require(row['wait_intervals']==cap_counts[task_id] and
            policy_ledger[task_id]['capacity_wait_intervals']==cap_counts[task_id] and
            policy_ledger[task_id]['policy_defer_intervals']==defer_counts[task_id],'Task waiting counters differ')
        near(policy_ledger[task_id]['observed_pending_minutes'],(stamp(row['start_time'])-stamp(row['arrival_time'])).total_seconds()/60,
            'Task observed waiting duration differs')
    require(summary['arrived']==summary['completed']==len(specs) and summary['pending']==summary['running']==
        summary['deadline_missed_unfinished']==summary['deadline_right_censored']==0,'Final completion/censoring differs')
    require(summary['deadline_missed_completed']==len(misses) and summary['deadline_met']==len(specs)-len(misses),
        'Deadline summary count differs')
    require(summary['intervals']==len(times) and stamp(summary['observation_end'])==common_end,'Summary observation scope differs')
    expected_drain=max(arrival_end,max(releases))
    require(stamp(summary['drained_at'])==expected_drain and
        stamp(common_row['own_observation_end'])==expected_drain,'Own Drain boundary differs')
    near(common_row['drain_hours'],(expected_drain-arrival_end).total_seconds()/3600,'Drain duration differs')
    for key,unit in (('carbon_unit','kgCO2e'),('energy_unit','kWh'),('power_unit','kW')):
        require(summary[key]==common_row[key]==unit,'Scientific units differ')
    for key,values in sums.items():
        near(summary[key],math.fsum(values),'Summary energy/carbon differs')
        near(common_row[key],math.fsum(values),'Common energy/carbon differs')
    require(int(common_row['deadline_missed_completed'])==len(misses),'Common miss count differs')
    near(common_row['deadline_violation_rate_observed'],len(misses)/len(specs),'Common DMR differs',absolute=1e-14)
    near(common_row['mean_jct_minutes_completed'],math.fsum(jct)/len(jct),'Common JCT differs')
    near(common_row['mean_wait_minutes_started'],math.fsum(wait)/len(wait),'Common waiting mean differs')
    for i,(resource,_) in enumerate(RESOURCES):
        near(common_row[resource+'_hours'],math.fsum(values[i] for values in hours),'Integrated task resource-hours differ')
        near(common_row[resource+'_fraction_mean'],math.fsum(float(r[resource+'_fraction']) for r in intervals)/len(times),
            'Mean resource utilization differs',absolute=1e-12)
        near(common_row[resource+'_fraction_max'],peak[i],'Peak resource utilization differs',absolute=1e-12)
    near(common_row['pending_mean'],math.fsum(len(post_pending[t]) for t in times)/len(times),'Mean pending count differs')
    require(int(common_row['pending_max'])==max(len(post_pending[t]) for t in times),'Peak pending count differs')
    require(int(common_row['capacity_wait_intervals'])==sum(cap_counts.values()) and
        int(common_row['policy_defer_intervals'])==sum(defer_counts.values()),'Common wait counters differ')
    actor=folder/'actions.csv';reward_count=None
    if actor.exists():
        actions=csv_rows(actor);seen=set();own_end=stamp(common_row['own_observation_end'])
        require(len(actions)==(own_end-arrival_start)//STEP,'Action tape must cover its own Drain exactly')
        for index,action in enumerate(actions):
            time=arrival_start+index*STEP;detail=pi[index]
            require(stamp(action['time'])==time and stamp(action['reward_event_time'])==time+STEP,
                'Actor action/reward boundary differs')
            require(int(action['pending'])==len(pre_pending[time]) and
                int(action['running'])==len(occupancy[time])-len(starts[time]),'Actor observed queue counts differ')
            near(action['ci_g_per_kwh'],samples[time]['ci_g_per_kwh'],'Actor current CI differs',absolute=1e-12)
            for action_field,detail_field in (('started_ids','started_ids'),('deferred_ids','policy_deferred_ids'),
                                             ('capacity_waiting_ids','capacity_waiting_ids')):
                require(lists(action[action_field])==detail[detail_field],'Actor log real decision set differs')
            new=lists(action['new_deadline_miss_ids'])
            require(len(new)==len(set(new)) and not seen.intersection(new),'Repeated first-miss reward penalty')
            expected={task_id for task_id,boundary in deadline_boundary.items() if boundary==time+STEP}
            require(set(new)==expected,'First-miss reward set/time differs from actual deadlines')
            seen.update(new)
            # Default registered reward, no terminal truncation because this is full Drain.
            expected_reward=-(float(intervals[index]['carbon_kg'])/1000+
                int(intervals[index]['pending'])*15/15000+10*len(new)/1000)
            near(action['reward'],expected_reward,'Reward unit/normalization differs',absolute=1e-12)
        require(seen==misses,'Actor first-miss set is not the final miss set')
        reward_count=len(seen)
    return dict(method=folder.name,tasks=len(specs),completed=len(specs),deadline_misses=len(misses),
        first_miss_penalties=reward_count,intervals=len(times),carbon_kg=math.fsum(sums['carbon_kg']),
        it_energy_kwh=math.fsum(sums['it_energy_kwh']),hvac_energy_kwh=math.fsum(sums['hvac_energy_kwh']),
        total_energy_kwh=math.fsum(sums['total_energy_kwh']),maximum_resource_error=max_resource_error,
        peak_cpu_utilization=peak[0],peak_gpu_utilization=peak[1],peak_mem_utilization=peak[2],
        policy_defer_intervals=sum(defer_counts.values()),capacity_wait_intervals=sum(cap_counts.values()),
        complete_drain=True,task_features_exact=True,grid_release_correct=True,capacity_conserved=True,
        current_frozen_ci_integral_correct=True,no_arrivals_during_drain=True)


def historical_regression(new,old):
    """Compare fixed scientific trajectories, never old totals of another scope."""
    new=Path(new);old=Path(old)
    core={name:file_hash(new/name)==file_hash(old/name) for name in CORE_FILES}
    require(all(core.values()),'Frozen external baseline task/decision trajectory changed: '+new.name)
    new_end=stamp(read(new/'summary.json')['observation_end']);old_end=stamp(read(old/'summary.json')['observation_end'])
    shared=min(new_end,old_end)
    new_intervals=[r for r in csv_rows(new/'intervals.csv') if stamp(r['interval_start'])<shared]
    old_intervals=[r for r in csv_rows(old/'intervals.csv') if stamp(r['interval_start'])<shared]
    new_policy=[r for r in jsonl(new/'policy_intervals.jsonl') if stamp(r['time'])<shared]
    old_policy=[r for r in jsonl(old/'policy_intervals.jsonl') if stamp(r['time'])<shared]
    require(new_intervals==old_intervals and new_policy==old_policy,'Frozen baseline shared physical/queue prefix changed')
    exact=new_end==old_end
    if exact:
        require(file_hash(new/'intervals.csv')==file_hash(old/'intervals.csv') and
            file_hash(new/'policy_intervals.jsonl')==file_hash(old/'policy_intervals.jsonl'),
            'Same-end baseline interval bytes changed')
    return dict(method=new.name,old_end=old_end.isoformat(),new_end=new_end.isoformat(),shared_end=shared.isoformat(),
        same_common_end=exact,ledger_events_policy_ledger_policy_events_bytes_exact=True,
        shared_interval_rows_exact=True,shared_policy_intervals_exact=True,shared_intervals=len(new_intervals),
        shared_carbon_kg=math.fsum(float(r['carbon_kg']) for r in new_intervals),
        old_total_carbon_kg=read(old/'summary.json')['carbon_kg'],new_total_carbon_kg=read(new/'summary.json')['carbon_kg'],
        comparison_scope='full same endpoint' if exact else 'complete task/decision trajectory plus shared physical prefix')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',default=str(NOTES/'validation-campaign-001'))
    parser.add_argument('--output',default=str(NOTES/'scientific-audit-001'))
    args=parser.parse_args(argv);campaign=Path(args.campaign);out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    receipt=read(campaign/'campaign-receipt.json');manifest=read(campaign/'campaign.json')
    require(receipt.get('passed') is True and manifest['mode']=='evaluate' and
        manifest['formal_test_evaluated'] is False and manifest['validation_training'] is False and
        manifest['total_core_results']==57,'Completed development campaign is required')
    require(receipt['campaign_manifest_sha256']==file_hash(campaign/'campaign.json'), 'Campaign manifest hash differs')
    from experiments.e1.rl_training_utils import load_bundle
    from run_validation_campaign import expected_methods
    require(len(manifest['inputs'])==3 and {r['factor'] for r in manifest['inputs']}=={1.,1.5,2.} and
        len(receipt['results'])==3,'Pressure factor matrix differs')
    protected={campaign/'campaign.json',campaign/'campaign-receipt.json'};jobs=[];expected_hashes={}
    campaign_results={r['factor']:r for r in receipt['results']}
    for record in manifest['inputs']:
        require(record['split']=='validation' and record['factor'] in (1.,1.5,2.),'Only development Validation permitted')
        bundle_root=Path(record['path']);bundle=load_bundle(bundle_root,allowed_split='validation')
        require(bundle.workload.sha256==record['workload_sha256'] and bundle.environment.sha256==record['environment_sha256'],
            'Frozen bundle identity differs')
        protected.update(bundle_root/name for name in record['files'])
        folder=campaign/('load-'+record['label']);summary_rows=csv_rows(folder/'common_summary.csv')
        require(len(summary_rows)==19 and {r['policy'] for r in summary_rows}==set(expected_methods()),'Method matrix differs')
        protected.update(folder/name for name in ('common_summary.csv','fairness.json','output_hashes.json','input-identity.json',
            'source-freeze.json','scientific_hashes.json'))
        for name,digest in campaign_results[record['factor']]['files'].items():
            expected_hashes[folder/name]=digest
        output_hashes=read(folder/'output_hashes.json')
        fairness=read(folder/'fairness.json');common=stamp(fairness['common_end'])
        for row in summary_rows:
            method=folder/row['policy'];protected.update(method/name for name in READ_FILES)
            if (method/'actions.csv').exists():protected.add(method/'actions.csv')
            for name in READ_FILES+(('actions.csv',) if (method/'actions.csv').exists() else ()):
                expected_hashes[method/name]=output_hashes[row['policy']+'/'+name]
            old=ROOT/'research_notes/E1/E1-6A/campaign-001'/('validation-'+record['label'])/row['policy']
            if row['policy'] in BASELINES:protected.update(old/name for name in READ_FILES)
            jobs.append((record,bundle,row,method,old,common))
    require(len(jobs)==57,'Incomplete pressure/method audit matrix')
    hashes={str(path):file_hash(path) for path in sorted(protected)}
    require(all(file_hash(path)==digest for path,digest in expected_hashes.items()),
        'Completed campaign output hash binding differs')
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'read-file-hashes-before.json',hashes)
    write_json(out/'audit-plan.json',dict(schema='e1-6b-independent-scientific-audit-v1',
        campaign_manifest_sha256=file_hash(campaign/'campaign.json'),driver_sha256=file_hash(Path(__file__)),
        task_count=57,formal_test_evaluated=False,validation_training=False,
        formulas='nonpreemptive [start,ceil(finish/15min)) occupancy; energy=kW*0.25h; carbon=kWh*CI(g/kWh)/1000',
        absolute_numeric_tolerance=1e-9,relative_numeric_tolerance=1e-12))
    results=[];regressions=[]
    try:
        for record,bundle,row,method,old,common in jobs:
            specs={s.task_id:s.to_dict() for s in bundle.workload.tasks}
            samples={s.time:dict(ci_g_per_kwh=s.ci_g_per_kwh,temperature_c=s.temperature_c,wet_bulb_c=s.wet_bulb_c)
                     for s in bundle.environment.samples}
            result=audit_method(specs,samples,bundle.capacity,bundle.workload.start,bundle.workload.end,common,method,row)
            result['factor']=record['factor'];results.append(result)
            if row['policy'] in BASELINES:
                regression=historical_regression(method,old);regression['factor']=record['factor'];regressions.append(regression)
            print(json.dumps(dict(event='audited',factor=record['factor'],method=method.name,
                tasks=result['tasks'],misses=result['deadline_misses']),sort_keys=True),flush=True)
        after={str(path):file_hash(path) for path in sorted(protected)}
        require(after==hashes,'Read-only input/historical/current output bytes changed during audit')
        write_json(out/'read-file-hashes-after.json',after)
        write_csv(out/'method-audit.csv',results);write_csv(out/'historical-regression.csv',regressions)
        require(len(results)==57 and len(regressions)==12,'Final audit matrix differs')
        final=dict(passed=True,methods=57,historical_regressions=12,all_read_files_unchanged=True,
            no_simulation_or_training=True,formal_test_evaluated=False,
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            files={p.name:file_hash(p) for p in out.iterdir() if p.is_file()})
        write_json(out/'audit-receipt.json',final)
        print(json.dumps(final,sort_keys=True),flush=True)
        return final
    except Exception as exc:
        write_json(out/'failure.json',dict(passed=False,error_type=type(exc).__name__,message=str(exc),completed_methods=results))
        raise


if __name__=='__main__':main()
