"""Phase-specific finite-cohort statistics, with explicit censoring denominators."""
import math
from collections import Counter

import pandas as pd

from .contracts import utc_time
from .metrics import summarize


def phase_rows(result, cutoff):
    cutoff=utc_time(cutoff,True)
    capacity_waits=Counter(e['task_id'] for e in result.policy_events
                          if e['event']=='capacity_waited' and pd.Timestamp(e['time'])<cutoff)
    rows=[]
    for source in result.tasks:
        if pd.Timestamp(source['arrival_time']) >= cutoff:
            continue
        row=dict(source)
        # The upstream task counter counts capacity failures, not policy deferrals.
        row['wait_intervals']=capacity_waits[row['task_id']]
        completed=(row['completion_observed_at'] is not None
                   and pd.Timestamp(row['completion_observed_at']) <= cutoff)
        started=row['start_time'] is not None and pd.Timestamp(row['start_time']) < cutoff
        row['state']='completed' if completed else ('running' if started else 'pending')
        if not completed:
            row['deadline_outcome']='missed_unfinished' if pd.Timestamp(row['sla_deadline'])<=cutoff else 'right_censored'
            row['sla_met']=None
            row['completion_observed_at']=None
        if not started:
            row['start_time']=row['finish_time']=None
            row['dest_dc_id']=None
        rows.append(row)
    return rows


def phase_summary(result, cutoff):
    cutoff=utc_time(cutoff,True)
    rows=phase_rows(result,cutoff)
    steps=[s for s in result.steps if pd.Timestamp(s['interval_end'])<=cutoff]
    if not steps:
        raise ValueError('Phase must include at least one complete interval')
    events=[e for e in result.policy_events if pd.Timestamp(e['time'])<cutoff]
    summary=summarize(rows,steps)
    summary.update(horizon='finite_cohort_phase', observation_end=cutoff.isoformat())
    n=summary['arrived']
    violations=summary['deadline_missed_completed']+summary['deadline_missed_unfinished']
    summary.update(completion_rate=summary['completed']/n if n else None,
        deadline_violation_rate_observed=violations/n if n else None,
        violation_denominator='all arrived tasks; unfinished future-deadline tasks remain censored',
        policy_defer_intervals=sum(e['event']=='policy_deferred' for e in events),
        capacity_wait_intervals=sum(e['event']=='capacity_waited' for e in events))
    pending=[s['pending'] for s in steps]
    summary.update(pending_mean=math.fsum(pending)/len(steps),pending_max=max(pending))
    for resource in ('cores','gpus','mem'):
        values=[s[resource+'_fraction'] for s in steps]
        summary[resource+'_fraction_mean']=math.fsum(values)/len(values)
        summary[resource+'_fraction_max']=max(values)
    def mean(values):
        return math.fsum(values)/len(values) if values else None
    def minutes(a,b):
        return (pd.Timestamp(a)-pd.Timestamp(b)).total_seconds()/60
    started=[r for r in rows if r['start_time'] is not None]
    completed=[r for r in rows if r['state']=='completed']
    summary['mean_wait_minutes_started']=mean([minutes(r['start_time'],r['arrival_time']) for r in started])
    summary['mean_observed_pending_minutes']=mean([minutes(r['start_time'] or cutoff,r['arrival_time']) for r in rows])
    summary['mean_jct_minutes_completed']=mean([minutes(r['finish_time'],r['arrival_time']) for r in completed])
    summary['mean_observed_jct_minutes_completed']=mean([minutes(r['completion_observed_at'],r['arrival_time']) for r in completed])
    for resource in ('cores','gpus','mem'):
        summary[resource+'_hours']=math.fsum(s['used_'+resource]*s['interval_hours'] for s in steps)
    return summary
