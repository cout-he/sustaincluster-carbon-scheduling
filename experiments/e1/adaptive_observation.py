"""Current-only continuous risk block for the independent four-action interface.

Population is *all current pending tasks*, including capacity-blocked tasks, at
the pre-action boundary. It differs from E1-6A's eligible candidate diagnostics.
No future workload, environment, action, outcome, or policy trajectory is an
input. This is a queue description, never a same-step allocation simulation.

EDF/static-work preparation costs O(n log n). Exact ordered predecessor fsum
queries cost O(n^2) worst case, and can-fit checks O(n*m) for m running tasks.
Memory is O(n+m). There is one EDF sort, not one sort per candidate. The actor
receives the 43-value float32 vector; per-task raw budgets are info/audit only.
"""
from dataclasses import asdict
import math

import numpy as np

from .cars_state import demand
from .contracts import finite
from .policies import EDFBackfill
from .task_risk_state import (BudgetParameters, _budget_from_pressures,
                              latest_grid_start, occupied_minutes)

RISK_SCHEMA = 'e1-6b-continuous-risk-observation-v1'
RISK_FIELDS = (
    'risk_u_cpu', 'risk_u_gpu', 'risk_u_mem',
    'risk_rho_cpu', 'risk_rho_gpu', 'risk_rho_mem',
    'risk_z_min', 'risk_z_mean', 'risk_z_p50', 'risk_z_p90', 'risk_z_max',
    'risk_z_cpu_mean', 'risk_z_cpu_present',
    'risk_z_gpu_mean', 'risk_z_gpu_present',
    'risk_z_mem_mean', 'risk_z_mem_present',
    'risk_z_tie_mean', 'risk_z_tie_present',
    'risk_z_zero_mean', 'risk_z_zero_present',
    'risk_solo_slack_min', 'risk_solo_slack_mean', 'risk_solo_slack_p10', 'risk_solo_slack_max',
    'risk_safe_slack_min', 'risk_safe_slack_mean', 'risk_safe_slack_p10', 'risk_safe_slack_max',
    'risk_solo_ge60', 'risk_solo_ge30', 'risk_solo_ge15',
    'risk_safe_ge60', 'risk_safe_ge30', 'risk_safe_ge15',
    'risk_pending_present', 'risk_history_known',
    'risk_buffer_mean', 'risk_buffer_max',
    'risk_z_bin_0_1', 'risk_z_bin_1_2', 'risk_z_bin_gt2',
    'risk_can_fit_fraction',
)


def _mean(values):
    return math.fsum(values)/len(values) if values else 0.


def _quantile(sorted_values, probability):
    if not sorted_values:
        return 0.
    position=(len(sorted_values)-1)*probability
    low,high=math.floor(position),math.ceil(position)
    return sorted_values[low]+(sorted_values[high]-sorted_values[low])*(position-low)


def risk_observation(snapshot, pressure, normalization=None):
    """Return (43 normalized actor features, raw current-state audit dict).

    u and fractions/flags remain in [0,1]. Continuous z and known historical
    rho map x/(1+abs(x)); solo/safe minutes map x/(declared_slack_scale+abs(x)),
    default120, retaining negative slack. Buffers map minutes/150. Unknown rho
    and empty statistics are zero; history_known/pending_present disambiguate.
    Passing legacy ObservationNormalization or its dict uses slack_minutes
    only. No normalization values are fitted to workload or held-out data.
    """
    if normalization is None:
        slack_scale=120.
    elif isinstance(normalization,dict):
        slack_scale=normalization.get('slack_minutes',120.)
    else:
        slack_scale=normalization.slack_minutes
    slack_scale=finite(slack_scale,'risk slack normalization',True)
    parameters=BudgetParameters()
    capacity=tuple(snapshot.capacity)
    rho=tuple(pressure.rho_resources)
    if len(rho)!=3 or ((pressure.effective_minutes==0 and not all(h is None for h in rho))
                       or (pressure.effective_minutes>0 and any(h is None for h in rho))):
        raise ValueError('History unknown/effective duration mismatch')
    known=pressure.effective_minutes>0
    ordered=EDFBackfill().order(snapshot.pending)
    if any(spec.arrival_time>snapshot.now for spec in ordered):
        raise ValueError('Cannot observe future pending task')
    running=tuple(snapshot.running)
    running_demands=tuple(tuple(row.demand[k] for row in running) for k in range(3))
    utilization=tuple(math.fsum(values)/capacity[k] for k,values in enumerate(running_demands))
    grid={spec.task_id:latest_grid_start(spec) for spec in ordered}
    urgent_work={spec.task_id:tuple(q*occupied_minutes(spec) for q in demand(spec))
                 for spec in ordered if
                 (grid[spec.task_id]-snapshot.now).total_seconds()/60<=parameters.history_minutes}
    work=[]
    budgets=[]
    tasks=[]
    for spec in ordered:
        queue=tuple(math.fsum(row[k] for row in work)/(capacity[k]*parameters.history_minutes)
                    for k in range(3))
        budget=_budget_from_pressures(spec,snapshot.now,capacity,utilization,queue,pressure,
                                      parameters,'weighted')
        # Same numerical capacity rule as LocalResourcePool: do not subtract
        # available demand (which loses a .2+.6+.2 boundary through rounding).
        can_fit=all(math.fsum(running_demands[k]+(q,))<=capacity[k]
                    for k,q in enumerate(demand(spec)))
        tasks.append(dict(task_id=spec.task_id,budget=asdict(budget),can_fit=can_fit))
        budgets.append(budget)
        if spec.task_id in urgent_work:
            work.append(urgent_work[spec.task_id])
    scores=sorted(b.score for b in budgets)
    solo=sorted(b.solo_slack_minutes for b in budgets)
    safe=sorted(b.safe_slack_minutes for b in budgets)
    buffers=[b.buffer_minutes for b in budgets]
    fields={name:0. for name in RISK_FIELDS}
    for resource,u,h in zip(('cpu','gpu','mem'),utilization,rho):
        fields['risk_u_'+resource]=u
        fields['risk_rho_'+resource]=h if known else 0.
    for suffix,value in zip(('min','mean','p50','p90','max'),
                            (scores[0] if scores else 0.,_mean(scores),_quantile(scores,.5),
                             _quantile(scores,.9),scores[-1] if scores else 0.)):
        fields['risk_z_'+suffix]=value
    for category in ('cpu','gpu','mem','tie','zero'):
        values=[b.score for b in budgets if b.dominant_resource==category]
        fields['risk_z_'+category+'_mean']=_mean(values)
        fields['risk_z_'+category+'_present']=float(bool(values))
    for label,values in (('solo',solo),('safe',safe)):
        for suffix,value in zip(('min','mean','p10','max'),
                                (values[0] if values else 0.,_mean(values),_quantile(values,.1),
                                 values[-1] if values else 0.)):
            fields['risk_'+label+'_slack_'+suffix]=value
        for gate in (60,30,15):
            fields['risk_'+label+'_ge'+str(gate)]=sum(v>=gate for v in values)/len(values) if values else 0.
    fields['risk_pending_present']=float(bool(ordered))
    fields['risk_history_known']=float(known)
    fields['risk_buffer_mean']=_mean(buffers)
    fields['risk_buffer_max']=max(buffers) if buffers else 0.
    fields['risk_z_bin_0_1']=sum(0<v<=1 for v in scores)/len(scores) if scores else 0.
    fields['risk_z_bin_1_2']=sum(1<v<=2 for v in scores)/len(scores) if scores else 0.
    fields['risk_z_bin_gt2']=sum(v>2 for v in scores)/len(scores) if scores else 0.
    fields['risk_can_fit_fraction']=sum(t['can_fit'] for t in tasks)/len(tasks) if tasks else 0.
    normalized=[]
    units={}
    for name,value in fields.items():
        if '_slack_' in name:
            normalized.append(value/(slack_scale+abs(value)));units[name]='minutes'
        elif name.startswith('risk_buffer_'):
            normalized.append(value/150.);units[name]='minutes'
        elif name.startswith('risk_rho_') or (name.startswith('risk_z_') and
                                            not name.endswith('_present') and '_bin_' not in name):
            normalized.append(value/(1+abs(value)));units[name]='dimensionless'
        else:
            normalized.append(value);units[name]='dimensionless'
    vector=np.asarray(normalized,dtype=np.float32)
    if not np.isfinite(vector).all() or np.max(np.abs(vector))>1:
        raise ValueError('Invalid current risk state or normalization')
    raw=dict(schema=RISK_SCHEMA,population='all_current_pending_pre_action',
             fields=fields,tasks=tasks,units=units,
             normalization=dict(slack_minutes=slack_scale,z_scale=1.,rho_scale=1.,buffer_minutes=150.))
    return vector,raw
