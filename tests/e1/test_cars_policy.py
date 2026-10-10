"""New policies share execution, receive observed immutable state only."""
from dataclasses import replace
import pandas as pd
import pytest
from .helpers import START, task
from .test_policies import execute_policy, rows


def policy(name,capacity=(1,1,1)):
    from experiments.e1.cars import make_cars_policy
    return make_cars_policy(name,{'threshold_g_per_kwh':200},START,capacity)


@pytest.mark.parametrize('name',['edf_carbon_threshold','cars_known_only','cars_risk_only','cars_full'])
def test_single_task_ci_boundary_slack_and_completed(name):
    p=policy(name);result=execute_policy([task('a',duration=15,sla=4)],p,minutes=60,ci=[201,200,200,200])
    # risk cold requires next15 + buffer30 <= latest45: equality allows.
    assert rows(result)['a']['start_time']==(START+pd.Timedelta(minutes=15)).isoformat()
    assert rows(result)['a']['deadline_outcome']=='met'
    assert p.diagnostics[0]['reason']=='high_ci_defer'


def test_known_only_prevents_additional_known_miss():
    p=policy('cars_known_only')
    # a now=> finishes15, b at15=>fin30<=deadline30.
    # a deferred=>b now30, a starts30=>fin45>deadline30.
    result=execute_policy([task('a',duration=15,sla=2),task('b',duration=30,sla=1)],p,
                          minutes=60,ci=[250]*4)
    assert rows(result)['a']['start_time']==START.isoformat()
    assert p.diagnostics[0]['reason']=='known_deadline_risk'


def test_same_step_started_state_visible_and_no_stale_capacity():
    p=policy('cars_full',capacity=(2,2,2))
    result=execute_policy([task('a',duration=30,sla=1,gpu=1),task('b',duration=15,sla=2)],p,
                          minutes=60,ci=[250]*4,capacity=(2,2,2))
    diag=p.diagnostics[1]
    assert diag['available']==[1,1,1]
    assert diag['running_count']==1 and diag['pending_count']==1
    assert max(s['used_cores'] for s in result.steps)<=2


def test_future_trace_and_ci_change_cannot_change_current_action():
    from .helpers import inputs, KnownPower
    from experiments.e1.engine import LocalExecutor
    first=task('a',duration=15,sla=4)
    decisions=[]
    for future,ci in [(task('x',arrival=15,cpu=.1),10),(task('y',arrival=15,cpu=50),999)]:
        w,e=inputs([first,future],60)
        e=replace(e,samples=tuple(replace(s,ci_g_per_kwh=250 if j==0 else ci) for j,s in enumerate(e.samples)))
        p=policy('cars_full');LocalExecutor(w,e,(1,1,1),KnownPower(),p).run()
        decisions.append(p.diagnostics[0])
    assert decisions[0]==decisions[1]


def test_expired_task_not_deferred_and_capacity_wait_not_a_policy_decision():
    p=policy('cars_full')
    result=execute_policy([task('a',duration=30,sla=1),task('b',arrival=15,duration=15,sla=1)],p,
                          minutes=60,ci=[250]*4)
    assert rows(result)['b']['deadline_outcome']=='missed'
    assert [d['reason'] for d in p.diagnostics]==['no_grid_slack','no_grid_slack']
    assert result.policy_tasks[1]['capacity_wait_intervals']==1


def test_old_registry_unchanged_and_risk_only_no_planner():
    from experiments.e1.policies import POLICY_NAMES,POLICY_SCHEMA
    assert len(POLICY_NAMES)==3 and POLICY_SCHEMA=='e1-1b-policy-v1'
    p=policy('cars_risk_only')
    execute_policy([task('a',duration=15,sla=2)],p,minutes=30,ci=[250,250])
    assert p.counters['planner_calls']==0
    assert p.diagnostics[0]['reason']=='risk_buffer_guard'
