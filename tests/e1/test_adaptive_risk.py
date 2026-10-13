"""Four-mode behavioral contracts; old TaskRiskPolicy remains frozen."""
from dataclasses import asdict
import pandas as pd
import pytest
from .helpers import START, task
from .test_task_risk import observe, budget, pressure
from .test_policies import execute_policy, rows


def new_policy(filtered=False):
    from experiments.e1.adaptive_risk import AdaptiveDelayPolicy
    return AdaptiveDelayPolicy({'threshold_g_per_kwh':161.075}, START, (1,1,1), risk_filter=filtered)


@pytest.mark.parametrize('action,threshold', [(1,60),(2,30),(3,15)])
@pytest.mark.parametrize('delta', [-15,0,15])
def test_slack_thresholds_are_eligibility_not_sleep_duration(action, threshold, delta):
    from experiments.e1.policies import DecisionContext
    s=task('a', duration=15, cpu=.1, mem=0, sla=(15+threshold+delta)/15)
    p=new_policy();p.set_action(action);p.on_candidate_resources(START,observe(p,(s,)))
    assert p.decide(s,DecisionContext(START,200)).defer is (delta>=0)
    assert p.diagnostics[-1]['action']==action
    assert p.diagnostics[-1]['solo_eligible'] is (delta>=0)


@pytest.mark.parametrize('action', [0,1,2,3])
def test_threshold_equality_and_expired_task_execute(action):
    from experiments.e1.policies import DecisionContext
    for s,ci in [(task('a',sla=20),161.075),(task('b',sla=.5),200)]:
        p=new_policy();p.set_action(action);p.on_candidate_resources(START,observe(p,(s,)))
        assert not p.decide(s,DecisionContext(START,ci)).defer


def test_filter_uses_frozen_budget_and_threshold_equality():
    from experiments.e1.policies import DecisionContext
    s=task('a', duration=15, cpu=.1, mem=0, sla=5) # grid solo60, cold30 ->safe30
    p=new_policy(True);p.set_action(2);p.on_candidate_resources(START,observe(p,(s,)))
    assert p.decide(s,DecisionContext(START,200)).defer
    assert p.diagnostics[-1]['budget']==asdict(budget(s,history=pressure(effective=0)))
    p=new_policy(True);p.set_action(1);p.on_candidate_resources(START,observe(p,(s,)))
    assert not p.decide(s,DecisionContext(START,200)).defer
    assert p.diagnostics[-1]['filter_veto']


def test_intra_step_start_updates_budget_for_next_candidate():
    # action2: a has S0 ->starts; b S45, u=.8 makes B15 and safe30 ->defer.
    # with b S30 the same new u changes B0 to15 ->no defer.
    from experiments.e1.cars_state import RunningTask
    from experiments.e1.policies import DecisionContext
    now=START+pd.Timedelta(minutes=120)
    a=task('a',arrival=120,duration=15,cpu=.8,mem=0,sla=1)
    b=task('b',arrival=120,duration=15,cpu=.1,mem=0,sla=3)
    p=new_policy(True);p.set_action(2);available=observe(p,(a,b),now=now)
    p.on_candidate_resources(now,available)
    assert not p.decide(a,DecisionContext(now,200)).defer
    p.on_task_started(RunningTask('a',(.8,0,0),now+pd.Timedelta(minutes=15)))
    p.on_candidate_resources(now,(.2,1,1))
    assert not p.decide(b,DecisionContext(now,200)).defer
    assert p.diagnostics[-1]['budget']['utilization'][0]==.8
    assert p.diagnostics[-1]['budget']['buffer_minutes']==15


def test_backfill_capacity_wait_is_not_risk_filter_veto():
    p=new_policy(True);p.set_action(0)
    result=execute_policy([task('a',duration=30,cpu=.8,mem=0,sla=1), task('b',cpu=.8,mem=0,sla=3),task('c',cpu=.2,mem=0,sla=4)],p,
                          minutes=60,ci=[200]*4)
    assert result.policy_steps[0]['started_ids']==['a','c']
    assert result.policy_steps[0]['capacity_waiting_ids']==['b']
    assert not any(d['task_id']=='b' and d['time']==START.isoformat() for d in p.diagnostics)
    assert all(not d['filter_veto'] for d in p.diagnostics)


@pytest.mark.parametrize('value', [True,-1,4,1.5])
def test_rejects_non_actions(value):
    with pytest.raises(ValueError):new_policy().set_action(value)


def test_rejects_changed_v1_coefficients_and_future_information():
    from experiments.e1.adaptive_risk import AdaptiveDelayPolicy
    for extra in ({'a_h':0},{'future_ci':True},{'threshold_g_per_kwh':200}):
        with pytest.raises(ValueError):
            AdaptiveDelayPolicy({'threshold_g_per_kwh':161.075,**extra},START,(1,1,1))
