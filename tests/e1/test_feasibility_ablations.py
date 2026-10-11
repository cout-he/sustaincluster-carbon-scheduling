"""Separate true shadow-risk decisions from queue-cap conservatism."""
from dataclasses import replace

import pandas as pd
import pytest

from .helpers import START, task
from .test_policies import execute_policy, rows


def make(name):
    from experiments.e1.feasibility_ablations import make_ablation_policy
    return make_ablation_policy(name, {'threshold_g_per_kwh':200}, START, (1,1,1))


def observed(policy, pending, now=START):
    from experiments.e1.cars_state import ResourceSnapshot
    policy.on_step_start(ResourceSnapshot(START,(1,1,1),(1,1,1),tuple(pending),(),tuple(pending)))
    if now!=START:
        policy.on_step_start(ResourceSnapshot(now,(1,1,1),(1,1,1),tuple(pending),()))


@pytest.mark.parametrize('n,execute',[(256,False),(257,True)])
def test_cap_only_strict_candidate_queue_boundary_and_no_planner(n,execute):
    from experiments.e1.policies import DecisionContext
    p=make('risk_cap_only')
    pending=[task(f'a{i:03}',cpu=.001,mem=.001,duration=15,sla=4) for i in range(n)]
    observed(p,pending)
    decision=p.decide(pending[0],DecisionContext(START,201))
    assert decision.defer is not execute
    assert p.counters['planner_calls']==0
    assert not hasattr(p,'planner')
    assert p.diagnostics[-1]['pending_count']==n
    assert p.diagnostics[-1]['reason']==('cap_execute_guard' if execute else 'high_ci_defer')


@pytest.mark.parametrize('name',['risk_cap_only','risk_shadow_only'])
def test_low_ci_and_history_guard_take_precedence_over_cap(name):
    from experiments.e1.policies import DecisionContext
    pending=[task(f'a{i:03}',cpu=.001,mem=.001,duration=15,sla=2) for i in range(257)]
    p=make(name);observed(p,pending)
    assert p.decide(pending[0],DecisionContext(START,200)).reason=='low_ci_execute'
    assert p.decide(pending[1],DecisionContext(START,201)).reason=='risk_buffer_guard'
    assert p.counters['planner_calls']==0


def test_shadow_over_cap_keeps_risk_defer_without_running_planner():
    from experiments.e1.policies import DecisionContext
    pending=[task(f'a{i:03}',cpu=.001,mem=.001,duration=15,sla=4) for i in range(257)]
    p=make('risk_shadow_only');observed(p,pending)
    decision=p.decide(pending[0],DecisionContext(START,201))
    assert decision.defer
    assert p.counters['planner_calls']==0
    assert p.diagnostics[-1]['planner']['status']=='uncertain'
    assert p.diagnostics[-1]['planner']['reason']=='pending_limit'
    assert p.diagnostics[-1]['shadow_planner_executed'] is False
    assert p.counters['uncertain_risk_fallbacks']==1


def test_shadow_only_blocks_explicit_added_deadline_miss():
    from experiments.e1.policies import DecisionContext
    a=task('a',duration=15,sla=10);b=task('b',duration=30,sla=5.5)
    now=START+pd.Timedelta(minutes=120)
    risk=make('cars_risk_only');shadow=make('risk_shadow_only')
    for p in (risk,shadow):observed(p,[a,b],now)
    assert risk.decide(a,DecisionContext(now,250)).defer
    decision=shadow.decide(a,DecisionContext(now,250))
    assert not decision.defer and decision.reason=='known_deadline_risk'
    d=shadow.diagnostics[-1]['planner']
    assert d['missed_now']==() and d['missed_defer']==('a',)
    assert shadow.counters['planner_calls']==2


def test_shadow_bounded_unknown_keeps_risk_defer_while_full_executes():
    from experiments.e1.policies import DecisionContext
    a=task('a',duration=300,sla=2);b=task('b',duration=300,sla=3)
    now=START+pd.Timedelta(minutes=120)
    full=make('cars_full');shadow=make('risk_shadow_only')
    for p in (full,shadow):observed(p,[a,b],now)
    assert not full.decide(a,DecisionContext(now,250)).defer
    assert shadow.decide(a,DecisionContext(now,250)).defer
    assert shadow.diagnostics[-1]['planner']['reason']=='bounded_horizon_unknown'
    assert shadow.counters['uncertain_risk_fallbacks']==1
    assert shadow.counters['conservative_fallbacks']==0


def test_cap_state_shrinks_after_real_same_step_start():
    tasks=[task(f'a{i:03}',cpu=.001,mem=.001,duration=15,sla=4) for i in range(257)]
    p=make('risk_cap_only')
    result=execute_policy(tasks,p,minutes=15,ci=[250])
    assert rows(result)['a000']['start_time']==START.isoformat()
    assert rows(result)['a001']['state']=='pending'
    assert result.policy_steps[0]['started_ids']==['a000']
    assert p.diagnostics[1]['pending_count']==256
    assert p.diagnostics[1]['available'][0]==pytest.approx(.999)


@pytest.mark.parametrize('name',['risk_cap_only','risk_shadow_only'])
def test_future_trace_and_ci_changes_preserve_current_decision(name):
    from .helpers import inputs, KnownPower
    from experiments.e1.engine import LocalExecutor
    first=task('a',duration=15,sla=4)
    decisions=[]
    for future,ci in [(task('x',arrival=15,cpu=.1),0),(task('y',arrival=15,cpu=20),999)]:
        w,e=inputs([first,future],60)
        e=replace(e,samples=tuple(replace(s,ci_g_per_kwh=250 if j==0 else ci) for j,s in enumerate(e.samples)))
        p=make(name);LocalExecutor(w,e,(1,1,1),KnownPower(),p).run()
        decisions.append(p.diagnostics[0])
    assert decisions[0]==decisions[1]
