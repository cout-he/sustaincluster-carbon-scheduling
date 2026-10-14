from dataclasses import asdict,replace
import pandas as pd
import pytest
from experiments.e1.carbon_gain_policy import CarbonGainEDF
from experiments.e1.deadline_guard import DeadlineGuard
from experiments.e1.online_ci_forecast import OnlineCIForecast
from experiments.e1.engine import LocalExecutor
from experiments.e1.policies import EDFBackfill
from .helpers import START,KnownPower,task,inputs
from .test_online_power_surrogate import model

def policy(mode='m2',**kwargs):
    return CarbonGainEDF(mode,START,(10,10,10),model(),prehistory=(),**kwargs)

def test_common_guard_cold_exact_and_disabled_solo_boundary():
    from experiments.e1.cars_state import ResourceSnapshot
    s=task('a',duration=90,sla=1.5)
    snap=ResourceSnapshot(START,(10,10,10),(10,10,10),(s,),(),(s,))
    g=DeadlineGuard(START,(10,10,10));g.observe(snap)
    assert g.check(s,START)['allowed'] and g.check(s,START)['buffer_steps']==2
    assert not g.check(replace(s,duration=89.999),START)['allowed']
    off=DeadlineGuard(START,(10,10,10),enabled=False);off.observe(snap)
    assert not off.check(task('b',duration=15,sla=1),START)['allowed']

def test_new_hooks_have_no_effect_without_subscriber():
    w,e=inputs([task('a'),task('b',arrival=15)],60)
    old=LocalExecutor(w,e,(2,1,3),KnownPower(),EDFBackfill(),True).run()
    p=policy();p.set_action(0)
    live=LocalExecutor(w,e,(2,1,3),KnownPower(),p,True).run()
    assert old.tasks==live.tasks and old.steps==live.steps
    assert all(r['completed_interval_end'] is None or pd.Timestamp(r['completed_interval_end'])<=pd.Timestamp(r['time']) for r in p.diagnostics)

def test_m0_matches_riskonly_and_capacity_first_backfill():
    from experiments.e1.cars import make_cars_policy
    w,e=inputs([task('a',duration=90),task('b',duration=15),task('c',arrival=30)],180)
    p=CarbonGainEDF('m0',START,(2,1,3),model(capacity=(2,1,3)))
    a=LocalExecutor(w,e,(2,1,3),KnownPower(),p,True).run()
    b=LocalExecutor(w,e,(2,1,3),KnownPower(),make_cars_policy('cars_risk_only',{'threshold_g_per_kwh':161.075},START,(2,1,3)),True).run()
    assert a.tasks==b.tasks and a.steps==b.steps

def test_same_step_running_snapshot_updates_and_causal_history():
    w,e=inputs([task('a',duration=90,cpu=1),task('b',duration=90,cpu=1)],120)
    p=policy();p.set_action(0)
    LocalExecutor(w,e,(10,10,10),KnownPower(),p,True).run()
    checks=[r for r in p.diagnostics if r['time']==START.isoformat()]
    assert [r['running_count'] for r in checks]==[0,1]
    assert len(p.forecaster.history)<=193

@pytest.mark.parametrize('mode,ci,expected',[('m0',161.075,False),('m0',161.076,True),('m1',100,True),('m2',100,True)])
def test_threshold_exact_and_low_ci_positive_execution_window(mode,ci,expected):
    from experiments.e1.cars_state import ResourceSnapshot
    from experiments.e1.environment import EnvironmentalSample
    from experiments.e1.policies import DecisionContext
    # duration90 => six slots; seasonal tail lower than the current CI.
    s=task('a',duration=90,mem=0)
    prior=[dict(time=START-pd.Timedelta(hours=24)+pd.Timedelta(minutes=15*k),ci=ci if k<6 else ci-20) for k in range(96)]
    p=CarbonGainEDF(mode,START,(10,10,10),model(),prehistory=prior)
    p.on_step_environment(EnvironmentalSample(START,ci,20,18))
    p.on_step_start(ResourceSnapshot(START,(10,10,10),(10,10,10),(s,),(),(s,)))
    assert p.decide(s,DecisionContext(START,ci)).defer==expected

def test_completed_hook_readonly_and_called_after_actual_measure():
    seen=[]
    class Observe(EDFBackfill):
        def on_interval_completed(self,row):
            with pytest.raises(TypeError):row['it_power_kw']=999
            seen.append(dict(row))
    w,e=inputs([task('a')],30)
    result=LocalExecutor(w,e,(2,1,3),KnownPower(),Observe(),True).run()
    assert tuple(seen)==result.steps

def test_moderate_and_aggressive_differ_without_guard_change():
    from experiments.e1.cars_state import ResourceSnapshot
    from experiments.e1.environment import EnvironmentalSample
    from experiments.e1.policies import DecisionContext
    s=task('a',duration=90,mem=0)
    prior=[dict(time=START-pd.Timedelta(hours=24)+pd.Timedelta(minutes=15*k),ci=200 if k<6 else 100) for k in range(96)]
    for action,expected in ((1,False),(2,True)):
        p=CarbonGainEDF('m2',START,(10,10,10),model(),prehistory=prior,moderate_threshold_kg=.1);p.set_action(action)
        p.on_step_environment(EnvironmentalSample(START,200,20,18))
        p.on_step_start(ResourceSnapshot(START,(10,10,10),(10,10,10),(s,),(),(s,)))
        assert p.decide(s,DecisionContext(START,200)).defer==expected
        assert p.diagnostics[-1]['buffer_steps']==2
