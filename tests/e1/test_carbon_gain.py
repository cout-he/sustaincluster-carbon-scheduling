from dataclasses import replace
import math
import pandas as pd
import pytest
from experiments.e1.carbon_gain import CarbonGainEstimator
from experiments.e1.cars_state import ResourceSnapshot,RunningTask
from experiments.e1.online_ci_forecast import OnlineCIForecast
from .helpers import START,task
from .test_online_power_surrogate import model

def forecast(values):
    p=OnlineCIForecast()
    for k,v in enumerate(values): p.observe(START-pd.Timedelta(hours=24)+pd.Timedelta(minutes=15*k),v)
    p.observe(START,values[0]);return p

def snap(s,running=(),capacity=(10,10,10)):
    used=tuple(math.fsum(t.demand[k] for t in running) for k in range(3))
    return ResourceSnapshot(START,capacity,tuple(c-u for c,u in zip(capacity,used)),(s,),running,(s,))

def estimate(s,values=(200,100,50),m=None,running=()):
    return CarbonGainEstimator(m or model(),forecast(values)).estimate(s,snap(s,running),20.,0.)

def test_full_occupied_duration_and_units_common_horizon():
    s=task('a',duration=15.0001,cpu=1,mem=0)
    r=estimate(s)
    assert r['status']=='ok' and r['occupied_steps']==2 and r['horizon_steps']==3
    assert r['gain_it_kg']==pytest.approx(1*.25*(200-50)/1000)
    assert r['gain_total_kg']==r['gain_it_kg']

def test_persistence_equal_gain_not_future_oracle():
    p=OnlineCIForecast('persistence');p.observe(START,200)
    s=task('a',mem=0)
    r=CarbonGainEstimator(model(),p).estimate(s,snap(s),20,0)
    assert r['gain_it_kg']==0 and r['gain_total_kg']==0

def test_hvac_reversal_and_lag_common_initial_state():
    s=task('a',duration=15,cpu=1,mem=0)
    # Background high only NOW; a deliberately inverse static response is a synthetic mechanism test.
    bg=RunningTask('bg',(8,0,0),START+pd.Timedelta(minutes=15))
    m=model(hvac=(0,0,-1e6,0,0))
    # Use a positive bounded inverse slope + offset to make NOW incremental cooling strongly negative.
    m=replace(m,hvac_coefficients=(1000,-30000,0,0,0))
    r=estimate(s,(200,100),m,(bg,))
    assert r['gain_it_kg']>0 and r['gain_total_kg']<0
    lag=replace(model(hvac=(100,1000,0,0,0)),alpha=.5)
    r=estimate(s,(200,100),lag)
    assert math.isfinite(r['gain_hvac_kg'])

def test_horizon_unknown_and_known_capacity_conflict():
    s=task('a',duration=1500,cpu=1,mem=0)
    assert estimate(s)['status']=='unknown_horizon'
    s=task('a',cpu=3,mem=0)
    bg=RunningTask('b',(8,0,0),START+pd.Timedelta(minutes=60))
    assert estimate(s,running=(bg,))['status']=='capacity_opportunity_unavailable'

def test_same_background_no_base_power_attribution():
    a=estimate(task('a',cpu=1,mem=0),(200,100))
    b=estimate(task('a',cpu=1,mem=0),(200,100),replace(model(),it_intercept_kw=9000))
    assert a['gain_it_kg']==b['gain_it_kg']

def test_release_grid_computed_once_per_task_per_candidate(monkeypatch):
    original=RunningTask.release_time.fget;calls=[]
    def counted(r):calls.append(r.task_id);return original(r)
    monkeypatch.setattr(RunningTask,'release_time',property(counted))
    s=task('a',duration=120,cpu=1,mem=0)
    running=tuple(RunningTask(str(k),(.1,.1,.1),START+pd.Timedelta(minutes=15*(k%3+1))) for k in range(20))
    result=estimate(s,tuple(range(200,104,-1)),running=running)
    assert result['status']=='ok' and len(calls)<=len(running)

@pytest.mark.parametrize('minutes',[15.,30.,45.,90.,150.])
def test_machine_epsilon_duration_matches_real_executor_release(minutes):
    import numpy as np
    s=task('a',duration=float(np.nextafter(minutes,np.inf)),cpu=1,mem=0)
    result=estimate(s,tuple([200.]*96))
    finish=START+pd.Timedelta(minutes=s.duration)
    slots=max(1,int((finish.ceil('15min')-START)/pd.Timedelta(minutes=15)))
    assert result['occupied_steps']==slots

def test_tiny_positive_duration_still_occupies_current_interval():
    s=task('a',duration=1e-15,cpu=1,mem=0)
    assert estimate(s,(200,100))['occupied_steps']==1

def test_same_step_zero_quantized_running_task_is_currently_allocated():
    s=task('a',cpu=3,mem=0)
    running=RunningTask('tiny',(8,0,0),START)
    assert estimate(s,running=(running,))['status']=='capacity_opportunity_unavailable'
