"""Behavioral RL contracts: causal sensors, persistent execution, honest rewards."""
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from experiments.e1.cars import make_cars_policy
from experiments.e1.engine import LocalExecutor
from experiments.e1.environment import EnvironmentalSample, FrozenEnvironment
from .helpers import START, KnownPower, inputs, task
from .test_stepwise import scenario

def env(w=None,e=None,capacity=(2,1,3),**kwargs):
    from experiments.e1.rl_env import CarbonSchedulingEnv
    if w is None: w,e=scenario()
    return CarbonSchedulingEnv(w,e,capacity,KnownPower,{'threshold_g_per_kwh':150},**kwargs)

@pytest.mark.parametrize('action,name',[(0,'edf_backfill'),(1,'edf_carbon_threshold'),(2,'cars_risk_only')])
def test_constant_modes_reproduce_complete_executor_outputs(action,name):
    w,e=scenario()
    expected=LocalExecutor(w,e,(2,1,3),KnownPower(),
        make_cars_policy(name,{'threshold_g_per_kwh':150},START,(2,1,3)),drain=True).run()
    live=env(w,e);live.reset(seed=7)
    done=False
    while not done:
        obs,r,term,trunc,info=live.step(action)
        assert live.observation_space.contains(obs) and np.isfinite(r)
        done=term or trunc
    assert asdict(live.executor.result())==asdict(expected)

def test_mode_switch_does_not_lose_arrival_history():
    w,e=inputs([task('a',duration=15,cpu=1),task('b',arrival=15,duration=15,cpu=1)],45)
    live=env(w,e,capacity=(1,1,1));live.reset(seed=7)
    assert live.observation_raw['history_observed_tasks']==1
    live.step(0)
    assert live.observation_raw['history_observed_tasks']==2
    assert live.observation_raw['history_resource_work_minutes']==(30.,0.,30.)
    live.step(1)
    assert live.observation_raw['history_observed_tasks']==2
    live.step(2)
    assert live.executor.result().summary['completed']==2

def test_mixed_mode_carbon_diagnostics_share_the_observed_arrival_history():
    w,e=inputs([task('a'),task('b',arrival=15)],45)
    live=env(w,e);live.reset(seed=1);live.step(0);live.step(1)
    assert live._policies[1].history.pressure(START+pd.Timedelta(minutes=30)).observed_tasks==2

def test_close_is_idempotent_and_reset_does_not_close_a_model_twice():
    created=[]
    class ClosingPower(KnownPower):
        def __init__(self): self.closes=0;created.append(self)
        def close(self):
            self.closes+=1
            if self.closes>1: raise RuntimeError('closed twice')
    from experiments.e1.rl_env import CarbonSchedulingEnv
    w,e=inputs([task('a')],30)
    live=CarbonSchedulingEnv(w,e,(1,1,1),ClosingPower,{'threshold_g_per_kwh':150})
    live.reset(seed=1);live.step(0);live.close();live.close()
    live.reset(seed=1);live.step(0);live.close()
    assert [p.closes for p in created]==[1,1]

def test_same_observed_history_and_action_ignore_different_future_trajectories():
    common=task('a',duration=20)
    wa,ea=inputs([common,task('b',arrival=30)],60)
    wb,eb=inputs([common,task('c',arrival=30,cpu=2,duration=60)],60)
    eb=replace(eb,samples=tuple(replace(s,ci_g_per_kwh=900,temperature_c=35)
                              if s.time>=START+pd.Timedelta(minutes=30) else s for s in eb.samples))
    a,b=env(wa,ea),env(wb,eb)
    oa,ia=a.reset(seed=7);ob,ib=b.reset(seed=7)
    np.testing.assert_array_equal(oa,ob)
    assert ia==ib
    oa,ra,ta,xa,ia=a.step(1);ob,rb,tb,xb,ib=b.step(1)
    np.testing.assert_array_equal(oa,ob)
    assert (ra,ta,xa,ia)==(rb,tb,xb,ib)
    assert a.executor.steps==b.executor.steps
    # The next returned sensor legitimately observes the now-arrived fork at 30.
    _,ra,_,_,ia=a.step(2);_,rb,_,_,ib=b.step(2)
    assert ra==rb and ia['interval']==ib['interval']

def test_terminal_observation_never_reads_next_future_ci_and_arrival():
    wa,ea=inputs([task('a'),task('b',arrival=15)],45)
    wb,eb=inputs([task('a'),task('c',arrival=15,duration=30)],45)
    eb=replace(eb,samples=tuple(replace(s,ci_g_per_kwh=999) if s.time>START else s for s in eb.samples))
    a,b=env(wa,ea,max_episode_steps=1),env(wb,eb,max_episode_steps=1)
    a.reset(seed=1);b.reset(seed=1)
    oa,ra,ta,xa,ia=a.step(0);ob,rb,tb,xb,ib=b.step(0)
    np.testing.assert_array_equal(oa,ob)
    assert ia==ib and (ra,ta,xa)==(rb,tb,xb)
    assert ia['observed_counts']['arrived']==1 and xa and not ta

def test_reward_units_new_miss_and_censored_terminal_liability():
    from experiments.e1.rl_env import RewardConfig
    w,e=inputs([task('a',duration=45),task('b',arrival=15,duration=15,sla=1)],45)
    reward=RewardConfig(carbon_scale_kg=.1,wait_scale_task_minutes=15,deadline_scale_tasks=1,
                        terminal_scale_tasks=1,alpha=1,beta=1,gamma=2,terminal_weight=3)
    live=env(w,e,capacity=(1,1,1),reward_config=reward,max_episode_steps=2)
    live.reset(seed=5)
    live.step(0)  # a is running before the earlier-deadline b arrives.
    _,r,term,trunc,info=live.step(0)
    terms=info['reward']
    assert not term and trunc
    assert terms['carbon_kg']==pytest.approx(.3)
    assert terms['wait_task_minutes']==15 and terms['new_deadline_misses']==1
    assert terms['new_deadline_miss_ids']==['b']
    assert terms['unfinished_tasks']==2 and terms['remaining_work_normalized']==.25
    assert terms['terminal_liability_normalized']==2.25
    assert r==pytest.approx(-12.75)
    assert live.executor.result().summary['deadline_right_censored']==1

def test_deadline_penalty_never_repeats_when_task_stays_pending():
    w,e=inputs([task('a',duration=45),task('b',arrival=15,duration=15,sla=1)],45)
    live=env(w,e,capacity=(1,1,1));live.reset(seed=5)
    live.step(0)
    first=live.step(0)[4]['reward'];second=live.step(0)[4]['reward']
    assert first['new_deadline_miss_ids']==['b']
    assert second['new_deadline_miss_ids']==[]
    assert second['unfinished_tasks']==1 and second['terminal_liability_normalized']>0

def test_exact_deadline_completion_released_later_is_not_penalized():
    w,e=inputs([task('a',duration=5,sla=1)],30)
    live=env(w,e);live.reset(seed=7)
    assert live.step(0)[4]['reward']['new_deadline_misses']==0
    assert live.step(0)[4]['reward']['new_deadline_misses']==0

def test_reset_keeps_frozen_tasks_and_creates_one_persistent_physics_object():
    created=[]
    class StatefulPower(KnownPower):
        def __init__(self): self.calls=0; created.append(self)
        def measure(self,*args): self.calls+=1;return super().measure(*args)
    from experiments.e1.rl_env import CarbonSchedulingEnv
    w,e=inputs([task('a')],30);original=w.payload()
    live=CarbonSchedulingEnv(w,e,(1,1,1),StatefulPower,{'threshold_g_per_kwh':150})
    live.reset(seed=1);live.step(0);live.step(0)
    assert len(created)==1 and created[0].calls==2
    live.reset(seed=1);live.step(0)
    assert len(created)==2 and created[1].calls==1 and w.payload()==original

def test_common_observation_window_includes_idle_energy_without_early_stop():
    w,e=inputs([task('a',duration=5)],15)
    _,extended=inputs([],45);extended=replace(extended,start=e.start)
    live=env(w,extended,capacity=(1,1,1));live.reset(seed=1)
    assert live.step(0)[2:4]==(False,False)
    assert live.step(0)[2:4]==(False,False)
    _,_,term,trunc,_=live.step(0)
    assert term and not trunc
    assert [s['total_energy_kwh'] for s in live.executor.steps]==[1.5,1.,1.]

def test_drain_stop_terminates_and_does_not_schedule_an_extra_interval():
    w,e=inputs([task('a',duration=5)],15);_,e=inputs([],45)
    live=env(w,e,stop_at_drain=True);live.reset(seed=1)
    _,_,term,trunc,_=live.step(0)
    assert term and not trunc and len(live.executor.steps)==1

def test_seeded_reset_step_and_random_complete_rollout():
    from gymnasium.utils.env_checker import check_env
    w,e=inputs([task('a'),task('b',arrival=15,cpu=.2)],60)
    live=env(w,e)
    check_env(live,skip_render_check=True)
    traces=[]
    for _ in range(2):
        live.reset(seed=123);trace=[];done=False
        while not done:
            action=live.action_space.sample()
            o,r,t,x,i=live.step(action);trace.append((action,o.tolist(),r,t,x,i));done=t or x
        traces.append(trace)
        assert live.executor.result().summary['completed']==2
        live.executor.check_ownership()
    assert traces[0]==traces[1]
    with pytest.raises(RuntimeError): live.step(0)

@pytest.mark.parametrize('action',[-1,3,1.5,True])
def test_invalid_action_has_no_execution_side_effect(action):
    live=env();live.reset(seed=1)
    with pytest.raises(ValueError): live.step(action)
    assert live.executor.steps==[]

def test_empty_queue_observation_is_finite_and_future_config_is_rejected():
    w,e=inputs([],30);live=env(w,e);obs,_=live.reset(seed=1)
    assert np.isfinite(obs).all() and live.observation_raw['solo_slack_minutes']==[]
    from experiments.e1.rl_env import CarbonSchedulingEnv,RewardConfig
    with pytest.raises(ValueError): RewardConfig(terminal_weight=0)
    with pytest.raises(ValueError): CarbonSchedulingEnv(w,e,(1,1,1),KnownPower,
                                                      {'threshold_g_per_kwh':150,'future_ci':True})

def test_history_window_is_fixed():
    from experiments.e1.rl_env import CarbonSchedulingEnv
    w,e=inputs([],30)
    with pytest.raises(ValueError,match='120'):
        CarbonSchedulingEnv(w,e,(1,1,1),KnownPower,
                            {'threshold_g_per_kwh':150,'history_minutes':240})

def test_numeric_configuration_is_canonical():
    from experiments.e1.rl_env import RewardConfig,ObservationNormalization
    w,e=inputs([],30)
    cfg=RewardConfig(carbon_scale_kg='1')
    norm=ObservationNormalization(ci_g_per_kwh='1000')
    assert isinstance(cfg.carbon_scale_kg,float)
    assert isinstance(norm.ci_g_per_kwh,float)
    live=env(w,e,reward_config=cfg,normalization=norm)
    live.reset(seed=1);assert np.isfinite(live.step(0)[1])

def test_miss_is_not_penalized_again_on_late_completion():
    w,e=inputs([task('a',duration=30,cpu=1),
                task('b',arrival=15,duration=15,sla=1)],60)
    live=env(w,e,capacity=(1,1,1));live.reset(seed=1)
    terms=[live.step(0)[4]['reward'] for _ in range(4)]
    assert [t['new_deadline_miss_ids'] for t in terms]==[[],['b'],[],[]]
    assert live.executor.result().summary['completed']==2
