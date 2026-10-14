from dataclasses import asdict,replace
import numpy as np
import pytest
from experiments.e1.carbon_gain_rl_env import CarbonGainSchedulingEnv,OBSERVATION_FIELDS
from experiments.e1.carbon_gain_policy import CarbonGainEDF
from experiments.e1.engine import LocalExecutor
from .helpers import START,KnownPower,inputs,task
from .test_online_power_surrogate import model

def env(w,e,**kw):
    return CarbonGainSchedulingEnv(w,e,(10,10,10),KnownPower,{'threshold_g_per_kwh':161.075},model(),moderate_threshold_kg=.01,**kw)

@pytest.mark.parametrize('action,threshold',[(0,None),(1,.01),(2,0.)])
def test_fixed_actions_match_same_executor(action,threshold):
    w,e=inputs([task('a',duration=90),task('b',arrival=30)],180)
    p=CarbonGainEDF('m2',START,(10,10,10),model(),moderate_threshold_kg=.01);p.set_action(action)
    expected=LocalExecutor(w,e,(10,10,10),KnownPower(),p,True).run()
    live=env(w,e);o,_=live.reset(seed=9)
    assert len(o)==len(OBSERVATION_FIELDS) and len(o)<84
    while True:
        o,r,t,x,info=live.step(action)
        assert np.isfinite(r) and live.observation_space.contains(o)
        if t or x:break
    actual=live.executor.result()
    assert actual.tasks==expected.tasks and actual.steps==expected.steps
    assert t and not x

def test_future_fork_same_current_and_current_step():
    w,e=inputs([task('a',duration=90),task('b',arrival=30)],180)
    w2=replace(w,tasks=(w.tasks[0],task('c',arrival=30,duration=120)))
    e2=replace(e,samples=tuple(replace(s,ci_g_per_kwh=999,temperature_c=39) if s.time>=START+__import__('pandas').Timedelta(minutes=30) else s for s in e.samples))
    a,b=env(w,e),env(w2,e2);oa,_=a.reset(seed=7);ob,_=b.reset(seed=7)
    np.testing.assert_array_equal(oa,ob)
    aa=a.step(2);bb=b.step(2)
    np.testing.assert_array_equal(aa[0],bb[0]);assert aa[1:4]==bb[1:4]
    assert a.executor.steps==b.executor.steps

def test_reset_random_determinism_and_gym_checker():
    from gymnasium.utils.env_checker import check_env
    w,e=inputs([task('a',duration=45),task('b',arrival=30)],90)
    live=env(w,e);check_env(live,skip_render_check=True)
    def rollout():
        live.reset(seed=51);rows=[]
        while True:
            a=live.action_space.sample();o,r,t,x,i=live.step(a);rows.append((a,o.tolist(),r,t,x))
            if t or x:return rows
    assert rollout()==rollout()

def test_truncation_preserves_old_terminal_debt_and_first_miss():
    w,e=inputs([task('a',duration=90)],120)
    live=env(w,e,max_episode_steps=1);live.reset(seed=1)
    o,r,t,x,i=live.step(0)
    assert x and not t and i['reward']['unfinished_tasks']==1

def test_sb3_smoke_updates_only_tiny_fixture():
    from stable_baselines3 import PPO
    import torch
    torch.set_num_threads(1)
    w,e=inputs([task('a',duration=45),task('b',arrival=30)],90)
    live=env(w,e)
    p=PPO('MlpPolicy',live,n_steps=8,batch_size=8,n_epochs=1,seed=71,
          policy_kwargs={'net_arch':[16,16]},device='cpu',verbose=0)
    before=[v.detach().clone() for v in p.policy.parameters()]
    p.learn(total_timesteps=16)
    assert p._n_updates==2 and any(not torch.equal(a,b) for a,b in zip(before,p.policy.parameters()))
    live.close()

def test_nontrivial_seasonal_rl_threshold_types_and_guard_separation():
    w,e=inputs([task('a',duration=90,mem=0)],180)
    e=replace(e,samples=tuple(replace(s,ci_g_per_kwh=200) for s in e.samples))
    prior=[dict(time=START-__import__('pandas').Timedelta(hours=24)+__import__('pandas').Timedelta(minutes=15*k),ci=200 if k<6 else 100) for k in range(96)]
    live=CarbonGainSchedulingEnv(w,e,(10,10,10),KnownPower,{'threshold_g_per_kwh':161.075},model(),prehistory=prior,moderate_threshold_kg='.1')
    obs,_=live.reset(seed=7)
    assert live.moderate_threshold_kg==.1 and live.observation_raw['carbon_gain']['candidates'][0]['gain_total_kg']>0
    live.step(1);assert live.executor.events[-1]['event']=='started'
    live.reset(seed=7);live.step(2)
    assert live.executor.policy_events[0]['event']=='policy_deferred'
    # A tight guard is not a forecast failure.
    w2,e2=inputs([task('b',duration=15,sla=1)],30);tight=env(w2,e2);o,_=tight.reset(seed=7)
    assert o[list(OBSERVATION_FIELDS).index('invalid_fraction')]==0
