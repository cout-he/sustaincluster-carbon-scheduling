"""Current-only risk sensors: fixed schema, frozen arithmetic, all backlog."""
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from experiments.e1.cars_state import Pressure, ResourceSnapshot, RunningTask
from experiments.e1.policies import EDFBackfill
from experiments.e1.task_risk_state import BudgetParameters, calculate_budget
from .helpers import START, inputs, task


def known_pressure(rho=(0.,0.,0.)):
    return Pressure(15.,(0.,0.,0.),rho,max(rho),0,0)


def snapshot(tasks=(),running=(),now=START,capacity=(10.,10.,10.),available=None):
    if available is None:
        import math
        available=tuple(c-math.fsum(r.demand[k] for r in running) for k,c in enumerate(capacity))
    return ResourceSnapshot(now,capacity,available,tuple(tasks),tuple(running))


def test_schema_and_hand_computed_continuous_risk_slack_fractions():
    from experiments.e1.adaptive_observation import RISK_FIELDS,risk_observation
    now=START+pd.Timedelta(minutes=15)
    urgent=task('a',duration=30,cpu=0,gpu=2,mem=0,sla=1)
    candidate=task('c',duration=60,cpu=0,gpu=1,mem=0,sla=2)
    long_slack=task('b',duration=30,cpu=0,gpu=8,mem=0,sla=20)
    running=(RunningTask('r',(0,5,0),now+pd.Timedelta(minutes=30)),)
    vector,raw=risk_observation(snapshot((long_slack,candidate,urgent),running,now),known_pressure((0,1.5,0)))
    assert vector.shape==(43,) and vector.dtype==np.float32
    assert len(RISK_FIELDS)==len(set(RISK_FIELDS))==43
    assert tuple(raw['fields'])==RISK_FIELDS
    expected=dict(risk_u_cpu=0.,risk_u_gpu=.5,risk_u_mem=0.,
                  risk_rho_cpu=0.,risk_rho_gpu=1.5,risk_rho_mem=0.,
                  risk_z_min=1.5,risk_z_mean=1.55,risk_z_p50=1.55,risk_z_p90=1.59,risk_z_max=1.6,
                  risk_z_gpu_mean=1.55,risk_z_gpu_present=1.,risk_z_cpu_present=0.,
                  risk_solo_slack_min=-15.,risk_solo_slack_mean=195.,risk_solo_slack_p10=-3.,risk_solo_slack_max=555.,
                  risk_safe_slack_min=-45.,risk_safe_slack_mean=165.,risk_safe_slack_p10=-33.,risk_safe_slack_max=525.,
                  risk_solo_ge60=1/3,risk_solo_ge30=2/3,risk_solo_ge15=2/3,
                  risk_safe_ge60=1/3,risk_safe_ge30=1/3,risk_safe_ge15=2/3,
                  risk_pending_present=1.,risk_history_known=1.,risk_buffer_mean=30.,risk_buffer_max=30.,
                  risk_z_bin_0_1=0.,risk_z_bin_1_2=1.,risk_z_bin_gt2=0.,risk_can_fit_fraction=2/3)
    for name,value in expected.items():
        assert raw['fields'][name]==pytest.approx(value),name
    assert vector[RISK_FIELDS.index('risk_u_gpu')]==pytest.approx(.5)
    assert vector[RISK_FIELDS.index('risk_rho_gpu')]==pytest.approx(.6)
    assert vector[RISK_FIELDS.index('risk_solo_slack_min')]==pytest.approx(-15/135)
    assert vector[RISK_FIELDS.index('risk_buffer_max')]==pytest.approx(.2)
    assert np.isfinite(vector).all() and np.max(np.abs(vector))<=1


def test_all_pending_budgets_exactly_match_frozen_reference_even_capacity_blocked():
    from experiments.e1.adaptive_observation import risk_observation
    specs=(task('b',duration=60,cpu=1,gpu=1,mem=0,sla=2),
           task('a',duration=16,cpu=11,gpu=0,mem=0,sla=1),
           task('z',duration=45,cpu=0,gpu=0,mem=0,sla=5))
    snap=snapshot(specs)
    _,raw=risk_observation(snap,known_pressure((1.4,.9,0)))
    ordered=EDFBackfill().order(specs)
    assert [r['task_id'] for r in raw['tasks']]==[s.task_id for s in ordered]
    for index,spec in enumerate(ordered):
        reference=calculate_budget(spec,START,snap.capacity,snap.running,ordered[:index],
                                   known_pressure((1.4,.9,0)),BudgetParameters())
        assert raw['tasks'][index]['budget']==asdict(reference)
    assert raw['tasks'][0]['can_fit'] is False
    assert len(raw['tasks'])==3 and raw['fields']['risk_can_fit_fraction']==pytest.approx(2/3)


@pytest.mark.parametrize('known',[False,True])
def test_empty_queue_zero_statistics_preserve_unknown_history_presence(known):
    from experiments.e1.adaptive_observation import RISK_FIELDS,risk_observation
    pressure=known_pressure() if known else Pressure(0.,(0.,0.,0.),(None,None,None),None,2,0)
    vector,raw=risk_observation(snapshot(),pressure)
    assert raw['tasks']==[]
    assert raw['fields']['risk_pending_present']==0
    assert raw['fields']['risk_history_known']==int(known)
    for index,name in enumerate(RISK_FIELDS):
        assert vector[index]==(int(known) if name=='risk_history_known' else 0.)


def test_tie_zero_and_resource_category_presence_not_raw_units():
    from experiments.e1.adaptive_observation import risk_observation
    specs=(task('a',duration=60,cpu=1,gpu=1,mem=0,sla=2),
           task('b',duration=60,cpu=0,gpu=0,mem=0,sla=2),
           task('c',duration=60,cpu=1,gpu=0,mem=0,sla=2))
    _,raw=risk_observation(snapshot(specs),known_pressure((2.,0.,0.)))
    assert raw['fields']['risk_z_tie_present']==1
    assert raw['fields']['risk_z_zero_present']==1
    assert raw['fields']['risk_z_cpu_present']==1
    assert raw['fields']['risk_z_gpu_present']==raw['fields']['risk_z_mem_present']==0
    assert raw['fields']['risk_z_zero_mean']==0
    assert raw['tasks'][1]['budget']['buffer_minutes']==0


def test_can_fit_matches_executor_fsum_boundary_not_free_resource_subtraction():
    from experiments.e1.adaptive_observation import risk_observation
    running=(RunningTask('r1',(.2,0,0),START+pd.Timedelta(minutes=30)),
             RunningTask('r2',(.6,0,0),START+pd.Timedelta(minutes=30)))
    spec=task('a',cpu=.2,gpu=0,mem=0,duration=60,sla=2)
    snap=snapshot((spec,),running,capacity=(1,1,1))
    assert snap.available[0]<spec.cores_req
    _,raw=risk_observation(snap,known_pressure())
    assert raw['tasks'][0]['can_fit'] is True
    assert raw['fields']['risk_can_fit_fraction']==1
    assert raw['fields']['risk_u_cpu']==.8


def test_unknown_history_cold_is_distinct_from_known_zero():
    from experiments.e1.adaptive_observation import risk_observation
    spec=task('a',duration=60,cpu=1,gpu=0,mem=0,sla=2)
    cold=Pressure(0,(60,0,0),(None,None,None),None,2,1)
    _,unknown=risk_observation(snapshot((spec,)),cold)
    _,quiet=risk_observation(snapshot((spec,)),known_pressure())
    assert unknown['tasks'][0]['budget']['score']==quiet['tasks'][0]['budget']['score']==0
    assert unknown['tasks'][0]['budget']['buffer_minutes']==30
    assert quiet['tasks'][0]['budget']['buffer_minutes']==0
    assert unknown['fields']['risk_history_known']==0 and quiet['fields']['risk_history_known']==1
    assert unknown['fields']['risk_safe_ge60']==0 and quiet['fields']['risk_safe_ge60']==1


def test_current_snapshot_after_real_start_updates_z_and_predecessor_membership():
    from experiments.e1.adaptive_observation import risk_observation
    early=task('a',duration=60,cpu=5,gpu=0,mem=0,sla=1)
    later=task('b',duration=60,cpu=1,gpu=0,mem=0,sla=3)
    _,before=risk_observation(snapshot((early,later)),known_pressure())
    running=RunningTask('a',(5,0,0),START+pd.Timedelta(minutes=60))
    _,after=risk_observation(snapshot((later,),(running,)),known_pressure())
    assert before['tasks'][1]['budget']['score']==.25
    assert after['tasks'][0]['budget']['score']==.5
    assert after['tasks'][0]['budget']['urgent_pressure']==(0,0,0)
    assert after['fields']['risk_u_cpu']==.5


def test_declared_slack_scale_is_used_without_fitting_to_input():
    from experiments.e1.adaptive_observation import RISK_FIELDS,risk_observation
    from experiments.e1.rl_env import ObservationNormalization
    spec=task('a',duration=60,cpu=1,gpu=0,mem=0,sla=2)
    default,_=risk_observation(snapshot((spec,)),known_pressure())
    changed,_=risk_observation(snapshot((spec,)),known_pressure(),ObservationNormalization(slack_minutes=240))
    dictionary,_=risk_observation(snapshot((spec,)),known_pressure(),{'slack_minutes':240})
    index=RISK_FIELDS.index('risk_solo_slack_max')
    assert default[index]==pytest.approx(1/3)
    assert changed[index]==pytest.approx(.2)
    np.testing.assert_array_equal(changed,dictionary)


def test_future_workload_and_weather_fork_cannot_enter_current_only_sensor():
    from experiments.e1.adaptive_observation import risk_observation
    common=task('a',duration=60,cpu=1,gpu=0,mem=0,sla=2)
    wa,ea=inputs([common,task('later-b',arrival=30)],60)
    wb,eb=inputs([common,task('later-c',arrival=30,duration=45,cpu=4)],60)
    eb=replace(eb,samples=tuple(replace(s,ci_g_per_kwh=999,temperature_c=45)
            if s.time>=START+pd.Timedelta(minutes=30) else s for s in eb.samples))
    a=snapshot(tuple(s for s in wa.tasks if s.arrival_time<=START))
    b=snapshot(tuple(s for s in wb.tasks if s.arrival_time<=START))
    va,ra=risk_observation(a,known_pressure())
    vb,rb=risk_observation(b,known_pressure())
    np.testing.assert_array_equal(va,vb)
    assert ra==rb
    assert [r['task_id'] for r in ra['tasks']]==['a']


def test_online_sensor_does_not_mutate_immutable_snapshot_or_pressure():
    from experiments.e1.adaptive_observation import risk_observation
    snap=snapshot((task('a',duration=60),));pressure=known_pressure((1,2,3))
    before=(asdict(snap),asdict(pressure))
    vector,raw=risk_observation(snap,pressure)
    assert (asdict(snap),asdict(pressure))==before
    raw['tasks'].clear()
    vector2,raw2=risk_observation(snap,pressure)
    np.testing.assert_array_equal(vector,vector2)
    assert len(raw2['tasks'])==1
