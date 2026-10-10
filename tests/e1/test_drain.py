"""Finite arrivals, genuine extended environment, shared drain and idle accounting."""
from dataclasses import replace

import pandas as pd
import pytest

from .helpers import START, KnownPower, inputs, task


def fixture(tasks, arrival_minutes=30, safety_minutes=120, ci=None):
    from experiments.e1.environment import EnvironmentalSample
    w, e = inputs(tasks, arrival_minutes)
    end = START + pd.Timedelta(minutes=safety_minutes)
    times = pd.date_range(START, end, freq='15min', inclusive='left')
    values = ci if ci is not None else [200 if t < w.end else 100 for t in times]
    e = replace(e, end=end, samples=tuple(EnvironmentalSample(t,c,20,18) for t,c in zip(times,values)))
    return w,e


def executor(tasks, policy=None, **kw):
    from experiments.e1.engine import LocalExecutor
    w,e = fixture(tasks, **kw)
    return LocalExecutor(w,e,(1,1,1),KnownPower(),policy=policy,drain=True)


def test_drain_stops_at_completion_release_boundary_without_loading_after_arrival_end():
    ex = executor([task('a', duration=35, sla=1.5)])
    result = ex.run(stop_at_drain=True)
    assert ex.drained_at == START + pd.Timedelta(minutes=45)
    assert ex.observed_end == ex.drained_at
    assert len(result.steps) == 3
    assert result.summary['completed'] == result.summary['arrived'] == 1
    assert result.tasks[0]['finish_time'] == (START + pd.Timedelta(minutes=35)).isoformat()
    assert result.tasks[0]['completion_observed_at'] == ex.drained_at.isoformat()
    assert all(e['time'] < ex.workload.end.isoformat() for e in result.events if e['event']=='arrived')
    assert result.summary['horizon'] == 'finite_cohort_drain'


def test_common_endpoint_integrates_same_idle_physics_and_reduces_matched_work_carbon_in_known_fixture():
    from experiments.e1.policies import CarbonThresholdFIFO
    fifo = executor([task('a', duration=30, sla=2)])
    carbon = executor([task('a', duration=30, sla=2)], CarbonThresholdFIFO(150))
    f0, c = fifo.run(stop_at_drain=True), carbon.run(stop_at_drain=True)
    assert fifo.drained_at == START + pd.Timedelta(minutes=30)
    assert carbon.drained_at == START + pd.Timedelta(minutes=60)
    f = fifo.extend_idle(carbon.drained_at)
    assert f.summary['completed'] == c.summary['completed'] == 1
    assert f.summary['total_energy_kwh'] == c.summary['total_energy_kwh'] == 5
    assert f.summary['carbon_kg'] == pytest.approx(.8)
    assert c.summary['carbon_kg'] == pytest.approx(.7)
    assert [s['running'] for s in f.steps] == [1,1,0,0]
    assert f0.summary['total_energy_kwh'] == 3
    assert all(s['it_power_kw']==2 and s['hvac_power_kw']==2 for s in f.steps[2:])
    assert f.summary['observation_end'] == c.summary['observation_end']
    assert carbon.policy.name == 'carbon_threshold_fifo'
    assert c.policy_summary['policy_defer_intervals'] == 2
    assert c.tasks[0]['deadline_outcome']=='met'


def test_drain_has_safety_cap_and_does_not_claim_oversized_task_completed():
    ex = executor([task('a',cpu=2,duration=15,sla=1)],safety_minutes=60)
    result = ex.run(stop_at_drain=True)
    assert ex.drained_at is None
    assert ex.observed_end == START + pd.Timedelta(minutes=60)
    assert result.summary['pending']==1 and result.summary['completed']==0
    assert result.tasks[0]['deadline_outcome']=='missed_unfinished'
    with pytest.raises(ValueError): ex.extend_idle(ex.observed_end)


def test_stop_at_drain_waits_for_arrival_window_even_if_queue_temporarily_empty():
    ex = executor([task('a',duration=5),task('b',arrival=15,duration=5)])
    result=ex.run(stop_at_drain=True)
    assert ex.drained_at==START+pd.Timedelta(minutes=30)
    assert result.summary['arrived']==result.summary['completed']==2


def test_idle_extension_requires_aligned_covered_future_endpoint_and_single_use_run():
    ex=executor([task('a')]); ex.run(stop_at_drain=True)
    for minutes in (15,31,135):
        with pytest.raises(ValueError): ex.extend_idle(START+pd.Timedelta(minutes=minutes))
    with pytest.raises(RuntimeError): ex.run(stop_at_drain=True)


def test_phase_metrics_separate_fixed_cutoff_completion_censoring_waiting_and_jct():
    from experiments.e1.policies import CarbonThresholdFIFO
    from experiments.e1.drain_metrics import phase_summary
    ex=executor([task('a',duration=30,sla=2)],CarbonThresholdFIFO(150))
    result=ex.run(stop_at_drain=True)
    fixed=phase_summary(result,ex.workload.end)
    final=phase_summary(result,ex.observed_end)
    assert fixed['arrived']==fixed['pending']==1 and fixed['completed']==0
    assert fixed['deadline_right_censored']==1 and fixed['deadline_violation_rate_observed']==0
    assert fixed['policy_defer_intervals']==2 and fixed['mean_observed_pending_minutes']==30
    assert fixed['mean_jct_minutes_completed'] is None
    assert final['completed']==1 and final['pending']==final['running']==0
    assert final['mean_wait_minutes_started']==30 and final['mean_jct_minutes_completed']==60
    assert final['deadline_violation_rate_observed']==0
    assert fixed['total_energy_kwh']==2 and final['total_energy_kwh']==5


def test_extended_future_ci_cannot_change_prefix_decisions():
    from experiments.e1.policies import CarbonThresholdFIFO
    prefix=[]
    for future in (0,999):
        ex=executor([task('a',duration=30,sla=2)],CarbonThresholdFIFO(150),ci=[200,200]+[future]*6)
        result=ex.run(stop_at_drain=True)
        prefix.append([e for e in result.policy_events if e['time'] < ex.workload.end.isoformat()])
    assert prefix[0]==prefix[1]


def test_fixed_ledger_does_not_inherit_future_capacity_waits_or_assignment():
    from experiments.e1.drain_metrics import phase_rows
    ex=executor([task('a',duration=45),task('b',arrival=15,duration=15)])
    result=ex.run(stop_at_drain=True)
    final=next(r for r in result.tasks if r['task_id']=='b')
    fixed=next(r for r in phase_rows(result,ex.workload.end) if r['task_id']=='b')
    assert final['wait_intervals']==2 and final['dest_dc_id']==1
    assert fixed['wait_intervals']==1
    assert fixed['state']=='pending' and fixed['dest_dc_id'] is None
    assert fixed['start_time'] is None and fixed['finish_time'] is None
