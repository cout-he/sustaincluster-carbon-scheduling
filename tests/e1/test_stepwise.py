"""Catch any change in boundary ordering or physics caused by kernel extraction."""
from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.e1.cars import make_cars_policy
from experiments.e1.engine import LocalExecutor
from .helpers import START, KnownPower, inputs, task

NAMES=('immediate_fifo','edf_backfill','carbon_threshold_fifo',
       'edf_carbon_threshold','cars_risk_only','cars_full')

def scenario():
    return inputs([task('a',duration=20,gpu=.6,sla=2),
                   task('b',duration=10,gpu=.4,sla=4),
                   task('c',duration=15,cpu=2,sla=3),
                   task('d',arrival=15,cpu=.5,mem=.4,sla=4),
                   task('e',arrival=30,duration=31,gpu=.9,mem=2,sla=1.5),
                   task('f',arrival=45,duration=12,cpu=3),
                   task('g',arrival=60,cpu=0,gpu=0,mem=0,sla=2)],90)

def executor(name='edf_backfill', workload=None, environment=None):
    if workload is None: workload, environment=scenario()
    params=dict(threshold_g_per_kwh=150)
    return LocalExecutor(workload,environment,(2,1,3),KnownPower(),
                         make_cars_policy(name,params,workload.start,(2,1,3)))

@pytest.mark.parametrize('name',NAMES)
def test_batch_and_stepwise_match_pre_refactor_golden(name):
    golden=json.loads((Path(__file__).parent/'fixtures/e1_5a_golden.json').read_text())['results'][name]
    batch=executor(name).run()
    assert json.loads(json.dumps(asdict(batch)))==golden
    live=executor(name)
    live.start()
    while not live.finished:
        interval=live.advance()
        assert interval['interval_hours']==.25
        live.check_ownership()
    assert json.loads(json.dumps(asdict(live.result())))==golden
    with pytest.raises(RuntimeError): live.advance()
    with pytest.raises(RuntimeError): live.run()

def test_prepared_boundary_has_arrivals_and_releases_but_no_scheduling():
    w,e=inputs([task('a',duration=5),task('b',arrival=15)],30)
    live=LocalExecutor(w,e,(1,1,1),KnownPower())
    live.start()
    assert list(live.ledger)==['a'] and not live.steps and not live.pool.running_tasks
    assert len(live.pool.pending_tasks)==1
    first=live.advance()
    assert first['used_cores']==1
    assert live.sample.time==START+pd.Timedelta(minutes=15)
    assert live.ledger['a']['state']=='completed'
    assert live.ledger['b']['state']=='pending'
    live.advance()
    assert live.finished
    assert live.result().tasks[0]['finish_time']=='2023-08-01T05:05:00+00:00'

def test_early_until_is_exclusive_and_returns_only_arrived_prefix():
    w,e=inputs([task('a',duration=45),task('b',arrival=30)],60)
    live=LocalExecutor(w,e,(1,1,1),KnownPower())
    live.start(until=START+pd.Timedelta(minutes=30))
    live.advance(); live.advance()
    result=live.result()
    assert live.finished and result.summary['arrived']==1
    assert len(result.steps)==2 and result.tasks[0]['state']=='running'
    assert result.tasks[0]['deadline_outcome']=='right_censored'
    assert all(event['task_id']!='b' for event in result.events)

def test_a_running_task_is_not_preempted_by_a_policy_switch():
    w,e=inputs([task('a',duration=30),task('b',arrival=15,cpu=.5)],45)
    live=LocalExecutor(w,e,(1,1,1),KnownPower())
    live.start(); live.advance()
    running=live.pool.running_tasks[0]
    live.advance(make_cars_policy('cars_risk_only',{'threshold_g_per_kwh':150},START,(1,1,1)))
    assert running.start_time==START and running.finish_time==START+pd.Timedelta(minutes=30)
    assert live.steps[1]['used_cores']==1
    live.advance()
    assert live.result().summary['completed']==2

def test_partial_cutoff_cannot_mark_whole_finite_cohort_drained():
    w,e=inputs([task('a'),task('b',arrival=30)],60)
    live=LocalExecutor(w,e,(1,1,1),KnownPower(),drain=True)
    live.start(until=START+pd.Timedelta(minutes=15));live.advance()
    assert live.result().summary['completed']==1
    assert live.drained_at is None
    with pytest.raises(ValueError): live.extend_idle(e.end)

def test_result_requires_at_least_one_accounted_interval():
    live=executor();live.start()
    with pytest.raises(RuntimeError,match='interval'): live.result()
