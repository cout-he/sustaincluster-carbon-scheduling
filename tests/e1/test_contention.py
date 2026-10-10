import json

import pandas as pd
import pytest

from .helpers import START, KnownPower, task
from .test_drain import fixture
from .test_run import configuration
from .test_temporal_run import fixture as temporal_fixture


def test_shared_stressed_cohort_drain_sla_resource_integrals_and_replay(tmp_path):
    from experiments.e1.contention import run_levels, replay_levels
    from experiments.e1 import temporal
    from experiments.e1.config import ROOT
    cfg,_=configuration(tmp_path)
    w,e=temporal_fixture([task('a',duration=30),task('b',arrival=15,duration=30)],safety_minutes=180)
    cfg['end']=w.end.isoformat();cfg['workload']['resource_multiplier']=1
    cfg['datacenter'].update(total_cores=1,total_gpus=1,total_mem=1)
    # Isolate power equations here; separate old tests cover the physical adapter.
    original=temporal.UpstreamPowerModel
    temporal.UpstreamPowerModel=lambda *args:KnownPower()
    try:
        out=tmp_path/'levels';replay=tmp_path/'replay'
        rows=run_levels(cfg,w,e,ROOT/'configs/dcs/dc_config.json',out,[1,2],123,150)
        replay_levels(out,replay)
    finally:temporal.UpstreamPowerModel=original
    assert len(rows)==18
    for factor in (1,2):
        common=[r for r in rows if r['factor']==factor and r['phase']=='common']
        assert len({r['observation_end'] for r in common})==1
        assert all(r['arrived']==r['completed']==2*factor for r in common)
        assert all(r['pending']==r['running']==r['deadline_right_censored']==0 for r in common)
        assert all(r['cores_fraction_max']<=1 and r['mem_fraction_max']<=1 for r in common)
        assert len({r['cores_hours'] for r in common})==1
        assert all(r['total_energy_kwh']==pytest.approx(r['it_energy_kwh']+r['hvac_energy_kwh']) for r in common)
        assert all(r['deadline_violation_rate_observed']==(r['deadline_missed_completed']+r['deadline_missed_unfinished'])/r['arrived'] for r in common)
    prov=json.loads((out/'provenance.json').read_text())
    for name,digest in prov['output_hashes'].items():
        assert (out/name).read_bytes()==(replay/name).read_bytes(),name
    with pytest.raises(FileExistsError):run_levels(cfg,w,e,ROOT/'configs/dcs/dc_config.json',out,[1],123,150)


def test_no_future_arrival_decisions_and_censoring_under_safety_cap():
    from experiments.e1.stress import stress_workload
    from experiments.e1.engine import LocalExecutor
    from experiments.e1.policies import CarbonThresholdFIFO, DecisionContext
    from dataclasses import fields
    w,e=fixture([task('a',cpu=2),task('b',arrival=15,duration=1000,sla=10)],safety_minutes=60)
    stressed,_=stress_workload(w,2,123)
    ex=LocalExecutor(stressed,e,(1,1,1),KnownPower(),policy=CarbonThresholdFIFO(150),drain=True)
    r=ex.run(stop_at_drain=True)
    by_id={t.task_id:t for t in stressed.tasks}
    assert {f.name for f in fields(DecisionContext)}=={'now','ci_g_per_kwh'}
    assert all(pd.Timestamp(x['time'])>=by_id[x['task_id']].arrival_time for x in r.policy_events)
    assert all(pd.Timestamp(x['time'])<w.end for x in r.events if x['event']=='arrived')
    assert r.summary['completed']+r.summary['pending']+r.summary['running']==4
    assert r.summary['deadline_missed_unfinished']==2 and r.summary['deadline_right_censored']==2


def test_failure_case_requires_temporal_order_and_both_comparators():
    from experiments.e1.contention import contention_cases
    met=dict(task_id='a',deadline_outcome='met',arrival_time=START.isoformat(),start_time=START.isoformat(),finish_time=(START+pd.Timedelta(minutes=30)).isoformat(),duration=30,cores_req=1,gpu_req=0,mem_req=1,sla_deadline=(START+pd.Timedelta(minutes=45)).isoformat())
    missed={**met,'deadline_outcome':'missed'}
    ledgers={p:[met] for p in ('immediate_fifo','edf_backfill')};ledgers['carbon_threshold_fifo']=[missed]
    events=[dict(task_id='a',event='policy_deferred',time=START.isoformat()),dict(task_id='a',event='capacity_waited',time=(START+pd.Timedelta(minutes=15)).isoformat())]
    counts,cases=contention_cases(ledgers,events)
    assert counts['carbon_only_missed_vs_both']==1 and len(cases)==1
    assert cases[0]['task_id']=='a'
    assert not contention_cases(ledgers,list(reversed([dict(events[0],time=events[1]['time']),dict(events[1],time=events[0]['time'])])))[1]
    ledgers['edf_backfill']=[missed]
    assert not contention_cases(ledgers,events)[1]
    ledgers['edf_backfill']=[dict(met,deadline_outcome='right_censored')]
    assert not contention_cases(ledgers,events)[1]
