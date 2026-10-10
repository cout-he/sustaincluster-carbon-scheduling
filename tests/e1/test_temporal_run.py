import json

import pytest

from .test_run import configuration
from .test_drain import fixture as legacy_fixture
from .helpers import task


def fixture(*args, **kwargs):
    # Synthetic observations are explicitly current-time values, not interpolated CI.
    from experiments.e1.selection import TemporalEnvironment
    w,e=legacy_fixture(*args, **kwargs)
    return w,TemporalEnvironment(**vars(e))


def test_driver_rejects_legacy_interpolated_environment_instead_of_relabeling(tmp_path):
    from experiments.e1.temporal import run_comparison
    from experiments.e1.config import ROOT
    cfg,_=configuration(tmp_path)
    w,e=legacy_fixture([task('a')])
    cfg['workload']['resource_multiplier']=1;cfg['end']=w.end.isoformat()
    with pytest.raises(ValueError,match='TemporalEnvironment'):
        run_comparison(cfg,w,e,ROOT/'configs/dcs/dc_config.json',tmp_path/'legacy',150)


def test_temporal_driver_equal_frozen_cohort_phase_summaries_and_strict_replay(tmp_path):
    from experiments.e1.temporal import run_comparison, replay_comparison
    from experiments.e1.config import ROOT
    cfg,_=configuration(tmp_path)
    w,e=fixture([task('a',duration=30,sla=2)])
    cfg['workload']['resource_multiplier']=1;cfg['workload']['sla_multiplier']=1.5
    # A valid fixture has the same config SLA as each TaskSpec.
    cfg['workload']['sla_multiplier']=2
    from dataclasses import replace
    w=replace(w,sla_multiplier=2)
    cfg['end']=w.end.isoformat()
    base=tmp_path/'base';replay=tmp_path/'replay'
    run_comparison(cfg,w,e,ROOT/'configs/dcs/dc_config.json',base,threshold_g_per_kwh=150)
    replay_comparison(base,replay)
    info=json.loads((base/'fairness.json').read_text())
    assert info['identical_frozen_inputs'] and info['all_tasks_completed'] and info['equal_workload_carbon_comparison_valid']
    assert info['common_end']=='2023-08-01T06:00:00+00:00'
    for policy in ('immediate_fifo','edf_backfill','carbon_threshold_fifo'):
        phases=json.loads((base/policy/'phases.json').read_text())
        assert phases['common']['completed']==1
        assert phases['common']['pending']==phases['common']['running']==0
        assert phases['fixed']['arrived']==1
    prov=json.loads((base/'provenance.json').read_text())
    for name,digest in prov['output_hashes'].items():
        assert (base/name).read_bytes()==(replay/name).read_bytes(),name
    with pytest.raises(FileExistsError): replay_comparison(base,base)


def test_temporal_driver_incomplete_cap_never_claims_equal_workload_carbon(tmp_path):
    from experiments.e1.temporal import run_comparison
    from experiments.e1.config import ROOT
    cfg,_=configuration(tmp_path)
    w,e=fixture([task('a',cpu=60000,sla=1.5)],safety_minutes=60)
    cfg['workload']['resource_multiplier']=1;cfg['end']=w.end.isoformat()
    result=run_comparison(cfg,w,e,ROOT/'configs/dcs/dc_config.json',tmp_path/'cap',150)
    info=json.loads((tmp_path/'cap/fairness.json').read_text())
    assert not info['all_tasks_completed'] and not info['equal_workload_carbon_comparison_valid']
    assert all(r['common']['pending']==1 and r['common']['deadline_missed_unfinished']==1 for r in result.values())


def test_actual_cli_archive_can_be_strictly_replayed(tmp_path,monkeypatch):
    from experiments.e1 import temporal
    cfg,_=configuration(tmp_path)
    w,e=fixture([task('a')])
    cfg['workload']['resource_multiplier']=1
    config=tmp_path/'cfg.json';config.write_text(json.dumps(cfg))
    rule=tmp_path/'rule.json';rule.write_text('{}')
    chosen=dict(selected=dict(start=w.start.isoformat(),end=w.end.isoformat()),threshold_g_per_kwh=150)
    monkeypatch.setattr(temporal,'select_window',lambda *args: chosen)
    monkeypatch.setattr(temporal,'load_workload',lambda *args: w)
    monkeypatch.setattr(temporal,'load_temporal_environment',lambda *args: e)
    monkeypatch.setattr('sys.argv',['temporal','--config',str(config),'--selection-config',str(rule),'--output',str(tmp_path/'cli')])
    temporal.main()
    temporal.replay_comparison(tmp_path/'cli',tmp_path/'replay')
