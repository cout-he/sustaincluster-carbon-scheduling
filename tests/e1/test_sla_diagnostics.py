"""Read-only diagnostic isolation and preserved history contracts."""
import json
from pathlib import Path

import pytest

from experiments.e1.contracts import file_hash, write_json


def test_test_diagnostic_rejected_before_missing_bundle_is_opened(tmp_path):
    from experiments.e1.sla_diagnostics import require_development_split
    with pytest.raises(ValueError, match='Test|test'):
        require_development_split('test')
    require_development_split('train')
    require_development_split('validation')


def test_historical_hash_drift_is_rejected_without_modifying_archive(tmp_path):
    from experiments.e1.sla_diagnostics import audit_historical_group
    group=tmp_path/'old';group.mkdir()
    write_json(group/'run_status.json',{'state':'complete'})
    write_json(group/'fairness.json',{'workload_sha256':'w','environment_sha256':'e'})
    write_json(group/'output_hashes.json',{'fairness.json':file_hash(group/'fairness.json')})
    (group/'fairness.json').write_text('{"tampered": true}')
    before=(group/'fairness.json').read_bytes()
    with pytest.raises(ValueError,match='hash|identity'):
        audit_historical_group(group,'w','e')
    assert (group/'fairness.json').read_bytes()==before


def test_history_inventory_rejects_escaping_hash_path(tmp_path):
    from experiments.e1.sla_diagnostics import audit_historical_group
    group=tmp_path/'old';group.mkdir()
    write_json(group/'run_status.json',{'state':'complete'})
    write_json(group/'output_hashes.json',{'../private.json':'0'*64})
    with pytest.raises(ValueError,match='path'):
        audit_historical_group(group,'w','e')


def test_refuse_output_overwrite_before_experiments_or_input_reads(tmp_path):
    from experiments.e1.sla_diagnostics import run_diagnostics
    out=tmp_path/'old';out.mkdir();marker=out/'keep.txt';marker.write_text('keep')
    with pytest.raises(FileExistsError):
        run_diagnostics({},out)
    assert marker.read_text()=='keep'
    assert list(out.iterdir())==[marker]


def test_pressure_tier_distinguishes_supported_from_near_target():
    from experiments.e1.sla_diagnostics import classify_pressure
    assert classify_pressure({'deadline_violation_rate_observed':0.,'capacity_wait_intervals':0})=='low'
    assert classify_pressure({'deadline_violation_rate_observed':.03,'capacity_wait_intervals':50})=='medium_supported_5pct'
    assert classify_pressure({'deadline_violation_rate_observed':.07,'capacity_wait_intervals':50})=='medium_near_target_not_supported'
    assert classify_pressure({'deadline_violation_rate_observed':.3,'capacity_wait_intervals':50})=='high'


def test_zero_task_reference_does_not_establish_sla_support():
    from experiments.e1.sla_diagnostics import classify_pressure
    assert classify_pressure({'deadline_violation_rate_observed':None,'capacity_wait_intervals':0})=='unclassified'


def test_complete_real_baseline_history_audits_without_rerun(tmp_path):
    from experiments.e1.sla_diagnostics import audit_historical_group
    from experiments.e1.rl_evaluate import EvaluationBundle, evaluate_group
    from .helpers import KnownPower, inputs, task
    # The earlier stepwise scenario intentionally has a short censoring cap.
    # This archive test instead supplies enough coverage for a finite cohort.
    workload,environment=inputs([task(name,duration=15,sla=1) for name in ('a','b','c')],120)
    group=tmp_path/'reference'
    bundle=EvaluationBundle('train',workload,environment,(2,1,3),KnownPower,
        {'threshold_g_per_kwh':161.075},510001,1)
    evaluate_group(bundle,group)
    before={path.relative_to(group):path.read_bytes() for path in group.rglob('*') if path.is_file()}
    audited=audit_historical_group(group,workload.sha256,environment.sha256)
    assert audited['fairness']['all_tasks_completed']
    assert audited['files_verified']>=29
    after={path.relative_to(group):path.read_bytes() for path in group.rglob('*') if path.is_file()}
    assert before==after


def test_sla_target_boundary_is_not_relaxed_for_high_pressure():
    from experiments.e1.sla_diagnostics import classify_pressure
    assert classify_pressure({'deadline_violation_rate_observed':.05,'capacity_wait_intervals':1})=='medium_supported_5pct'
    assert classify_pressure({'deadline_violation_rate_observed':.05001,'capacity_wait_intervals':1})=='medium_near_target_not_supported'
    assert classify_pressure({'deadline_violation_rate_observed':.35,'capacity_wait_intervals':1})=='high'
