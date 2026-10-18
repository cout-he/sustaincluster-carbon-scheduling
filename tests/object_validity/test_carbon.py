import numpy as np
import pytest
from types import SimpleNamespace
import pandas as pd

from experiments.object_validity.a0_carbon import integral_opportunity, segmented_acf, read_existing_rules, write_csv


def test_full_runtime_integral_can_reverse_start_instant_ranking():
    # Start CI favours slot 1, but its full two-slot run includes the spike.
    row = integral_opportunity(np.array([100., 50., 500., 20.]), 0, 30., 15.)
    assert row['best_offset_slots'] == 0
    assert row['immediate_ci_integral_g_per_kwh_h'] == pytest.approx(37.5)
    assert row['saving_ci_integral_g_per_kwh_h'] == 0


def test_integral_uses_rounded_full_slots_and_legal_slack_only():
    row = integral_opportunity(np.array([200., 100., 50., 0.]), 0, 16., 15.)
    assert row['runtime_slots'] == 2
    assert row['best_offset_slots'] == 1
    assert row['saving_ci_integral_g_per_kwh_h'] == pytest.approx(37.5)
    assert row['legal_wait_slots'] == 1


def test_censored_candidate_window_is_not_claimed_complete():
    row = integral_opportunity(np.array([200., 100.]), 0, 16., 15.)
    assert row['window_complete'] is False
    assert row['candidates_observed'] == 1
    assert row['candidates_required'] == 2


def test_acf_does_not_join_non_contiguous_fragments():
    segments = [np.array([1., 2., 3.]), np.array([10., 9., 8.])]
    row = segmented_acf(segments, 1)
    assert row['pairs'] == 4
    assert row['correlation'] == pytest.approx(np.corrcoef([1,2,10,9], [2,3,9,8])[0,1])


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        integral_opportunity(np.array([100.]), 0, 0., 15.)
    with pytest.raises(ValueError):
        integral_opportunity(np.array([100.]), 0, 15., -1.)


def test_constant_ci_has_no_migration_opportunity():
    row=integral_opportunity(np.full(100,150.),3,47.,30.)
    assert row['runtime_slots']==4
    assert row['saving_fraction']==0
    assert row['window_complete'] is True


def test_reuse_rules_requires_exact_frozen_data_identity(tmp_path):
    s=SimpleNamespace(scenario_id='train-a',identity=dict(workload_sha256='work',environment_sha256='env'),
                      environment=SimpleNamespace(end=pd.Timestamp('2023-01-03T00:00Z')),
                      workload=SimpleNamespace(tasks=(1,2)))
    path=tmp_path/'rules.csv'
    row=dict(scenario_id='train-a',workload_sha256='work',environment_sha256='env',
             window_end=s.environment.end.isoformat(),fixed_setpoint_c=22,arrived=2,policy='edf_backfill')
    write_csv(path,[row])
    assert len(read_existing_rules(path,[s]))==1
    row['environment_sha256']='changed'
    write_csv(path,[row])
    with pytest.raises(ValueError,match='identity/design'):
        read_existing_rules(path,[s])
