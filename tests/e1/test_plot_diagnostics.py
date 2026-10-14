import json

import pytest

from experiments.e1.contracts import file_hash
from experiments.e1.plot_diagnostics import (
    choose_a_display, choose_c_displays, verify_campaign, plot_risk_scatter,
)


def test_a_uses_only_real_verified_a2_and_stable_largest_gap():
    rows = [dict(case_id='synthetic', mode='A1', factor=None, edf_misses=99, oracle_misses=0),
            dict(case_id='b', mode='A2', factor=2, edf_misses=7, oracle_misses=3),
            dict(case_id='a', mode='A2', factor=1.5, edf_misses=5, oracle_misses=1)]
    assert choose_a_display(rows) == 'a'


def test_a_zero_gain_is_preserved_and_empty_rejected():
    assert choose_a_display([dict(case_id='z', mode='A2', factor=1, edf_misses=0, oracle_misses=0)]) == 'z'
    with pytest.raises(ValueError):
        choose_a_display([])


def test_c_selection_keeps_zero_case_and_absent_classes():
    selected, absent = choose_c_displays([
        dict(case_id='b', delta_misses=0, delta_carbon=0),
        dict(case_id='a', delta_misses=0, delta_carbon=0)])
    assert selected == ['a']
    assert absent == ['extra_miss', 'carbon_sla_tradeoff']


def test_c_stable_extreme_and_tradeoff_is_display_only():
    selected, absent = choose_c_displays([
        dict(case_id='a', delta_misses=2, delta_carbon=1),
        dict(case_id='b', delta_misses=1, delta_carbon=-1),
        dict(case_id='c', delta_misses=0, delta_carbon=0)])
    assert selected == ['a', 'c', 'b']
    assert absent == []


def fixture_campaign(path, **overrides):
    path.mkdir()
    (path/'data.csv').write_text('value\n1\n', encoding='utf8')
    receipt = dict(passed=True, split='train', test_access=False,
                   output_hashes={'data.csv': file_hash(path/'data.csv')})
    receipt.update(overrides)
    (path/'receipt.json').write_text(json.dumps(receipt), encoding='utf8')
    (path/'run_status.json').write_text(json.dumps({'status':'complete'}), encoding='utf8')


def test_reader_checks_role_before_following_payload_paths(tmp_path):
    fixture_campaign(tmp_path/'bad', split='test', output_hashes={'missing': '0'*64})
    with pytest.raises(ValueError, match='Train'):
        verify_campaign(tmp_path/'bad')


def test_reader_verifies_content_and_refuses_path_escape(tmp_path):
    fixture_campaign(tmp_path/'ok')
    assert verify_campaign(tmp_path/'ok')['passed']
    (tmp_path/'ok/data.csv').write_text('changed', encoding='utf8')
    with pytest.raises(ValueError, match='hash'):
        verify_campaign(tmp_path/'ok')
    fixture_campaign(tmp_path/'escape', output_hashes={'../other': '0'*64})
    with pytest.raises(ValueError, match='path'):
        verify_campaign(tmp_path/'escape')


def test_reader_rejects_incomplete_run(tmp_path):
    fixture_campaign(tmp_path/'partial')
    (tmp_path/'partial/run_status.json').write_text('{"status":"failed"}', encoding='utf8')
    with pytest.raises(ValueError, match='complete'):
        verify_campaign(tmp_path/'partial')


def test_scatter_exports_all_zero_points_csv_and_editable_formats(tmp_path):
    records = [dict(case_id='fixture-a', factor=1.5, score=.1, delta_misses=0, delta_carbon=0),
               dict(case_id='fixture-b', factor=2, score=1.2, delta_misses=0, delta_carbon=0)]
    result = plot_risk_scatter(records, tmp_path)
    assert result['n'] == 2
    for suffix in ('csv', 'png', 'pdf', 'svg'):
        assert (tmp_path/('C-risk-scatter.'+suffix)).stat().st_size > 0
    assert 'fixture-a' in (tmp_path/'C-risk-scatter.csv').read_text()
    with pytest.raises(FileExistsError):
        plot_risk_scatter(records, tmp_path)


def test_cumulative_integral_is_timestamped_at_interval_end():
    import pandas as pd
    from experiments.e1 import plot_diagnostics as module
    frame=pd.DataFrame([dict(interval_start='2023-05-01T05:00:00Z',
                             interval_end='2023-05-01T05:15:00Z',carbon_kg=2),
                        dict(interval_start='2023-05-01T05:15:00Z',
                             interval_end='2023-05-01T05:30:00Z',carbon_kg=3)])
    times,values=module.cumulative_carbon_curve(frame)
    assert times.iloc[0]==pd.Timestamp('2023-05-01T05:15:00Z')
    assert list(values)==[2,5]
