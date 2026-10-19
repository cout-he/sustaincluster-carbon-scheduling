"""Synthetic boundary cases: no source task/environment payload is read."""
from dataclasses import replace
import csv
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from experiments.e1.environment import EnvironmentalSample, FrozenEnvironment
from experiments.e1.workload import FrozenWorkload, TaskSpec


def scene(durations=(16, 1, 1), gpu=(1, 1, 0), sla=2, ci=200):
    start = pd.Timestamp('2023-01-01T00:00:00Z')
    end = start + pd.Timedelta(hours=12)
    tasks = tuple(TaskSpec(str(i), str(i), 0, i, start, d, .1, g, .1,
                           0, sla, 1, tuple(range(12)))
                  for i, (d, g) in enumerate(zip(durations, gpu)))
    workload = FrozenWorkload(start, start + pd.Timedelta(minutes=15), 'synthetic', 5, sla, 2023, tasks)
    samples = tuple(EnvironmentalSample(t, ci, 20, 15) for t in
                    pd.date_range(start, end, freq='15min', inclusive='left'))
    env = FrozenEnvironment(start, end, 'synthetic-ci', 'synthetic-weather', samples)
    return SimpleNamespace(scenario_id='synthetic', role='train', workload=workload,
                           environment=env, capacity=(10, 1, 10),
                           identity={'role': 'train', 'resource_pressure_factor': 1,
                                     'source_fragment_sha256': 'synthetic-fragment'})


def test_capacity_and_active_wait_are_separate_and_backfill_is_preserved():
    from experiments.object_model_a1.rules import diagnose
    row, details = diagnose(scene(), 'fifo', (10, 1, 10))
    by_id = {r['task_id']: r for r in details['tasks']}
    assert by_id['0']['start_time'] == by_id['2']['start_time']
    assert by_id['1']['capacity_wait_minutes'] == 30
    assert row['capacity_waited_tasks'] == 1
    assert row['policy_deferred_tasks'] == 0
    assert row['gpu_bottleneck_attempts'] == 2
    assert row['cpu_bottleneck_attempts'] == row['mem_bottleneck_attempts'] == 0
    assert row['completed_fraction'] == 1
    assert by_id['0']['finish_time'] == '2023-01-01T00:16:00+00:00'
    assert by_id['0']['completion_observed_at'] == '2023-01-01T00:30:00+00:00'
    assert not any('energy' in key or 'carbon_kg' in key or 'power_kw' in key for key in row)


def test_edf_uses_original_deadline_and_retains_every_task():
    from experiments.object_model_a1.rules import diagnose
    row, details = diagnose(scene(), 'edf', (10, 1, 10))
    by_id = {r['task_id']: r for r in details['tasks']}
    assert by_id['1']['capacity_wait_minutes'] == 0
    assert by_id['0']['capacity_wait_minutes'] == 15
    assert row['policy_defer_minutes_total'] == 0
    assert len(by_id) == 3


def test_threshold_uses_current_ci_and_can_exhaust_individually_safe_slack():
    from experiments.object_model_a1.rules import diagnose
    row, details = diagnose(scene(durations=(60, 60), gpu=(1, 1)),
                            'carbon_threshold_fifo', (10, 1, 10), threshold=100)
    # Both jobs have 60 minutes of theoretical Slack. The rule defers both to
    # the same latest-start boundary; only one fits, so the other misses.
    assert row['policy_deferred_tasks'] == 2
    assert row['capacity_waited_tasks'] == 1
    assert row['deadline_misses'] == 1
    assert row['long_slack_capacity_waited_tasks'] == 1
    assert row['long_slack_capacity_missed_tasks'] == 1
    assert row['completed_fraction'] == 1
    assert all(t['duration_minutes'] == 60 for t in details['tasks'])


def test_actual_duration_ci_integral_excludes_rounded_release_tail():
    from experiments.object_model_a1.rules import diagnose
    s = scene(durations=(16,), gpu=(1,), ci=200)
    row, _ = diagnose(s, 'fifo', (10, 1, 10))
    assert row['gpu_ci_exposure_proxy'] == pytest.approx(200 * 16 / 60)
    assert row['gpu_resource_hours'] == pytest.approx(16 / 60)
    assert row['gpu_mean_execution_ci_g_per_kwh'] == pytest.approx(200)
    assert row['gpu_reserved_ci_exposure_proxy'] == pytest.approx(200 * 30 / 60)


def test_structurally_infeasible_task_fails_without_deletion_or_busy_loop():
    from experiments.object_model_a1.rules import diagnose
    with pytest.raises(ValueError, match='exceeds candidate capacity'):
        diagnose(scene(durations=(60,), gpu=(2,)), 'fifo', (10, 1, 10))


def test_nontrain_rejected_before_executor_access():
    from experiments.object_model_a1.rules import diagnose
    s = scene()
    s.role = 'development'
    with pytest.raises(ValueError, match='Train'):
        diagnose(s, 'fifo', (10, 1, 10))


def test_remaining_solo_slack_distinguishes_missed_from_zero_slot():
    from experiments.object_model_a1.rules import diagnose
    row, _ = diagnose(scene(durations=(60, 60), gpu=(1, 1)),
                      'carbon_threshold_fifo', (10, 1, 10), threshold=100)
    assert row['observed_remaining_legal_wait_slots_negative_tasks'] == 1
    assert row['observed_remaining_legal_wait_slots_0_tasks'] == 1
    assert row['observed_remaining_legal_wait_slots_ge4_tasks'] == 0


def test_fragment_identity_repair_preserves_all_simulation_metrics(tmp_path):
    from experiments.object_model_a1.rules import repair_fragment_ids
    from experiments.e1.contracts import file_hash
    source = tmp_path/'source.csv'
    source.write_text('scenario_id,source_fragment_id,arrival_start\n'
                      'train-20230101T0000-r1-s1.5,real-fragment,2023-01-01T00:00:00+00:00\n',encoding='utf8')
    out = tmp_path/'out'
    out.mkdir()
    rows = out/'scenario-summary.csv'
    rows.write_text('original_scenario_id,arrival_start,source_fragment_id,dmr,deadline_misses\n'
                    'train-20230101T0000-r0.75-s2,2023-01-01T00:00:00+00:00,,0.125,1\n',encoding='utf8')
    (out/'metadata.json').write_text(json.dumps({'script_sha256':'producer-rules-hash'}),encoding='utf8')
    before = file_hash(rows)
    result = repair_fragment_ids(out,source,'producer-commit')
    corrected = list(csv.DictReader(rows.open(encoding='utf8')))[0]
    assert corrected['source_fragment_id'] == 'real-fragment'
    assert corrected['dmr'] == '0.125'
    assert corrected['deadline_misses'] == '1'
    assert result['input_csv_sha256'] == before
    assert result['non_fragment_columns_sha256_before'] == result['non_fragment_columns_sha256_after']
    assert result['repaired_rows'] == 1
    assert json.loads((out/'metadata.json').read_text())['producer_run_rules_sha256'] == 'producer-rules-hash'
