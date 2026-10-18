from types import SimpleNamespace

import pandas as pd
import pytest

from experiments.e1.environment import EnvironmentalSample, FrozenEnvironment
from experiments.e1.workload import FrozenWorkload, TaskSpec
from experiments.object_validity.a0_resources import diagnose_scenario, task_static_rows, unconstrained_demand


def scenario(durations=(16, 1, 1), gpu=(1, 1, 0)):
    start = pd.Timestamp('2023-01-01T00:00:00Z')
    end = start + pd.Timedelta(hours=2)
    tasks = tuple(TaskSpec(str(i), str(i), 0, i, start, d, .1, g, .1,
                           0, 2, 1, tuple(range(12)))
                  for i, (d, g) in enumerate(zip(durations, gpu)))
    workload = FrozenWorkload(start, start + pd.Timedelta(minutes=15), 'synthetic', 1, 2, 2023, tasks)
    samples = tuple(EnvironmentalSample(t, 200, 20, 15) for t in
                    pd.date_range(start, end, freq='15min', inclusive='left'))
    env = FrozenEnvironment(start, end, 'ci', 'weather', samples)
    return SimpleNamespace(scenario_id='synthetic', role='train', workload=workload,
                           environment=env, capacity=(10, 1, 10), identity={}, statistics={})


def test_capacity_wait_is_gpu_only_and_backfill_starts_fitting_task():
    row, details = diagnose_scenario(scenario(), 'fifo')
    tasks = {r['task_id']: r for r in details['tasks']}
    assert tasks['0']['start_time'] == tasks['2']['start_time']
    assert tasks['1']['capacity_wait_minutes'] == 30
    assert tasks['1']['start_time'] == '2023-01-01T00:30:00+00:00'
    assert row['capacity_waited_tasks'] == 1
    assert row['gpu_bottleneck_attempts'] == 2
    assert row['cpu_bottleneck_attempts'] == row['mem_bottleneck_attempts'] == 0
    assert row['policy_deferred_tasks'] == 0
    assert row['dmr'] == pytest.approx(1/3)
    # Actual finish uses 16min; resource release and left-hold use 30min.
    assert tasks['0']['finish_time'] == '2023-01-01T00:16:00+00:00'
    assert tasks['0']['completion_observed_at'] == '2023-01-01T00:30:00+00:00'
    assert row['gpu_fraction_mean_full'] == pytest.approx(3/8)


def test_edf_orders_by_deadline_and_no_policy_wait():
    row, details = diagnose_scenario(scenario(), 'edf')
    tasks = {r['task_id']: r for r in details['tasks']}
    assert tasks['1']['capacity_wait_minutes'] == 0
    assert tasks['0']['capacity_wait_minutes'] == 15
    assert row['policy_defer_minutes_total'] == 0
    assert row['completed_fraction'] == 1


def test_slack_slot_floor_retains_subslot_tasks_and_sla_counterfactual():
    rows = task_static_rows(scenario(durations=(1, 15, 30), gpu=(0, 0, 0)))
    assert [r['legal_wait_slots'] for r in rows] == [0, 1, 2]
    assert [r['legal_wait_slots_sla1_5'] for r in rows] == [0, 0, 1]
    assert [r['legal_wait_slots_sla2_0'] for r in rows] == [0, 1, 2]
    assert len(rows) == 3


def test_unconstrained_demand_does_not_enforce_gpu_capacity_and_uses_rounded_release():
    rows, curves = unconstrained_demand(scenario())
    gpu = next(r for r in rows if r['resource'] == 'gpu' and r['window'] == 'arrival')
    assert gpu['demand_max'] == 2
    assert gpu['fraction_of_current_capacity_max'] == 2
    # Curve ends at exclusive 30min drain, so the empty [30,45) slot is excluded.
    assert curves['gpu'] == pytest.approx([2, 1])
