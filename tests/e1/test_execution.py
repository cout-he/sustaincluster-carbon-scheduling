"""Real Task/resource execution with independent tiny scheduling fixtures."""
import pandas as pd
import pytest

from .helpers import START, execute, inputs, task, KnownPower


def rows(result):
    return {r['task_id']: r for r in result.tasks}


def test_immediate_start_release_before_next_arrival_and_resource_conservation():
    result = execute([task('a'), task('b', arrival=15)])
    ledger = rows(result)
    assert ledger['a']['start_time'] == START.isoformat()
    assert ledger['b']['start_time'] == (START + pd.Timedelta(minutes=15)).isoformat()
    assert ledger['a']['state'] == ledger['b']['state'] == 'completed'
    assert result.summary['arrived'] == 2 and result.summary['completed'] == 2
    assert all(s['used_cores'] == 1 and s['available_cores'] == 0 for s in result.steps)


def test_fifo_skips_infeasible_head_and_backfills():
    result = execute([task('a', duration=30), task('b', cpu=2), task('c')], capacity=(2, 1, 3))
    ledger = rows(result)
    assert ledger['a']['start_time'] == ledger['c']['start_time'] == START.isoformat()
    assert ledger['b']['state'] == 'pending' and ledger['b']['start_time'] is None
    assert ledger['b']['wait_intervals'] == 2
    assert result.steps[0]['used_cores'] == 2


@pytest.mark.parametrize('resource', ['gpu', 'mem'])
def test_backfill_and_release_enforce_gpu_and_memory_independently(resource):
    demands = {'cpu': 0.1, 'gpu': 0.1, 'mem': 0.1}
    demands[resource] = 0.6
    small = dict(demands); small[resource] = 0.4
    result = execute([task('a', **demands), task('b', **demands), task('c', **small)],
                     capacity=(1, 1, 1))
    ledger = rows(result)
    assert ledger['a']['start_time'] == ledger['c']['start_time'] == START.isoformat()
    assert ledger['b']['start_time'] == (START + pd.Timedelta(minutes=15)).isoformat()
    assert all(r['state'] == 'completed' for r in result.tasks)
    for step in result.steps:
        for key in ('cores', 'gpus', 'mem'):
            assert 0 <= step[f'used_{key}'] <= 1
            assert step[f'used_{key}'] + step[f'available_{key}'] == pytest.approx(1)


@pytest.mark.parametrize('cpu,gpu,mem,capacity', [(2, 0, 1, (1, 1, 1)),
                                                (1, 2, 1, (1, 1, 1)),
                                                (1, 0, 2, (1, 1, 1))])
def test_oversized_each_resource_stays_in_ledger_without_overallocation(cpu, gpu, mem, capacity):
    result = execute([task('a', cpu=cpu, gpu=gpu, mem=mem, sla=1)], capacity=capacity)
    row = result.tasks[0]
    assert row['state'] == 'pending' and row['structurally_infeasible']
    assert row['deadline_outcome'] == 'missed_unfinished'
    assert all(s['running'] == 0 and s['used_cores'] == 0 and s['used_gpus'] == 0 and s['used_mem'] == 0
               for s in result.steps)


def test_exact_finish_for_deadline_and_grid_resource_release():
    result = execute([task('a', duration=5, sla=1), task('b', duration=5, sla=1)])
    ledger = rows(result)
    assert ledger['a']['finish_time'] == '2023-08-01T05:05:00+00:00'
    assert ledger['a']['completion_observed_at'] == '2023-08-01T05:15:00+00:00'
    assert ledger['a']['deadline_outcome'] == 'met'
    assert ledger['b']['start_time'] == '2023-08-01T05:15:00+00:00'
    assert ledger['b']['deadline_outcome'] == 'missed'


def test_horizon_completion_flush_missed_and_right_censored_states():
    result = execute([task('a', duration=30, sla=1), task('b', duration=45, sla=0.5),
                      task('c', duration=60, sla=2)], capacity=(3, 1, 3))
    ledger = rows(result)
    assert ledger['a']['state'] == 'completed' and ledger['a']['deadline_outcome'] == 'met'
    assert ledger['b']['state'] == 'running' and ledger['b']['deadline_outcome'] == 'missed_unfinished'
    assert ledger['c']['state'] == 'running' and ledger['c']['deadline_outcome'] == 'right_censored'
    assert result.summary['completed'] + result.summary['running'] + result.summary['pending'] == 3
    assert len(result.steps) == 2  # No extra interval at T.


def test_pending_deadline_equal_end_is_missed_and_not_silently_cancelled():
    result = execute([task('a', duration=60), task('b', duration=15, sla=2)])
    assert rows(result)['b']['state'] == 'pending'
    assert rows(result)['b']['deadline_outcome'] == 'missed_unfinished'


def test_deterministic_repeated_executor_and_empty_workload():
    a = execute([task('a', cpu=0.1, gpu=0.2, mem=0.3)])
    b = execute([task('a', cpu=0.1, gpu=0.2, mem=0.3)])
    assert a == b
    result = execute([])
    assert result.summary['arrived'] == 0 and result.tasks == ()
    assert result.summary['total_energy_kwh'] == 2  # Idle IT+HVAC still charged.


@pytest.mark.parametrize('resource,field', [('cpu', 'cores'), ('gpu', 'gpus'), ('mem', 'mem')])
def test_fractional_exact_fit_does_not_introduce_one_interval_delay(resource, field):
    tasks = []
    for name, value in zip('abc', (0.2, 0.6, 0.2)):
        demands = dict(cpu=0, gpu=0, mem=0); demands[resource] = value
        tasks.append(task(name, **demands))
    result = execute(tasks)
    assert all(r['start_time'] == START.isoformat() for r in result.tasks)
    assert result.steps[0][f'used_{field}'] == 1 and result.steps[0][f'available_{field}'] == 0
    assert all(r['deadline_outcome'] == 'met' for r in result.tasks)


def test_every_task_records_local_zero_transfer_semantics():
    result = execute([task('a'), task('b', cpu=2)])
    assert all(row[key] == 0 for row in result.tasks for key in (
        'transmission_delay_seconds', 'transmission_energy_kwh', 'transmission_carbon_kg'))


def test_mismatched_environment_window_and_duplicate_task_ids_fail():
    from experiments.e1.engine import LocalExecutor
    workload, _ = inputs([task('a')], 30)
    _, other_environment = inputs([], 15)
    with pytest.raises(ValueError, match='window'):
        LocalExecutor(workload, other_environment, (1, 1, 1), KnownPower())
    with pytest.raises(ValueError, match='Duplicate'):
        inputs([task('a'), task('a')])


def test_power_failure_propagates_instead_of_reusing_old_info():
    class BrokenPower:
        def measure(self, *args):
            raise RuntimeError('physical failure')
    with pytest.raises(RuntimeError, match='physical failure'):
        execute([task('a')], power=BrokenPower())
