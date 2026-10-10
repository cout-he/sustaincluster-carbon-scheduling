"""Independent scheduling fixtures: policy order/eligibility, shared execution."""
from dataclasses import FrozenInstanceError, replace

import pandas as pd
import pytest

from .helpers import START, KnownPower, inputs, task


def policies():
    from experiments.e1.policies import ImmediateFIFO, EDFBackfill, CarbonThresholdFIFO
    return (ImmediateFIFO(), EDFBackfill(), CarbonThresholdFIFO(200))


def execute_policy(tasks, policy, minutes=30, ci=None, capacity=(1, 1, 1)):
    from experiments.e1.engine import LocalExecutor
    workload, environment = inputs(tasks, minutes)
    if ci is not None:
        environment = replace(environment, samples=tuple(replace(s, ci_g_per_kwh=value)
                              for s, value in zip(environment.samples, ci)))
    return LocalExecutor(workload, environment, capacity, KnownPower(), policy=policy).run()


def rows(result):
    return {r['task_id']: r for r in result.tasks}


def test_edf_reorders_old_backlog_together_with_new_arrivals_every_step():
    from experiments.e1.policies import EDFBackfill
    result = execute_policy([task('z', duration=15, sla=1), task('a', duration=30, sla=5),
                             task('b', arrival=15, duration=15, sla=1)], EDFBackfill(), minutes=60)
    assert result.policy_steps[1]['ordered_waiting_ids'] == ['b', 'a']
    assert rows(result)['b']['start_time'] == (START + pd.Timedelta(minutes=15)).isoformat()
    assert rows(result)['a']['start_time'] == (START + pd.Timedelta(minutes=30)).isoformat()


def test_edf_stable_tie_break_is_deadline_arrival_then_lexical_stable_id():
    from experiments.e1.policies import EDFBackfill
    a, b = task('a10', sla=2), task('a2', sla=2)
    c = task('c', arrival=15, sla=1)  # same deadline, later arrival
    assert [s.task_id for s in EDFBackfill().order((c, b, a))] == ['a10', 'a2', 'c']


def test_edf_backfills_later_tasks_behind_oversized_earliest_deadline():
    from experiments.e1.policies import EDFBackfill
    result = execute_policy([task('a', cpu=3, sla=1), task('b', duration=30, sla=1),
                             task('c', sla=3)], EDFBackfill(), capacity=(2, 1, 2))
    assert result.policy_steps[0]['ordered_waiting_ids'] == ['a', 'b', 'c']
    assert result.policy_steps[0]['started_ids'] == ['b', 'c']
    assert rows(result)['a']['state'] == 'pending'
    assert rows(result)['a']['deadline_outcome'] == 'missed_unfinished'


@pytest.mark.parametrize('ci,expected', [(199, False), (200, False), (201, True)])
def test_threshold_strict_boundary_and_no_future_access(ci, expected):
    from experiments.e1.policies import CarbonThresholdFIFO, DecisionContext
    ctx = DecisionContext(START, ci)
    assert CarbonThresholdFIFO(200).decide(task('a', sla=3), ctx).defer is expected
    assert not hasattr(ctx, 'forecast') and not hasattr(ctx, 'environment')


@pytest.mark.parametrize('duration,sla,expected_start,defer_count', [
    (10, 2.5, 15, 1),  # next start equals exact latest feasible start
    (10, 2, 0, 0),     # only 10 min slack, cannot safely wait a 15 min step
    (5, 1, 0, 0),      # latest start is now
    (5, 0.5, 0, 0),    # latest start already passed
])
def test_latest_start_exact_equality_off_grid_and_expired(duration, sla, expected_start, defer_count):
    from experiments.e1.policies import CarbonThresholdFIFO
    result = execute_policy([task('a', duration=duration, sla=sla)], CarbonThresholdFIFO(200), ci=[201, 201])
    assert rows(result)['a']['start_time'] == (START + pd.Timedelta(minutes=expected_start)).isoformat()
    assert result.policy_tasks[0]['policy_defer_intervals'] == defer_count
    assert result.policy_tasks[0]['capacity_wait_intervals'] == 0


def test_capacity_waiting_is_not_claimed_as_policy_defer_even_with_carbon_slack():
    from experiments.e1.policies import CarbonThresholdFIFO
    result = execute_policy([task('a', cpu=2, sla=10), task('b', sla=10)],
                            CarbonThresholdFIFO(200), ci=[250, 250])
    diag = {t['task_id']: t for t in result.policy_tasks}
    assert diag['a']['capacity_wait_intervals'] == 2 and diag['a']['policy_defer_intervals'] == 0
    assert diag['b']['capacity_wait_intervals'] == 0 and diag['b']['policy_defer_intervals'] == 2
    assert rows(result)['a']['wait_intervals'] == 2 and rows(result)['b']['wait_intervals'] == 0
    assert all(s['capacity_waiting_ids'] == ['a'] and s['policy_deferred_ids'] == ['b']
               and s['pending_end_ids'] == ['a', 'b'] for s in result.policy_steps)
    assert {e['event'] for e in result.policy_events} == {'capacity_waited', 'policy_deferred'}
    assert all(e['can_fit_at_decision'] is (e['event'] != 'capacity_waited') for e in result.policy_events)


def test_expired_waiting_task_is_started_when_capacity_frees_without_cancellation():
    from experiments.e1.policies import CarbonThresholdFIFO
    result = execute_policy([task('a', duration=45, sla=1), task('b', duration=15, sla=1)],
                            CarbonThresholdFIFO(200), minutes=60, ci=[250] * 4)
    assert rows(result)['b']['start_time'] == (START + pd.Timedelta(minutes=45)).isoformat()
    assert rows(result)['b']['deadline_outcome'] == 'missed'
    diag = {t['task_id']: t for t in result.policy_tasks}
    assert diag['b']['capacity_wait_intervals'] == 3 and diag['b']['policy_defer_intervals'] == 0


def test_policy_defer_then_low_carbon_start_and_current_load_integration():
    from experiments.e1.policies import CarbonThresholdFIFO
    result = execute_policy([task('a', sla=3)], CarbonThresholdFIFO(200), ci=[201, 200])
    assert rows(result)['a']['start_time'] == (START + pd.Timedelta(minutes=15)).isoformat()
    assert rows(result)['a']['state'] == 'completed' and rows(result)['a']['deadline_outcome'] == 'met'
    assert result.policy_tasks[0]['policy_defer_intervals'] == 1
    assert result.policy_tasks[0]['observed_pending_minutes'] == 15
    assert result.policy_summary['policy_defer_intervals'] == 1
    assert [s['it_power_kw'] for s in result.steps] == [2, 4]
    assert result.summary['total_energy_kwh'] == 2.5
    assert result.summary['carbon_kg'] == pytest.approx(1 * 201 / 1000 + 1.5 * 200 / 1000)


def test_all_policies_use_identical_immutable_inputs_and_conserve_all_resources():
    specs = [task('a', cpu=0.6, gpu=0.6, mem=0.6, duration=30, sla=2),
             task('b', cpu=0.6, gpu=0.6, mem=0.6, sla=4),
             task('c', cpu=0.4, gpu=0.4, mem=0.4, sla=4)]
    original = [s.to_dict() for s in specs]
    for policy in policies():
        result = execute_policy(specs, policy, minutes=60, ci=[250, 150, 250, 150])
        assert [s.to_dict() for s in specs] == original
        assert [r['task_id'] for r in result.tasks] == [s.task_id for s in specs]
        for spec, row in zip(specs, result.tasks):
            for key in ('duration', 'cores_req', 'gpu_req', 'mem_req', 'arrival_time', 'raw_fields'):
                assert row[key] == spec.to_dict()[key]
        for step in result.steps:
            for resource in ('cores', 'gpus', 'mem'):
                assert 0 <= step[f'used_{resource}'] <= 1
                assert step[f'used_{resource}'] + step[f'available_{resource}'] == pytest.approx(1)


def test_three_policy_terminal_censoring_and_no_terminal_energy_or_forced_start():
    for policy in policies():
        result = execute_policy([task('a', sla=5), task('b', duration=60, sla=2),
                                 task('c', duration=45, sla=0.5)], policy,
                                capacity=(3, 1, 3), ci=[250, 250])
        assert len(result.steps) == len(result.policy_steps) == 2
        assert sum(result.summary[k] for k in ('completed', 'pending', 'running')) == 3
        assert rows(result)['c']['deadline_outcome'] == 'missed_unfinished'
        assert rows(result)['b']['deadline_outcome'] == 'right_censored'
        assert all(e['time'] < (START + pd.Timedelta(minutes=30)).isoformat() for e in result.policy_events)
    assert rows(result)['a']['state'] == 'pending'  # Carbon may wait beyond arrival window.


@pytest.mark.parametrize('kind', ['drop', 'duplicate', 'replace', 'mutate'])
def test_policy_cannot_drop_duplicate_replace_or_mutate_waiting_tasks(kind):
    from experiments.e1.policies import ImmediateFIFO
    class BadPolicy(ImmediateFIFO):
        def order(self, waiting):
            if kind == 'drop': return ()
            if kind == 'duplicate': return waiting + waiting
            if kind == 'replace': return (replace(waiting[0], duration=999),)
            waiting[0].duration = 999
    with pytest.raises((ValueError, FrozenInstanceError)):
        execute_policy([task('a')], BadPolicy())


@pytest.mark.parametrize('threshold', [-1, float('nan'), float('inf'), True])
def test_threshold_must_be_explicit_finite_nonnegative_literal(threshold):
    from experiments.e1.policies import CarbonThresholdFIFO
    with pytest.raises(ValueError): CarbonThresholdFIFO(threshold)
