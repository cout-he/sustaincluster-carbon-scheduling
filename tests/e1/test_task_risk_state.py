"""Pure timing and formula contracts, written before the budget implementation."""
from dataclasses import replace

import pandas as pd
import pytest

from .helpers import START, task


def test_raw_duration_latest_grid_does_not_use_rounded_release():
    from experiments.e1.task_risk_state import latest_grid_start, occupied_minutes
    s = task('a', duration=16, sla=35/16)
    assert occupied_minutes(s) == 30
    assert latest_grid_start(s) == START + pd.Timedelta(minutes=15)


def test_weighted_budget_uses_current_running_and_only_urgent_predecessors():
    from experiments.e1.task_risk_state import BudgetParameters, calculate_budget
    from experiments.e1.cars_state import Pressure, RunningTask
    now = START + pd.Timedelta(minutes=15)
    s = task('c', duration=60, cpu=0, gpu=1, mem=0, sla=2)
    late = task('a', duration=30, cpu=0, gpu=2, mem=0, sla=1)
    nonurgent = task('b', duration=30, cpu=0, gpu=8, mem=0, sla=20)
    running = (RunningTask('r', (0, 5, 0), now+pd.Timedelta(minutes=30)),)
    p = Pressure(15, (0, 0, 0), (0, 1.5, 0), 1.5, 2, 1)
    b = calculate_budget(s, now, (10, 10, 10), running, (late, nonurgent), p, BudgetParameters())
    assert b.weights == (0, 1, 0)
    assert b.utilization == (0, .5, 0)
    assert b.urgent_pressure == (0, .05, 0)
    assert b.score == pytest.approx(1.55)
    assert b.buffer_minutes == 30
    assert b.solo_slack_minutes == 45
    assert b.safe_slack_minutes == 15


def test_no_history_omits_numeric_pressure_but_keeps_common_cold():
    from experiments.e1.task_risk_state import BudgetParameters, calculate_budget
    from experiments.e1.cars_state import Pressure
    s = task('a', duration=60, cpu=1, mem=0, sla=2)
    cold = Pressure(0, (60, 0, 0), (None, None, None), None, 2, 1)
    for params in (BudgetParameters(), BudgetParameters(a_h=0)):
        b = calculate_budget(s, START, (10, 10, 10), (), (), cold, params)
        assert b.buffer_minutes == 30 and b.cold_steps == 2
    hot = replace(cold, effective_minutes=15, rho_resources=(2, 0, 0), rho=2)
    assert calculate_budget(s, START, (10, 10, 10), (), (), hot, BudgetParameters()).buffer_minutes == 30
    assert calculate_budget(s, START, (10, 10, 10), (), (), hot, BudgetParameters(a_h=0)).buffer_minutes == 0


@pytest.mark.parametrize('name', ['task_risk_full', 'task_risk_no_history', 'task_risk_no_urgent',
                                  'task_risk_no_utilization', 'task_risk_max_required'])
def test_static_work_cache_matches_reference_each_live_candidate_bit_for_bit(name):
    from experiments.e1.task_risk import TaskRiskPolicy
    from experiments.e1.cars_state import ResourceSnapshot, RunningTask
    from experiments.e1.policies import DecisionContext
    tasks = tuple(task(f't-{i:03d}', duration=16+i%5*15, cpu=.01, gpu=.001,
                       mem=.001, sla=1.5+i%4) for i in range(45))
    policies = [TaskRiskPolicy(name, dict(threshold_g_per_kwh=161.075), START, (1, 1, 1),
                               query_backend=mode) for mode in ('reference', 'cached_work')]
    ordered = policies[0].order(tasks)
    now = START+pd.Timedelta(minutes=15)
    snap = ResourceSnapshot(now, (1, 1, 1), (1, 1, 1), ordered, (), ())
    for p in policies:
        p.history.observe(START, tasks)
        p.on_step_start(snap)
    for index, spec in enumerate(ordered):
        if index % 7 == 0:
            continue  # capacity-failed predecessor: no callback or deletion
        decisions = []
        for p in policies:
            p.on_candidate_resources(now, (1, 1, 1))
            decisions.append(p.decide(spec, DecisionContext(now, 250)))
        assert decisions[0] == decisions[1]
        assert policies[0].diagnostics[-1] == policies[1].diagnostics[-1]
        for p in policies:
            if index % 3 == 0:
                p.on_task_started(RunningTask(spec.task_id, (.01,.001,.001), now+pd.Timedelta(minutes=spec.duration)))
            else:
                p.on_task_deferred(spec.task_id, now+pd.Timedelta(minutes=15))


def test_cache_and_reference_full_executor_science_are_identical():
    from experiments.e1.task_risk import TaskRiskPolicy
    from .test_policies import execute_policy
    tasks = [task('a', duration=45, sla=1, cpu=.6, gpu=.2),
             task('b', duration=30, sla=3, cpu=.6, gpu=.1),
             task('c', arrival=15, duration=16, sla=2, cpu=.3, gpu=.1)]
    results = [execute_policy(tasks, TaskRiskPolicy('task_risk_full', {'threshold_g_per_kwh':161.075},
        START, (1,1,2), query_backend=mode), minutes=120, ci=[250]*4+[100]*4, capacity=(1,1,2))
        for mode in ('reference','cached_work')]
    assert results[0] == results[1]
