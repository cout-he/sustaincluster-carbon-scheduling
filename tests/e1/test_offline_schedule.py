"""Finite mandatory CP-SAT schedules versus independently enumerated starts."""
from itertools import product
import math

import pandas as pd
import pytest

from .helpers import START, KnownPower, inputs, task
from .test_offline_schedule_cases import case


def solve(frozen, **kwargs):
    from experiments.e1.offline_schedule import SolverBudget, solve_deadline_case
    return solve_deadline_case(frozen, SolverBudget(**kwargs))


def exhaustive_misses(frozen):
    horizon = int((frozen.end - frozen.start) / pd.Timedelta(minutes=15))
    choices = []
    for spec in frozen.tasks:
        length = int(pd.Timedelta(minutes=spec.duration).ceil('15min') / pd.Timedelta(minutes=15))
        first = max(0, int((spec.arrival_time - frozen.start) / pd.Timedelta(minutes=15)))
        choices.append(range(first, horizon - length + 1))
    best = len(frozen.tasks)
    for slots in product(*choices):
        legal = True
        for t in range(horizon):
            demands = [spec for spec, s in zip(frozen.tasks, slots)
                       if s <= t < s + int(pd.Timedelta(minutes=spec.duration).ceil('15min')
                                          / pd.Timedelta(minutes=15))]
            if any(math.fsum(getattr(spec, name) for spec in demands) > cap
                   for name, cap in zip(('cores_req', 'gpu_req', 'mem_req'), frozen.capacity)):
                legal = False
                break
        if legal:
            misses = sum(frozen.start + s * pd.Timedelta(minutes=15) + pd.Timedelta(minutes=spec.duration)
                         > spec.sla_deadline for spec, s in zip(frozen.tasks, slots))
            best = min(best, misses)
    return best


def test_sixteen_minute_task_finishes_on_time_but_occupies_two_slots():
    frozen = case([task('a', duration=16), task('b', duration=16)])
    answer = solve(frozen)
    assert answer.status == 'OPTIMAL'
    assert answer.miss_count == 1
    assert answer.verification['completed_tasks'] == 2
    assert answer.verification['occupied_resource_hours'] == [1, 0, 1]
    assert answer.best_bound == 1


def test_missed_jobs_are_mandatory_instead_of_removed():
    frozen = case([task('a', sla=.5), task('b', sla=.5), task('c', sla=.5)])
    answer = solve(frozen)
    assert answer.status == 'OPTIMAL'
    assert answer.miss_count == 3
    assert set(dict(answer.starts)) == {'a', 'b', 'c'}
    assert answer.verification['occupied_resource_hours'] == [.75, 0, .75]


def test_cumulative_constraints_prevent_independent_resource_schedules():
    frozen = case([task('a', cpu=.5, gpu=.75, mem=.5),
                   task('b', cpu=.5, gpu=.5, mem=.5)])
    answer = solve(frozen)
    assert answer.status == 'OPTIMAL'
    assert answer.miss_count == 1
    assert dict(answer.starts)['a'] != dict(answer.starts)['b']


def test_arrival_is_preserved_and_task_cannot_start_early():
    frozen = case([task('a', arrival=30, sla=1)])
    answer = solve(frozen)
    assert dict(answer.starts)['a'] == START + pd.Timedelta(minutes=30)
    assert answer.miss_count == 0


def test_solver_distinguishes_infeasible_and_unknown_without_incumbent():
    bad = solve(case([task('too_large', cpu=2)]))
    assert bad.status == 'INFEASIBLE'
    assert bad.starts == () and bad.miss_count is None
    timed = solve(case([task('a'), task('b')]), max_seconds=1e-9)
    assert timed.status == 'UNKNOWN'
    assert timed.miss_count is None and timed.starts == ()


def test_precision_uses_exact_binary_mapping_when_safe():
    from experiments.e1.offline_schedule import SolverBudget, integer_resources
    maps = integer_resources(case([task('a', cpu=.25)]), SolverBudget())
    assert maps[0]['mode'] == 'exact_binary'
    assert maps[0]['candidate_demands']['a'] * 4 == maps[0]['capacity_integer']
    assert maps[0]['maximum_demand_error'] == 0


def test_int64_overflow_falls_back_to_fixed_conservative_scale():
    from experiments.e1.offline_schedule import SolverBudget, integer_resources
    frozen = case([task('a', cpu=1e-8)], capacity=(50000, 1000, 80000))
    maps = integer_resources(frozen, SolverBudget())
    assert maps[0]['mode'] == 'conservative_decimal'
    assert maps[0]['scale'] == 1_000_000
    assert maps[0]['candidate_demands']['a'] == 1
    assert 0 < maps[0]['maximum_demand_error'] <= 1e-6


def test_exact_rational_model_does_not_claim_every_fsum_boundary_equivalent():
    frozen = case([task(str(i), cpu=q, gpu=0, mem=0) for i, q in enumerate((.1, .2, .3, .4))])
    answer = solve(frozen)
    assert answer.miss_count == 1
    assert answer.precision[0]['mode'] == 'exact_binary'
    assert 'fsum' in answer.optimality_scope


def test_solution_roundtrip_preserves_all_result_fields():
    from experiments.e1.offline_schedule import ScheduleSolution
    answer = solve(case([task('a')]))
    copy = ScheduleSolution.from_dict(answer.to_dict())
    assert copy.to_dict() == answer.to_dict()


def test_tiny_optimum_agrees_with_independent_exhaustive_schedule():
    specs = [task('a', duration=30, cpu=.75, mem=0), task('b', duration=30, cpu=.5, mem=0),
             task('c', duration=30, cpu=.5, mem=0), task('d', duration=60, cpu=.25, mem=0)]
    frozen = case(specs)
    assert exhaustive_misses(frozen) == solve(frozen).miss_count == 1


def test_eight_fixed_seed_tiny_instances_match_exhaustive_three_resource_optima():
    import random
    generator = random.Random(710001)
    for instance in range(8):
        specs = [task(f'{index}', arrival=generator.choice((0, 15)),
                      duration=generator.choice((15, 30)),
                      cpu=generator.choice((.25, .5, .75)),
                      gpu=generator.choice((0, .25, .5)),
                      mem=generator.choice((.25, .5)), sla=generator.choice((1, 1.5, 2)))
                 for index in range(4)]
        frozen = case(specs, case_id=f'enumeration-{instance}')
        answer = solve(frozen)
        assert answer.status == 'OPTIMAL'
        assert answer.miss_count == exhaustive_misses(frozen)


def test_twelve_task_microcase_demonstrates_local_edf_headroom():
    from experiments.e1.offline_schedule_cases import synthetic_reservation_case
    from experiments.e1.offline_schedule import run_conditional_edf
    frozen = synthetic_reservation_case()
    reference = run_conditional_edf(frozen)
    answer = solve(frozen)
    assert reference.miss_count == 6
    assert answer.status == 'OPTIMAL' and answer.miss_count == 3
    assert answer.verification['completed_tasks'] == reference.verification['completed_tasks'] == 12


def test_schedule_replay_uses_local_executor_task_clocks_and_energy():
    from experiments.e1.offline_schedule import replay_schedule
    specs = [task('a', duration=16), task('b', duration=15)]
    workload, environment = inputs(specs, 120)
    frozen = case(specs)
    answer = solve(frozen)
    executed = replay_schedule(frozen, dict(answer.starts), workload, environment, KnownPower())
    assert all(row['state'] == 'completed' for row in executed.tasks)
    assert {row['task_id']: pd.Timestamp(row['start_time']) for row in executed.tasks} == dict(answer.starts)
    assert sum(row['total_energy_kwh'] for row in executed.steps) > 0
    assert executed.summary['completed'] == 2


def test_conditional_edf_preserves_future_background_reservation_and_backfills():
    from experiments.e1.offline_schedule_cases import FixedOccupancy
    from experiments.e1.offline_schedule import run_conditional_edf, replay_schedule
    specs = [task('candidate', duration=30, cpu=1, mem=0),
             task('small', duration=15, cpu=.25, mem=0),
             task('bg', arrival=15, duration=15, cpu=.75, mem=0)]
    workload, environment = inputs(specs, 120)
    bg = FixedOccupancy('bg', START + pd.Timedelta(minutes=15),
                        START + pd.Timedelta(minutes=30), (.75, 0, 0))
    frozen = case(specs[:2], mode='A2', background=(bg,))
    reference = run_conditional_edf(frozen, workload=workload, environment=environment,
                                    power_model=KnownPower())
    assert dict(reference.starts)['small'] == START
    assert dict(reference.starts)['candidate'] == START + pd.Timedelta(minutes=30)
    executed = replay_schedule(frozen, dict(reference.starts), workload, environment, KnownPower())
    assert next(row for row in executed.tasks if row['task_id'] == 'bg')['start_time'] == bg.start.isoformat()


def test_A2_replay_keeps_original_running_background_and_candidate_arrival():
    from experiments.e1.offline_schedule_cases import FixedOccupancy
    from experiments.e1.offline_schedule import replay_schedule
    specs = [task('bg', duration=16), task('candidate', duration=15)]
    workload, environment = inputs(specs, 120)
    bg = FixedOccupancy('bg', START, START + pd.Timedelta(minutes=30), (1, 0, 1))
    frozen = case([specs[1]], mode='A2', start=START + pd.Timedelta(minutes=15),
                  background=(bg,))
    answer = solve(frozen)
    executed = replay_schedule(frozen, dict(answer.starts), workload, environment, KnownPower())
    row = next(r for r in executed.tasks if r['task_id'] == 'candidate')
    assert row['arrival_time'] == START.isoformat()
    assert row['start_time'] == (START + pd.Timedelta(minutes=30)).isoformat()
    assert row['deadline_outcome'] == 'missed'


def test_conditional_edf_reports_measured_execution_time():
    from experiments.e1.offline_schedule import run_conditional_edf
    reference = run_conditional_edf(case([task('a')]))
    assert reference.wall_seconds > 0


def test_solver_uses_original_nanoseconds_at_near_fifteen_minute_boundary():
    # math.ceil(raw_float/15) says 2, but original Task Timedelta is exactly
    # 15 minutes, therefore LocalExecutor releases this task after one slot.
    duration = math.nextafter(15.0, math.inf)
    assert math.ceil(duration / 15) == 2
    assert pd.Timedelta(minutes=duration) == pd.Timedelta(minutes=15)
    specs = [task('a', duration=duration, sla=1), task('b', arrival=15, sla=1)]
    answer = solve(case(specs))
    assert answer.status == 'OPTIMAL' and answer.miss_count == 0
    assert dict(answer.starts)['b'] == START + pd.Timedelta(minutes=15)


def test_fractional_raw_deadline_matches_solver_original_finish():
    frozen = case([task('a', duration=16, sla=31 / 16)],
                  candidate_not_before=START + pd.Timedelta(minutes=15))
    answer = solve(frozen)
    assert answer.status == 'OPTIMAL' and answer.miss_count == 0
    assert dict(answer.starts)['a'] == START + pd.Timedelta(minutes=15)


def test_conservative_model_can_be_infeasible_while_original_pool_is_feasible():
    from experiments.e1.offline_schedule import run_conditional_edf
    specs = [task('full', cpu=.3, mem=0), task('tiny', cpu=1e-8, mem=0)]
    frozen = case(specs, capacity=(.3, 1, 1))
    reference = run_conditional_edf(frozen)
    answer = solve(frozen)
    assert reference.verification['completed_tasks'] == 2
    assert answer.status == 'INFEASIBLE'
    assert answer.precision[0]['mode'] == 'conservative_decimal'
    assert answer.precision[0]['capacity_loss'] > 0
    assert answer.miss_count is None


def test_solver_budget_rejects_unregistered_threads_seed_or_precision():
    from experiments.e1.offline_schedule import SolverBudget
    for params in ({'workers': 2}, {'seed': 1}, {'conservative_scale': 1000}, {'max_seconds': 0},
                   {'max_seconds': 121}, {'workers': True}, {'seed': 710001.0},
                   {'conservative_scale': 1_000_000.0}):
        with pytest.raises(ValueError):
            SolverBudget(**params)


def test_A2_replay_rejects_background_redefinition_and_candidate_mutation():
    from dataclasses import replace
    from experiments.e1.offline_schedule_cases import FixedOccupancy
    from experiments.e1.offline_schedule import replay_schedule
    specs = [task('bg', duration=15), task('a', duration=15)]
    workload, environment = inputs(specs, 120)
    bad = case([specs[1]], mode='A2', background=(FixedOccupancy(
        'bg', START, START + pd.Timedelta(minutes=30), (1, 0, 1)),))
    with pytest.raises(ValueError, match='background'):
        replay_schedule(bad, {'a': START + pd.Timedelta(minutes=30)}, workload, environment, KnownPower())
    altered = replace(workload, tasks=(specs[0], replace(specs[1], duration=16)))
    valid = case([specs[1]], mode='A2', background=(FixedOccupancy(
        'bg', START, START + pd.Timedelta(minutes=15), (1, 0, 1)),))
    with pytest.raises(ValueError, match='candidate'):
        replay_schedule(valid, {'a': START + pd.Timedelta(minutes=30)}, altered, environment, KnownPower())


def test_A2_replay_prefix_power_and_starts_equal_original_edf():
    from experiments.e1.offline_schedule_cases import FixedOccupancy
    from experiments.e1.offline_schedule import replay_schedule
    from experiments.e1.engine import LocalExecutor
    from experiments.e1.policies import EDFBackfill
    specs = [task('bg', duration=16), task('a', duration=30)]
    workload, environment = inputs(specs, 120)
    original = LocalExecutor(workload, environment, (1, 1, 1), KnownPower(), EDFBackfill(), drain=True).run()
    anchor = START + pd.Timedelta(minutes=15)
    frozen = case([specs[1]], mode='A2', start=anchor, background=(FixedOccupancy(
        'bg', START, START + pd.Timedelta(minutes=30), (1, 0, 1)),))
    replay = replay_schedule(frozen, {'a': START + pd.Timedelta(minutes=45)}, workload, environment, KnownPower())
    assert original.steps[0] == replay.steps[0]
    assert [event for event in original.events if pd.Timestamp(event['time']) < anchor] == [
        event for event in replay.events if pd.Timestamp(event['time']) < anchor]


@pytest.mark.parametrize('method', ['replay_schedule', 'run_conditional_edf'])
def test_explicit_frozen_inputs_require_an_explicit_power_model(method):
    from experiments.e1 import offline_schedule
    workload, environment = inputs([task('a')], 120)
    frozen = case(workload.tasks)
    with pytest.raises(ValueError, match='power model'):
        if method == 'replay_schedule':
            offline_schedule.replay_schedule(frozen, {'a': START}, workload, environment)
        else:
            offline_schedule.run_conditional_edf(frozen, workload=workload, environment=environment)


def test_default_closed_case_zero_energy_is_explicitly_marked_diagnostic():
    from experiments.e1.offline_schedule import replay_schedule
    executed = replay_schedule(case([task('a')]), {'a': START})
    assert executed.summary['offline_zero_power_diagnostic'] is True
    assert executed.summary['is_cluster_carbon_result'] is False
    assert executed.summary['carbon_kg'] == 0
