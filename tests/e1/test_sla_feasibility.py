"""Offline necessary bounds checked against real and exact discrete schedules."""
from dataclasses import replace
from itertools import product
import random

import pandas as pd
import pytest

from experiments.e1.contracts import content_hash
from .helpers import START, KnownPower, inputs, task


def analysis(tasks, capacity=(1, 1, 1), **kwargs):
    from experiments.e1.sla_feasibility import analyze_sla_feasibility
    workload, _ = inputs(tasks, 120)
    return analyze_sla_feasibility(workload, capacity, **kwargs)


def exact_minimum_misses(tasks, capacity=(1, 1, 1)):
    """Enumerate legal grid starts or a miss; no use of the bound implementation."""
    choices = []
    last_slot = 0
    for spec in tasks:
        first = int((spec.arrival_time - START).total_seconds() // 900)
        latest = int(((spec.sla_deadline - pd.Timedelta(minutes=spec.duration))
                      - START).total_seconds() // 900)
        occupied = int(pd.Timedelta(minutes=spec.duration).ceil('15min').total_seconds() // 900)
        choices.append([None] + list(range(first, latest + 1)))
        last_slot = max(last_slot, latest + occupied + 1)
    optimum = len(tasks)
    for starts in product(*choices):
        misses = sum(start is None for start in starts)
        if misses >= optimum:
            continue
        occupancy = [[0.0, 0.0, 0.0] for _ in range(last_slot)]
        legal = True
        for spec, start in zip(tasks, starts):
            if start is None:
                continue
            length = int(pd.Timedelta(minutes=spec.duration).ceil('15min').total_seconds() // 900)
            for slot in range(start, start + length):
                occupancy[slot] = [a + b for a, b in zip(occupancy[slot],
                                     (spec.cores_req, spec.gpu_req, spec.mem_req))]
                if any(a > b for a, b in zip(occupancy[slot], capacity)):
                    legal = False
        if legal:
            optimum = misses
    return optimum


def test_empty_cohort_has_zero_bound_and_no_witness():
    result = analysis([])
    assert result['task_count'] == result['minimum_deadline_misses_lower_bound'] == 0
    assert result['violation_rate_lower_bound'] == 0
    assert result['checked_interval_count'] == 0
    assert result['witnesses'] == []


def test_workload_excess_requires_removing_largest_jobs_until_capacity_fits():
    specs = [task('a', cpu=.7, gpu=0, mem=0, sla=1),
             task('b', cpu=.6, gpu=0, mem=0, sla=1),
             task('c', cpu=.6, gpu=0, mem=0, sla=1),
             task('d', cpu=.6, gpu=0, mem=0, sla=1)]
    result = analysis(specs, include_task_ids=True)
    assert result['minimum_deadline_misses_lower_bound'] == 3
    witness = result['witnesses'][0]
    assert witness['resource'] == 'cpu'
    assert witness['required_resource_minutes'] == pytest.approx(37.5)
    assert witness['capacity_resource_minutes'] == 15
    assert witness['minimum_misses_lower_bound'] == 3
    assert len(witness['greedy_removal_task_ids']) == 3
    assert set(witness['task_ids']) == {'a', 'b', 'c', 'd'}
    assert result['minimum_deadline_misses_lower_bound'] <= exact_minimum_misses(specs)


def test_fractional_finish_sla_is_not_rounded_up_to_resource_release_time():
    from experiments.e1.engine import LocalExecutor
    from experiments.e1.sla_feasibility import task_timing
    spec = task('fractional', duration=16, sla=1.5)
    timing = task_timing(spec)
    assert timing['occupied_minutes'] == 30
    assert timing['deadline'] == START + pd.Timedelta(minutes=24)
    assert timing['latest_grid_start'] == START
    assert timing['latest_occupied_release'] == START + pd.Timedelta(minutes=30)
    result = analysis([spec])
    assert result['intrinsically_impossible_count'] == 0
    assert result['minimum_deadline_misses_lower_bound'] == 0
    workload, environment = inputs([spec], 60)
    executed = LocalExecutor(workload, environment, (1, 1, 1), KnownPower()).run()
    assert executed.tasks[0]['deadline_outcome'] == 'met'
    assert executed.tasks[0]['finish_time'] == (START + pd.Timedelta(minutes=16)).isoformat()
    assert executed.tasks[0]['completion_observed_at'] == (START + pd.Timedelta(minutes=30)).isoformat()
    assert [row['used_cores'] for row in executed.steps] == [1, 1, 0, 0]


def test_rounded_resource_occupancy_detects_competition_hidden_by_raw_duration():
    specs = [task('a', duration=16, sla=1.5), task('b', duration=16, sla=1.5)]
    result = analysis(specs)
    assert result['minimum_deadline_misses_lower_bound'] == 1
    assert result['witnesses'][0]['end'] == (START + pd.Timedelta(minutes=30)).isoformat()
    assert result['witnesses'][0]['required_resource_minutes'] == 60
    assert exact_minimum_misses(specs) == 1


def test_raw_J_endpoint_includes_deadline_equality_and_excludes_later_arrival():
    specs = [task('a', sla=1), task('b', sla=1), task('later', arrival=15, sla=1)]
    result = analysis(specs, include_task_ids=True)
    witness = next(w for w in result['witnesses'] if w['start'] == START.isoformat()
                   and w['end'] == (START + pd.Timedelta(minutes=15)).isoformat()
                   and w['resource'] == 'cpu')
    assert witness['task_count'] == 2
    assert set(witness['task_ids']) == {'a', 'b'}
    assert witness['minimum_misses_lower_bound'] == 1


def test_all_resources_are_checked_but_bounds_are_not_added():
    specs = [task('cpu1', cpu=1, gpu=0, mem=0, sla=1),
             task('cpu2', cpu=1, gpu=0, mem=0, sla=1),
             task('gpu1', cpu=0, gpu=1, mem=0, sla=1),
             task('gpu2', cpu=0, gpu=1, mem=0, sla=1)]
    result = analysis(specs)
    assert result['resource_results']['cpu']['minimum_deadline_misses_lower_bound'] == 1
    assert result['resource_results']['gpu']['minimum_deadline_misses_lower_bound'] == 1
    assert result['resource_results']['mem']['minimum_deadline_misses_lower_bound'] == 0
    assert result['minimum_deadline_misses_lower_bound'] == 1  # Deliberately conservative maximum.
    assert exact_minimum_misses(specs) == 2


def test_intrinsic_impossibility_counts_unique_ids_without_adding_interval_bound():
    specs = [task('too_large', cpu=2), task('too_late', duration=30, sla=.5),
             task('both', cpu=2, duration=30, sla=.5), task('normal', cpu=0)]
    result = analysis(specs, include_task_ids=True)
    assert result['intrinsically_impossible_count'] == 3
    assert result['intrinsically_impossible_task_ids'] == ['both', 'too_large', 'too_late']
    assert result['minimum_deadline_misses_lower_bound'] == 3


def test_no_detected_overload_does_not_imply_nonpreemptive_feasibility():
    specs = [task(f't{i}', cpu=.6, gpu=0, mem=0, sla=2) for i in range(3)]
    result = analysis(specs)
    assert result['minimum_deadline_misses_lower_bound'] == 0
    assert exact_minimum_misses(specs) == 1


def test_occupied_deadline_strengthening_is_separate_from_original_raw_J():
    specs = [task('a', duration=14, sla=16/14), task('b', duration=14, sla=16/14)]
    raw = analysis(specs)
    occupied = analysis(specs, interval_mode='occupied_deadline')
    assert raw['minimum_deadline_misses_lower_bound'] == 0
    assert occupied['minimum_deadline_misses_lower_bound'] == 1
    assert occupied['interval_mode'] == 'occupied_deadline'
    assert exact_minimum_misses(specs) == 1


def test_bound_and_input_identity_are_invariant_under_input_permutation():
    specs = [task('c', duration=16, sla=2), task('a', duration=5, sla=4),
             task('b', arrival=15, duration=15, sla=1)]
    before = content_hash([s.to_dict() for s in specs])
    first = analysis(specs, include_task_ids=True)
    second = analysis(list(reversed(specs)), include_task_ids=True)
    assert first['minimum_deadline_misses_lower_bound'] == second['minimum_deadline_misses_lower_bound']
    assert first['witnesses'] == second['witnesses']
    assert content_hash([s.to_dict() for s in specs]) == before


@pytest.mark.parametrize('mode', ['raw_deadline', 'occupied_deadline'])
@pytest.mark.parametrize('specs', [
    [task('a', duration=15, sla=1), task('b', duration=15, sla=1)],
    [task('a', duration=16, sla=1.5), task('b', duration=5, sla=4)],
    [task('a', cpu=.6, gpu=.3, duration=15, sla=2),
     task('b', cpu=.6, gpu=.3, duration=15, sla=2),
     task('c', cpu=.6, gpu=.8, arrival=15, duration=15, sla=1)],
    [task('a', cpu=.2, gpu=1, duration=30, sla=1),
     task('b', cpu=.2, gpu=1, arrival=15, duration=15, sla=1),
     task('c', cpu=1, gpu=0, arrival=30, duration=15, sla=1)],
])
def test_lower_bound_never_exceeds_exact_small_instance_optimum(mode, specs):
    result = analysis(specs, interval_mode=mode)
    assert result['minimum_deadline_misses_lower_bound'] <= exact_minimum_misses(specs)


def test_numerical_capacity_equality_does_not_create_false_positive():
    specs = [task('a', cpu=.1, gpu=0, mem=0, sla=1),
             task('b', cpu=.2, gpu=0, mem=0, sla=1)]
    result = analysis(specs, capacity=(.3, 1, 1))
    assert result['minimum_deadline_misses_lower_bound'] == 0


def test_both_offline_bounds_remain_conservative_on_seeded_small_exact_instances():
    generator = random.Random(510005)
    for instance in range(30):
        specs = []
        for index in range(4):
            duration = generator.choice((5, 14, 15, 16, 25, 30))
            budget = duration + generator.choice((0, 10, 15, 30))
            specs.append(task(f'{instance}/{index}', arrival=generator.choice((0, 15, 30)),
                duration=duration, sla=budget / duration,
                cpu=generator.choice((0, .25, .5, 1)),
                gpu=generator.choice((0, .25, .5, 1)),
                mem=generator.choice((0, .25, .5, 1))))
        optimum = exact_minimum_misses(specs)
        for mode in ('raw_deadline', 'occupied_deadline'):
            result = analysis(specs, interval_mode=mode)
            assert result['minimum_deadline_misses_lower_bound'] <= optimum, (instance, mode)


@pytest.mark.parametrize('capacity', [(0, 1, 1), (True, 1, 1), (1, 1), (1, float('nan'), 1)])
def test_invalid_capacity_is_rejected(capacity):
    with pytest.raises(ValueError):
        analysis([], capacity=capacity)


def test_unknown_interval_mode_and_negative_witness_limit_are_rejected():
    with pytest.raises(ValueError):
        analysis([], interval_mode='future_forecast')
    with pytest.raises(ValueError):
        analysis([], top_witnesses=-1)
