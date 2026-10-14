"""Independent original-clock/resource checks, not CP-SAT model assertions."""
import pandas as pd
import pytest

from .helpers import START, task
from .test_offline_schedule_cases import case


def verify(frozen, starts):
    from experiments.e1.offline_schedule_verify import verify_schedule
    return verify_schedule(frozen, starts)


def test_actual_finish_sla_and_ceil_resource_release_are_distinct():
    frozen = case([task('a', duration=16, sla=1.5)])
    checked = verify(frozen, {'a': START})
    assert checked['verified'] is True
    assert checked['miss_count'] == 0
    row = checked['tasks'][0]
    assert row['finish_time'] == (START + pd.Timedelta(minutes=16)).isoformat()
    assert row['release_time'] == (START + pd.Timedelta(minutes=30)).isoformat()
    assert checked['occupied_resource_hours'] == [.5, 0, .5]
    assert checked['actual_resource_hours'] == pytest.approx([16 / 60, 0, 16 / 60])


def test_late_task_still_has_full_resource_occupancy_and_unique_outcome():
    frozen = case([task('a', duration=16, sla=1)])
    checked = verify(frozen, {'a': START + pd.Timedelta(minutes=15)})
    assert checked['miss_count'] == 1
    assert checked['task_count'] == checked['completed_tasks'] == 1
    assert checked['unfinished_tasks'] == 0
    assert checked['occupied_resource_hours'] == [.5, 0, .5]


def test_deadline_equality_is_on_time():
    assert verify(case([task('a', duration=15, sla=2)]),
                  {'a': START + pd.Timedelta(minutes=15)})['miss_count'] == 0


def test_fractional_deadline_does_not_get_rounded_up():
    assert verify(case([task('a', duration=15, sla=1.999999)]),
                  {'a': START + pd.Timedelta(minutes=15)})['miss_count'] == 1


def test_rejects_missing_extra_duplicate_and_non_grid_starts():
    frozen = case([task('a')])
    for starts in ({}, {'a': START, 'extra': START}, [('a', START), ('a', START)]):
        with pytest.raises(ValueError):
            verify(frozen, starts)
    with pytest.raises(ValueError, match='grid'):
        verify(frozen, {'a': START + pd.Timedelta(minutes=1)})


def test_arrival_and_anchor_are_start_lower_bounds():
    with pytest.raises(ValueError, match='arrival'):
        verify(case([task('a', arrival=15)]), {'a': START})
    anchor = START + pd.Timedelta(minutes=30)
    with pytest.raises(ValueError, match='anchor'):
        verify(case([task('a')], mode='A2', candidate_not_before=anchor), {'a': START})


def test_resources_are_held_until_next_grid_not_actual_finish():
    frozen = case([task('a', duration=16), task('b', duration=15)])
    with pytest.raises(ValueError, match='capacity'):
        verify(frozen, {'a': START, 'b': START + pd.Timedelta(minutes=15)})
    assert verify(frozen, {'a': START, 'b': START + pd.Timedelta(minutes=30)})['completed_tasks'] == 2


def test_each_resource_dimension_has_the_same_occupancy_interval():
    specs = [task('a', cpu=.5, gpu=.75, mem=.5), task('b', cpu=.5, gpu=.5, mem=.5)]
    with pytest.raises(ValueError, match='gpu'):
        verify(case(specs), {'a': START, 'b': START})


def test_original_fsum_boundary_is_checked_without_epsilon():
    specs = [task(str(i), cpu=q, gpu=0, mem=0) for i, q in enumerate((.1, .2, .3, .4))]
    assert verify(case(specs), {s.task_id: START for s in specs})['peak_used_resources'][0] == 1
    specs.append(task('too_much', cpu=1e-12, mem=0))
    with pytest.raises(ValueError, match='capacity'):
        verify(case(specs), {s.task_id: START for s in specs})


def test_fixed_background_is_counted_once_and_excludes_candidate_ids():
    from experiments.e1.offline_schedule_cases import FixedOccupancy
    bg = FixedOccupancy('bg', START, START + pd.Timedelta(minutes=30), (.75, 0, 0))
    frozen = case([task('a', cpu=.5, mem=0)], mode='A2', background=(bg,))
    with pytest.raises(ValueError, match='capacity'):
        verify(frozen, {'a': START})
    assert verify(frozen, {'a': START + pd.Timedelta(minutes=30)})['miss_count'] == 1


def test_horizon_requires_resource_release_not_just_finish():
    frozen = case([task('a', duration=16)], end=START + pd.Timedelta(minutes=30))
    with pytest.raises(ValueError, match='horizon'):
        verify(frozen, {'a': START + pd.Timedelta(minutes=15)})
