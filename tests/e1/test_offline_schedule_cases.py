"""Frozen finite cases preserve original tasks and fixed external occupancy."""
from dataclasses import replace

import pandas as pd
import pytest

from .helpers import START, task


def case(tasks=(), **kwargs):
    from experiments.e1.offline_schedule_cases import OfflineCase
    values = dict(case_id='tiny', mode='A1', tasks=tuple(tasks), start=START,
                  end=START + pd.Timedelta(minutes=120), capacity=(1, 1, 1),
                  source_workload_sha256='original-input')
    values.update(kwargs)
    return OfflineCase(**values)


def test_case_roundtrip_keeps_original_specs_and_deadlines():
    from experiments.e1.offline_schedule_cases import OfflineCase
    spec = task('a', duration=16.25)
    original = spec.to_dict()
    frozen = case([spec])
    replay = OfflineCase.from_dict(frozen.to_dict())
    assert frozen.tasks[0] is spec
    assert replay.to_dict() == frozen.to_dict()
    assert replay.sha256 == frozen.sha256
    assert spec.to_dict() == original


def test_duplicate_candidates_fail_instead_of_double_counting():
    spec = task('a')
    with pytest.raises(ValueError, match='Duplicate'):
        case([spec, spec])


def test_background_candidate_overlap_is_rejected():
    from experiments.e1.offline_schedule_cases import FixedOccupancy
    bg = FixedOccupancy('a', START, START + pd.Timedelta(minutes=30), (.5, 0, .5))
    with pytest.raises(ValueError, match='overlap'):
        case([task('a')], mode='A2', background=(bg,))


def test_closed_case_cannot_silently_include_background():
    from experiments.e1.offline_schedule_cases import FixedOccupancy
    bg = FixedOccupancy('bg', START, START + pd.Timedelta(minutes=30), (.5, 0, .5))
    with pytest.raises(ValueError, match='A1'):
        case([], background=(bg,))


def test_fixed_background_requires_grid_and_positive_occupancy():
    from experiments.e1.offline_schedule_cases import FixedOccupancy
    with pytest.raises(ValueError, match='grid'):
        FixedOccupancy('bg', START, START + pd.Timedelta(minutes=16), (1, 0, 1))
    with pytest.raises(ValueError, match='release'):
        FixedOccupancy('bg', START, START, (1, 0, 1))


def test_A2_candidates_may_have_arrived_before_anchor_without_mutation():
    anchor = START + pd.Timedelta(minutes=30)
    spec = task('old')
    frozen = case([spec], mode='A2', start=anchor, candidate_not_before=anchor)
    assert frozen.tasks[0].arrival_time == START
    assert frozen.candidate_not_before == anchor


def test_case_rejects_candidate_after_horizon_or_wrong_dc():
    with pytest.raises(ValueError, match='arrival'):
        case([task('future', arrival=120)])
    with pytest.raises(ValueError, match='DC1'):
        case([replace(task('a'), dc_id=2)])


def test_synthetic_reservation_microcase_is_twelve_tasks_with_original_sla():
    from experiments.e1.offline_schedule_cases import synthetic_reservation_case
    frozen = synthetic_reservation_case()
    assert len(frozen.tasks) == 12
    assert all(spec.sla_multiplier == 1.5 for spec in frozen.tasks)
    assert len({spec.task_id for spec in frozen.tasks}) == 12
    assert frozen.mode == 'A1'


def test_background_extraction_preserves_every_unselected_reference_task():
    from experiments.e1.offline_schedule_cases import background_from_reference
    from experiments.e1.engine import LocalExecutor
    from experiments.e1.policies import EDFBackfill
    from .helpers import KnownPower, inputs
    specs = [task('bg', duration=16), task('candidate', duration=30)]
    workload, environment = inputs(specs, 120)
    reference = LocalExecutor(workload, environment, (1, 1, 1), KnownPower(), EDFBackfill()).run()
    background = background_from_reference(workload, reference.tasks, {'candidate'})
    assert len(background) == 1
    assert background[0].task_id == 'bg'
    assert background[0].start == START
    assert background[0].release == START + pd.Timedelta(minutes=30)
    assert background[0].demand == (1, 0, 1)


def test_background_extraction_rejects_incomplete_reference_or_unknown_candidate():
    from experiments.e1.offline_schedule_cases import background_from_reference
    from experiments.e1.engine import LocalExecutor
    from .helpers import KnownPower, inputs
    workload, environment = inputs([task('long', duration=180)], 120)
    reference = LocalExecutor(workload, environment, (1, 1, 1), KnownPower()).run()
    with pytest.raises(ValueError, match='completed'):
        background_from_reference(workload, reference.tasks, set())
    with pytest.raises(ValueError, match='candidate'):
        background_from_reference(workload, reference.tasks, {'absent'})
