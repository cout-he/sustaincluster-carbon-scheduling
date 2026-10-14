"""Candidate selection observes current reference events, never their outcomes."""
from dataclasses import replace
import pytest
from .helpers import START, task
from .test_counterfactual import cohort


def reference(tasks):
    from experiments.e1.engine import LocalExecutor
    from experiments.e1.policies import EDFBackfill
    from .helpers import KnownPower
    w, e = cohort(tasks)
    return w, LocalExecutor(w, e, (1, 1, 1), KnownPower(), EDFBackfill(), drain=True).run(stop_at_drain=True)


def test_candidates_are_arrived_can_fit_with_grid_slack_and_stable_ids():
    from experiments.e1.counterfactual_cases import candidate_records
    w, r = reference([task('a', duration=60, sla=2), task('b', duration=60, sla=2)])
    rows = candidate_records(w, r, (1, 1, 1))
    assert rows[0]['task_id'] == 'a'
    assert rows[0]['candidate_occurrence'] == 1
    assert rows[0]['solo_slack_minutes'] == 60
    assert not any(row['task_id'] == 'b' and row['time'] == START.isoformat() for row in rows)


def test_candidate_selection_ignores_realized_deadline_and_carbon_outcomes():
    from experiments.e1.counterfactual_cases import candidate_records
    w, r = reference([task('a', duration=60, sla=2)])
    changed = replace(r, tasks=tuple(dict(row, deadline_outcome='fiction', finish_time=None) for row in r.tasks),
                      summary={'carbon_kg': -999})
    assert candidate_records(w, r, (1, 1, 1)) == candidate_records(w, changed, (1, 1, 1))


def test_malformed_or_future_candidate_rejected():
    from experiments.e1.counterfactual_cases import candidate_records
    w, r = reference([task('a', duration=60, sla=2)])
    changed = replace(r, policy_events=(dict(r.policy_events[0], time=(START.replace(year=2022)).isoformat()),))
    with pytest.raises(ValueError, match='arrival|candidate'):
        candidate_records(w, changed, (1, 1, 1))


def test_known_release_screen_uses_release_boundary_and_small_gpu_conflict():
    from experiments.e1.counterfactual_cases import known_release_screen
    from experiments.e1.cars_state import RunningTask
    import pandas as pd
    candidate = task('candidate', duration=16, cpu=0, gpu=.2, mem=0, sla=3)
    predecessor = task('predecessor', duration=15, cpu=0, gpu=.9, mem=0, sla=3)
    released = RunningTask('released', (0, .1, 0), START + pd.Timedelta(minutes=29))
    screen = known_release_screen(candidate, START, (1, 1, 1), [released], [predecessor])
    assert screen['known_release_screen_checked_pred_count'] == 1
    assert screen['known_release_screen_conflicts_now'] == 0
    assert screen['known_release_screen_conflicts_defer'] == 1
    assert screen['known_release_screen_conflict_delta'] == 1
    assert screen['existing_reservations'] == 0
    assert screen['reservation_conflict_assessed'] is False


def test_known_release_screen_keeps_running_until_grid_release_and_excludes_self_successor():
    from experiments.e1.counterfactual_cases import known_release_screen
    from experiments.e1.cars_state import RunningTask
    import pandas as pd
    candidate = task('candidate', duration=16, cpu=0, gpu=.2, mem=0, sla=3)
    predecessor = task('predecessor', duration=15, cpu=0, gpu=.9, mem=0, sla=3)
    successor = task('successor', duration=60, cpu=0, gpu=.9, mem=0, sla=3)
    occupied = RunningTask('still_occupied', (0, .2, 0), START + pd.Timedelta(minutes=30.001))
    screen = known_release_screen(candidate, START, (1, 1, 1), [occupied], [candidate, successor, predecessor])
    assert screen['known_release_screen_checked_pred_count'] == 1
    assert screen['known_release_screen_conflicts_now'] == 1
    assert screen['known_release_screen_conflicts_defer'] == 1


def test_known_release_screen_horizon_inclusive_and_no_future_predecessor():
    from experiments.e1.counterfactual_cases import known_release_screen
    candidate = task('candidate', duration=300, cpu=.6, sla=2)
    boundary = task('boundary', duration=120, cpu=.5, sla=2)
    outside = task('outside', duration=135, cpu=.5, sla=2)
    screen = known_release_screen(candidate, START, (1, 1, 1), [], [boundary, outside])
    assert screen['known_release_screen_checked_pred_count'] == 1
    future = task('future', arrival=15, duration=30)
    with pytest.raises(ValueError, match='future'):
        known_release_screen(candidate, START, (1, 1, 1), [], [future])
