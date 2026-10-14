import copy
import importlib

import pandas as pd
import pytest

from tests.e1.helpers import START, inputs, task


def module():
    return importlib.import_module('experiments.e1.diagnostic_selection')


def reference():
    tasks = [task('a', duration=90, cpu=.2, gpu=.2), task('b', arrival=15, duration=120)]
    w, _ = inputs(tasks)
    events = [dict(time=START.isoformat(), task_id='a', decision_rank=0,
                   can_fit_at_decision=True, event='started', ci_g_per_kwh=200,
                   available_before=dict(cores=1, gpus=1, mem=1))]
    steps = [dict(time=START.isoformat(), decision_pending_ids=['a'], ordered_waiting_ids=['a'])]
    return w, events, steps


def test_records_use_only_current_arrivals_and_turn_resources():
    w, events, steps = reference()
    records = module().build_selection_records(w, (1, 1, 1), events, steps)
    candidate = records['candidates'][0]
    assert candidate['solo_slack_minutes'] == 45
    assert candidate['cpu_utilization'] == 0
    assert candidate['gpu_utilization'] == 0
    assert candidate['history_observed_tasks'] == 1
    assert records['snapshots'][0]['pending_ids'] == ['a']


def test_future_fork_does_not_change_current_records():
    w, events, steps = reference()
    from dataclasses import replace
    other = replace(w, tasks=(w.tasks[0], task('z', arrival=15, duration=500, gpu=1)))
    assert module().build_selection_records(w, (1, 1, 1), events, steps) == module().build_selection_records(
        other, (1, 1, 1), events, steps)


@pytest.mark.parametrize('split', ['validation', 'test', None])
def test_wrong_split_rejected_before_input_access(split):
    class Forbidden:
        def __getattribute__(self, name):
            raise AssertionError('Input accessed')
    with pytest.raises(ValueError, match='Train'):
        module().build_selection_records(Forbidden(), (1, 1, 1), [], [], split=split)


def test_capacity_wait_and_short_slack_not_candidates():
    w, events, steps = reference()
    events[0]['can_fit_at_decision'] = False
    assert not module().build_selection_records(w, (1, 1, 1), events, steps)['candidates']
    events[0].update(can_fit_at_decision=True, time=(START+pd.Timedelta(minutes=45)).isoformat())
    steps[0]['time'] = events[0]['time']
    assert not module().build_selection_records(w, (1, 1, 1), events, steps)['candidates']


def test_selection_is_stable_and_missing_classes_explicit():
    records = {'snapshots': [], 'candidates': []}
    result = module().select_diagnostic_cases(records)
    assert result['A'] == [] and result['C'] == []
    assert len(result['missing']) == 10


def test_selection_uses_no_outcomes_and_preserves_input():
    w, events, steps = reference()
    records = module().build_selection_records(w, (1, 1, 1), events, steps)
    frozen = copy.deepcopy(records)
    first = module().select_diagnostic_cases(records, min_tasks=1)
    assert records == frozen
    for row in records['candidates']:
        row.update(delta_misses=999, finish_time='2099-01-01', carbon_gain=-999)
    assert module().select_diagnostic_cases(records, min_tasks=1) == first


def test_invalid_pending_future_task_fails_closed():
    w, events, steps = reference()
    steps[0]['decision_pending_ids'].append('b')
    steps[0]['ordered_waiting_ids'].append('b')
    with pytest.raises(ValueError, match='future'):
        module().build_selection_records(w, (1, 1, 1), events, steps)
