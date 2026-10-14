import importlib
from dataclasses import replace

import pandas as pd
import pytest

from experiments.e1.engine import LocalExecutor
from experiments.e1.policies import EDFBackfill
from tests.e1.helpers import KnownPower, START, inputs, task


def module():
    return importlib.import_module('experiments.e1.diagnostic_audit')


def complete():
    w, e = inputs([task('a'), task('b', arrival=15)], minutes=60)
    ex = LocalExecutor(w, e, (1, 1, 1), KnownPower(), EDFBackfill(), drain=True)
    ex.run(stop_at_drain=True)
    return w, e, ex.extend_idle(e.end)


def test_identical_complete_results_fair_and_have_stable_core_hash():
    w, e, result = complete()
    receipt = module().audit_fair_results({'a': result, 'b': result}, w, e, (1, 1, 1))
    assert receipt['passed'] and receipt['equal_resource_hours']
    assert module().core_scientific_hash(result) == module().core_scientific_hash(result)


def test_different_accounting_end_refused():
    w, e, result = complete()
    other = replace(result, steps=result.steps[:-1])
    with pytest.raises(RuntimeError, match='common'):
        module().audit_fair_results({'a': result, 'b': other}, w, e, (1, 1, 1))


def test_unfinished_work_refused():
    w, e, result = complete()
    tasks = list(result.tasks)
    tasks[0] = dict(tasks[0], state='running')
    with pytest.raises(RuntimeError, match='complete'):
        module().audit_fair_results({'a': replace(result, tasks=tuple(tasks))}, w, e, (1, 1, 1))


def test_changed_energy_is_not_accepted_from_summary_alone():
    w, e, result = complete()
    steps = list(result.steps)
    steps[0] = dict(steps[0], carbon_kg=steps[0]['carbon_kg']+1)
    with pytest.raises(RuntimeError):
        module().audit_fair_results({'a': replace(result, steps=tuple(steps))}, w, e, (1, 1, 1))


def test_policy_event_annotation_does_not_change_core_but_changes_full_hash():
    _, _, result = complete()
    events = [dict(event) for event in result.policy_events]
    events[0]['reason'] = 'force_execute'
    other = replace(result, policy_events=tuple(events))
    assert module().core_scientific_hash(result) == module().core_scientific_hash(other)
    assert module().full_scientific_hash(result) != module().full_scientific_hash(other)


def test_summary_fields_from_actual_finished_tasks_and_integrals():
    w, e, result = complete()
    row = module().diagnostic_summary(result, w.end)
    assert row['completed'] == 2 and row['unfinished'] == 0
    assert row['misses'] == 0 and row['dmr'] == 0
    assert row['it_energy_kwh'] + row['hvac_energy_kwh'] == row['total_energy_kwh']
    assert row['mean_jct_minutes'] == 15
    assert row['mean_wait_minutes'] == 0


def test_existing_accounting_end_after_drain_includes_idle():
    w, e, result = complete()
    row = module().diagnostic_summary(result, w.end)
    assert row['common_end'] == e.end.isoformat()
    assert len(result.steps) == 4
