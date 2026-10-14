"""One eligible decision differs; all execution remains the public E1 path."""
from dataclasses import replace
import pandas as pd
import pytest

from .helpers import START, KnownPower, inputs, task


def cohort(tasks, arrival_minutes=30, environment_minutes=120):
    from experiments.e1.contracts import interval_times
    from experiments.e1.environment import EnvironmentalSample
    w, e = inputs(tasks, arrival_minutes)
    e = replace(e, end=START + pd.Timedelta(minutes=environment_minutes), samples=tuple(
        EnvironmentalSample(t, 250 if i < 2 else 100, 20, 18)
        for i, t in enumerate(interval_times(START, START + pd.Timedelta(minutes=environment_minutes)))))
    return w, e


def test_paired_once_equal_prefix_and_real_different_start_with_non_grid_duration():
    from experiments.e1.counterfactual import InterventionTarget, run_paired_case
    a = task('a', duration=16, cpu=.5, mem=.5, sla=3)
    b = task('b', arrival=15, duration=15, cpu=1, mem=1, sla=2)
    w, e = cohort([a, b])
    original = w.sha256
    paired = run_paired_case(w, e, (1, 1, 1), KnownPower,
        InterventionTarget('micro', START, 'a'), minimum_common_end=START + pd.Timedelta(minutes=90))
    assert paired.prefix_equal
    assert paired.prefixes['force_execute'] == paired.prefixes['force_defer_once']
    assert paired.common_end == START + pd.Timedelta(minutes=90)
    assert w.sha256 == original
    assert len(paired.interventions['force_execute']) == len(paired.interventions['force_defer_once']) == 1
    rows = {name: {r['task_id']: r for r in result.tasks} for name, result in paired.results.items()}
    assert rows['force_execute']['a']['start_time'] == START.isoformat()
    # Defer once forbids this interval only; the later, earlier-deadline b
    # consumes capacity next interval, so the candidate actually starts at +30.
    assert rows['force_defer_once']['a']['start_time'] == (START + pd.Timedelta(minutes=30)).isoformat()
    assert rows['force_defer_once']['a']['wait_intervals'] == 1
    assert rows['force_execute']['a']['finish_time'] == (START + pd.Timedelta(minutes=16)).isoformat()
    assert rows['force_execute']['a']['completion_observed_at'] == (START + pd.Timedelta(minutes=30)).isoformat()
    assert all(r['state'] == 'completed' for result in paired.results.values() for r in result.tasks)
    assert any(r['partition'] == 'post_intervention_arrival' and r['task_id'] == 'b' for r in paired.task_impacts)


def test_reference_capture_is_exact_edf_and_updates_same_step_running():
    from experiments.e1.counterfactual import InterventionTarget, capture_reference
    from experiments.e1.engine import LocalExecutor
    from experiments.e1.policies import EDFBackfill
    a = task('a', duration=15, cpu=.5, mem=.5, sla=3)
    b = task('b', duration=60, cpu=.5, mem=.5, sla=2)
    w, e = cohort([a, b])
    captured = capture_reference(w, e, (1, 1, 1), KnownPower,
        [InterventionTarget('a', START, 'a'), InterventionTarget('b', START, 'b')])
    plain = LocalExecutor(w, e, (1, 1, 1), KnownPower(), EDFBackfill(), drain=True).run(stop_at_drain=True)
    assert captured.result == plain
    assert captured.records['a']['features']['utilization'] == [0, 0, 0]
    assert captured.records['b']['features']['utilization'] == [.5, 0, .5]
    assert captured.records['b']['features']['history_observed_total'] == 2


@pytest.mark.parametrize('target', [
    ('missing', START, 'missing', 1), ('not_fit', START, 'b', 1),
    ('missing_occurrence', START, 'a', 2), ('short_slack', START, 'short', 1)])
def test_invalid_or_capacity_blocked_interventions_fail_closed(target):
    from experiments.e1.counterfactual import InterventionTarget, run_paired_case
    w, e = cohort([task('a', duration=60, sla=2), task('b', duration=60, sla=2), task('short', duration=15)])
    with pytest.raises((ValueError, RuntimeError), match='target|Target|slack|exactly once'):
        run_paired_case(w, e, (1, 1, 1), KnownPower, InterventionTarget(*target))


def test_future_fork_same_online_prefix_and_current_interval():
    from experiments.e1.counterfactual import InterventionTarget, run_paired_case
    a = task('a', duration=30, cpu=.5, mem=.5, sla=3)
    w, e = cohort([a, task('b', arrival=15, duration=15)])
    fork = replace(w, tasks=(a, task('future', arrival=15, duration=45)))
    later = replace(e, samples=(e.samples[0],) + tuple(replace(s, ci_g_per_kwh=333) for s in e.samples[1:]))
    first = run_paired_case(w, e, (1, 1, 1), KnownPower, InterventionTarget('x', START, 'a'))
    second = run_paired_case(fork, later, (1, 1, 1), KnownPower, InterventionTarget('x', START, 'a'))
    assert first.features == second.features
    assert first.prefixes['force_execute'] == second.prefixes['force_execute']
    assert first.results['force_execute'].steps[0] == second.results['force_execute'].steps[0]


def test_stateful_power_prefix_mismatch_invalidates_pair():
    from experiments.e1.counterfactual import InterventionTarget, run_paired_case
    class StatefulPower(KnownPower):
        def __init__(self, state): self.state = state
    states = iter([1, 2])
    w, e = cohort([task('a', duration=30, sla=3)])
    with pytest.raises(RuntimeError, match='prefix'):
        run_paired_case(w, e, (1, 1, 1), lambda: StatefulPower(next(states)), InterventionTarget('x', START, 'a'))


def test_common_end_outside_coverage_and_nonempty_truncation_rejected():
    from experiments.e1.counterfactual import InterventionTarget, run_paired_case
    w, e = cohort([task('a', duration=30, sla=3)])
    with pytest.raises(ValueError, match='coverage|endpoint'):
        run_paired_case(w, e, (1, 1, 1), KnownPower, InterventionTarget('x', START, 'a'),
            minimum_common_end=e.end + pd.Timedelta(minutes=15))
    w, e = cohort([task('a', duration=180, sla=3)], environment_minutes=60)
    with pytest.raises(RuntimeError, match='drain|completed'):
        run_paired_case(w, e, (1, 1, 1), KnownPower, InterventionTarget('x', START, 'a'))


@pytest.mark.parametrize('duration,accepted', [(30, True), (29.999, False), (16, False)])
def test_one_step_slack_uses_exact_grid_start_boundary(duration, accepted):
    from experiments.e1.counterfactual import InterventionTarget, run_paired_case
    w, e = cohort([task('a', duration=duration, sla=1.5)])
    if not accepted:
        with pytest.raises(ValueError, match='slack'):
            run_paired_case(w, e, (1, 1, 1), KnownPower, InterventionTarget('x', START, 'a'))
    else:
        paired = run_paired_case(w, e, (1, 1, 1), KnownPower, InterventionTarget('x', START, 'a'))
        assert paired.features['solo_slack_minutes'] == 15
        assert paired.results['force_defer_once'].tasks[0]['start_time'] == (START + pd.Timedelta(minutes=15)).isoformat()
        assert paired.results['force_defer_once'].tasks[0]['deadline_outcome'] == 'met'
