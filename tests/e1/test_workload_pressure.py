"""Behavioral diagnostics: exact E1 resource holding, no policy changes."""
import pytest

from experiments.e1.contracts import STEP
from .helpers import START, KnownPower, inputs, task


def pressure(tasks=(), minutes=60, capacity=(10, 2, 20), history_minutes=120):
    from experiments.e1.workload_pressure import summarize
    w, _ = inputs(tasks, minutes)
    return summarize(w, capacity, history_minutes)


def test_all_arrival_ticks_include_zero_buckets_and_endpoint_is_excluded():
    p = pressure([task('a', duration=31, cpu=2, gpu=1, mem=3)])
    assert len(p.arrivals) == 4
    assert [r['arriving_tasks'] for r in p.arrivals] == [1, 0, 0, 0]
    assert p.arrivals[-1]['interval_end'] == (START + 4 * STEP).isoformat()
    assert p.summary['tasks'] == 1
    assert p.summary['arrivals']['mean'] == .25


def test_raw_duration_and_rounded_resource_occupancy_are_distinct():
    p = pressure([task('a', duration=31, cpu=2, gpu=1, mem=3)])
    row = p.arrivals[0]
    assert row['cores_demand'] == 2
    assert row['gpus_work_minutes_raw'] == 31
    assert row['gpus_work_minutes_occupied'] == 45
    assert row['cores_work_minutes_occupied'] == 90
    assert row['mem_work_minutes_occupied'] == 135
    assert row['gpus_pressure_15m'] == 45 / (2 * 15)
    assert row['gpus_pressure_120m'] == 45 / (2 * 120)
    assert p.summary['gpus']['window_work_minutes_occupied'] == 45
    assert p.summary['gpus']['arrival_window_load'] == 45 / (2 * 60)


def test_sla_uses_raw_finish_not_rounded_release_and_urgent_is_one_step_budget():
    p = pressure([task('a', duration=31), task('b', duration=10),
                  task('c', duration=30)])
    details = {r['task_id']: r for r in p.task_details}
    assert details['a']['solo_slack_minutes'] == 15.5
    assert details['a']['latest_grid_start'] == (START + STEP).isoformat()
    assert details['a']['rounded_occupancy_minutes'] == 45
    assert not details['a']['urgent_on_arrival']
    assert details['b']['solo_slack_minutes'] == 5
    assert details['b']['usable_grid_slack_minutes'] == 0
    assert details['b']['urgent_on_arrival']
    assert not details['c']['urgent_on_arrival']  # equality permits one step
    assert p.summary['urgent_on_arrival_tasks'] == 1


def test_history_pressure_only_uses_current_and_previous_observed_buckets():
    common = [task('a', arrival=0, duration=15, gpu=1)]
    a = pressure(common, history_minutes=30)
    b = pressure(common + [task('b', arrival=30, duration=150, gpu=1)],
                 history_minutes=30)
    assert a.arrivals[:2] == b.arrivals[:2]
    assert [r['gpus_pressure_30m'] for r in a.arrivals] == [.25, .25, 0, 0]
    assert a.arrivals[0]['history_observed_minutes'] == 15
    assert a.arrivals[1]['history_observed_minutes'] == 30


def test_two_x_seeds_change_identity_not_resource_arrival_shape():
    from experiments.e1.stress import stress_workload
    from experiments.e1.workload_pressure import summarize
    w, _ = inputs([task('a', duration=10, gpu=.5),
                   task('b', arrival=15, duration=31, gpu=1),
                   task('c', arrival=30, duration=90, gpu=.75)], 60)
    a, _ = stress_workload(w, 2, 510001)
    b, _ = stress_workload(w, 2, 510002)
    assert a.sha256 != b.sha256
    assert {t.task_id for t in a.tasks} != {t.task_id for t in b.tasks}
    assert summarize(a, (10, 2, 20)).arrivals == summarize(b, (10, 2, 20)).arrivals
    base_hash = w.sha256
    low, _ = stress_workload(w, 1.25, 510001)
    high, _ = stress_workload(w, 1.5, 510001)
    assert {t.task_id for t in low.tasks} <= {t.task_id for t in high.tasks}
    assert {t.task_id for t in high.tasks} <= {t.task_id for t in a.tasks}
    assert w.sha256 == base_hash
    assert a.tasks[:len(w.tasks)] == w.tasks


def test_empty_tasks_have_finite_zero_summaries():
    p = pressure()
    assert p.summary['tasks'] == 0
    assert p.summary['duration_minutes']['mean'] == 0
    assert p.summary['urgent_on_arrival_fraction'] == 0
    assert p.summary['gpus']['arrival_window_load'] == 0
    assert all(r['arriving_tasks'] == 0 for r in p.arrivals)


def test_fractional_replication_is_seed_dependent_but_not_independent_real_trace():
    from experiments.e1.stress import stress_workload
    from experiments.e1.workload_pressure import summarize
    originals = [task(chr(97+i), arrival=15*(i % 4), duration=15*(i+1),
                      gpu=(i+1)/10) for i in range(10)]
    w, _ = inputs(originals, 60)
    a, _ = stress_workload(w, 1.5, 510001)
    b, _ = stress_workload(w, 1.5, 510002)
    assert len(a.tasks) == len(b.tasks) == 15
    assert a.tasks[:10] == b.tasks[:10] == w.tasks
    assert summarize(a, (10, 2, 20)).arrivals != summarize(b, (10, 2, 20)).arrivals
    parents = {t.task_id: t for t in originals}
    for copy in a.tasks[10:] + b.tasks[10:]:
        parent = parents[copy.task_id.split('~e1-2b-')[0]]
        assert copy.arrival_time == parent.arrival_time
        assert copy.duration == parent.duration
        assert copy.gpu_req == parent.gpu_req


def test_rounded_occupancy_respects_exact_release_grid_without_extra_tick():
    from experiments.e1.workload_pressure import occupied_minutes
    assert occupied_minutes(15) == 15
    assert occupied_minutes(30) == 30
    assert occupied_minutes(30.000001) == 45


@pytest.mark.parametrize('capacity', [(0, 1, 1), (1, 1), (1, -1, 1)])
def test_capacity_validation(capacity):
    with pytest.raises(ValueError, match='capacity|CPU'):
        pressure(capacity=capacity)


@pytest.mark.parametrize('minutes', [0, 17, -15])
def test_history_must_be_positive_multiple_of_step(minutes):
    with pytest.raises(ValueError, match='history'):
        pressure(history_minutes=minutes)


def test_edf_alignment_uses_actual_capacity_events_and_raw_sla_finish():
    from experiments.e1.engine import LocalExecutor
    from experiments.e1.policies import EDFBackfill
    from experiments.e1.workload_pressure import summarize, align_edf
    tasks = [task('a', duration=16, cpu=1, gpu=1, mem=1),
             task('b', duration=15, cpu=1, gpu=1, mem=1)]
    w, e = inputs(tasks, 60)
    result = LocalExecutor(w, e, (1, 1, 1), KnownPower(), EDFBackfill()).run()
    aligned = align_edf(summarize(w, (1, 1, 1)), result.steps,
                        result.tasks, result.policy_steps)
    rows = aligned['series']
    assert [r['capacity_waiting_tasks'] for r in rows] == [1, 0, 0, 0]
    assert [r['pending'] for r in rows] == [1, 0, 0, 0]
    assert [r['new_completed_misses_observed'] for r in rows] == [0, 0, 0, 1]
    assert aligned['summary']['capacity_wait_intervals'] == 1
    assert aligned['summary']['completed_misses'] == 1
    assert [r['new_latest_start_misses_known'] for r in rows] == [0, 1, 0, 0]
    assert aligned['audit']['all_task_ids_match']
    assert aligned['audit']['resource_capacity_conserved']


def test_edf_alignment_preserves_terminal_completion_boundary_and_zero_drain_arrivals():
    from experiments.e1.engine import LocalExecutor
    from experiments.e1.policies import EDFBackfill
    from experiments.e1.selection import TemporalEnvironment
    from experiments.e1.workload_pressure import summarize, align_edf
    w, _ = inputs([task('a', duration=45, sla=.5)], 30)
    _, full = inputs([], 60)
    e = TemporalEnvironment(w.start, full.end, 'ci', 'weather', full.samples)
    executor = LocalExecutor(w, e, (1, 1, 1), KnownPower(), EDFBackfill(), drain=True)
    result = executor.run(stop_at_drain=True)
    aligned = align_edf(summarize(w, (1, 1, 1)), result.steps,
                        result.tasks, result.policy_steps)
    assert len(aligned['series']) == 3
    assert aligned['series'][2]['arriving_tasks'] == 0
    assert aligned['terminal_boundary']['time'] == (START + 3 * STEP).isoformat()
    assert aligned['terminal_boundary']['new_completed_misses_observed'] == 1
    assert sum(r['new_completed_misses_observed'] for r in aligned['series']) == 0
    assert aligned['summary']['completed_misses'] == 1
    assert [r['new_latest_start_misses_known'] for r in aligned['series']] == [1, 0, 0]
    assert aligned['audit']['no_new_drain_arrivals']


def test_alignment_rejects_mismatched_ledger_and_events():
    from experiments.e1.workload_pressure import summarize, align_edf
    from experiments.e1.engine import LocalExecutor
    from experiments.e1.policies import EDFBackfill
    w, e = inputs([task('a')], 30)
    result = LocalExecutor(w, e, (1, 1, 1), KnownPower(), EDFBackfill()).run()
    with pytest.raises(ValueError, match='task|ledger'):
        align_edf(summarize(w, (1, 1, 1)), result.steps, (), result.policy_steps)
    bad = [dict(r) for r in result.policy_steps]
    bad[0]['capacity_waiting_ids'] = ['unknown']
    with pytest.raises(ValueError, match='unknown|event'):
        align_edf(summarize(w, (1, 1, 1)), result.steps, result.tasks, bad)


def test_alignment_resource_occupancy_must_match_actual_running_task_demands():
    from experiments.e1.workload_pressure import summarize, align_edf
    from experiments.e1.engine import LocalExecutor
    from experiments.e1.policies import EDFBackfill
    w, e = inputs([task('a', gpu=1)], 30)
    result = LocalExecutor(w, e, (1, 1, 1), KnownPower(), EDFBackfill()).run()
    steps = [dict(r) for r in result.steps]
    steps[0].update(used_gpus=0, available_gpus=1, gpus_fraction=0)
    with pytest.raises(ValueError, match='running|occupancy'):
        align_edf(summarize(w, (1, 1, 1)), steps, result.tasks, result.policy_steps)


def test_alignment_candidate_events_must_form_a_partition_of_waiting_tasks():
    from experiments.e1.workload_pressure import summarize, align_edf
    from experiments.e1.engine import LocalExecutor
    from experiments.e1.policies import EDFBackfill
    w, e = inputs([task('a')], 30)
    result = LocalExecutor(w, e, (1, 1, 1), KnownPower(), EDFBackfill()).run()
    details = [dict(r) for r in result.policy_steps]
    details[0]['started_ids'] = []
    with pytest.raises(ValueError, match='partition|candidate'):
        align_edf(summarize(w, (1, 1, 1)), result.steps, result.tasks, details)


def test_cli_excludes_test_before_opening_any_input(tmp_path):
    from experiments.e1.workload_pressure import main
    with pytest.raises(SystemExit):
        main(['--bundle', str(tmp_path/'missing-test'), '--split', 'test',
              '--output', str(tmp_path/'out')])
    assert not (tmp_path/'out').exists()
