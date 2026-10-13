"""Synthetic micro-tests of the offline audit; these are not research runs."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

path = Path(__file__).with_name('filter_diagnostic_audit.py')
spec = importlib.util.spec_from_file_location('filter_diagnostic_audit', path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
START = pd.Timestamp('2023-01-01T00:00:00Z')


def task(name, arrival=0, cpu=0, gpu=0, mem=0, duration=15, slack=0):
    # Explicit unit-fixture SLA, independent of real Train/Validation inputs.
    return dict(task_id=name, arrival_time=(START + pd.Timedelta(minutes=arrival)).isoformat(),
                duration=duration, sla_multiplier=(duration + slack) / duration,
                cores_req=cpu, gpu_req=gpu, mem_req=mem)


def test_frozen_formula_hand_computation_and_history_open_left_endpoint():
    now = START + pd.Timedelta(minutes=120)
    candidate = task('b', arrival=120, cpu=1, duration=15, slack=60)
    urgent = task('a', arrival=120, cpu=2, duration=30, slack=15)
    old = task('old', arrival=0, cpu=10, duration=150)
    recent = task('recent', arrival=15, cpu=10, duration=150)
    running = task('running', arrival=30, cpu=5, duration=180)
    budget = audit.risk_budget(candidate, now, START, (10, 10, 10),
                               (running,), (urgent,), (old, recent, urgent, candidate))
    # CPU u=.5, urgent=60/1200=.05; open-left history excludes old.
    # rho=(1500+60+15)/1200=1.3125, z=.5+.05+.8125=1.3625, B=30.
    assert budget['utilization'] == pytest.approx((.5, 0, 0))
    assert budget['urgent_pressure'] == pytest.approx((.05, 0, 0))
    assert budget['history_pressure'] == pytest.approx((1.3125, 0, 0))
    assert budget['score'] == pytest.approx(1.3625)
    assert budget['buffer_minutes'] == 30
    assert budget['solo_slack_minutes'] == 60
    assert budget['safe_slack_minutes'] == 30


def test_cold_zero_and_grid_duration_are_frozen():
    s = task('a', cpu=1, duration=16, slack=59.99)
    cold = audit.risk_budget(s, START, START, (10, 10, 10), (), (), (s,))
    assert cold['cold_steps'] == 2 and cold['buffer_minutes'] == 30
    assert cold['history_pressure'] == (None, None, None)
    assert cold['solo_slack_minutes'] == 45 and cold['safe_slack_minutes'] == 15
    zero = task('zero', duration=16, slack=59.99)
    budget = audit.risk_budget(zero, START, START, (10, 10, 10), (), (), (zero,))
    assert budget['score'] == budget['buffer_minutes'] == 0
    assert budget['dominant_resource'] == 'zero'


def test_future_task_cannot_be_passed_as_observed_history():
    with pytest.raises(ValueError, match='future'):
        audit.risk_budget(task('now', cpu=1), START, START, (10, 10, 10), (), (),
                          (task('future', arrival=15, cpu=1),))


def test_cached_current_history_matches_direct_formula_and_resource_units():
    now = START + pd.Timedelta(minutes=120)
    candidate = task('a', arrival=120, cpu=10, gpu=1, slack=60)
    running = task('r', arrival=0, cpu=50, gpu=.4, duration=180)
    capacity = (100, 1, 1000)
    direct = audit.risk_budget(candidate, now, START, capacity, (running,), (), ())
    pressure = audit.history_pressure(now, START, capacity, ())
    cached = audit.risk_budget(candidate, now, START, capacity, (running,), (), (),
                               _observed_pressure=pressure)
    assert direct == cached
    assert direct['weights'] == pytest.approx((1 / 11, 10 / 11, 0))
    assert direct['score'] == pytest.approx(.5 / 11 + .4 * 10 / 11)
    assert direct['dominant_resource'] == 'gpu' and direct['buffer_minutes'] == 15


def test_fsum_can_fit_boundary_matches_executor_without_subtraction_epsilon():
    now = START + pd.Timedelta(minutes=120)
    specs = {s['task_id']: s for s in (task('a', arrival=120, cpu=.2),
             task('r1', cpu=.2, duration=180), task('r2', cpu=.6, duration=180))}
    available = 1 - (.2 + .6)
    assert available < .2
    event = dict(task_id='a', time=now.isoformat(), event='started', decision_rank=0,
                 can_fit_at_decision=True, available_before=dict(cores=available, gpus=1, mem=1))
    diagnostic = dict(task_id='a', time=now.isoformat(), action=0, slack_threshold_minutes=None,
        ci_g_per_kwh=200, solo_slack_minutes=0, solo_eligible=False, risk_filter=True,
        filter_veto=False, budget=None, available=[available, 1, 1], pending_count=1,
        running_count=2, defer=False, reason='adaptive_edf_execute')
    counted = audit.audit_step(specs=specs, now=now, start=START, capacity=(1, 1, 1),
        action=0, ci=200, risk_filter=True, pending_ids=['a'], running_ids=['r1', 'r2'],
        history_ids=['r1', 'r2', 'a'], events=[event], diagnostics=[diagnostic])
    assert counted['candidate_checks'] == counted['started'] == 1


def fixture():
    now = START + pd.Timedelta(minutes=120)
    a = task('a', arrival=120, cpu=8, slack=0)
    b = task('b', arrival=120, cpu=1, slack=30)
    c = task('c', arrival=120, cpu=8, slack=60)
    specs = {s['task_id']: s for s in (a, b, c)}
    # a starts; b loses eligibility after same-step a start (u=.8 ->B15).
    budget = dict(weights=(1., 0., 0.), utilization=(.8, 0., 0.),
                  urgent_pressure=(0., 0., 0.), history_pressure=(.2125, 0., 0.),
                  resource_pressure=(.8, 0., 0.), score=.8, buffer_minutes=15,
                  solo_slack_minutes=30., safe_slack_minutes=15., cold_steps=0,
                  dominant_resource='cpu')
    events = [dict(task_id=name, event=kind, decision_rank=index,
                   can_fit_at_decision=fit, available_before=dict(cores=available, gpus=10, mem=10))
              for index, (name, kind, fit, available) in enumerate((
                  ('a', 'started', True, 10), ('b', 'started', True, 2),
                  ('c', 'capacity_waited', False, 1)))]
    diagnostics = [dict(task_id='a', action=2, slack_threshold_minutes=30,
        ci_g_per_kwh=200, solo_slack_minutes=0, solo_eligible=False, risk_filter=True,
        filter_veto=False, budget=None, available=[10, 10, 10], pending_count=3,
        running_count=0, defer=False, reason='below_action_slack'),
        dict(task_id='b', action=2, slack_threshold_minutes=30, ci_g_per_kwh=200,
        solo_slack_minutes=30, solo_eligible=True, risk_filter=True, filter_veto=True,
        budget=budget, available=[2, 10, 10], pending_count=2, running_count=1,
        defer=False, reason='adaptive_risk_veto')]
    for row in events + diagnostics:
        row['time'] = now.isoformat()
    return dict(specs=specs, now=now, start=START, capacity=(10, 10, 10),
                action=2, ci=200, risk_filter=True, pending_ids=['a', 'b', 'c'],
                running_ids=[], history_ids=['a', 'b', 'c'], events=events,
                diagnostics=diagnostics)


def test_same_step_start_changes_risk_and_capacity_failure_is_not_candidate():
    result = audit.audit_step(**fixture())
    assert result == dict(candidate_checks=2, solo_eligible=1, filter_veto=1,
                          deferred=0, capacity_waiting=1, started=2,
                          same_step_updated_budget_checks=1)


def test_stale_before_start_budget_rejected():
    data = fixture()
    data['diagnostics'][1]['budget']['utilization'] = (0., 0., 0.)
    with pytest.raises(ValueError, match='budget'):
        audit.audit_step(**data)


def test_capacity_failed_diagnostic_rejected():
    data = fixture()
    extra = deepcopy(data['diagnostics'][-1]); extra['task_id'] = 'c'
    data['diagnostics'].append(extra)
    with pytest.raises(ValueError, match='capacity|candidate|diagnostic'):
        audit.audit_step(**data)


def test_ci_equal_execute_and_unfiltered_fraction_at_boundary():
    data = fixture()
    # Keep only the first task, with exactly30 minutes of solo slack and no filter.
    data['specs']['a'] = task('a', arrival=120, cpu=8, slack=30)
    data.update(pending_ids=['a'], history_ids=['a'], events=data['events'][:1],
                diagnostics=data['diagnostics'][:1], risk_filter=False, ci=161.075)
    row = data['diagnostics'][0]
    row.update(ci_g_per_kwh=161.075, solo_slack_minutes=30, risk_filter=False,
               pending_count=1, reason='low_ci_execute')
    assert audit.audit_step(**data)['solo_eligible'] == 0
    data['ci'] = 200; row.update(ci_g_per_kwh=200, solo_eligible=True,
                              defer=True, reason='adaptive_high_ci_defer')
    data['events'][0]['event'] = 'policy_deferred'
    result = audit.audit_step(**data)
    assert result['solo_eligible'] == result['deferred'] == 1 and result['filter_veto'] == 0


def test_future_pending_and_invalid_action_rejected():
    data = fixture(); data['action'] = 4
    with pytest.raises(ValueError, match='action'):
        audit.audit_step(**data)
    data = fixture(); data['specs']['b']['arrival_time'] = (data['now'] + pd.Timedelta(minutes=15)).isoformat()
    with pytest.raises(ValueError, match='future'):
        audit.audit_step(**data)


def test_test_role_rejected_before_opening_any_bundle_or_output(tmp_path):
    import json
    campaign = tmp_path / 'campaign'; campaign.mkdir()
    (campaign / 'campaign.json').write_text(json.dumps(dict(mode='evaluate', formal_test_evaluated=True,
        validation_training=False, total_core_results=57, inputs=[dict(split='test')])))
    (campaign / 'campaign-receipt.json').write_text(json.dumps(dict(passed=True)))
    output = tmp_path / 'forbidden'
    with pytest.raises(ValueError, match='Test|development'):
        audit.run(campaign, output)
    assert not output.exists()
