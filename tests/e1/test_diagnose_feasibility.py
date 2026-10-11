"""Small, independent checks for E1-4A archived-result diagnosis."""
from types import SimpleNamespace
import json
import csv

import pandas as pd
import pytest


NOW = pd.Timestamp('2023-07-01T10:15:00Z')


def test_known_risk_pairs_only_new_miss_ids_and_counts_repeated_warnings():
    # Catches using total miss counts, deduplicating decision occurrences, or
    # treating a met final outcome as a real miss.
    from experiments.e1.diagnose_feasibility import pair_known_risk, risk_outcome_counts
    rows = [dict(time='2023-07-01T10:15:00Z', task_id='candidate',
                 reason='known_deadline_risk', planner=dict(
                     status='risk', missed_now=['common'],
                     missed_defer=['common', 'affected', 'met'])),
            dict(time='2023-07-01T10:30:00Z', task_id='other',
                 reason='known_deadline_risk', planner=dict(
                     status='risk', missed_now=[], missed_defer=['affected'])),
            dict(time='2023-07-01T10:45:00Z', task_id='ignored',
                 reason='known_feasibility_uncertain', planner=dict(
                     status='uncertain', missed_now=[], missed_defer=['met']))]
    ledger = {key: dict(task_id=key, deadline_outcome=outcome)
              for key, outcome in [('common', 'missed'), ('affected', 'missed'), ('met', 'met')]}
    paired = pair_known_risk(rows, ledger)
    assert [p['risk_task_id'] for p in paired] == ['affected', 'met', 'affected']
    assert [p['actual_miss'] for p in paired] == [True, False, True]
    assert risk_outcome_counts(paired) == dict(
        prediction_occurrences=3, unique_predicted_tasks=2,
        actual_miss_occurrences=2, unique_actual_miss_tasks=1,
        actual_met_occurrences=1, other_outcome_occurrences=0)


def test_missing_predicted_task_is_reported_instead_of_silently_counted_met():
    from experiments.e1.diagnose_feasibility import pair_known_risk
    rows = [dict(time='2023-07-01T10:15:00Z', task_id='candidate',
                 reason='known_deadline_risk', planner=dict(
                     status='risk', missed_now=[], missed_defer=['absent']))]
    with pytest.raises(ValueError, match='absent'):
        pair_known_risk(rows, {})


def test_categories_separate_history_guards_explicit_risk_and_pending_cap():
    from experiments.e1.diagnose_feasibility import decision_counts
    rows = [dict(reason='risk_buffer_guard'), dict(reason='known_deadline_risk'),
            dict(reason='known_feasibility_uncertain', pending_count=257,
                 planner=dict(status='uncertain', reason='pending_limit')),
            dict(reason='known_feasibility_uncertain', pending_count=7,
                 planner=dict(status='uncertain', reason='bounded_horizon_unknown')),
            dict(reason='high_ci_defer')]
    expected = dict(candidate_checks=5,
        history_risk_buffer_guards=1, explicit_shadow_risk_guards=1,
        pending_cap_fallbacks=1, other_uncertain_fallbacks=1,
        reason_counts=dict(risk_buffer_guard=1, known_deadline_risk=1,
            known_feasibility_uncertain=2, high_ci_defer=1),
        planner_uncertain_reasons=dict(pending_limit=1, bounded_horizon_unknown=1))
    counts = decision_counts(rows)
    assert {key: counts[key] for key in expected} == expected


def test_fixed_case_selection_is_time_id_order_and_retains_missing_slots():
    from experiments.e1.diagnose_feasibility import select_cases
    rows = [dict(time='2023-07-01T10:30:00Z', task_id='b', reason='known_deadline_risk'),
            dict(time='2023-07-01T10:15:00Z', task_id='z', reason='known_deadline_risk'),
            dict(time='2023-07-01T10:15:00Z', task_id='a', reason='known_deadline_risk')]
    selected = select_cases(rows)
    assert selected['first_known_risk']['task_id'] == 'a'
    assert selected['first_pending_cap_fallback'] is None
    assert selected['first_history_buffer_guard'] is None
    assert selected['historical_17475_189_copy1_1015'] is None


def test_current_boundary_shadow_starts_respect_prior_blocks_and_real_later_actions():
    # All jobs have arrived at this boundary. Future arrivals are unnecessary
    # for shadow to disagree with a later real carbon defer at the SAME boundary.
    from experiments.e1.cars_state import ResourceSnapshot
    from experiments.e1.diagnose_feasibility import compare_same_step
    def job(key, deadline):
        return SimpleNamespace(task_id=key, cores_req=1, gpu_req=0, mem_req=0,
            duration=15, arrival_time=NOW, sla_deadline=NOW + pd.Timedelta(minutes=deadline))
    candidate, later, blocked = job('candidate', 30), job('later', 45), job('blocked', 60)
    snapshot = ResourceSnapshot(NOW, (3, 1, 1), (3, 1, 1),
        (candidate, later, blocked), (), blocked_until=(('blocked', NOW+pd.Timedelta(minutes=15)),))
    actual = [dict(task_id='candidate', event='started', decision_rank=0),
              dict(task_id='later', event='policy_deferred', decision_rank=1),
              dict(task_id='blocked', event='policy_deferred', decision_rank=-1)]
    result = compare_same_step(candidate, snapshot, actual, 0)
    assert result['shadow_now_started_ids'] == ['candidate', 'later']
    assert result['actual_remaining_started_ids'] == ['candidate']
    assert result['shadow_start_actual_deferred_ids'] == ['later']
    assert result['shadow_start_actual_capacity_waited_ids'] == []
    assert result['current_boundary_arrival_count'] == 3
    assert result['same_step_start_set_mismatch'] is True


def test_reference_analysis_preserves_input_and_exports_all_predicted_occurrences(tmp_path):
    from experiments.e1.diagnose_feasibility import analyze_reference
    reference = tmp_path/'reference'
    row = dict(time=NOW.isoformat(), task_id='candidate', reason='known_deadline_risk',
        available=[3, 1, 1], pending_count=2, running_count=0,
        planner=dict(status='risk', reason='additional_known_deadline_miss',
            missed_now=[], missed_defer=['affected'], unknown_now=[], unknown_defer=[]))
    ledger = [dict(task_id=key, arrival_time=NOW.isoformat(), duration=15,
        cores_req=1, gpu_req=0, mem_req=0,
        sla_deadline=(NOW+pd.Timedelta(minutes=30)).isoformat(),
        start_time=(NOW+pd.Timedelta(minutes=delay)).isoformat(),
        finish_time=(NOW+pd.Timedelta(minutes=delay+15)).isoformat(),
        deadline_outcome=outcome, source_row=1, source_task_index=idx)
        for idx, (key, delay, outcome) in enumerate([
            ('candidate', 0, 'met'), ('affected', 30, 'missed')])]
    events = [dict(time=NOW.isoformat(), task_id='candidate', event='started', decision_rank=0),
              dict(time=NOW.isoformat(), task_id='affected', event='policy_deferred', decision_rank=1)]
    config = dict(datacenter=dict(total_cores=3, total_gpus=1, total_mem=1),
        policy_params=dict(horizon_steps=8, max_pending=256))
    for load in ('load_1p50', 'load_2p00'):
        full = reference/load/'cars_full'
        risk = reference/load/'cars_risk_only'
        full.mkdir(parents=True); risk.mkdir()
        for name, values in [('diagnostics', [row]), ('ledger', ledger), ('policy_events', events),
                             ('policy_intervals', [dict(time=NOW.isoformat(), ordered_waiting_ids=['candidate', 'affected'])])]:
            (full/f'{name}.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in values))
        (full/'resolved_config.json').write_text(json.dumps(config))
        (risk/'ledger.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in ledger))
    original = {p: p.read_bytes() for p in reference.rglob('*') if p.is_file()}
    output = tmp_path/'E1-4A'/'audit'
    result = analyze_reference(reference, output)
    assert result['loads']['load_1p50']['risk_outcomes']['actual_miss_occurrences'] == 1
    assert result['loads']['load_2p00']['decision_counts']['explicit_shadow_risk_guards'] == 1
    assert (output/'known_risk_outcomes.csv').read_text().count('affected') == 2
    assert json.loads((output/'representative_cases.json').read_text())['load_1p50']['first_pending_cap_fallback'] is None
    assert {p: p.read_bytes() for p in reference.rglob('*') if p.is_file()} == original


def test_reference_analysis_rejects_output_within_reference_before_writing(tmp_path):
    from experiments.e1.diagnose_feasibility import analyze_reference
    reference = tmp_path/'reference'
    reference.mkdir()
    with pytest.raises(ValueError, match='reference'):
        analyze_reference(reference, reference/'E1-4A')
    assert list(reference.iterdir()) == []


def test_campaign_analysis_pairs_misses_and_start_times_against_risk_only(tmp_path):
    from experiments.e1.diagnose_feasibility import analyze_campaign
    campaign = tmp_path/'campaign'
    policies = ('edf_carbon_threshold', 'cars_risk_only', 'cars_full', 'risk_cap_only', 'risk_shadow_only')
    for load in ('load_1p50', 'load_2p00'):
        folder = campaign/load
        folder.mkdir(parents=True)
        (folder/'fairness.json').write_text(json.dumps(dict(passed=True, marker=load)))
        with (folder/'comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['policy', 'phase', 'carbon_kg'])
            writer.writeheader()
            for policy in policies:
                writer.writerow(dict(policy=policy, phase='fixed', carbon_kg=99))
                writer.writerow(dict(policy=policy, phase='common', carbon_kg=100))
        for policy in policies:
            policy_dir = folder/policy
            policy_dir.mkdir()
            base = policy == 'cars_risk_only'
            rows = [dict(task_id='a', arrival_time=NOW.isoformat(),
                         start_time=(NOW+pd.Timedelta(minutes=15 if base else 0)).isoformat(),
                         deadline_outcome='met' if base else 'missed'),
                    dict(task_id='b', arrival_time=NOW.isoformat(),
                         start_time=(NOW+pd.Timedelta(minutes=0 if base else 15)).isoformat(),
                         deadline_outcome='missed' if base else 'met')]
            (policy_dir/'ledger.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            (policy_dir/'diagnostics.jsonl').write_text('')
    output = tmp_path/'E1-4A'/'campaign-analysis'
    result = analyze_campaign(campaign, output)
    full = result['loads']['load_1p50']['policies']['cars_full']
    assert full['paired_misses']['risk_only_met_policy_missed'] == 1
    assert full['paired_misses']['risk_only_missed_policy_met'] == 1
    assert full['start_differences']['earlier_count'] == 1
    assert full['start_differences']['later_count'] == 1
    assert full['start_differences']['mean_delta_minutes'] == 0
    with (output/'common_metrics.csv').open() as stream:
        metrics = list(csv.DictReader(stream))
    assert len(metrics) == 10 and {row['phase'] for row in metrics} == {'common'}
    assert result['loads']['load_2p00']['fairness']['marker'] == 'load_2p00'


@pytest.mark.parametrize('policy,reason,defer,planner,executed,want', [
    ('risk_cap_only', 'cap_execute_guard', False, None, False, (1, 0, 0, 0)),
    ('cars_full', 'known_feasibility_uncertain', False,
     dict(status='uncertain', reason='pending_limit', planner_calls=0), None, (1, 0, 1, 0)),
    ('risk_shadow_only', 'shadow_uncertain_risk_fallback', True,
     dict(status='uncertain', reason='pending_limit', planner_calls=0), False, (0, 1, 1, 0)),
])
def test_cap_execute_and_uncertain_risk_defer_are_distinct_guard_outcomes(
        policy, reason, defer, planner, executed, want):
    # Catches CapOnly's absent planner being omitted, and ShadowOnly's reported
    # uncertain being subtracted from conservative execute into a negative count.
    from experiments.e1.diagnose_feasibility import decision_counts
    row = dict(policy=policy, reason=reason, defer=defer, pending_count=257,
               pending_limit_exceeded=True, risk_only_defer=True)
    if planner is not None:
        row['planner'] = planner
    if executed is not None:
        row['shadow_planner_executed'] = executed
    counts = decision_counts([row])
    assert (counts['cap_execute_guards'], counts['uncertain_risk_fallbacks'],
            counts['planner_uncertain_status_count'], counts['planner_actual_execution_decisions']) == want
    assert counts['pending_cap_fallbacks'] == want[0]
    assert counts['other_uncertain_fallbacks'] == 0
    assert counts['queue_limit_all_candidate_checks'] == 1
    assert counts['queue_limit_eligible_candidate_checks'] == 1
    assert counts['risk_only_eligible_candidate_checks'] == 1


def test_reported_uncertain_bypass_does_not_dilute_actual_planner_status_proportions():
    from experiments.e1.diagnose_feasibility import planner_status_counts
    bypass = dict(shadow_planner_executed=False,
        planner=dict(status='uncertain', reason='pending_limit', planner_calls=0))
    executed = dict(shadow_planner_executed=True,
        planner=dict(status='safe', reason='no_additional_known_deadline_miss', planner_calls=2))
    unknown = dict(shadow_planner_executed=True,
        planner=dict(status='uncertain', reason='bounded_horizon_unknown', planner_calls=2))
    stats = planner_status_counts([bypass, executed, unknown])
    assert stats['planner_reported_status_decisions'] == 3
    assert stats['planner_status_bypass_count'] == 1
    assert stats['planner_uncertain_proportion'] == pytest.approx(2/3)
    assert stats['planner_actual_execution_decisions'] == 2
    assert stats['planner_actual_safe_proportion'] == .5
    assert stats['planner_actual_uncertain_proportion'] == .5
    assert stats['planner_actual_scheduling_passes'] == 4


def test_queue_limit_all_candidates_includes_risk_only_ineligible_decisions():
    from experiments.e1.diagnose_feasibility import decision_counts
    rows = [dict(reason='risk_buffer_guard', pending_count=400,
                 pending_limit_exceeded=True, risk_only_defer=False),
            dict(reason='shadow_uncertain_risk_fallback', defer=True, pending_count=300,
                 pending_limit_exceeded=True, risk_only_defer=True, shadow_planner_executed=False,
                 planner=dict(status='uncertain', reason='pending_limit', planner_calls=0)),
            dict(reason='high_ci_defer', defer=True, pending_count=5,
                 pending_limit_exceeded=False, risk_only_defer=True, shadow_planner_executed=True,
                 planner=dict(status='safe', reason='no_additional_known_deadline_miss', planner_calls=2))]
    counts = decision_counts(rows)
    assert counts['queue_limit_all_candidate_checks'] == 2
    assert counts['risk_only_eligible_candidate_checks'] == 2
    assert counts['queue_limit_eligible_candidate_checks'] == 1
    assert counts['cap_execute_guards'] == 0
