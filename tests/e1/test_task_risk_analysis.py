"""Diagnostic denominators and matched comparisons must not imply causation."""
import pytest


def _synthetic_campaign(tmp_path):
    """Internally matched file fixture; never substituted for scientific traces."""
    import json
    from pathlib import Path
    import pandas as pd
    from experiments.e1.contracts import content_hash, file_hash, write_json
    from experiments.e1.temporal import write_csv
    from experiments.e1.task_risk_experiment import CONFIG_SCHEMA, GROUP_POLICIES, PARAMETERS
    root = tmp_path / 'campaign'
    root.mkdir()
    cohorts = [dict(split=split, factor=factor) for split in ('train', 'validation')
               for factor in (1., 1.5, 2.)]
    config = dict(schema=CONFIG_SCHEMA, test_evaluation=False, new_training=False,
                  parameters=PARAMETERS, policies=list(GROUP_POLICIES), sla_target=.05,
                  cohorts=cohorts)
    write_json(root / 'campaign-freeze.json', dict(config=config, config_sha256=content_hash(config),
               sources={}, source_sha256=content_hash({}), formal_test_evaluated=False))
    start = pd.Timestamp('2023-05-01T00:00:00Z')
    arrival_end = start + pd.Timedelta(minutes=60)
    endpoint = start + pd.Timedelta(minutes=180)
    def iso(minutes):
        return (start + pd.Timedelta(minutes=minutes)).isoformat()
    def jsonl(path, values):
        Path(path).write_text(''.join(json.dumps(value) + '\n' for value in values), encoding='utf8')
    for cohort in cohorts:
        folder = root / f"{cohort['split']}-{cohort['factor']:g}"
        folder.mkdir()
        write_json(folder / 'run_status.json', dict(state='complete'))
        write_json(folder / 'source-freeze.json', dict(sources={}, source_sha256=content_hash({})))
        write_json(folder / 'fairness.json', dict(**cohort, policies=list(GROUP_POLICIES),
            workload_sha256='fixture-workload', environment_sha256='fixture-environment',
            identical_frozen_inputs=True, all_tasks_completed=True, equal_resource_hours=True,
            equal_workload_carbon_comparison_valid=True, idle_included=True,
            common_end=endpoint.isoformat(), arrival_end=arrival_end.isoformat(), arrived=3,
            parameters=PARAMETERS, formal_test_evaluated=False, policy_future_ci_access=False))
        write_json(folder / 'baseline-regression.json', dict(matched=True))
        common, phases, runtimes, paired = [], [], [], []
        for name in GROUP_POLICIES:
            sub = folder / name
            sub.mkdir()
            ledger = []
            for task_id in 'abc':
                minute = (165 if task_id == 'b' else 0)
                if name == 'task_risk_full':
                    minute = {'a': 165, 'b': 0, 'c': 15}[task_id]
                finish = minute + 15
                ledger.append(dict(task_id=task_id, arrival_time=iso(0), duration=15,
                    cores_req=.1, gpu_req=.2, mem_req=.3, sla_deadline=iso(150),
                    start_time=iso(minute), finish_time=iso(finish), state='completed',
                    completion_observed_at=iso(finish),
                    deadline_outcome='missed' if finish > 150 else 'met'))
            jsonl(sub / 'ledger.jsonl', ledger)
            jsonl(sub / 'policy_ledger.jsonl', [dict(task_id=r['task_id'],
                capacity_wait_intervals=0, policy_defer_intervals=1 if r['task_id'] == 'c' else 0,
                observed_pending_minutes=(pd.Timestamp(r['start_time']) - start).total_seconds()/60)
                for r in ledger])
            diagnostics = [dict(task_id='a', time=iso(0), eligible=False, budget=None,
                                reason='low_ci_execute', defer=False)]
            if name.startswith('task_risk_'):
                diagnostics.append(dict(task_id='c', time=iso(0), eligible=True, defer=True,
                    reason='high_ci_defer', budget=dict(buffer_minutes=30, dominant_resource='mem',
                    weights=(1/6, 1/3, 1/2), utilization=(0, 0, 0), urgent_pressure=(0, 0, 0),
                    history_pressure=(None, None, None), resource_pressure=(0, 0, 0), score=0,
                    solo_slack_minutes=135, safe_slack_minutes=105, cold_steps=2)))
            jsonl(sub / 'diagnostics.jsonl', diagnostics)
            jsonl(sub / 'policy_events.jsonl', [dict(task_id='a', reason='low_ci_execute', event='started'),
                dict(task_id='c', reason='high_ci_defer', event='policy_deferred')])
            write_json(sub / 'diagnostics-summary.json', dict(policy=name, counters={}))
            summary = dict(**cohort, policy=name, phase='common', arrived=3, completed=3,
                pending=0, running=0, deadline_missed_completed=1, deadline_missed_unfinished=0,
                deadline_right_censored=0, deadline_violation_rate_observed=1/3,
                observation_end=endpoint.isoformat(), own_observation_end=endpoint.isoformat(),
                drain_hours=2, termination='all_completed', it_energy_kwh=2, hvac_energy_kwh=1,
                total_energy_kwh=3, carbon_kg=.499 if name == 'task_risk_full' else .5,
                policy_defer_intervals=1, capacity_wait_intervals=0, pending_mean=.5, pending_max=1,
                mean_wait_minutes_started=sum((pd.Timestamp(r['start_time'])-start).total_seconds()/60
                                             for r in ledger)/3,
                mean_jct_minutes_completed=sum((pd.Timestamp(r['finish_time'])-start).total_seconds()/60
                                              for r in ledger)/3,
                cores_fraction_mean=.1, cores_fraction_max=.1, gpus_fraction_mean=.2,
                gpus_fraction_max=.2, mem_fraction_mean=.3, mem_fraction_max=.3,
                cores_hours=.075, gpus_hours=.15, mem_hours=.225)
            common.append(summary)
            phase_document = {phase:dict(summary, phase=phase) for phase in ('fixed', 'own', 'common')}
            write_json(sub / 'phases.json', phase_document)
            phases.extend(phase_document.values())
            runtimes.append(dict(policy=name, run_seconds=1.5, order_seconds=.1,
                decision_seconds=.2, decide_calls=len(diagnostics), step_state_seconds=.01,
                candidate_state_seconds=.01, start_notification_seconds=.01, defer_notification_seconds=.01))
        from experiments.e1.task_risk_analysis import paired_tasks
        reference_ledgers = {name: [json.loads(line) for line in
            (folder/name/'ledger.jsonl').read_text(encoding='utf8').splitlines()] for name in GROUP_POLICIES}
        for name in GROUP_POLICIES:
            for reference in ('edf_backfill', 'cars_risk_only'):
                pairs = paired_tasks(reference_ledgers[reference], reference_ledgers[name])
                paired.append(dict(policy=name, reference=reference,
                    extra_misses=sum(r['additional_miss'] for r in pairs),
                    avoided_misses=sum(r['avoided_miss'] for r in pairs),
                    both_missed=sum(r['shared_miss'] for r in pairs),
                    started_differently=sum(r['shift_minutes'] != 0 for r in pairs)))
        write_csv(folder/'common_summary.csv', common)
        write_csv(folder/'comparison.csv', phases)
        write_csv(folder/'runtime.csv', runtimes)
        write_csv(folder/'paired-miss-summary.csv', paired)
        manifest = {p.relative_to(folder).as_posix():file_hash(p) for p in folder.rglob('*')
                    if p.is_file() and p.name not in ('run_status.json', 'runtime.csv')}
        write_json(folder/'scientific-hashes.json', manifest)
    return root


def test_campaign_analysis_requires_complete_registered_cohorts_and_refuses_overwrite(tmp_path):
    from experiments.e1.task_risk_analysis import analyze_campaign
    import pandas as pd
    root = _synthetic_campaign(tmp_path)
    out = tmp_path/'analysis'
    result = analyze_campaign(root, out, plots=False)
    assert result['rows'] == 48 and result['formal_test_evaluated'] is False
    summary = pd.read_csv(out/'summary.csv')
    assert len(summary) == 48 and set(summary['split']) == {'train', 'validation'}
    assert summary['completed'].eq(3).all()
    assert summary['misses'].eq(1).all() and summary['dmr'].eq(1/3).all()
    assert summary['equal_workload_carbon_comparison_valid'].all()
    assert len(pd.read_csv(out/'phase-comparison.csv')) == 144
    with pytest.raises(FileExistsError):
        analyze_campaign(root, out, plots=False)


def test_campaign_analysis_validates_hashes_and_does_not_publish_partial_groups(tmp_path):
    from experiments.e1.task_risk_analysis import analyze_campaign
    root = _synthetic_campaign(tmp_path)
    ledger = root/'validation-2'/'task_risk_full'/'ledger.jsonl'
    ledger.write_text(ledger.read_text(encoding='utf8') + '{}\n', encoding='utf8')
    out = tmp_path/'analysis'
    with pytest.raises(ValueError, match='hash'):
        analyze_campaign(root, out, plots=False)
    assert not out.exists()


def test_campaign_analysis_rejects_test_config_even_if_named_as_development(tmp_path):
    import json
    from experiments.e1.task_risk_analysis import analyze_campaign
    from experiments.e1.contracts import content_hash, write_json
    root = _synthetic_campaign(tmp_path)
    path = root/'campaign-freeze.json'
    doc = json.loads(path.read_text(encoding='utf8'))
    doc['config']['cohorts'][-1]['split'] = 'test'
    doc['config_sha256'] = content_hash(doc['config'])
    write_json(path, doc)
    with pytest.raises(ValueError):
        analyze_campaign(root, tmp_path/'analysis', plots=False)


def test_representative_cases_include_added_avoided_and_changed_outcome_without_cherry_picking(tmp_path):
    import pandas as pd
    from experiments.e1.task_risk_analysis import analyze_campaign
    root = _synthetic_campaign(tmp_path)
    out = tmp_path/'analysis'
    analyze_campaign(root, out, plots=False)
    cases = pd.read_csv(out/'representative-case-selection.csv')
    assert len(cases) == 18 and cases['found'].all()
    assert cases.groupby('case')['task_id'].unique().to_dict() == {
        'added_miss': ['a'], 'avoided_miss': ['b'], 'changedstart_sameoutcome': ['c']}
    detail = pd.read_csv(out/'representative-tasks.csv')
    assert len(detail) == 18*8
    assert detail.groupby(['split','factor','case']).size().eq(8).all()
    budgets = pd.read_csv(out/'budget-statistics.csv')
    assert len(budgets) == 30
    assert budgets['candidate_checks'].eq(2).all()
    assert budgets['eligible_checks'].eq(1).all() and budgets['first_eligible_tasks'].eq(1).all()
    assert len(pd.read_csv(out/'buffer-histogram.csv')) == 30*11*2
    assert len(pd.read_csv(out/'paired-miss-summary.csv')) == 96


def test_paper_figures_export_png_pdf_and_source_data(tmp_path):
    import json
    from experiments.e1.task_risk_analysis import analyze_campaign
    root = _synthetic_campaign(tmp_path)
    out = tmp_path/'analysis'
    analyze_campaign(root, out)
    for stem in ('deadline-miss-rate','carbon-relative-edf','high-pressure-pareto',
                 'first-eligible-buffer-distribution','buffer-by-demand-category'):
        assert (out/(stem+'.png')).stat().st_size > 1000
        assert (out/(stem+'.pdf')).read_bytes().startswith(b'%PDF')
    assert (out/'figure-source-summary.csv').is_file()
    assert (out/'figure-source-buffer.csv').is_file()
    assert (out/'figure-source-resource.csv').is_file()
    provenance = json.loads((out/'analysis-provenance.json').read_text(encoding='utf8'))
    assert provenance['single_deterministic_run_per_cohort'] is True
    assert provenance['validation_is_development'] is True
    assert provenance['formal_test_evaluated'] is False
    assert (out/'analysis-hashes.json').is_file()


def test_budget_denominator_excludes_early_exit_and_first_task_is_unique():
    from experiments.e1.task_risk_analysis import budget_statistics
    rows = [dict(task_id='a', eligible=False, budget=None, reason='low_ci_execute'),
            dict(task_id='b', eligible=True, budget=dict(buffer_minutes=30, dominant_resource='gpu'), reason='task_risk_guard'),
            dict(task_id='b', eligible=True, budget=dict(buffer_minutes=15, dominant_resource='gpu'), reason='high_ci_defer'),
            dict(task_id='c', eligible=True, budget=dict(buffer_minutes=0, dominant_resource='cpu'), reason='high_ci_defer')]
    stat = budget_statistics(rows)
    assert stat['candidate_checks'] == 4 and stat['eligible_checks'] == 3
    assert stat['first_eligible_tasks'] == 2
    assert stat['all_buffer_mean_minutes'] == 15
    assert stat['first_buffer_mean_minutes'] == 15
    assert stat['first_resource']['gpu']['mean_minutes'] == 30
    assert stat['defer_fraction_eligible'] == pytest.approx(2/3)


def test_pareto_never_compares_different_cohorts_or_equal_points_as_dominating():
    from experiments.e1.task_risk_analysis import pareto_flags
    rows = [dict(split='train', factor=1, policy='a', carbon_kg=10, dmr=.1),
            dict(split='train', factor=1, policy='b', carbon_kg=11, dmr=.2),
            dict(split='train', factor=1, policy='c', carbon_kg=10, dmr=.1),
            dict(split='validation', factor=1, policy='d', carbon_kg=1, dmr=0)]
    assert pareto_flags(rows) == [True, False, True, True]


def test_paired_start_and_miss_difference_does_not_drop_task_identity():
    from experiments.e1.task_risk_analysis import paired_tasks
    base = [dict(task_id='a', start_time='2023-05-01T00:00:00Z', deadline_outcome='met')]
    later = [dict(task_id='a', start_time='2023-05-01T00:15:00Z', deadline_outcome='missed')]
    rows = paired_tasks(base, later)
    assert rows[0]['shift_minutes'] == 15
    assert rows[0]['additional_miss'] and not rows[0]['avoided_miss']
    with pytest.raises(ValueError, match='identity'):
        paired_tasks(base, [])
