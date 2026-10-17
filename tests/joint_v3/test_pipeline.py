"""Frozen finite dispatch, prelaunch integrity and actual-result aggregation."""
import csv
import json

import pytest


def pool(tmp_path):
    folder = tmp_path/'pool'
    folder.mkdir()
    (folder/'manifest.json').write_text(json.dumps(dict(test_access=False, scenarios=[
        dict(scenario_id='train-a', role='train'),
        dict(scenario_id='dev-z', role='development'),
        dict(scenario_id='dev-a', role='development'),
        dict(scenario_id='july', role='validation')])), encoding='utf8')
    return folder


def test_prepare_freezes_eighteen_equal_budget_jobs_with_full_seeds_first(tmp_path):
    from experiments.joint_v3.pipeline import prepare, validate_protocol
    path = prepare(pool(tmp_path), tmp_path/'out')
    frozen = validate_protocol(path)
    assert len(frozen['jobs']) == 18
    assert [j['variant'] for j in frozen['jobs'][:3]] == ['full'] * 3
    assert [j['seed'] for j in frozen['jobs'][:3]] == [20261017, 20261018, 20261019]
    assert frozen['evaluation']['role'] == 'development'
    assert frozen['evaluation']['selection_scenario_ids'] == ['dev-a', 'dev-z']
    assert len(frozen['evaluation']['scenario_ids']) == 12
    assert frozen['evaluation']['scenario_ids'][0] == 'dev-a-eval-r0.75-s1.5'
    for job in frozen['jobs']:
        assert job['name'] == job['job_id']
        config = json.loads((path.parent/job['config']).read_text(encoding='utf8'))
        assert config['budget'] == dict(total_physical_steps=65536,
                                        max_wall_seconds=21600., max_episodes=None)
        assert config['variant'] == job['variant']
    assert frozen['test_access'] is False


def test_protocol_or_config_tampering_is_rejected_before_launch(tmp_path):
    from experiments.joint_v3.pipeline import prepare, run
    path = prepare(pool(tmp_path), tmp_path/'out')
    protocol = json.loads(path.read_text(encoding='utf8'))
    config_path = path.parent/protocol['jobs'][0]['config']
    config = json.loads(config_path.read_text(encoding='utf8'))
    config['budget']['total_physical_steps'] = 65537
    config_path.write_text(json.dumps(config), encoding='utf8')
    with pytest.raises(ValueError, match='config'):
        run(path, jobs=3)
    assert not (path.parent/'pipeline-status.json').exists()


def test_modified_protocol_or_pool_fails_identity_validation(tmp_path):
    from experiments.joint_v3.pipeline import prepare, validate_protocol
    folder = pool(tmp_path)
    path = prepare(folder, tmp_path/'out')
    protocol = json.loads(path.read_text(encoding='utf8'))
    protocol['jobs'][0]['seed'] = 1
    path.write_text(json.dumps(protocol), encoding='utf8')
    with pytest.raises(ValueError, match='protocol'):
        validate_protocol(path)


def test_changed_pool_metadata_is_rejected_before_launch(tmp_path):
    from experiments.joint_v3.pipeline import prepare, validate_protocol
    folder = pool(tmp_path)
    path = prepare(folder, tmp_path/'out')
    value = json.loads((folder/'manifest.json').read_text(encoding='utf8'))
    value['new_metadata'] = 'changed'
    (folder/'manifest.json').write_text(json.dumps(value), encoding='utf8')
    with pytest.raises(ValueError, match='pool'):
        validate_protocol(path)


def test_analyzer_uses_complete_rows_and_displays_n_without_inventing_missing_seeds(tmp_path):
    from experiments.joint_v3.analyze import analyze
    for seed, carbon, complete in [(20261017, 100., True), (20261018, 120., False)]:
        output = tmp_path/'jobs'/f'full-{seed}'/'evaluation'
        output.mkdir(parents=True)
        rows = [dict(seed=seed, variant='full', comparison_method='B5',
            scenario_id='dev-a', scenario_role='development', arrived=10, completed=10,
            deadline_misses=1, carbon_kg=carbon, incremental_carbon_kg=20.,
            it_carbon_kg=60., hvac_carbon_kg=40., total_energy_kwh=1.)]
        (output/'results.json').write_text(json.dumps(dict(complete=complete, results=rows,
                                                         test_access=False)), encoding='utf8')
    result = analyze(tmp_path, figures=False)
    with (tmp_path/'analysis'/'aggregate.csv').open(encoding='utf8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert result['evaluation_files'] == 1
    assert len(rows) == 1 and rows[0]['n'] == '1'
    assert float(rows[0]['carbon_kg_mean']) == 100.
    assert float(rows[0]['dmr_mean']) == .1
    assert rows[0]['carbon_kg_std'] == ''


def test_three_distinct_seeds_enable_sample_std_after_summing_scenarios(tmp_path):
    from experiments.joint_v3.analyze import analyze
    for index, seed in enumerate((20261017, 20261018, 20261019)):
        output = tmp_path/'jobs'/f'full-{seed}'/'evaluation'
        output.mkdir(parents=True)
        rows = [dict(seed=seed, variant='full', comparison_method='B5', scenario_id=scene,
            scenario_role='development', arrived=arrived, completed=arrived,
            deadline_misses=missed, carbon_kg=carbon)
            for scene, arrived, missed, carbon in [('a', 10, 1, 10.+index), ('b', 30, 1, 20.+index)]]
        (output/'results.json').write_text(json.dumps(dict(complete=True, results=rows,
                                                         test_access=False)), encoding='utf8')
    analyze(tmp_path, figures=False)
    with (tmp_path/'analysis'/'aggregate.csv').open(encoding='utf8', newline='') as stream:
        row = next(csv.DictReader(stream))
    assert row['n'] == '3'
    assert float(row['carbon_kg_mean']) == 32.
    assert float(row['carbon_kg_std']) == 2.
    assert float(row['dmr_mean']) == .05


def test_analysis_reports_actual_training_steps_without_rounding_to_target(tmp_path):
    from experiments.joint_v3.analyze import analyze
    folder = tmp_path/'jobs'/'full-20261017'/'training'
    folder.mkdir(parents=True)
    (folder/'run-result.json').write_text(json.dumps(dict(seed=20261017, variant='full',
        physical_steps=31, completed_episodes=1, elapsed_seconds=14.,
        last_run=dict(stop_reason='interrupted_uncommitted_episode_rolled_back'))), encoding='utf8')
    analyze(tmp_path, figures=False)
    with (tmp_path/'analysis'/'training-progress.csv').open(encoding='utf8', newline='') as stream:
        row = next(csv.DictReader(stream))
    assert row['actual_physical_steps'] == '31'
    assert row['target_reached'] == 'False'


def test_complete_frozen_jobs_are_not_dispatched_again(tmp_path, monkeypatch):
    from experiments.joint_v3 import pipeline
    path = pipeline.prepare(pool(tmp_path), tmp_path/'out')
    protocol = json.loads(path.read_text(encoding='utf8'))
    for job in protocol['jobs']:
        folder = path.parent/'jobs'/job['name']
        training = folder/'training'
        training.mkdir(parents=True)
        (training/'identity.json').write_text(json.dumps(dict(config_identity=job['config_identity'],
            code_identity=protocol['source_identity'])), encoding='utf8')
        (training/'run-result.json').write_text(json.dumps(dict(seed=job['seed'], variant=job['variant'],
            physical_steps=65536, elapsed_seconds=100., completed_episodes=8,
            last_run=dict(stop_reason='step_budget'))), encoding='utf8')
        evaluation = folder/'evaluation'
        evaluation.mkdir()
        rows = [dict(scenario_id=scene, comparison_method=method, seed=job['seed'],
            variant=job['variant'], scenario_role='development', arrived=1, completed=1,
            deadline_misses=0, carbon_kg=1.) for scene in protocol['evaluation']['scenario_ids']
            for method in (pipeline.METHODS if job['variant'] == 'full' else ('B5',))]
        (evaluation/'results.json').write_text(json.dumps(dict(complete=True, results=rows,
            test_access=False)), encoding='utf8')
    real_popen = pipeline.subprocess.Popen
    def forbidden_dispatch(args, **kwargs):
        if '-m' in args and any(str(a).startswith('experiments.joint_v3.') for a in args):
            raise AssertionError('A completed job must not launch another job process')
        return real_popen(args, **kwargs)
    monkeypatch.setattr(pipeline.subprocess, 'Popen', forbidden_dispatch)
    status = pipeline.run(path)
    assert status['stage'] == 'complete' and status['completed_runs'] == 18


def test_interrupted_zero_exit_training_is_not_evaluated_as_budget_complete(tmp_path, monkeypatch):
    from experiments.joint_v3 import analyze as analysis, pipeline
    path = pipeline.prepare(pool(tmp_path), tmp_path/'out')
    protocol = json.loads(path.read_text(encoding='utf8'))
    for index, job in enumerate(protocol['jobs']):
        folder = path.parent/'jobs'/job['name']
        training = folder/'training'
        training.mkdir(parents=True)
        (training/'identity.json').write_text(json.dumps(dict(config_identity=job['config_identity'],
            code_identity=protocol['source_identity'])), encoding='utf8')
        if index == 0:
            (training/'checkpoints').mkdir()
            (training/'checkpoints'/'best_deadline.json').write_text('{}', encoding='utf8')
            (training/'run-result.json').write_text(json.dumps(dict(seed=job['seed'], variant=job['variant'],
                physical_steps=31, elapsed_seconds=14., completed_episodes=1,
                last_run=dict(stop_reason='interrupted_uncommitted_episode_rolled_back'))), encoding='utf8')
        else:
            (training/'run-result.json').write_text(json.dumps(dict(seed=job['seed'], variant=job['variant'],
                physical_steps=65536, elapsed_seconds=100., completed_episodes=8,
                last_run=dict(stop_reason='step_budget'))), encoding='utf8')
            evaluation = folder/'evaluation'
            evaluation.mkdir()
            rows = [dict(scenario_id=scene, comparison_method=method, seed=job['seed'],
                variant=job['variant'], scenario_role='development', arrived=1, completed=1,
                deadline_misses=0, carbon_kg=1.) for scene in protocol['evaluation']['scenario_ids']
                for method in (pipeline.METHODS if job['variant'] == 'full' else ('B5',))]
            (evaluation/'results.json').write_text(json.dumps(dict(complete=True, results=rows,
                test_access=False)), encoding='utf8')
    class StoppedWorker:
        pid = 999999
        def poll(self):
            return 0
    monkeypatch.setattr(pipeline, 'code_identity', lambda: dict(sha256=protocol['source_identity']))
    monkeypatch.setattr(pipeline.subprocess, 'Popen', lambda *args, **kwargs: StoppedWorker())
    real_analyze = analysis.analyze
    monkeypatch.setattr(analysis, 'analyze', lambda root: real_analyze(root, figures=False))
    status = pipeline.run(path)
    interrupted = status['jobs'][protocol['jobs'][0]['job_id']]
    assert interrupted['stage'] == 'failed'
    assert interrupted.get('evaluation_output') is None
