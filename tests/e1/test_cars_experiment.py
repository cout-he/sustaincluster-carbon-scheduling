"""Seven-policy finite cohorts, historical bytes, and sealed replay behavior."""
import json
from dataclasses import replace

import pandas as pd
import pytest

from .helpers import KnownPower, task
from .test_run import configuration
from .test_temporal_run import fixture

NAMES = ('immediate_fifo', 'edf_backfill', 'carbon_threshold_fifo',
         'edf_carbon_threshold', 'cars_known_only', 'cars_risk_only', 'cars_full')
THRESHOLD = 137.64999999999998


def read(path):
    return json.loads(path.read_text())


def setup_reference(tmp_path, monkeypatch, tasks=None, factors=None):
    from experiments.e1 import temporal
    from experiments.e1.config import ROOT
    cfg, _ = configuration(tmp_path)
    w, e = fixture(tasks or [task('a', duration=30, sla=2),
                            task('b', arrival=15, duration=30, sla=2)],
                   safety_minutes=180)
    w = replace(w, sla_multiplier=w.tasks[0].sla_multiplier)
    cfg['start'], cfg['end'] = w.start.isoformat(), w.end.isoformat()
    cfg['workload'].update(resource_multiplier=1, sla_multiplier=w.sla_multiplier)
    cfg['datacenter'].update(total_cores=1, total_gpus=1, total_mem=1)
    monkeypatch.setattr(temporal, 'UpstreamPowerModel', lambda *args: KnownPower())
    selection = {'fixture': True, 'threshold_g_per_kwh': THRESHOLD}
    reference = tmp_path / 'reference'
    if factors:
        from experiments.e1.contention import run_levels
        run_levels(cfg, w, e, ROOT/'configs/dcs/dc_config.json', reference,
                   [1, 1.25, 1.5, 2], 20261010, THRESHOLD, selection)
    else:
        temporal.run_comparison(cfg, w, e, ROOT/'configs/dcs/dc_config.json',
                                reference, THRESHOLD, selection)
    parameters = read(ROOT/'experiments/e1/configs/e1_3_cars.json')
    return cfg, w, e, reference, parameters, selection


def run_case(tmp_path, monkeypatch, tasks=None):
    from experiments.e1 import cars_experiment as driver
    cfg, w, e, reference, parameters, selection = setup_reference(tmp_path, monkeypatch, tasks)
    monkeypatch.setattr(driver, 'UpstreamPowerModel', lambda *args: KnownPower())
    out = tmp_path/'cars'
    phases = driver.run_cars_comparison(cfg, w, e, reference/'dc_config.json', out,
                                       parameters, THRESHOLD, selection, reference)
    return driver, out, phases, w, reference


def test_seven_frozen_strategies_share_work_and_preserve_legacy_three(tmp_path, monkeypatch):
    # Catches altered task features, wrong common cutoff, or rewriting old formatting.
    driver, out, phases, w, reference = run_case(tmp_path, monkeypatch)
    assert set(phases) == set(NAMES)
    fair = read(out/'fairness.json')
    assert fair['all_tasks_completed'] and fair['equal_resource_hours']
    assert fair['equal_workload_carbon_comparison_valid']
    assert fair['policy_future_ci_access'] is False
    assert len(fair['resource_hours']['cores']) == 7
    assert len({p['common']['observation_end'] for p in phases.values()}) == 1
    assert all(p['common']['arrived'] == p['common']['completed'] == 2 for p in phases.values())
    assert all(p['common']['pending'] == p['common']['running'] == 0 for p in phases.values())
    assert read(out/'baseline_regression.json')['files_checked'] == 37
    numerical = read(reference/'provenance.json')['output_hashes']
    numerical = [n for n in numerical if n in ('comparison.csv','timeseries.csv',
                 'paired_task_shifts.csv','fairness.json') or n.split('/')[0] in NAMES[:3]]
    assert len(numerical) == 37
    for name in numerical:
        assert (reference/name).read_bytes() == (out/'legacy-three'/name).read_bytes(), name
    ids = {t.task_id: t for t in w.tasks}
    for name in NAMES:
        events = [json.loads(line) for line in (out/name/'policy_events.jsonl').read_text().splitlines()]
        assert all(pd.Timestamp(row['time']) >= ids[row['task_id']].arrival_time for row in events)
        fixed = read(out/name/'fixed_ledger.json')
        assert all(row['start_time'] is None or pd.Timestamp(row['start_time']) < w.end for row in fixed)
    with pytest.raises(FileExistsError):
        driver.replay_comparison(out, out)


def test_sealed_level_replay_excludes_runtime_and_has_identical_science(tmp_path, monkeypatch):
    # Catches clock timings being included in deterministic output hashes.
    driver, out, _, _, _ = run_case(tmp_path, monkeypatch)
    replay = tmp_path/'replay'
    driver.replay_comparison(out, replay)
    prov = read(out/'provenance.json')
    assert prov['input_hashes'] and prov['sources'] and prov['source_sha256']
    assert 'runtime.json' not in prov['output_hashes']
    assert len(read(out/'runtime.json')['policies']) == 7
    for policy in read(out/'runtime.json')['policies'].values():
        assert policy['scheduler']['order_calls'] > 0
        assert policy['scheduler']['decide_calls'] > 0
        assert policy['scheduler']['order_seconds'] >= 0
        assert policy['scheduler']['decision_seconds'] >= 0
        assert policy['scheduler']['planner_seconds'] >= 0
    for name in prov['output_hashes']:
        assert (out/name).read_bytes() == (replay/name).read_bytes(), name


@pytest.mark.parametrize('damage', ['source', 'manifest', 'input', 'input_manifest', 'runtime'])
def test_level_replay_mismatch_gets_failed_fresh_lifecycle(tmp_path, monkeypatch, damage):
    # Catches accepting damaged archives or rejecting before recording failure.
    driver, out, _, _, _ = run_case(tmp_path, monkeypatch)
    prov = read(out/'provenance.json')
    if damage == 'source':
        name = next(iter(prov['sources']))
        with (out/'source'/name).open('a') as stream:
            stream.write('\n# changed\n')
    elif damage == 'input':
        with (out/'dc_config.json').open('a') as stream:
            stream.write('\n')
    elif damage == 'input_manifest':
        prov['input_hashes']['workload_snapshot'] = 'incorrect'
        (out/'provenance.json').write_text(json.dumps(prov))
    elif damage == 'runtime':
        with (out/'runtime.json').open('a') as stream: stream.write('\n')
    else:
        del prov['output_hashes']['fairness.json']
        (out/'provenance.json').write_text(json.dumps(prov))
    replay = tmp_path/'replay'
    with pytest.raises((ValueError, RuntimeError)):
        driver.replay_comparison(out, replay)
    assert read(replay/'run_status.json')['state'] == 'failed'


def test_incomplete_cohort_keeps_denominator_and_disables_matched_carbon(tmp_path, monkeypatch):
    # Catches falsely claiming equal completed work for a structurally oversized task.
    _, out, phases, _, _ = run_case(tmp_path, monkeypatch, [task('a', cpu=2)])
    assert not read(out/'fairness.json')['equal_workload_carbon_comparison_valid']
    assert all(p['common']['arrived'] == p['common']['pending'] == 1 for p in phases.values())
    assert all(p['common']['deadline_missed_unfinished'] == 1 for p in phases.values())


def test_campaign_imports_frozen_stressed_child_without_generator_and_replays_offline(tmp_path, monkeypatch):
    # Catches rebuilding the synthetic workload instead of importing the sealed child.
    from experiments.e1 import cars_experiment as driver
    from experiments.e1 import stress
    _, _, _, reference, parameters, _ = setup_reference(tmp_path, monkeypatch, factors=[1, 2])
    monkeypatch.setattr(driver, 'UpstreamPowerModel', lambda *args: KnownPower())
    def forbidden(*args):
        raise AssertionError('A frozen reference must not be regenerated')
    monkeypatch.setattr(stress, 'stress_workload', forbidden)
    out = tmp_path/'campaign'
    rows = driver.run_reference_levels(reference, parameters, out, factors=[2])
    assert len(rows) == 21 and {r['factor'] for r in rows} == {2.0}
    child = out/'load_2p00'
    assert (child/'workload.json').read_bytes() == (reference/'load_2p00/workload.json').read_bytes()
    assert read(child/'baseline_regression.json')['files_checked'] == 37
    reference.rename(tmp_path/'removed-reference')
    replay = tmp_path/'campaign-replay'
    driver.replay_levels(out, replay)
    prov = read(out/'provenance.json')
    for name in prov['output_hashes']:
        assert (out/name).read_bytes() == (replay/name).read_bytes(), name
    assert read(replay/'load_2p00/run_status.json')['state'] == 'complete'


def test_campaign_requires_child_manifest_and_records_failure(tmp_path, monkeypatch):
    # Catches incomplete campaign manifests slipping through parent verification.
    from experiments.e1 import cars_experiment as driver
    _, _, _, reference, parameters, _ = setup_reference(tmp_path, monkeypatch, factors=[1, 2])
    monkeypatch.setattr(driver, 'UpstreamPowerModel', lambda *args: KnownPower())
    out = tmp_path/'campaign'
    driver.run_reference_levels(reference, parameters, out, factors=[2])
    prov = read(out/'provenance.json')
    del prov['child_output_manifests']['load_2p00']
    (out/'provenance.json').write_text(json.dumps(prov))
    replay = tmp_path/'campaign-replay'
    with pytest.raises(ValueError):
        driver.replay_levels(out, replay)
    assert read(replay/'run_status.json')['state'] == 'failed'


def test_legacy_endpoint_is_preserved_when_a_new_policy_drains_later(tmp_path, monkeypatch):
    # Catches extending old science to the seven-policy common end before regression.
    from experiments.e1 import cars_experiment as driver
    from experiments.e1.policies import ImmediateFIFO, PolicyDecision
    original = driver.make_cars_policy
    class DelayedPolicy(ImmediateFIFO):
        name = 'cars_full'
        def decide(self, spec, context):
            return PolicyDecision(context.now < spec.arrival_time+pd.Timedelta(minutes=90), 'fixture_delay')
    monkeypatch.setattr(driver, 'make_cars_policy', lambda name, *args:
                        DelayedPolicy() if name == 'cars_full' else original(name, *args))
    _, out, phases, _, _ = run_case(tmp_path, monkeypatch, [task('a',duration=30,sla=10)])
    old = read(out/'legacy-three/fairness.json')['common_end']
    assert old == '2023-08-01T06:00:00+00:00'
    assert read(out/'fairness.json')['common_end'] == '2023-08-01T07:00:00+00:00'
    assert phases['immediate_fifo']['own']['observation_end'] == '2023-08-01T05:30:00+00:00'
    assert read(out/'baseline_regression.json')['matched']


def test_cli_missing_manifest_records_failed_lifecycle(tmp_path, monkeypatch):
    # Catches CLI dispatch opening provenance before entering the fresh-run lifecycle.
    from experiments.e1 import cars_experiment as driver
    archive = tmp_path/'damaged'
    archive.mkdir()
    (archive/'config.json').write_text('{}')
    replay = tmp_path/'replay'
    monkeypatch.setattr('sys.argv', ['cars','--replay',str(archive),'--output',str(replay)])
    with pytest.raises(FileNotFoundError):
        driver.main()
    assert read(replay/'run_status.json')['state'] == 'failed'


def test_reference_parent_manifest_is_mandatory_before_any_child_execution(tmp_path, monkeypatch):
    # Catches importing from an incomplete E1-2B parent manifest.
    from experiments.e1 import cars_experiment as driver
    _, _, _, reference, parameters, _ = setup_reference(tmp_path, monkeypatch, factors=[1, 2])
    monkeypatch.setattr(driver, 'UpstreamPowerModel', lambda *args: KnownPower())
    prov = read(reference/'provenance.json')
    del prov['output_hashes']['config.json']
    (reference/'provenance.json').write_text(json.dumps(prov))
    out = tmp_path/'campaign'
    with pytest.raises(ValueError, match='manifest'):
        driver.run_reference_levels(reference, parameters, out, factors=[2])
    assert read(out/'run_status.json')['state'] == 'failed'


def test_approved_archive_head_preserves_research_provenance_not_publication_sha():
    from experiments.e1 import cars_experiment as driver
    from experiments.e1.config import ROOT
    folder=ROOT/'research_notes/E1/E1-2B/results-001/load_1p50'
    if not folder.exists(): pytest.skip('Private historical archive is an external data dependency')
    cfg,w,_,_,identity=driver._load_reference(folder)
    parameters=read(ROOT/'experiments/e1/configs/e1_3_cars.json')
    replication=read(folder.parent/'load_1p50_replication.json')
    archived_head=read(folder/'provenance.json')['git']['head']
    assert archived_head=='3f6ea95cb835b89ba50b0ef76d66d14b8037643e'
    assert archived_head!=parameters['reference_source_commit']
    driver.validate_approved_reference(cfg,w,identity,replication,parameters,archived_head)
    with pytest.raises(ValueError,match='approved'):
        driver.validate_approved_reference(cfg,w,identity,replication,parameters,'wrong')


def test_fixed_parameters_cannot_silently_enable_future_ci(tmp_path, monkeypatch):
    # Catches accepting an unimplemented or forbidden change to the declared design.
    from experiments.e1 import cars_experiment as driver
    from experiments.e1.config import ROOT
    cfg, _ = configuration(tmp_path)
    w, e = fixture([task('a')])
    cfg['end'] = w.end.isoformat()
    cfg['workload']['resource_multiplier'] = 1
    parameters = read(ROOT/'experiments/e1/configs/e1_3_cars.json')
    parameters['future_ci'] = True
    monkeypatch.setattr(driver, 'UpstreamPowerModel', lambda *args: KnownPower())
    out = tmp_path/'changed-parameters'
    with pytest.raises(ValueError, match='parameter'):
        driver.run_cars_comparison(cfg,w,e,ROOT/'configs/dcs/dc_config.json',out,parameters,THRESHOLD)
    assert read(out/'run_status.json')['state'] == 'failed'
