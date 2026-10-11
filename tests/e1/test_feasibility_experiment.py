"""E1-4A imports sealed E1-3 cohorts and preserves its controls exactly."""
import json

import pandas as pd
import pytest

from .helpers import KnownPower, task
from .test_cars_experiment import THRESHOLD, setup_reference


NAMES = ('edf_carbon_threshold', 'cars_risk_only', 'cars_full',
         'risk_cap_only', 'risk_shadow_only')


def read(path):
    return json.loads(path.read_text())


def test_campaign_inventory_rejects_unmanifested_scientific_directory(tmp_path):
    # Strict replay must not silently drop additional scientific evidence.
    from experiments.e1 import feasibility_experiment as driver
    (tmp_path/'config.json').write_text('{}')
    child = tmp_path/'load_1p50'
    child.mkdir()
    (child/'summary.json').write_text('{}')
    driver._verify_campaign_inventory(tmp_path, {'config.json'}, {'load_1p50': {'summary.json'}})
    extra = tmp_path/'untracked'
    extra.mkdir()
    (extra/'result.csv').write_text('value\n1\n')
    with pytest.raises(ValueError, match='inventory'):
        driver._verify_campaign_inventory(tmp_path, {'config.json'}, {'load_1p50': {'summary.json'}})


def reference(tmp_path, monkeypatch, campaign=False, tasks=None):
    from experiments.e1 import cars_experiment as old
    cfg, w, e, previous, parameters, selection = setup_reference(
        tmp_path, monkeypatch, tasks or [task('a', duration=30, sla=4),
                                         task('b', arrival=15, duration=30, sla=4)],
        factors=[1, 2] if campaign else None)
    monkeypatch.setattr(old, 'UpstreamPowerModel', lambda *args: KnownPower())
    out = tmp_path/'e1-3'
    if campaign:
        old.run_reference_levels(previous, parameters, out)
    else:
        old.run_cars_comparison(cfg, w, e, previous/'dc_config.json', out,
                               parameters, THRESHOLD, selection, previous)
    return cfg, w, e, out, read(out/'parameters.json'), selection


def run_case(tmp_path, monkeypatch, tasks=None):
    from experiments.e1 import feasibility_experiment as driver
    cfg, w, e, old, parameters, selection = reference(tmp_path, monkeypatch, tasks=tasks)
    monkeypatch.setattr(driver, 'UpstreamPowerModel', lambda *args: KnownPower())
    out = tmp_path/'e1-4a'
    phases = driver.run_ablation_comparison(cfg, w, e, old/'dc_config.json', out,
        parameters, THRESHOLD, selection, reference=old)
    return driver, out, old, phases


def test_five_policies_share_frozen_cohort_and_legacy_control_bytes(tmp_path, monkeypatch):
    # Changing a control decision, its endpoint, or the common drain breaks this test.
    driver, out, old, phases = run_case(tmp_path, monkeypatch)
    assert tuple(phases) == NAMES
    fair = read(out/'fairness.json')
    assert fair['all_tasks_completed'] and fair['equal_resource_hours']
    assert fair['policy_future_ci_access'] is False and fair['idle_included']
    assert len({p['common']['observation_end'] for p in phases.values()}) == 1
    assert all(p['common']['completed'] == p['common']['arrived'] == 2 for p in phases.values())
    regression = read(out/'baseline_regression.json')
    assert regression['matched']
    for policy in NAMES[:3]:
        for filename in (*driver.POLICY_FILES, 'diagnostics.jsonl'):
            assert (old/policy/filename).read_bytes() == (out/'legacy-controls'/policy/filename).read_bytes()
    for filename in ('workload.json', 'environment.json', 'dc_config.json',
                     'parameters.json', 'selection.json', 'resolved_config.json'):
        assert (out/filename).read_bytes() == (old/filename).read_bytes()


def test_level_strict_replay_is_identical_and_runtime_has_separate_integrity(tmp_path, monkeypatch):
    # A non-deterministic timing in science or ignored timing integrity breaks this test.
    driver, out, _, _ = run_case(tmp_path, monkeypatch)
    replay = tmp_path/'replay'
    driver.replay_comparison(out, replay)
    prov = read(out/'provenance.json')
    assert set(prov['output_hashes']) == driver.LEVEL_FILES
    assert 'runtime.json' not in prov['output_hashes']
    assert set(prov['runtime_hashes']) == {'runtime.json'}
    assert set(read(out/'runtime.json')['policies']) == set(NAMES)
    assert prov['input_hashes'] and prov['sources'] and prov['model_source_version']
    for filename in prov['output_hashes']:
        assert (out/filename).read_bytes() == (replay/filename).read_bytes(), filename
    with (out/'runtime.json').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='runtime'):
        driver.replay_comparison(out, tmp_path/'bad-runtime')
    assert read(tmp_path/'bad-runtime/run_status.json')['state'] == 'failed'


@pytest.mark.parametrize('damage', ['manifest', 'input', 'source', 'parameters'])
def test_cross_version_import_checks_complete_old_archive_before_execution(tmp_path, monkeypatch, damage):
    # A missing scientific key, wrong source, or changed frozen input must fail closed.
    from experiments.e1 import feasibility_experiment as driver
    cfg, w, e, old, parameters, selection = reference(tmp_path, monkeypatch)
    prov = read(old/'provenance.json')
    if damage == 'manifest':
        del prov['output_hashes']['fairness.json']
        (old/'provenance.json').write_text(json.dumps(prov))
    elif damage == 'input':
        prov['input_hashes']['selection'] = 'wrong'
        (old/'provenance.json').write_text(json.dumps(prov))
    elif damage == 'source':
        source = next(iter(prov['sources']))
        with (old/'source'/source).open('a') as stream:
            stream.write('\n# damaged\n')
    else:
        parameters['max_pending'] = 128
    out = tmp_path/'failed-import'
    with pytest.raises((ValueError, RuntimeError)):
        driver.run_ablation_comparison(cfg, w, e, old/'dc_config.json', out,
            parameters, THRESHOLD, selection, reference=old)
    assert read(out/'run_status.json')['state'] == 'failed'
    assert not (out/NAMES[0]).exists()


def test_campaign_offline_replay_preserves_children_and_parent_manifest(tmp_path, monkeypatch):
    # Re-reading the reference or rebuilding synthetic workloads during replay breaks it.
    from experiments.e1 import feasibility_experiment as driver
    from experiments.e1 import stress
    _, _, _, old, _, _ = reference(tmp_path, monkeypatch, campaign=True)
    monkeypatch.setattr(driver, 'UpstreamPowerModel', lambda *args: KnownPower())
    def forbidden(*args, **kwargs):
        raise AssertionError('Frozen workload cannot be regenerated')
    monkeypatch.setattr(stress, 'stress_workload', forbidden)
    out = tmp_path/'campaign'
    rows = driver.run_reference_levels(old, out, factors=[1.5, 2.0])
    assert len(rows) == 30
    assert {r['factor'] for r in rows} == {1.5, 2.0}
    assert {r['policy'] for r in rows} == set(NAMES)
    old.rename(tmp_path/'reference-moved')
    replay = tmp_path/'campaign-replay'
    driver.replay_levels(out, replay)
    prov = read(out/'provenance.json')
    assert set(prov['output_hashes']) == driver.CAMPAIGN_FILES
    for filename in prov['output_hashes']:
        assert (out/filename).read_bytes() == (replay/filename).read_bytes(), filename
    for child in ('load_1p50', 'load_2p00'):
        for filename in prov['child_output_manifests'][child]['output_hashes']:
            assert (out/child/filename).read_bytes() == (replay/child/filename).read_bytes()


def test_legacy_control_cutoff_is_independent_of_new_policy_own_drain(tmp_path, monkeypatch):
    # Extending controls first to a later E1-4A end would destroy old carbon accounting.
    from experiments.e1 import feasibility_experiment as driver
    from experiments.e1.policies import ImmediateFIFO, PolicyDecision
    original = driver.make_ablation_policy
    class Delayed(ImmediateFIFO):
        name = 'risk_cap_only'
        def decide(self, spec, context):
            return PolicyDecision(context.now < spec.arrival_time+pd.Timedelta(minutes=90), 'fixture_delay')
    monkeypatch.setattr(driver, 'make_ablation_policy', lambda name, *args:
        Delayed() if name == 'risk_cap_only' else original(name, *args))
    _, out, old, phases = run_case(tmp_path, monkeypatch, [task('a', duration=30, sla=10)])
    old_end = read(old/'fairness.json')['common_end']
    assert pd.Timestamp(read(out/'fairness.json')['common_end']) > pd.Timestamp(old_end)
    assert read(out/'baseline_regression.json')['reference_common_end'] == old_end
    assert phases['cars_full']['own']['observation_end'] <= old_end


def test_legacy_control_cutoff_can_be_later_than_five_policy_common_end(tmp_path, monkeypatch):
    # A physical model cannot be rewound after idle extension to the old longer end.
    from experiments.e1 import cars_experiment as old
    from experiments.e1.policies import ImmediateFIFO, PolicyDecision
    original = old.make_cars_policy
    class DelayedOldKnown(ImmediateFIFO):
        name = 'cars_known_only'
        def decide(self, spec, context):
            return PolicyDecision(context.now < spec.arrival_time+pd.Timedelta(minutes=90), 'fixture_delay')
    monkeypatch.setattr(old, 'make_cars_policy', lambda name, *args:
        DelayedOldKnown() if name == 'cars_known_only' else original(name, *args))
    driver, out, previous, phases = run_case(tmp_path, monkeypatch, [task('a', duration=30, sla=10)])
    old_end = read(previous/'fairness.json')['common_end']
    common = read(out/'fairness.json')['common_end']
    assert pd.Timestamp(common) < pd.Timestamp(old_end)
    assert all(p['common']['observation_end'] == common for p in phases.values())
    for policy in NAMES[:3]:
        for filename in (*driver.POLICY_FILES, 'diagnostics.jsonl'):
            assert (previous/policy/filename).read_bytes() == (out/'legacy-controls'/policy/filename).read_bytes()
