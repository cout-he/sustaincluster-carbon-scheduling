"""Guarded inputs and episode transactions, using a real tiny scenario."""
import json
import numpy as np
import pytest


def test_test_role_rejected_before_any_pool_access(tmp_path):
    from experiments.joint_v3.evaluate import evaluate
    with pytest.raises(ValueError, match='sealed'):
        evaluate(tmp_path/'missing-pool', tmp_path/'missing-checkpoint',
                 tmp_path/'out', role='test')
    assert not (tmp_path/'out').exists()


def test_september_once_entry_requires_authorization_before_opening_any_source(tmp_path):
    from experiments.joint_v3.test_once import run
    with pytest.raises(ValueError, match='sealed'):
        run(tmp_path/'missing-freeze', None, tmp_path/'missing-source', tmp_path/'out',
            marker_path=tmp_path/'attempt.json')
    assert not (tmp_path/'out').exists()
    assert not (tmp_path/'attempt.json').exists()


def test_fixed_baselines_include_same_full_idle_window():
    from experiments.joint_v3.evaluate import run_method
    from experiments.joint_v3.config import defaults
    from tests.joint_v2.test_env import fixture_scenario
    scenario = fixture_scenario(arrival_minutes=30, total_minutes=180)
    a = run_method('B0', scenario, defaults())
    b = run_method('B1', scenario, defaults())
    assert a['physical_steps'] == b['physical_steps'] == 12
    assert a['completed'] == b['completed'] == 0
    assert b['incremental_carbon_kg'] == pytest.approx(0., abs=1e-9)
    assert a['carbon_kg'] > b['carbon_kg']


def test_cooling_only_baseline_really_starts_each_resource_feasible_edf_candidate():
    from experiments.joint_v3.evaluate import run_method
    from experiments.joint_v3.config import defaults
    from experiments.joint_v3.mappo import RACSystem
    from tests.joint_v2.test_env import fixture_scenario
    from tests.e1.helpers import task
    scenario = fixture_scenario([task('blocker', duration=30, cpu=1, mem=1, sla=1),
        task('urgent', arrival=15, duration=15, cpu=2, mem=1, sla=2),
        task('flexible', arrival=15, duration=60, cpu=1, mem=1, sla=4)], total_minutes=150)
    result = run_method('B2', scenario, defaults(), RACSystem(seed=17))
    assert result['deadline_guard_triggers'] == 0
    assert result['task_action_distribution'].get('1', 0) == 0


def test_checkpoint_rejects_changed_source_before_loading_weights(tmp_path):
    from experiments.joint_v3.checkpoint import Store
    store = Store(tmp_path, 'config', 'data', 'code')
    store.save(dict(completed_episodes=1, test_access=False, sample=[1, 2]))
    assert store.load()['sample'] == [1, 2]
    with pytest.raises(ValueError, match='code'):
        Store(tmp_path, 'config', 'data', 'other').load()


def test_seed_budget_and_protocol_cannot_silently_expand():
    from experiments.joint_v3.config import defaults, load_config
    cfg = defaults()
    cfg['budget']['total_physical_steps'] = 65537
    with pytest.raises(ValueError): load_config(cfg)
    cfg = defaults(); cfg['validation']['heldout_final_check'] = True
    with pytest.raises(ValueError): load_config(cfg)


def test_small_training_resume_commits_only_whole_episodes(tmp_path):
    from experiments.joint_v3.train import run
    from experiments.joint_v3.config import defaults
    from tests.joint_v2.test_env import fixture_scenario
    cfg = defaults(seed=17)
    cfg['budget'].update(total_physical_steps=30, max_wall_seconds=60., max_episodes=1)
    cfg['pretrain']['cooling']['enabled'] = False
    cfg['pretrain']['task']['enabled'] = False
    scenario = fixture_scenario(arrival_minutes=30, total_minutes=180)
    scenario.identity = dict(resource_pressure_factor=1., sla_multiplier=1.5)
    dev = fixture_scenario(arrival_minutes=30, total_minutes=180)
    dev.role, dev.scenario_id = 'development', 'dev-fixture'
    out = tmp_path/'run'
    a = run(None, out, config=cfg, scenarios=([scenario], [dev]))
    assert a['completed_episodes'] == 1
    assert a['physical_steps'] == 6
    b = run(None, out, config=cfg, resume=True, scenarios=([scenario], [dev]))
    assert b['physical_steps'] == a['physical_steps']
    assert b['parameter_hashes'] == a['parameter_hashes']
    before = (out/'config.json').read_bytes()
    changed = json.loads(before)
    changed['ppo']['learning_rate'] *= 2
    with pytest.raises(ValueError, match='config'):
        run(None, out, config=changed, resume=True, scenarios=([scenario], [dev]))
    assert (out/'config.json').read_bytes() == before
