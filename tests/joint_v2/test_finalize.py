"""Durable final-validation consumption and immutable model association.

These tests use no scientific source payload or physical simulation. The real
checkpoint store is exercised around a one-slot synthetic training fixture.
"""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.e1.contracts import file_hash
from experiments.joint_v2 import train_v2
from experiments.joint_v2.config import load_config


class _Agent:
    def __init__(self, obs_dim, action_dim, config=None, seed=0):
        self.obs_dim, self.action_dim, self.seed = obs_dim, action_dim, seed
        self.updates = 0

    def component_hashes(self):
        return dict(actor=f'actor-{self.seed}-{self.updates}',
                    critic=f'critic-{self.seed}-{self.updates}')

    def export_state(self):
        return dict(obs_dim=self.obs_dim, action_dim=self.action_dim,
                    seed=self.seed, updates=self.updates)

    @classmethod
    def from_state(cls, state):
        agent = cls(state['obs_dim'], state['action_dim'], seed=state['seed'])
        agent.updates = state['updates']
        return agent

    def act(self, observation, mask, deterministic=False):
        return 0

    def value(self, observation):
        return 0.

    def update(self, trajectory, bootstrap_value=None, *, deadline=None):
        self.updates += 1
        return dict(optimizer_steps=1, actor_optimizer_steps=1,
                    critic_optimizer_steps=1)


def _summary(scenario):
    return dict(scenario_id=scenario.scenario_id, scenario_role=scenario.role,
                arrived=1, completed=1, unfinished=0, deadline_right_censored=0,
                temperature_violations=0, dmr=0., deadline_misses=0,
                carbon_kg=10., incremental_carbon_kg=1.,
                fixed_window_complete=True, all_arrivals_observed=True,
                complete_episode_for_constraint=True,
                task_action_count=1, cooling_action_count=1)


class _Env:
    def __init__(self, scenario, config, **kwargs):
        self.scenario = scenario
        self.done = False

    def set_deadline_multiplier(self, value):
        pass

    def reset(self, seed=None):
        self.done = False

    def step(self, task_actor, cooling_actor):
        self.done = True
        return SimpleNamespace(task_boundary_observation=np.zeros(1),
                               cooling_boundary_observation=np.zeros(1),
                               task_decisions=(), cooling_decisions=(),
                               reward=-1., done=True)

    def summary(self):
        return _summary(self.scenario)

    def close(self):
        pass


class _Monitor:
    def __init__(self, output, **kwargs):
        pass

    def flush(self, state):
        pass

    def close(self):
        pass


@pytest.fixture
def finalize_run(monkeypatch, tmp_path):
    pool = tmp_path/'pool'
    pool.mkdir()
    (pool/'manifest.json').write_text('{"fixture":"no scientific payload"}')
    (pool/'config.json').write_text('{"sampler":{}}')
    output = tmp_path/'training'

    def scenario(role):
        return SimpleNamespace(scenario_id='synthetic-'+role, role=role,
                               identity=dict(source_fragment_id=role),
                               statistics=dict(capacity_pressure=.1, synthetic=True))

    train, development, heldout = map(scenario, ('train', 'development', 'validation'))
    monkeypatch.setattr(train_v2, '_role_sets', lambda _: ([train], [development]))
    monkeypatch.setattr(train_v2, 'IPPOAgent', _Agent)
    monkeypatch.setattr(train_v2, 'V2Env', _Env)
    monkeypatch.setattr(train_v2, 'Monitor', _Monitor)
    monkeypatch.setattr(train_v2, 'plot', lambda _: None)
    monkeypatch.setattr(train_v2, 'capacity_audit', lambda _: {})
    monkeypatch.setattr(train_v2, 'step_record', lambda outcome, **kwargs: kwargs)
    monkeypatch.setattr(train_v2, 'episode_record',
                        lambda index, scenario, *args: dict(episode=index, synthetic=True,
                            source_fragment_id=scenario.identity['source_fragment_id']))
    monkeypatch.setattr(train_v2, '_code_provenance',
                        lambda: dict(code_manifest_sha256='synthetic-code-identity'))
    cfg = load_config()
    cfg['monitor']['tensorboard'] = False
    cfg['validation']['heldout_final_check'] = True
    calls = dict(loads=0, evaluations=0)
    behavior = dict(mode='completed')

    def read_attempt():
        return json.loads((output/'heldout-validation-once.json').read_text(encoding='utf8'))

    def load_pool(path, roles):
        assert roles == ('validation',)
        # Publication must precede even the held-out payload load.
        marker = read_attempt()
        assert marker['status'] == 'attempted'
        pointer = marker['checkpoint']
        assert file_hash(output/'checkpoints'/pointer['file']) == pointer['sha256']
        calls['loads'] += 1
        return [heldout]

    def evaluate(agents, scenarios, config, deadline_multiplier, *, deadline=None):
        if scenarios[0].role == 'development':
            return [_summary(scenarios[0])]
        assert read_attempt()['status'] == 'attempted'
        calls['evaluations'] += 1
        if behavior['mode'] == 'interrupted':
            raise KeyboardInterrupt('synthetic interruption during final evaluation')
        if behavior['mode'] == 'failed':
            raise RuntimeError('synthetic failure during final evaluation')
        if behavior['mode'] == 'budget_exhausted':
            return None
        return [_summary(scenarios[0])]

    monkeypatch.setattr(train_v2, 'load_pool', load_pool)
    monkeypatch.setattr(train_v2, 'evaluate_scenarios', evaluate)

    def run(*, resume=None, max_episodes=1):
        return train_v2.run(pool, output, config=cfg, resume=resume, total_steps=20,
                            wall_seconds=60, max_episodes=max_episodes, skip_pretrain=True)

    return SimpleNamespace(run=run, output=output, calls=calls,
                           behavior=behavior, read_attempt=read_attempt)


@pytest.mark.parametrize('mode', ['budget_exhausted', 'interrupted', 'failed'])
def test_july_attempt_is_durable_before_access_and_not_retried_on_resume(finalize_run, mode):
    fixture = finalize_run
    fixture.behavior['mode'] = mode
    if mode == 'failed':
        with pytest.raises(RuntimeError, match='synthetic failure'):
            fixture.run()
    elif mode == 'interrupted':
        # Either clean finalization or propagation is acceptable; consumption
        # must survive both the interruption and absent checkpoint state flags.
        try:
            fixture.run()
        except KeyboardInterrupt:
            pass
    else:
        fixture.run()
    attempt = fixture.read_attempt()
    assert attempt['status'] == dict(interrupted='interrupted', failed='attempted',
                                     budget_exhausted='budget_exhausted')[mode]
    assert fixture.calls == dict(loads=1, evaluations=1)
    original_attempt = deepcopy(attempt)
    fixture.behavior['mode'] = 'completed'
    resumed = fixture.run(resume='latest', max_episodes=2)
    assert resumed['completed_episodes'] == 2
    assert fixture.calls == dict(loads=1, evaluations=1)
    assert fixture.read_attempt() == original_attempt


def test_july_result_keeps_original_generation_when_final_alias_advances(finalize_run):
    fixture = finalize_run
    first = fixture.run()
    attempt = fixture.read_attempt()
    report = json.loads((fixture.output/'heldout-validation-once.json').read_text(encoding='utf8'))
    assert attempt['status'] == report['status'] == 'completed'
    pointer = attempt['checkpoint']
    assert report['checkpoint'] == pointer
    assert first['heldout_validation']['checkpoint'] == pointer
    assert pointer['completed_episodes'] == 1
    generation = fixture.output/'checkpoints'/pointer['file']
    assert file_hash(generation) == pointer['sha256']
    original_report = deepcopy(report)

    resumed = fixture.run(resume='latest', max_episodes=2)
    latest_final = json.loads((fixture.output/'checkpoints/final.json').read_text(encoding='utf8'))
    assert first['component_hashes'] != resumed['component_hashes']
    assert latest_final['completed_episodes'] == 2
    assert latest_final['sha256'] != pointer['sha256']
    assert file_hash(generation) == pointer['sha256']
    assert json.loads((fixture.output/'heldout-validation-once.json').read_text(encoding='utf8')) == original_report
    assert fixture.read_attempt() == attempt
    assert fixture.calls == dict(loads=1, evaluations=1)


def test_reported_wall_time_includes_plotting_and_july_evaluation(finalize_run, monkeypatch):
    fixture = finalize_run
    clock = [0.]
    monkeypatch.setattr(train_v2.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(train_v2, 'plot', lambda _: clock.__setitem__(0, clock[0]+3.))
    original_evaluate = train_v2.evaluate_scenarios

    def evaluate(agents, scenarios, config, deadline_multiplier, *, deadline=None):
        if scenarios[0].role == 'validation':
            clock[0] += 7.
        return original_evaluate(agents, scenarios, config, deadline_multiplier, deadline=deadline)

    monkeypatch.setattr(train_v2, 'evaluate_scenarios', evaluate)
    result = fixture.run()
    assert fixture.read_attempt()['status'] == 'completed'
    assert result['last_run']['wall_seconds'] == 10.
