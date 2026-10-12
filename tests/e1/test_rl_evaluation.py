"""Fair PPO evaluation must compare a finite cohort over one physical horizon."""
from dataclasses import asdict, replace
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from experiments.e1.engine import LocalExecutor
from experiments.e1.cars import make_cars_policy
from .helpers import START, KnownPower, inputs, task


def bundle(split='validation', power_factory=KnownPower, tasks=None, minutes=30):
    from experiments.e1.rl_evaluate import EvaluationBundle
    w, _ = inputs(tasks or [task('a', duration=30, sla=2)], minutes)
    _, e = inputs([], 90)
    return EvaluationBundle(split, w, e, (1, 1, 1), power_factory,
                            {'threshold_g_per_kwh':50}, stress_seed=520001, factor=1.)


def test_group_extends_persistent_idle_physics_to_slowest_complete_run(tmp_path):
    from experiments.e1.rl_evaluate import EvaluationCandidate, evaluate_group
    created=[]
    class Stateful(KnownPower):
        def __init__(self): self.calls=0; self.closed=False; created.append(self)
        def measure(self,*args):
            assert not self.closed
            self.calls+=1
            return super().measure(*args)
        def close(self): self.closed=True
    b=bundle(power_factory=Stateful)
    result=evaluate_group(b,tmp_path/'group',candidates=[EvaluationCandidate('always-carbon',lambda obs:1)])
    rows={r['policy']:r for r in result['common_rows']}
    assert result['fairness']['common_end']==(START+pd.Timedelta(minutes=60)).isoformat()
    assert rows['edf_backfill']['total_energy_kwh']==5.
    assert rows['edf_backfill']['carbon_kg']==1.15
    assert rows['always-carbon']['total_energy_kwh']==5.
    assert rows['always-carbon']['carbon_kg']==1.35
    assert all(r['completed']==1 and r['pending']==r['running']==0 for r in rows.values())
    assert all(r['cores_hours']==.5 for r in rows.values())
    assert result['fairness']['equal_workload_carbon_comparison_valid']
    assert len(created)==4 and all(p.calls==4 and p.closed for p in created)
    assert rows['edf_backfill']['own_observation_end']==(START+pd.Timedelta(minutes=30)).isoformat()
    assert rows['always-carbon']['own_observation_end']==(START+pd.Timedelta(minutes=60)).isoformat()


@pytest.mark.parametrize('action,policy',[(0,'edf_backfill'),(1,'edf_carbon_threshold'),(2,'cars_risk_only')])
def test_fixed_modes_match_direct_ledger_resources_and_all_energy_fields(tmp_path,action,policy):
    from experiments.e1.rl_evaluate import evaluate_group
    b=bundle(tasks=[task('a',duration=30,sla=2),task('b',arrival=15,duration=15)],minutes=30)
    data=evaluate_group(b,tmp_path/f'fixed{action}',fixed_actions=(action,))
    assert data['equivalence'][str(action)]['all_execution_fields_exact']
    direct=data['results'][policy]
    fixed=data['results'][f'fixed_action_{action}']
    assert asdict(fixed)==asdict(direct)
    assert len({t['task_id'] for t in fixed.tasks})==2
    assert max(s['cores_fraction'] for s in fixed.steps)<=1


def test_test_split_rejected_before_creating_models_or_output(tmp_path):
    from experiments.e1.rl_evaluate import evaluate_group
    called=[]
    def forbidden(): called.append(True);raise AssertionError('physical Test model created')
    with pytest.raises(ValueError,match='Test'):
        evaluate_group(bundle('test',power_factory=forbidden),tmp_path/'forbidden')
    assert not called and not (tmp_path/'forbidden').exists()


def test_group_never_overwrites_prior_results(tmp_path):
    from experiments.e1.rl_evaluate import evaluate_group
    target=tmp_path/'kept';target.mkdir();(target/'evidence').write_text('old')
    with pytest.raises(FileExistsError):evaluate_group(bundle(),target)
    assert (target/'evidence').read_text()=='old'


def test_invalid_predictor_action_raises_and_closes_every_live_model(tmp_path):
    from experiments.e1.rl_evaluate import EvaluationCandidate,evaluate_group
    created=[]
    class ClosingPower(KnownPower):
        def __init__(self):self.closed=False;created.append(self)
        def close(self):self.closed=True
    with pytest.raises(ValueError,match='action'):
        evaluate_group(bundle(power_factory=ClosingPower),tmp_path/'bad',
                       candidates=[EvaluationCandidate('bad',lambda obs:3)])
    assert created and all(p.closed for p in created)
    assert json.loads((tmp_path/'bad/run_status.json').read_text())['state']=='failed'


def test_truncated_cohort_is_censored_and_cannot_claim_equal_work_carbon(tmp_path):
    from experiments.e1.rl_evaluate import evaluate_group
    b=bundle(tasks=[task('a',duration=200,sla=1)],minutes=30)
    data=evaluate_group(b,tmp_path/'incomplete')
    assert not data['fairness']['equal_workload_carbon_comparison_valid']
    for row in data['common_rows']:
        assert row['termination']=='safety_cap_incomplete'
        assert row['completed']==0 and row['running']==1
        assert row['deadline_right_censored']==1
        assert row['observation_end']==b.environment.end.isoformat()


def test_prediction_receives_only_current_sensor_and_actions_have_causal_labels(tmp_path):
    from experiments.e1.rl_evaluate import EvaluationCandidate,evaluate_group
    seen=[]
    def actor(obs):
        assert isinstance(obs,np.ndarray) and obs.shape==(40,)
        seen.append(obs.copy());return 0
    data=evaluate_group(bundle(),tmp_path/'online',
                        candidates=[EvaluationCandidate('ppo',actor,seed=7,training_steps=1024)])
    assert len(seen)==2
    trace=data['action_rows']['ppo']
    assert trace[0]['time']==START.isoformat()
    assert trace[0]['ci_g_per_kwh']==100 and trace[0]['action']==0
    assert trace[0]['pending_tasks']==1 and trace[0]['running_tasks']==0
    assert data['common_rows'][-1]['training_steps']==1024


def test_sla_first_selection_retains_every_seed_and_refuses_incomplete_cheapest():
    from experiments.e1.rl_evaluate import select_candidate
    rows=[dict(policy='s1',seed=1,arrived=100,completed=100,pending=0,running=0,
               deadline_violation_rate_observed=.06,carbon_kg=1),
          dict(policy='s2',seed=2,arrived=100,completed=100,pending=0,running=0,
               deadline_violation_rate_observed=.04,carbon_kg=3),
          dict(policy='s3',seed=3,arrived=100,completed=99,pending=1,running=0,
               deadline_violation_rate_observed=.01,carbon_kg=.1)]
    chosen=select_candidate(rows,max_deadline_violation_rate=.05)
    assert chosen['selected_policy']=='s2'
    assert chosen['candidates_considered']==3 and len(chosen['candidate_audit'])==3
    assert select_candidate([rows[0]],max_deadline_violation_rate=.05)['selected_policy'] is None
    assert select_candidate(rows,max_deadline_violation_rate=0)['status']=='no_eligible_candidate'


def test_selection_is_deterministic_on_carbon_miss_seed_ties():
    from experiments.e1.rl_evaluate import select_candidate
    base=dict(arrived=10,completed=10,pending=0,running=0,
              deadline_violation_rate_observed=0.,carbon_kg=10.)
    a=dict(base,policy='p2',seed=2);b=dict(base,policy='p1',seed=1)
    assert select_candidate([a,b],max_deadline_violation_rate=.05)['selected_policy']=='p1'
    with pytest.raises(ValueError):select_candidate([a],max_deadline_violation_rate=float('nan'))


def test_actor_and_current_interval_ignore_trajectory_forks_not_yet_observed(tmp_path):
    from experiments.e1.rl_evaluate import EvaluationCandidate,evaluate_group
    common=task('a',duration=15)
    first=bundle(tasks=[common,task('b',arrival=30,duration=15)],minutes=45)
    second=bundle(tasks=[common,task('c',arrival=30,duration=45)],minutes=45)
    changed=replace(second.environment,samples=tuple(
        replace(s,ci_g_per_kwh=999,temperature_c=40) if s.time>=START+pd.Timedelta(minutes=30)
        else s for s in second.environment.samples))
    second=replace(second,environment=changed)
    def online(obs):return int(obs[0]>.2)
    a=evaluate_group(first,tmp_path/'fork-a',candidates=[EvaluationCandidate('ppo',online)])
    b=evaluate_group(second,tmp_path/'fork-b',candidates=[EvaluationCandidate('ppo',online)])
    assert a['action_rows']['ppo'][:2]==b['action_rows']['ppo'][:2]
    assert a['results']['ppo'].steps[:2]==b['results']['ppo'].steps[:2]


def test_metadata_pin_default_reward_and_observation_normalization(tmp_path):
    from experiments.e1.rl_evaluate import evaluate_group
    data=evaluate_group(bundle(),tmp_path/'definitions')
    assert data['fairness']['reward_config']['carbon_scale_kg']==1000
    assert data['fairness']['reward_config']['terminal_weight']==10
    assert data['fairness']['normalization']['slack_minutes']==120


def test_load_candidate_test_guard_precedes_any_checkpoint_file_read():
    from experiments.e1.rl_evaluate import load_ppo_candidate
    with pytest.raises(ValueError,match='Test'):
        load_ppo_candidate('/does-not-exist.zip',{'name':'forbidden'},split='test')


def test_checkpoint_identity_is_required_before_any_file_read():
    from experiments.e1.rl_evaluate import load_ppo_candidate
    with pytest.raises(ValueError,match='SHA'):
        load_ppo_candidate('/does-not-exist.zip',{'name':'missing-identity'})


@pytest.fixture(scope='module')
def trained_checkpoint(tmp_path_factory):
    from experiments.e1.rl_env import RewardConfig,ObservationNormalization
    from experiments.e1.rl_train import train_ppo
    from .test_rl_training import tiny_config,tiny_factory
    folder=tmp_path_factory.mktemp('evaluation-checkpoint')/'trained'
    config=tiny_config()
    config.update(schema='e1-5b-ppo-v0',threshold_g_per_kwh=161.075,
        training_identity={'split':'train','stress_seed':510001,'factor':2.,'identity':{}},
        reward=asdict(RewardConfig()),normalization=asdict(ObservationNormalization()))
    summary=train_ppo(tiny_factory(),config,seed=55101,total_timesteps=64,output_dir=folder)
    return folder,summary


def checkpoint_copy(source,tmp_path):
    directory=tmp_path/'checkpoint'
    shutil.copytree(source,directory)
    return directory


def reseal_training_config(directory,config):
    """Tampered declarations remain byte-consistent to test semantic rejection."""
    from experiments.e1.contracts import file_hash,write_json
    write_json(directory/'training_config.json',config)
    summary=json.loads((directory/'training_summary.json').read_text())
    summary['config_sha256']=file_hash(directory/'training_config.json')
    summary['files']['training_config.json']=summary['config_sha256']
    write_json(directory/'training_summary.json',summary)


def candidate_metadata(summary):
    return dict(name='ppo_55101',seed=55101,training_steps=64,
                checkpoint_sha256=summary['checkpoint_sha256'])


def test_load_authenticated_trained_candidate_preserves_deterministic_actions_and_config_identity(trained_checkpoint):
    from experiments.e1.rl_evaluate import load_ppo_candidate
    directory,summary=trained_checkpoint
    candidate=load_ppo_candidate(directory/'model.zip',candidate_metadata(summary))
    records=json.loads((directory/'roundtrip.json').read_text())
    actions=[int(candidate.predictor(np.asarray(o,dtype=np.float32))) for o in records['observations']]
    assert actions==records['actions_after_load']
    assert candidate.training_config_sha256==summary['config_sha256']


@pytest.mark.parametrize('field,value',[('seed',55102),('training_steps',128)])
def test_load_rejects_candidate_metadata_inconsistent_with_training_summary(trained_checkpoint,field,value):
    from experiments.e1.rl_evaluate import load_ppo_candidate
    directory,summary=trained_checkpoint
    metadata=candidate_metadata(summary);metadata[field]=value
    with pytest.raises(ValueError,match='metadata'):
        load_ppo_candidate(directory/'model.zip',metadata)


def test_load_rejects_training_config_byte_tamper(trained_checkpoint,tmp_path):
    from experiments.e1.rl_evaluate import load_ppo_candidate
    directory,summary=trained_checkpoint
    copied=checkpoint_copy(directory,tmp_path)
    with (copied/'training_config.json').open('a') as stream:stream.write(' ')
    with pytest.raises(ValueError,match='hash'):
        load_ppo_candidate(copied/'model.zip',candidate_metadata(summary))


@pytest.mark.parametrize('invalid',['validation-training','wrong-threshold','modified-reward','modified-normalization'])
def test_load_rejects_resealed_invalid_training_protocol(trained_checkpoint,tmp_path,invalid):
    from experiments.e1.rl_evaluate import load_ppo_candidate
    directory,summary=trained_checkpoint
    copied=checkpoint_copy(directory,tmp_path)
    config=json.loads((copied/'training_config.json').read_text())
    if invalid=='validation-training':config['training_identity']['split']='validation'
    if invalid=='wrong-threshold':config['threshold_g_per_kwh']=137.65
    if invalid=='modified-reward':config['reward']['gamma']=1
    if invalid=='modified-normalization':config['normalization']['count_tasks']=10
    reseal_training_config(copied,config)
    with pytest.raises(ValueError,match='protocol'):
        load_ppo_candidate(copied/'model.zip',candidate_metadata(summary))


def test_checkpoint_hash_is_final_selection_tie_break_before_name():
    from experiments.e1.rl_evaluate import select_candidate
    base=dict(arrived=10,completed=10,pending=0,running=0,seed=55101,
              deadline_violation_rate_observed=0.,carbon_kg=10.)
    first=dict(base,policy='z-low-hash',checkpoint_sha256='a'*64)
    second=dict(base,policy='a-high-hash',checkpoint_sha256='b'*64)
    assert select_candidate([first,second],max_deadline_violation_rate=.05)['selected_policy']=='z-low-hash'
