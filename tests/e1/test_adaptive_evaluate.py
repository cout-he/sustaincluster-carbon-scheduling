from dataclasses import replace
import numpy as np
import pytest
from .helpers import START, KnownPower, inputs, task
from experiments.e1.rl_evaluate import EvaluationBundle


def bundle():
    from experiments.e1.environment import FrozenEnvironment,EnvironmentalSample
    w,e=inputs([task('a',duration=15,cpu=.4,mem=0,sla=5),task('b',arrival=15,cpu=.4,mem=0,sla=2)],30)
    e=FrozenEnvironment(e.start,START+__import__('pandas').Timedelta(minutes=150),e.carbon_source_sha256,e.weather_source_sha256,
        tuple(EnvironmentalSample(START+__import__('pandas').Timedelta(minutes=m),200 if m<60 else 100,20,18) for m in range(0,150,15)))
    return EvaluationBundle('validation',w,e,(1,1,1),KnownPower,{'threshold_g_per_kwh':161.075},520001,1)


def test_common_idle_tail_task_resource_and_carbon_integrals(tmp_path):
    from experiments.e1.adaptive_evaluate import AdaptiveCandidate,evaluate_adaptive_group
    b=bundle()
    result=evaluate_adaptive_group(b,tmp_path/'first',candidates=(AdaptiveCandidate('plain','plain',lambda obs:3),))
    assert result['fairness']['all_tasks_completed']
    assert result['fairness']['equal_resource_hours']
    assert len({r['observation_end'] for r in result['common_rows']})==1
    assert len(result['common_rows'])==5
    edf=result['results']['edf_backfill'];slow=result['results']['plain']
    assert len(edf.steps)==len(slow.steps)
    assert edf.steps[-1]['running']==edf.steps[-1]['pending']==0
    for r in result['results'].values():
        assert r.summary['carbon_kg']==pytest.approx(sum(s['total_energy_kwh']*s['ci_g_per_kwh']/1000 for s in r.steps))
        assert len(r.tasks)==2
    assert result['fairness']['policy_future_ci_access'] is False


def test_same_actions_replay_ledger_and_energy_exact(tmp_path):
    from experiments.e1.adaptive_evaluate import AdaptiveCandidate,evaluate_adaptive_group,scientific_hashes,replay_adaptive_group
    def run(folder):
        return evaluate_adaptive_group(bundle(),folder,candidates=(AdaptiveCandidate('combined','combined',lambda obs:3),))
    first=run(tmp_path/'a');second=run(tmp_path/'b')
    assert scientific_hashes(first['results'])==scientific_hashes(second['results'])
    replay=replay_adaptive_group(tmp_path/'a',tmp_path/'replay',bundle=bundle())
    assert replay['replay']['scientific_outputs_exact']
    with (tmp_path/'a'/'combined'/'ledger.jsonl').open('a') as s:s.write('{}\n')
    with pytest.raises(ValueError,match='hash'):
        replay_adaptive_group(tmp_path/'a',tmp_path/'bad',bundle=bundle())
    with pytest.raises(FileExistsError):run(tmp_path/'a')


def test_test_role_rejected_before_model_or_physics(tmp_path):
    from experiments.e1.adaptive_evaluate import evaluate_adaptive_group,load_adaptive_candidate
    with pytest.raises(ValueError,match='Test'):
        evaluate_adaptive_group(replace(bundle(),split='test'),tmp_path/'forbidden')
    with pytest.raises(ValueError,match='Test'):
        load_adaptive_candidate(tmp_path/'nonexistent',split='test')
    assert not (tmp_path/'forbidden').exists()


def test_no_duplicate_names_and_invalid_action_is_rejected(tmp_path):
    from experiments.e1.adaptive_evaluate import AdaptiveCandidate,evaluate_adaptive_group
    with pytest.raises(ValueError):
        evaluate_adaptive_group(bundle(),tmp_path/'invalid',candidates=(AdaptiveCandidate('x','plain',lambda obs:4),))
    with pytest.raises(ValueError):
        evaluate_adaptive_group(bundle(),tmp_path/'dupe',candidates=(AdaptiveCandidate('edf_backfill','plain',lambda obs:0),))


def test_checkpoint_protocol_requires_registered_reward_action_schema_and_train_only():
    from pathlib import Path
    from copy import deepcopy
    import json
    from experiments.e1.adaptive_evaluate import validate_checkpoint_protocol
    from experiments.e1.config import ROOT
    c=json.loads((ROOT/'experiments/e1/configs/e1_6b_ppo.json').read_text())
    c.update(group='plain',seed=55101,total_timesteps=20480,training_identities=[
        dict(split='train',factor=f) for f in (1.5,2)])
    summary=dict(group='plain',seed=55101,num_timesteps=20480,risk_state=False,risk_filter=False,
                 observation_size=83,action_size=4,training_identities=c['training_identities'])
    assert validate_checkpoint_protocol(c,summary)
    for key,value in [('reward',{'gamma':11}),('action_slack_minutes',[None,90,30,15]),
                      ('observation_schema','old'),('training_identities',[dict(split='validation',factor=f) for f in (1.5,2)])]:
        modified=deepcopy(c);modified[key]=value
        with pytest.raises(ValueError):validate_checkpoint_protocol(modified,summary)
