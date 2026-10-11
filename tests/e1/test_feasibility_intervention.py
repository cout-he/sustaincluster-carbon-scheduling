"""Single-action intervention is separate from unchanged Full baselines."""
import pandas as pd
import pytest

from .helpers import START, task
from .test_policies import execute_policy, rows


def make(task_id='a', at=START):
    from experiments.e1.feasibility_intervention import SingleDeferFull
    return SingleDeferFull({'threshold_g_per_kwh':200}, START, (1,1,1), task_id, at)


def observe(p, pending, now=START):
    from experiments.e1.cars_state import ResourceSnapshot
    p.on_step_start(ResourceSnapshot(now,(1,1,1),(1,1,1),tuple(pending),(),tuple(pending)))


def test_only_exact_predeclared_action_changes_and_subsequent_full_restored():
    from experiments.e1.policies import DecisionContext
    from experiments.e1.cars import make_cars_policy
    a=task('a',duration=15,sla=2)
    p=make();base=make_cars_policy('cars_full',{'threshold_g_per_kwh':200},START,(1,1,1))
    for policy in (p,base):observe(policy,[a])
    assert base.decide(a,DecisionContext(START,250)).reason=='risk_buffer_guard'
    assert p.decide(a,DecisionContext(START,250)).defer
    p.on_task_deferred('a',START+pd.Timedelta(minutes=15))
    observe(p,[a],START+pd.Timedelta(minutes=15))
    assert not p.decide(a,DecisionContext(START+pd.Timedelta(minutes=15),250)).defer
    assert len(p.interventions)==1
    assert p.interventions[0]['original_reason']=='risk_buffer_guard'
    p.assert_applied_once()


def test_wrong_task_is_unchanged_and_missing_target_is_rejected():
    from experiments.e1.policies import DecisionContext
    p=make('b');a=task('a',duration=15,sla=2);observe(p,[a])
    assert not p.decide(a,DecisionContext(START,250)).defer
    with pytest.raises(RuntimeError,match='exactly once'):p.assert_applied_once()


def test_original_defer_cannot_be_reported_as_an_execute_to_defer_intervention():
    from experiments.e1.policies import DecisionContext
    p=make();a=task('a',duration=15,sla=4);observe(p,[a])
    with pytest.raises(ValueError,match='original Execute'):p.decide(a,DecisionContext(START,250))


def test_intervention_runs_real_executor_without_altering_input_or_ownership():
    a=task('a',duration=15,sla=2);p=make()
    result=execute_policy([a],p,minutes=45,ci=[250,250,250])
    p.assert_applied_once()
    row=rows(result)['a']
    assert row['start_time']==(START+pd.Timedelta(minutes=15)).isoformat()
    assert row['state']=='completed' and row['deadline_outcome']=='met'
    assert result.policy_steps[0]['policy_deferred_ids']==['a']
    assert result.policy_steps[1]['started_ids']==['a']
    assert row['cores_req']==a.cores_req and row['duration']==a.duration


def test_intervention_rejects_other_cohort_before_execution():
    from experiments.e1.feasibility_intervention import validate_intervention_cohort
    from .helpers import inputs
    workload,_=inputs([task('a')],30)
    with pytest.raises(ValueError,match='approved 1.5'):
        validate_intervention_cohort(workload,{'factor':1.5})
