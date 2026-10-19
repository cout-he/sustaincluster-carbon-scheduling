from dataclasses import replace
from types import SimpleNamespace
import json
from pathlib import Path
import pandas as pd
import pytest
from experiments.e1.workload import FrozenWorkload, TaskSpec
from experiments.object_model_a1.config import load_candidate, adapt_scenario

CONFIGS = Path(__file__).resolve().parents[2]/'experiments/object_model_a1/configs'


def source():
    start=pd.Timestamp('2023-01-01T00:00:00Z')
    raw=('job',0,0,0,30,200.,50.,3.,0,0,0,0)
    task=TaskSpec('identity','job',0,0,start,30,10,2.5,15,0,1.5,1,raw)
    w=FrozenWorkload(start,start+pd.Timedelta(days=1),'source',5,1.5,2020,(task,))
    return SimpleNamespace(scenario_id='train-synthetic-r1-s1.5',role='train',workload=w,
        environment=SimpleNamespace(start=start,end=start+pd.Timedelta(days=31)),
        identity={'role':'train','resource_pressure_factor':1},parameters={},statistics={})


def test_observed_adapter_preserves_identity_time_duration_and_environment():
    s=source(); c=load_candidate(CONFIGS/'OM.json'); a=adapt_scenario(s,c)
    assert a.capacity==(10000,200,16000)
    t=a.workload.tasks[0]
    assert (t.cores_req,t.gpu_req,t.mem_req)==(2,.5,3)
    assert t.task_id==s.workload.tasks[0].task_id
    assert t.arrival_time==s.workload.tasks[0].arrival_time and t.duration==30
    assert a.environment is s.environment and s.workload.tasks[0].gpu_req==2.5
    assert not hasattr(a,'model_path') and not hasattr(a,'save')


def test_legacy_mapping_is_unchanged_and_capacity_only():
    s=source(); a=adapt_scenario(s,load_candidate(CONFIGS/'M.json'))
    assert a.workload.tasks==s.workload.tasks and a.capacity==(50000,1000,80000)


def test_role_and_frozen_mapping_mismatch_fail_before_adaptation():
    s=source();s.role='validation'
    with pytest.raises(ValueError,match='Train'): adapt_scenario(s,load_candidate(CONFIGS/'OM.json'))
    s=source();s.workload=replace(s.workload,tasks=(replace(s.workload.tasks[0],gpu_req=999),))
    with pytest.raises(ValueError,match='raw'): adapt_scenario(s,load_candidate(CONFIGS/'OM.json'))


def test_unknown_request_units_and_physics_are_rejected(tmp_path):
    c=json.loads((CONFIGS/'OM.json').read_text())
    c['mapping_version']='requested_plan_v1'
    p=tmp_path/'c.json';p.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='mapping'): load_candidate(p)
    c=json.loads((CONFIGS/'OM.json').read_text());c['power_model']='legacy_joint'
    p.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='power'): load_candidate(p)


def test_unit_rescaling_preserves_normalized_load_without_rounding():
    s=source();l=load_candidate(CONFIGS/'M.json');o=load_candidate(CONFIGS/'OM.json')
    a,b=adapt_scenario(s,l),adapt_scenario(s,o)
    ta,tb=a.workload.tasks[0],b.workload.tasks[0]
    assert [ta.cores_req/a.capacity[0],ta.gpu_req/a.capacity[1],ta.mem_req/a.capacity[2]]==pytest.approx(
        [tb.cores_req/b.capacity[0],tb.gpu_req/b.capacity[1],tb.mem_req/b.capacity[2]])


@pytest.mark.parametrize('key,value',[
    ('pressure_factors',[1]),('sla_multipliers',[1.5]),
    ('ci_version','future_linear'),('base_source_sha','unknown'),
    ('final_research_object_frozen',True)])
def test_execution_declarations_cannot_silently_disagree(tmp_path,key,value):
    c=json.loads((CONFIGS/'OM.json').read_text());c[key]=value
    path=tmp_path/'bad.json';path.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='declaration'): load_candidate(path)


def test_derived_candidate_does_not_expose_source_statistics_as_new_metrics():
    s=source();s.statistics={'capacity_pressure':.123,'label':'old'}
    a=adapt_scenario(s,load_candidate(CONFIGS/'OM.json'))
    assert 'capacity_pressure' not in a.statistics
    assert a.statistics['source_statistics']==s.statistics
    assert a.statistics['derived_statistics_available'] is False
