from dataclasses import replace

import pytest

from .helpers import inputs, task


def base():
    return inputs([task(chr(97+i), arrival=15*(i%2)) for i in range(8)])[0]


def test_original_specs_and_one_x_identity_are_preserved():
    from experiments.e1.stress import stress_workload
    w=base()
    one,m=stress_workload(w,1,20261010)
    assert one is w and m['extra_tasks']==0
    two,m=stress_workload(w,2,20261010)
    assert two.tasks[:len(w.tasks)]==w.tasks
    for copy,parent in zip(two.tasks[len(w.tasks):],m['copies']):
        original=next(t for t in w.tasks if t.task_id==parent['parent_task_id'])
        assert replace(copy,task_id=original.task_id)==original
    assert len({t.task_id for t in two.tasks})==16
    assert two.resource_multiplier==w.resource_multiplier


def test_fixed_seed_floor_counts_reproducible_nested_prefixes():
    from experiments.e1.stress import stress_workload
    w=base()
    previous=[]
    for f,n in [(1,8),(1.25,10),(1.5,12),(2,16)]:
        result,m=stress_workload(w,f,123)
        assert len(result.tasks)==n and result.tasks[:len(previous)]==tuple(previous)
        assert stress_workload(w,f,123)[0].sha256==result.sha256
        previous=list(result.tasks)
    assert stress_workload(w,1.25,124)[0].sha256!=stress_workload(w,1.25,123)[0].sha256
    odd=replace(w,tasks=w.tasks[:3])
    assert len(stress_workload(odd,1.25,123)[0].tasks)==3
    assert len(stress_workload(odd,1.5,123)[0].tasks)==4


@pytest.mark.parametrize('factor',[0.9,2.1,True,float('nan'),float('inf')])
def test_invalid_stress_requests_fail(factor):
    from experiments.e1.stress import stress_workload
    with pytest.raises(ValueError):stress_workload(base(),factor,123)


def test_seed_and_clone_id_collision_fail():
    from experiments.e1.stress import stress_workload
    with pytest.raises(ValueError):stress_workload(base(),2,True)
    w=base();two,m=stress_workload(w,2,123)
    damaged=replace(w,tasks=w.tasks+(two.tasks[-1],))
    with pytest.raises(ValueError,match='ID'):stress_workload(damaged,2,123)
