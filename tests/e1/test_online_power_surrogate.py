import numpy as np
import pytest
from experiments.e1.online_power_surrogate import PowerSurrogate, fit_surrogate

def model(capacity=(10,10,10),hvac=(0,0,0,0,0)):
    return PowerSurrogate(10.,(1.,2.,.5),hvac,capacity,10000.,(0.,10000.),(-30.,50.))

def test_power_units_and_nonnegative_bounded_response():
    m=model(hvac=(0,-1,1,0,0))
    assert m.it((1,2,3))==16.5
    assert np.isfinite(m.hvac(100,20,50)) and 0<=m.hvac(100,20,50)<=10000
    with pytest.raises(ValueError): m.it((11,0,0))
    with pytest.raises(ValueError): m.hvac(float('nan'),20,0)

def test_dynamic_synthetic_response_and_static_v1():
    m=model(hvac=(100,0,0,0,0))
    assert m.hvac(10,20,200)==100
    from dataclasses import replace
    lag=replace(m,alpha=.5)
    assert lag.hvac(10,20,200)==150

def test_fit_train_only_rank_and_serialization():
    rows=[]
    for k in range(60):
        q=(k%7,(k*k)%11,(k*13)%17);it=10+q[0]+2*q[1]+.5*q[2];temp=10+k%9
        rows.append(dict(used_cores=q[0],used_gpus=q[1],used_mem=q[2],it_power_kw=it,
                         hvac_power_kw=50+.01*it*it+temp,temperature_c=temp))
    m,report=fit_surrogate(rows,rows,capacity=(100,100,100),role='train')
    assert report['it']['rank']==4 and report['hvac']['rank']==5
    assert PowerSurrogate.from_dict(m.to_dict())==m
    with pytest.raises(ValueError):fit_surrogate(rows,rows,capacity=(100,100,100),role='validation')

def test_nonfinite_negative_coefficients_input_contract():
    with pytest.raises(ValueError):model(hvac=(0,0,float('inf'),0,0))
    with pytest.raises(ValueError):PowerSurrogate(10,(-1,0,0),(0,0,0,0,0),(10,10,10),100)

@pytest.mark.parametrize('key',['split','role','data_role'])
def test_explicit_nontrain_row_labels_always_rejected(key):
    with pytest.raises(ValueError,match='label|Train'):
        fit_surrogate([{key:'test'}],[{key:'test'}],capacity=(10,10,10),role='train')
