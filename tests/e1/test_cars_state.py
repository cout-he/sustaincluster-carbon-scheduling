"""Online history uses observed arrivals once, never a full future trace."""
from dataclasses import FrozenInstanceError
import pandas as pd
import pytest
from .helpers import START, task


def test_snapshot_is_immutable_and_rejects_future_arrivals():
    from experiments.e1.cars_state import ResourceSnapshot
    snapshot = ResourceSnapshot(START, (1,1,1), (1,1,1), (task('a'),), ())
    with pytest.raises(FrozenInstanceError): snapshot.now = START
    with pytest.raises(ValueError, match='future'):
        ResourceSnapshot(START, (1,1,1), (1,1,1), (task('future',arrival=15),), ())


@pytest.mark.parametrize('cpu,expected', [(0.49,0),(0.5,1),(0.99,1),(1,2),(2,2)])
def test_pressure_bins_and_grid_occupation(cpu,expected):
    from experiments.e1.cars_state import ArrivalHistory
    h = ArrivalHistory(START, (1,1,1))
    s = task('a', duration=14, cpu=cpu, gpu=0, mem=0)
    h.observe(START,(s,))
    cold=h.pressure(START)
    assert cold.rho is None and cold.buffer_steps==2
    risk=h.pressure(START+pd.Timedelta(minutes=15))
    assert risk.resource_work_minutes == (cpu*15,0,0)
    assert risk.rho==pytest.approx(cpu) and risk.buffer_steps==expected


def test_history_no_duplicate_or_future_and_open_left_expiry_drain():
    from experiments.e1.cars_state import ArrivalHistory
    h=ArrivalHistory(START,(1,1,1)); s=task('a',duration=15)
    h.observe(START,(s,));h.observe(START,(s,))
    assert h.pressure(START).observed_tasks==1
    with pytest.raises(ValueError,match='future'): h.observe(START,(task('b',arrival=15),))
    assert h.pressure(START+pd.Timedelta(minutes=120)).observed_tasks==0
    assert h.pressure(START+pd.Timedelta(minutes=135)).rho==0


@pytest.mark.parametrize('resource', ['cpu','gpu','mem'])
def test_each_resource_can_determine_max_pressure(resource):
    from experiments.e1.cars_state import ArrivalHistory
    q=dict(cpu=0,gpu=0,mem=0);q[resource]=1
    h=ArrivalHistory(START,(1,1,1));h.observe(START,(task('a',duration=15,**q),))
    assert h.pressure(START+pd.Timedelta(minutes=15)).buffer_steps==2
