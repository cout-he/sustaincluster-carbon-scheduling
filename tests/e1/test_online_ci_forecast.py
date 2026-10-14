from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from experiments.e1.online_ci_forecast import OnlineCIForecast, extract_prehistory, evaluate_forecasts
from .helpers import START

def warm(method='seasonal'):
    p=OnlineCIForecast(method)
    for k in range(96): p.observe(START-pd.Timedelta(hours=24)+pd.Timedelta(minutes=15*k),100+k)
    p.observe(START,120)
    return p

def test_seasonal_exact_legal_sources_and_current_actual():
    p=warm();f=p.forecast(START,5)
    assert f.ci==tuple(range(120,125)) and not f.cold
    assert f.times[0]==START and all(t<=START for group in f.sources for t in group)
    assert f.methods[0]=='observed_current'

def test_persistence_equal_and_duplicate_idempotence():
    p=warm('persistence');p.observe(START,120)
    assert p.forecast(START,12).ci==(120.,)*12
    with pytest.raises(ValueError): p.observe(START,121)
    with pytest.raises(ValueError): p.observe(START-pd.Timedelta(minutes=15),100)

def test_cold_missing_future_and_horizon_rejected():
    p=OnlineCIForecast();p.observe(START,10)
    f=p.forecast(START,96)
    assert f.cold and f.ci==(10.,)*96 and f.fallback_count==95
    for h in (0,97,True):
        with pytest.raises(ValueError): p.forecast(START,h)
    with pytest.raises(ValueError): p.forecast(START+pd.Timedelta(minutes=15),1)

def test_negative_seasonal_prediction_clamped():
    p=warm();p.observe(START+pd.Timedelta(minutes=15),0)
    assert min(p.forecast(START+pd.Timedelta(minutes=15),10).ci)>=0

def test_prehistory_previous_hour_only_and_strict_before_start(tmp_path):
    from experiments.e1.environment import CI_COLUMN
    dates=pd.date_range(START-pd.Timedelta(hours=25),START+pd.Timedelta(hours=2),freq='h')
    path=tmp_path/'ci.csv';pd.DataFrame({'Datetime (UTC)':dates,CI_COLUMN:range(len(dates))}).to_csv(path,index=False)
    rows=extract_prehistory(path,START)
    assert len(rows)==96 and all(r['time']<START and r['source_time']<=r['time'] for r in rows)
    assert [r['ci'] for r in rows[:4]]==[1.,1.,1.,1.]

def test_forecast_evaluation_uses_separate_future_labels():
    samples=[{'time':START+pd.Timedelta(minutes=15*k),'ci':float(k)} for k in range(20)]
    result=evaluate_forecasts(samples,(),horizons=(4,8,16))
    assert {r['horizon_steps'] for r in result['summary']}=={4,8,16}
    assert all(r['rmse']>=r['mae'] for r in result['summary'])

def test_raw_ci_declared_utc_column_with_naive_timestamps(tmp_path):
    from experiments.e1.environment import CI_COLUMN
    dates=pd.date_range(START-pd.Timedelta(hours=24),START,freq='h',inclusive='left').tz_localize(None)
    p=tmp_path/'raw.csv';pd.DataFrame({'Datetime (UTC)':dates,CI_COLUMN:100.}).to_csv(p,index=False)
    assert len(extract_prehistory(p,START))==96

def test_prehistory_stops_before_future_rows_and_binds_source(tmp_path):
    from experiments.e1.environment import CI_COLUMN
    dates=pd.date_range(START-pd.Timedelta(hours=24),START,freq='h')
    p=tmp_path/'raw.csv';pd.DataFrame({'Datetime (UTC)':dates,CI_COLUMN:100}).to_csv(p,index=False)
    with p.open('a') as stream:stream.write('invalid_future_timestamp,private_future_value\n')
    assert len(extract_prehistory(p,START))==96
    with pytest.raises(ValueError):extract_prehistory(p,START,expected_sha256='0'*64)

def test_unsorted_historical_prefix_fails(tmp_path):
    from experiments.e1.environment import CI_COLUMN
    dates=list(pd.date_range(START-pd.Timedelta(hours=24),START,freq='h',inclusive='left'))
    dates[:2]=reversed(dates[:2]);p=tmp_path/'unordered.csv'
    pd.DataFrame({'Datetime (UTC)':dates,CI_COLUMN:100}).to_csv(p,index=False)
    with pytest.raises(ValueError,match='order'):extract_prehistory(p,START)
