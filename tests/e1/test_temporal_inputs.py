import json

import pandas as pd
import pytest

from .test_inputs import START, environment_files


def test_ci_previous_observation_does_not_use_next_actual_hour_and_roundtrips(tmp_path):
    from experiments.e1.selection import load_temporal_environment, TemporalEnvironment
    ci,weather=environment_files(tmp_path)
    first=load_temporal_environment(ci,weather,START,START+pd.Timedelta(hours=1))
    assert [s.ci_g_per_kwh for s in first.samples]==[100]*4
    frame=pd.read_csv(ci);frame.iloc[1,1]=999;frame.to_csv(ci,index=False)
    second=load_temporal_environment(ci,weather,START,START+pd.Timedelta(hours=1))
    assert [s.ci_g_per_kwh for s in second.samples]==[100]*4
    path=tmp_path/'frozen.json';first.save(path)
    assert TemporalEnvironment.load(path)==first
    assert json.loads(path.read_text())['payload']['interpolation']=='ci_previous_observation_weather_linear_utc'


def test_calibration_is_fixed_from_nonoverlap_history_and_rejects_gap():
    from experiments.e1.selection import calibrate_threshold
    from experiments.e1.environment import CI_COLUMN
    frame=pd.DataFrame({'Datetime (UTC)':pd.date_range(START,periods=5,freq='1h'),CI_COLUMN:[100,200,0,999,999]})
    assert calibrate_threshold(frame,START,START+pd.Timedelta(hours=2))==150
    with pytest.raises(ValueError,match='coverage'):
        calibrate_threshold(frame.drop(index=1),START,START+pd.Timedelta(hours=2))


def test_temporal_inputs_reject_internal_weather_or_carbon_gap(tmp_path):
    from experiments.e1.selection import load_temporal_environment
    from experiments.e1.environment import CI_COLUMN
    ci,weather=environment_files(tmp_path)
    pd.DataFrame({'Datetime (UTC)':pd.date_range(START,periods=4,freq='1h'),CI_COLUMN:[100]*4}).to_csv(ci,index=False)
    raw=json.loads(weather.read_text());raw['hourly']['time']=[(START+pd.Timedelta(hours=i)).tz_localize(None).isoformat() for i in (0,2,3)]
    raw['hourly']['temperature_2m']=[20,20,20];weather.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='coverage'):
        load_temporal_environment(ci,weather,START,START+pd.Timedelta(hours=3))
