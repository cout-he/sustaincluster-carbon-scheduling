"""Outcome-independent window selection, historical calibration and causal CI input."""
import json
from dataclasses import replace
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve_path
from .contracts import file_hash, interval_times, read_snapshot, save_snapshot, utc_time
from .environment import CI_COLUMN, EnvironmentalSample, FrozenEnvironment, load_environment

CI_BASIS='ci_previous_observation_weather_linear_utc'


def require_hourly_coverage(times, values, start, end, name):
    dates=pd.DatetimeIndex(pd.to_datetime(times,utc=True))
    if dates.has_duplicates or dates.hasnans or not np.isfinite(np.asarray(values,dtype=float)).all():
        raise ValueError(f'{name} invalid timestamps or values')
    expected=pd.date_range(start.floor('1h'),end.ceil('1h'),freq='1h')
    if not set(expected)<=set(dates):
        raise ValueError(f'{name} hourly coverage gap')


def calibrate_threshold(frame,start,end):
    start,end=utc_time(start,True),utc_time(end,True)
    times=pd.to_datetime(frame['Datetime (UTC)'],utc=True)
    selected=frame[(times>=start)&(times<end)]
    dates=pd.DatetimeIndex(pd.to_datetime(selected['Datetime (UTC)'],utc=True))
    expected=pd.date_range(start,end,freq='1h',inclusive='left')
    if len(dates)!=len(expected) or set(dates)!=set(expected) or not np.isfinite(selected[CI_COLUMN]).all():
        raise ValueError('Calibration hourly coverage must be complete')
    return float(selected[CI_COLUMN].median())


class TemporalEnvironment(FrozenEnvironment):
    def payload(self):
        payload=super().payload();payload['interpolation']=CI_BASIS
        return payload

    def save(self,path):
        save_snapshot(path,'e1-temporal-environment-v1',self.payload())

    @classmethod
    def load(cls,path):
        data=read_snapshot(path,'e1-temporal-environment-v1')
        if data.pop('interpolation')!=CI_BASIS or data.pop('wet_bulb_transform')!='upstream_temperature_times_0.9':
            raise ValueError('Unexpected temporal environment transform')
        data['start'],data['end']=utc_time(data['start'],True),utc_time(data['end'],True)
        data['samples']=tuple(EnvironmentalSample(utc_time(s['time'],True),s['ci_g_per_kwh'],
                              s['temperature_c'],s['wet_bulb_c']) for s in data['samples'])
        return cls(**data)


def load_temporal_environment(carbon_path,weather_path,start,end):
    start,end=utc_time(start,True),utc_time(end,True)
    frame=pd.read_csv(carbon_path)
    dates=pd.DatetimeIndex(pd.to_datetime(frame['Datetime (UTC)'],utc=True))
    require_hourly_coverage(dates,frame[CI_COLUMN],start,end,'Carbon')
    weather=json.loads(Path(weather_path).read_text(encoding='utf-8'))
    times=pd.DatetimeIndex(pd.to_datetime(weather['hourly']['time']))
    if times.tz is None:
        times=times.tz_localize(timezone(timedelta(seconds=weather['utc_offset_seconds'])))
    require_hourly_coverage(times,weather['hourly']['temperature_2m'],start,end,'Weather')
    original=load_environment(carbon_path,weather_path,start,end)
    order=np.argsort(dates.asi8);stamps=dates.asi8[order]
    # Only the latest sample at or BEFORE each decision time is exposed as CI.
    indices=np.searchsorted(stamps,pd.DatetimeIndex(interval_times(start,end)).asi8,side='right')-1
    if (indices<0).any(): raise ValueError('Carbon has no past observation')
    ci=np.asarray(frame[CI_COLUMN],dtype=float)[order][indices]
    samples=tuple(replace(s,ci_g_per_kwh=float(c)) for s,c in zip(original.samples,ci))
    data={**vars(original),'samples':samples}
    return TemporalEnvironment(**data)


def select_window(cfg,rule):
    carbon_path=resolve_path(cfg['environment']['carbon_path'])
    weather_path=resolve_path(cfg['environment']['weather_path'])
    frame=pd.read_csv(carbon_path)
    threshold=calibrate_threshold(frame,rule['calibration_start'],rule['calibration_end'])
    first,last=map(pd.Timestamp,rule['candidate_start_range'])
    if pd.Timestamp(rule['calibration_end'])>first:
        raise ValueError('Calibration overlaps test candidate range')
    candidates=[]
    for start in pd.date_range(first,last,freq=f"{rule['candidate_stride_hours']}h"):
        end=start+pd.Timedelta(hours=rule['arrival_window_hours'])
        env=load_temporal_environment(carbon_path,weather_path,start,end)
        values=np.array([s.ci_g_per_kwh for s in env.samples]);high=values>threshold
        item=dict(start=start.isoformat(),end=end.isoformat(),ci_min=float(values.min()),ci_max=float(values.max()),
                  ci_range=float(np.ptp(values)),high_hours=float(high.sum()/4),low_hours=float((~high).sum()/4),
                  high_to_low_transitions=int(np.count_nonzero(high[:-1]&~high[1:])))
        item['qualifies']=(item['ci_range']>=rule['minimum_ci_range_g_per_kwh']
            and min(item['high_hours'],item['low_hours'])>=rule['minimum_hours_each_side_of_threshold']
            and item['high_to_low_transitions']>=rule['minimum_high_to_low_transitions'])
        candidates.append(item)
    chosen=next((c for c in candidates if c['qualifies']),None)
    if chosen is None: raise ValueError('No candidate meets preregistered CI variation criteria')
    return dict(rule=rule,threshold_g_per_kwh=threshold,selected=chosen,candidates=candidates,
                carbon_source_sha256=file_hash(carbon_path),weather_source_sha256=file_hash(weather_path),
                ci_basis=CI_BASIS,recorded_before_policy_runs=True)
