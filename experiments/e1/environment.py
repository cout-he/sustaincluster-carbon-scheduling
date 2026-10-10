"""Frozen UTC-indexed exogenous inputs, without upstream random offsets."""
import json
import math
from dataclasses import dataclass
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import content_hash, file_hash, finite, interval_times, read_snapshot, save_snapshot, utc_time, window

CI_COLUMN = 'Carbon Intensity gCO₂eq/kWh (direct)'


@dataclass(frozen=True)
class EnvironmentalSample:
    time: pd.Timestamp
    ci_g_per_kwh: float
    temperature_c: float
    wet_bulb_c: float

    def __post_init__(self):
        utc_time(self.time, True)
        finite(self.ci_g_per_kwh, 'carbon intensity')
        if not all(math.isfinite(float(v)) for v in (self.temperature_c, self.wet_bulb_c)):
            raise ValueError('Weather must be finite')

    def to_dict(self):
        return dict(time=self.time.isoformat(), ci_g_per_kwh=self.ci_g_per_kwh,
                    temperature_c=self.temperature_c, wet_bulb_c=self.wet_bulb_c)


@dataclass(frozen=True)
class FrozenEnvironment:
    start: pd.Timestamp
    end: pd.Timestamp
    carbon_source_sha256: str
    weather_source_sha256: str
    samples: tuple
    weather_source_utc_offset_seconds: int = 0
    weather_source_timezone: str = 'UTC'
    weather_time_basis: str = 'metadata_fixed_offset'

    def __post_init__(self):
        offset = self.weather_source_utc_offset_seconds
        if isinstance(offset, bool) or not isinstance(offset, int) or abs(offset) >= 86400:
            raise ValueError('Weather UTC offset must be integer seconds within one day')
        if self.weather_time_basis != 'metadata_fixed_offset':
            raise ValueError('Unsupported weather time basis')
        if tuple(s.time for s in self.samples) != interval_times(self.start, self.end):
            raise ValueError('Environmental coverage must match every interval exactly')

    def payload(self):
        return dict(start=self.start.isoformat(), end=self.end.isoformat(),
                    carbon_source_sha256=self.carbon_source_sha256,
                    weather_source_sha256=self.weather_source_sha256,
                    interpolation='linear_utc', wet_bulb_transform='upstream_temperature_times_0.9',
                    weather_source_utc_offset_seconds=self.weather_source_utc_offset_seconds,
                    weather_source_timezone=self.weather_source_timezone, weather_time_basis=self.weather_time_basis,
                    samples=[s.to_dict() for s in self.samples])

    @property
    def sha256(self):
        return content_hash(self.payload())

    def save(self, path):
        save_snapshot(path, 'e1-environment-v1', self.payload())

    @classmethod
    def load(cls, path):
        data = read_snapshot(path, 'e1-environment-v1')
        if data.pop('interpolation') != 'linear_utc' or data.pop('wet_bulb_transform') != 'upstream_temperature_times_0.9':
            raise ValueError('Unsupported frozen environmental transform')
        data['start'], data['end'] = window(data['start'], data['end'])
        data['samples'] = tuple(EnvironmentalSample(utc_time(s['time'], True), s['ci_g_per_kwh'],
                                                   s['temperature_c'], s['wet_bulb_c']) for s in data['samples'])
        return cls(**data)


def _interpolate(times, values, grid, name):
    times = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    values = np.asarray(values, dtype=float)
    if len(times) != len(values) or len(times) < 2 or times.hasnans or times.has_duplicates:
        raise ValueError(f'Invalid {name} source timestamps/length')
    if not np.isfinite(values).all():
        raise ValueError(f'{name} source must be finite')
    order = np.argsort(times.asi8)
    stamps, values = times.asi8[order], values[order]
    targets = pd.DatetimeIndex(grid).asi8
    if targets[0] < stamps[0] or targets[-1] > stamps[-1]:
        raise ValueError(f'{name} source coverage excludes requested UTC window')
    # Subtract origin before float conversion to retain sub-hour precision.
    return np.interp((targets - stamps[0]) / 1e9, (stamps - stamps[0]) / 1e9, values)


def load_environment(carbon_path, weather_path, start, end):
    start, end = window(start, end)
    grid = interval_times(start, end)
    frame = pd.read_csv(carbon_path)
    carbon = _interpolate(frame['Datetime (UTC)'], frame[CI_COLUMN], grid, 'carbon')
    raw = json.loads(Path(weather_path).read_text(encoding='utf-8'))
    offset = raw.get('utc_offset_seconds')
    if isinstance(offset, bool) or not isinstance(offset, int) or abs(offset) >= 86400:
        raise ValueError('Weather source requires an explicit integer UTC offset')
    hourly = raw['hourly']
    times = pd.DatetimeIndex(pd.to_datetime(hourly['time']))
    if times.tz is None:
        times = times.tz_localize(timezone(timedelta(seconds=offset)))
    elif any(t.utcoffset().total_seconds() != offset for t in times):
        raise ValueError('Weather timestamp/metadata UTC offset mismatch')
    temperature = _interpolate(times.tz_convert('UTC'), hourly['temperature_2m'], grid, 'weather')
    samples = tuple(EnvironmentalSample(t, float(ci), float(temp), float(temp * 0.9))
                    for t, ci, temp in zip(grid, carbon, temperature))
    return FrozenEnvironment(start, end, file_hash(carbon_path), file_hash(weather_path), samples,
                             offset, str(raw.get('timezone', 'unspecified')))
