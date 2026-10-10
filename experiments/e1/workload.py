"""Immutable trace-derived tasks: observed usage, synthetic SLA, no augmentation."""
import random
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rl_components.task import Task

from .contracts import content_hash, file_hash, finite, interval_times, read_snapshot, save_snapshot, utc_time, window

COLUMNS = ('job_name', 'start_time', 'end_time', 'start_dt', 'duration_min',
           'cpu_usage', 'gpu_wrk_util', 'avg_mem', 'avg_gpu_wrk_mem',
           'bandwidth_gb', 'weekday_name', 'weekday_num')


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    source_job_name: str
    source_row: int
    source_task_index: int
    arrival_time: pd.Timestamp
    duration: float
    cores_req: float
    gpu_req: float
    mem_req: float
    bandwidth_gb: float
    sla_multiplier: float
    dc_id: int
    raw_fields: tuple

    def __post_init__(self):
        utc_time(self.arrival_time, True)
        finite(self.duration, 'duration', True)
        finite(self.sla_multiplier, 'sla_multiplier', True)
        for name in ('cores_req', 'gpu_req', 'mem_req', 'bandwidth_gb'):
            finite(getattr(self, name), name)
        if len(self.raw_fields) != 12 or not self.task_id:
            raise ValueError('Task requires stable ID and 12 raw fields')

    @property
    def sla_deadline(self):
        return self.arrival_time + pd.Timedelta(minutes=self.sla_multiplier * self.duration)

    def make_task(self):
        # The upstream constructor consumes global random for its suffix. Preserve
        # that stream and replace the suffix with a manifest-derived unique ID.
        state = random.getstate()
        try:
            task = Task(self.task_id, self.arrival_time, self.duration, self.cores_req,
                        self.gpu_req, self.mem_req, self.bandwidth_gb, self.sla_multiplier)
        finally:
            random.setstate(state)
        task.job_name = self.task_id
        task.origin_dc_id = self.dc_id
        return task

    def to_dict(self):
        return dict(task_id=self.task_id, source_job_name=self.source_job_name,
                    source_row=self.source_row, source_task_index=self.source_task_index,
                    arrival_time=self.arrival_time.isoformat(), duration=self.duration,
                    cores_req=self.cores_req, gpu_req=self.gpu_req, mem_req=self.mem_req,
                    bandwidth_gb=self.bandwidth_gb, sla_multiplier=self.sla_multiplier,
                    dc_id=self.dc_id, raw_fields=list(self.raw_fields))

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        data['arrival_time'] = utc_time(data['arrival_time'], True)
        data['raw_fields'] = tuple(data['raw_fields'])
        return cls(**data)


@dataclass(frozen=True)
class FrozenWorkload:
    start: pd.Timestamp
    end: pd.Timestamp
    source_sha256: str
    resource_multiplier: float
    sla_multiplier: float
    trace_year: int
    tasks: tuple

    def __post_init__(self):
        window(self.start, self.end)
        if len({t.task_id for t in self.tasks}) != len(self.tasks):
            raise ValueError('Duplicate stable task ID')
        if any(not self.start <= t.arrival_time < self.end for t in self.tasks):
            raise ValueError('Task outside arrival cohort')

    def payload(self):
        return dict(start=self.start.isoformat(), end=self.end.isoformat(),
                    source_sha256=self.source_sha256, columns=list(COLUMNS),
                    resource_multiplier=self.resource_multiplier, sla_multiplier=self.sla_multiplier,
                    trace_year=self.trace_year, tasks=[t.to_dict() for t in self.tasks])

    @property
    def sha256(self):
        return content_hash(self.payload())

    def save(self, path):
        save_snapshot(path, 'e1-workload-v1', self.payload())

    @classmethod
    def load(cls, path):
        data = read_snapshot(path, 'e1-workload-v1')
        if tuple(data.pop('columns')) != COLUMNS:
            raise ValueError('Frozen 12-column schema mismatch')
        data['start'], data['end'] = window(data['start'], data['end'])
        data['tasks'] = tuple(TaskSpec.from_dict(t) for t in data['tasks'])
        return cls(**data)


def _raw_value(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None  # JSON null preserves missing metadata; required demands still fail finite().
    if not isinstance(value, (str, int, float, bool)) and value is not None:
        return str(value)
    return value


def load_workload(path, start, end, resource_multiplier=5, sla_multiplier=1.5, dc_id=1, trace_year=2020):
    start, end = window(start, end)
    scale = finite(resource_multiplier, 'resource_multiplier', True)
    sla = finite(sla_multiplier, 'sla_multiplier', True)
    source_hash = file_hash(path)
    frame = pd.read_pickle(Path(path))
    if not {'interval_15m', 'tasks_matrix'} <= set(frame.columns) or frame.empty:
        raise ValueError('Workload requires interval_15m and tasks_matrix with source coverage')
    times = [utc_time(t, True) for t in frame['interval_15m']]
    if len(set(times)) != len(times):
        raise ValueError('duplicate workload buckets')
    rows = {t: (i, matrix) for i, (t, matrix) in enumerate(zip(times, frame['tasks_matrix']))}
    source_start, source_end = min(times), max(times)
    specs = []
    for arrival in interval_times(start, end):
        query = arrival.replace(year=trace_year)
        if query < source_start or query > source_end:
            raise ValueError('Workload source coverage does not include requested window')
        if query not in rows:
            continue  # A covered but absent bucket has no arrivals; never wraps data.
        row_index, matrix = rows[query]
        for task_index, raw in enumerate(matrix):
            if len(raw) != 12:
                raise ValueError('Workload schema must contain exactly 12 columns')
            raw = tuple(_raw_value(v) for v in raw)
            specs.append(TaskSpec(
                task_id=f'{source_hash[:16]}/{arrival.isoformat()}/{row_index}/{task_index}',
                source_job_name=str(raw[0]), source_row=row_index, source_task_index=task_index,
                arrival_time=arrival, duration=finite(raw[4], 'duration', True),
                cores_req=scale * finite(raw[5], 'cpu_usage') / 100,
                gpu_req=scale * finite(raw[6], 'gpu_wrk_util') / 100,
                mem_req=scale * finite(raw[7], 'avg_mem'), bandwidth_gb=finite(raw[9], 'bandwidth_gb'),
                sla_multiplier=sla, dc_id=dc_id, raw_fields=raw))
    return FrozenWorkload(start, end, source_hash, scale, sla, trace_year, tuple(specs))
