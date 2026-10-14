"""Immutable finite offline cases; never registered as an online policy input."""
from dataclasses import dataclass, field

import pandas as pd

from .contracts import STEP, content_hash, finite, utc_time, window
from .workload import TaskSpec

CASE_SCHEMA = 'e1-7a-offline-case-v1'
RESOURCE_FIELDS = ('cores_req', 'gpu_req', 'mem_req')
RESOURCE_NAMES = ('cpu', 'gpu', 'mem')


@dataclass(frozen=True)
class FixedOccupancy:
    task_id: str
    start: pd.Timestamp
    release: pd.Timestamp
    demand: tuple

    def __post_init__(self):
        object.__setattr__(self, 'start', utc_time(self.start, True))
        object.__setattr__(self, 'release', utc_time(self.release, True))
        object.__setattr__(self, 'demand', tuple(finite(v, 'background demand') for v in self.demand))
        if not self.task_id or self.release <= self.start:
            raise ValueError('Background requires ID and release after start')
        if len(self.demand) != 3:
            raise ValueError('Background requires CPU/GPU/MEM')

    def to_dict(self):
        return dict(task_id=self.task_id, start=self.start.isoformat(),
                    release=self.release.isoformat(), demand=list(self.demand))

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True)
class OfflineCase:
    case_id: str
    mode: str
    tasks: tuple
    start: pd.Timestamp
    end: pd.Timestamp
    capacity: tuple
    candidate_not_before: object = None
    background: tuple = ()
    source_workload_sha256: str = ''
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        start, end = window(self.start, self.end)
        object.__setattr__(self, 'start', start)
        object.__setattr__(self, 'end', end)
        anchor = start if self.candidate_not_before is None else utc_time(self.candidate_not_before, True)
        object.__setattr__(self, 'candidate_not_before', anchor)
        object.__setattr__(self, 'tasks', tuple(self.tasks))
        object.__setattr__(self, 'background', tuple(self.background))
        object.__setattr__(self, 'capacity', tuple(finite(v, 'capacity', True) for v in self.capacity))
        if not self.case_id or self.mode not in ('A1', 'A2') or len(self.capacity) != 3:
            raise ValueError('Case requires ID, A1/A2 and CPU/GPU/MEM capacity')
        if not start <= anchor < end:
            raise ValueError('Candidate anchor outside case horizon')
        ids = [s.task_id for s in self.tasks]
        bg_ids = [b.task_id for b in self.background]
        if len(set(ids)) != len(ids) or len(set(bg_ids)) != len(bg_ids):
            raise ValueError('Duplicate candidate or background ID')
        if set(ids) & set(bg_ids):
            raise ValueError('Candidate/background ID overlap')
        if self.mode == 'A1' and self.background:
            raise ValueError('A1 closed case cannot include background')
        if any(s.dc_id != 1 for s in self.tasks):
            raise ValueError('Offline candidates must originate at DC1')
        if any(s.arrival_time >= end for s in self.tasks):
            raise ValueError('Candidate arrival outside registered horizon')
        if self.mode == 'A1' and any(s.arrival_time < start for s in self.tasks):
            raise ValueError('A1 candidate arrival before case start')
        if any(pd.Timedelta(minutes=s.duration).value <= 0 for s in self.tasks):
            raise ValueError('Duration must be positive at original nanosecond precision')

    def to_dict(self):
        return dict(schema=CASE_SCHEMA, case_id=self.case_id, mode=self.mode,
                    tasks=[s.to_dict() for s in self.tasks], start=self.start.isoformat(),
                    end=self.end.isoformat(), capacity=list(self.capacity),
                    candidate_not_before=self.candidate_not_before.isoformat(),
                    background=[b.to_dict() for b in self.background],
                    source_workload_sha256=self.source_workload_sha256, metadata=self.metadata)

    @property
    def sha256(self):
        return content_hash(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        if data.pop('schema', None) != CASE_SCHEMA:
            raise ValueError('Offline case schema mismatch')
        data['tasks'] = tuple(TaskSpec.from_dict(s) for s in data['tasks'])
        data['background'] = tuple(FixedOccupancy.from_dict(b) for b in data['background'])
        return cls(**data)


def background_from_reference(workload, task_rows, candidate_ids):
    """Retain every noncandidate original task's fixed reference occupancy.

    The caller freezes the candidate IDs before solving. No task features are
    inferred or regenerated, and unsuccessful/incomplete reference runs fail.
    """
    candidate_ids = set(candidate_ids)
    specs = {s.task_id: s for s in workload.tasks}
    rows = tuple(task_rows)
    if len(rows) != len(specs) or {r['task_id'] for r in rows} != set(specs):
        raise ValueError('Complete unique reference task ledger required')
    if not candidate_ids <= set(specs):
        raise ValueError('Unknown candidate ID')
    result = []
    for row in sorted(rows, key=lambda r: r['task_id']):
        if row['state'] != 'completed' or row['start_time'] is None:
            raise ValueError('Reference background must be fully completed')
        spec = specs[row['task_id']]
        start = utc_time(row['start_time'], True)
        finish = start + pd.Timedelta(minutes=spec.duration)
        if finish != utc_time(row['finish_time']):
            raise ValueError('Reference task finish/duration mismatch')
        if spec.task_id not in candidate_ids:
            result.append(FixedOccupancy(spec.task_id, start, finish.ceil('15min'),
                                        tuple(getattr(spec, f) for f in RESOURCE_FIELDS)))
    return tuple(result)


def synthetic_reservation_case():
    """Registered 12-task mechanism example, not a real workload trace.

    Three separated blocks share capacity one. EDF starts a .75/30min task
    before two .5/30min tasks and backfills a .25/60min task. A reservation can
    instead start the two .5 tasks together, sacrificing just the .75 task.
    """
    origin = pd.Timestamp('2023-05-01T05:00:00Z')
    specs = []
    for block in range(3):
        arrival = origin + block * pd.Timedelta(minutes=120)
        for index, (name, duration, cpu) in enumerate((('a', 30, .75), ('b', 30, .5),
                                                      ('c', 30, .5), ('d', 60, .25))):
            task_id = f'synthetic-block-{block}-{name}'
            specs.append(TaskSpec(task_id, task_id, block, index, arrival, duration,
                                  cpu, 0, 0, 0, 1.5, 1, tuple(range(12))))
    return OfflineCase('synthetic-reservation-12', 'A1', tuple(specs), origin,
                       origin + pd.Timedelta(minutes=360), (1, 1, 1),
                       source_workload_sha256=content_hash([s.to_dict() for s in specs]),
                       metadata={'role': 'synthetic_mechanism_validation_not_real_trace'})
