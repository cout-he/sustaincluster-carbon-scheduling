"""Read-only observed state and online, past-arrival resource work pressure."""
import math
from collections import deque
from dataclasses import dataclass

from .contracts import STEP, finite, utc_time


def demand(spec):
    return (spec.cores_req, spec.gpu_req, spec.mem_req)


@dataclass(frozen=True)
class RunningTask:
    task_id: str
    demand: tuple
    finish_time: object
    deadline: object = None

    def __post_init__(self):
        object.__setattr__(self, 'demand', tuple(finite(v,'running demand') for v in self.demand))
        if len(self.demand)!=3: raise ValueError('CPU/GPU/MEM required')
        object.__setattr__(self, 'finish_time', utc_time(self.finish_time))
        if self.deadline is not None:
            object.__setattr__(self, 'deadline', utc_time(self.deadline))

    @property
    def release_time(self):
        return self.finish_time.ceil('15min')


@dataclass(frozen=True)
class ResourceSnapshot:
    now: object
    capacity: tuple
    available: tuple
    pending: tuple
    running: tuple
    arrivals: tuple = ()
    blocked_until: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, 'now', utc_time(self.now,True))
        for name in ('capacity','available','pending','running','arrivals','blocked_until'):
            object.__setattr__(self,name,tuple(getattr(self,name)))
        if len(self.capacity)!=3 or len(self.available)!=3: raise ValueError('CPU/GPU/MEM required')
        for v in self.capacity: finite(v,'capacity',True)
        ids=[s.task_id for s in self.pending]+[s.task_id for s in self.running]
        if len(ids)!=len(set(ids)): raise ValueError('Duplicate known task')
        if any(s.arrival_time>self.now for s in self.pending+self.arrivals):
            raise ValueError('Snapshot contains future arrival')


@dataclass(frozen=True)
class Pressure:
    effective_minutes: float
    resource_work_minutes: tuple
    rho_resources: tuple
    rho: object
    buffer_steps: int
    observed_tasks: int


class ArrivalHistory:
    def __init__(self, experiment_start, capacity, history_minutes=120,
                 breakpoints=(.5,1.), buffers=(0,1,2), cold_start_buffer_steps=2):
        self.start=utc_time(experiment_start,True)
        self.capacity=tuple(finite(v,'capacity',True) for v in capacity)
        if len(self.capacity)!=3: raise ValueError('CPU/GPU/MEM required')
        self.window=finite(history_minutes,'history window',True)
        self.breakpoints=tuple(breakpoints);self.buffers=tuple(buffers)
        if self.breakpoints!=(.5,1.) or self.buffers!=(0,1,2) or cold_start_buffer_steps!=2:
            raise ValueError('CARS-v1 fixed risk mapping')
        self.cold=cold_start_buffer_steps
        self.records=deque();self.seen=set();self.last=self.start

    def _advance(self, now):
        now=utc_time(now,True)
        if now<self.last: raise ValueError('History time cannot move backward')
        self.last=now
        cutoff=now.value-int(self.window*60*1e9)
        while self.records and self.records[0][0]<=cutoff: self.records.popleft()
        return now

    def observe(self,now,arrivals):
        arrivals=tuple(arrivals)
        if any(s.arrival_time>now for s in arrivals): raise ValueError('Cannot observe future arrival')
        if any(s.arrival_time<self.start for s in arrivals): raise ValueError('No prehistory allowed')
        now=self._advance(now)
        for s in sorted(arrivals,key=lambda s:(s.arrival_time,s.task_id)):
            if s.task_id in self.seen: continue
            self.seen.add(s.task_id)
            occupied=math.ceil(s.duration/15)*15
            self.records.append((s.arrival_time.value,tuple(v*occupied for v in demand(s))))
        # Normally arrivals are exactly current; enforce sorting for delayed ingestion.
        self.records=deque(sorted(self.records,key=lambda r:r[0]))
        self._advance(now)

    def pressure(self,now):
        now=self._advance(now)
        elapsed=min(self.window,(now-self.start).total_seconds()/60)
        work=tuple(math.fsum(r[1][k] for r in self.records) for k in range(3))
        if elapsed==0:
            return Pressure(0,work,(None,None,None),None,self.cold,len(self.records))
        resources=tuple(v/(c*elapsed) for v,c in zip(work,self.capacity))
        rho=max(resources)
        buffer=self.buffers[0 if rho<.5 else 1 if rho<1 else 2]
        return Pressure(elapsed,work,resources,rho,buffer,len(self.records))
