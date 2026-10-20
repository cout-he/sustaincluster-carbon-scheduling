"""Pure EDF calendar for *already observed* jobs; no scenario/environment I/O.

Minutes and CPU/GPU-equivalent/GB units are inherited from V2. Deadline uses
raw finish; capacity releases use V2's grid rounding. No horizon truncation.
"""
from dataclasses import dataclass
import math
from time import perf_counter_ns

from experiments.platform_v1.data import Task

STEP = 15
DEADLINE_EPS = 1e-9


def grid_release(finish):
    return STEP * math.ceil(finish / STEP - 1e-12)


@dataclass(frozen=True)
class Running:
    task: Task
    finish: float


@dataclass(frozen=True)
class Snapshot:
    minute: float
    capacity: tuple
    sla: float
    pending: tuple
    running: tuple = ()


@dataclass(frozen=True)
class Timing:
    task_id: str
    deadline: float
    start: float
    finish: float
    release: float
    already_running: bool


@dataclass(frozen=True)
class SlotPreview:
    started: tuple
    policy_wait: tuple
    capacity_wait: tuple
    blocked_resources: tuple
    used: tuple


@dataclass(frozen=True)
class Plan:
    snapshot: Snapshot
    forced_defer: frozenset
    records: tuple
    first_slot: SlotPreview
    dispatch_slots: int
    task_checks: int
    elapsed_ns: int

    @property
    def by_id(self):
        return {r.task_id: r for r in self.records}

    @property
    def on_time_ids(self):
        return frozenset(r.task_id for r in self.records if r.finish <= r.deadline + DEADLINE_EPS)

    @property
    def late_ids(self):
        return frozenset(r.task_id for r in self.records) - self.on_time_ids

    @property
    def max_release(self):
        # Empty known task set vacuously satisfies CG, including after 66h.
        return max((r.release for r in self.records), default=0)

    def total_lateness(self, identities):
        records = self.by_id
        return math.fsum(max(records[i].finish - records[i].deadline, 0) for i in sorted(identities))


def edf_key(task, sla):
    return (task.arrival_minute + sla * task.duration_minutes, task.arrival_minute, task.task_id)


def _running_start(r):
    # V2 observations omit start. All real starts lie on the grid; subtraction
    # may yield 14.999999999999993 for an actual start of 15 (D=60.01).
    if not math.isfinite(r.finish): raise ValueError('Nonfinite running finish')
    raw = r.finish - r.task.duration_minutes
    start = STEP * round(raw / STEP)
    if abs(start - raw) > DEADLINE_EPS:
        raise ValueError('Running start is off the V2 scheduling grid')
    return start


def _validate(s):
    if (not math.isfinite(s.minute) or s.minute < 0 or s.minute % STEP
            or not math.isfinite(s.sla) or s.sla < 1 or len(s.capacity) != 3
            or any(not math.isfinite(c) or c <= 0 for c in s.capacity)):
        raise ValueError('Invalid known-state clock, capacity or synthetic SLA')
    tasks = s.pending + tuple(r.task for r in s.running)
    if len({t.task_id for t in tasks}) != len(tasks):
        raise ValueError('Duplicate known task identity')
    for t in tasks:
        if (not isinstance(t, Task) or not math.isfinite(t.arrival_minute)
                or t.arrival_minute < 0 or t.arrival_minute > s.minute or t.arrival_minute % STEP
                or not math.isfinite(t.duration_minutes) or t.duration_minutes <= 0
                or len(t.demand) != 3 or any(not math.isfinite(x) or x < 0 or x > c
                                           for x, c in zip(t.demand, s.capacity))):
            raise ValueError('Invalid/unknown-future/intrinsically infeasible known task; never drop it')
    for r in s.running:
        start = _running_start(r)
        if (start > s.minute or start < r.task.arrival_minute or grid_release(r.finish) <= s.minute):
            raise ValueError('Running record is inconsistent with a prepared V2 observation')
    if any(math.fsum(r.task.demand[a] for r in s.running) > s.capacity[a] + 1e-9 for a in range(3)):
        raise ValueError('Running capacity exceeds logical inventory')


def _dispatch(minute, pending, active, capacity, defer):
    # Match frozen executor's summation and comparison order, including NO
    # capacity tolerance in the per-task fit check and capacity-before-defer.
    used = [math.fsum(t.demand[a] for t, _ in active) for a in range(3)]
    started, policy, blocked, resources, remaining = [], [], [], [], []
    for t in pending:
        axes = tuple(a for a in range(3) if math.fsum([used[a], t.demand[a]]) > capacity[a])
        if axes:
            blocked.append(t.task_id); resources.append((t.task_id, axes)); remaining.append(t)
        elif t.task_id in defer:
            policy.append(t.task_id); remaining.append(t)
        else:
            started.append(t)
            used = [math.fsum([used[a], t.demand[a]]) for a in range(3)]
    return (tuple(remaining), tuple(started), SlotPreview(
        tuple(t.task_id for t in started), tuple(policy), tuple(blocked), tuple(resources), tuple(used)))


def preview_now(snapshot, force_defer=frozenset()):
    _validate(snapshot)
    if not set(force_defer) <= {t.task_id for t in snapshot.pending}:
        raise ValueError('Defer must contain only known pending identities')
    pending = tuple(sorted(snapshot.pending, key=lambda t: edf_key(t, snapshot.sla)))
    active = [(r.task, grid_release(r.finish)) for r in snapshot.running]
    return _dispatch(snapshot.minute, pending, active, snapshot.capacity, force_defer)[2]


def simulate(snapshot, force_defer=frozenset()):
    begin = perf_counter_ns()
    _validate(snapshot)
    force_defer = frozenset(force_defer)
    if not force_defer <= {t.task_id for t in snapshot.pending}:
        raise ValueError('Defer must contain only known pending identities')
    pending = tuple(sorted(snapshot.pending, key=lambda t: edf_key(t, snapshot.sla)))
    active = [(r.task, grid_release(r.finish)) for r in snapshot.running]
    records = [Timing(r.task.task_id, edf_key(r.task, snapshot.sla)[0],
                      _running_start(r), r.finish, grid_release(r.finish), True)
               for r in snapshot.running]
    minute = snapshot.minute
    first = None; ticks = 0; checks = 0
    while True:
        active = [(t, release) for t, release in active if release > minute]
        checks += len(pending); ticks += 1
        remaining, starts, preview = _dispatch(minute, pending, active, snapshot.capacity,
                                              force_defer if first is None else frozenset())
        if first is None: first = preview
        for t in starts:
            finish = minute + t.duration_minutes
            release = grid_release(finish)
            records.append(Timing(t.task_id, edf_key(t, snapshot.sla)[0], minute, finish, release, False))
            active.append((t, release))
        pending = remaining
        if not pending: break  # all finishes/releases are already fixed; no needless drain ticks
        minute += STEP
    return Plan(snapshot, force_defer, tuple(records), first, ticks, checks, perf_counter_ns()-begin)
