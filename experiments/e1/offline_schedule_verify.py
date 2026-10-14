"""Independent original E1 resource/clock verification of complete schedules."""
import math

import pandas as pd

from .contracts import STEP, utc_time
from .offline_schedule_cases import RESOURCE_FIELDS, RESOURCE_NAMES


def normalize_starts(starts):
    pairs = tuple(starts.items() if isinstance(starts, dict) else starts)
    if len({key for key, _ in pairs}) != len(pairs):
        raise ValueError('Duplicate schedule task ID')
    return {key: utc_time(value, True) for key, value in pairs}


def verify_schedule(case, starts):
    """Fail on any illegal start or resource hold; never relax capacity.

    Original binary-float demands are individually passed to math.fsum just as
    LocalResourcePool does. Solver integer coefficients are not used here.
    """
    starts = normalize_starts(starts)
    if set(starts) != {spec.task_id for spec in case.tasks}:
        raise ValueError('Schedule must include every candidate exactly once')
    rows, holds = [], []
    for spec in case.tasks:
        start = starts[spec.task_id]
        if start < spec.arrival_time:
            raise ValueError('Schedule starts before original arrival')
        if start < case.candidate_not_before or start < case.start:
            raise ValueError('Schedule starts before registered candidate anchor')
        finish = start + pd.Timedelta(minutes=spec.duration)
        release = finish.ceil('15min')
        if release > case.end:
            raise ValueError('Schedule does not fully release within registered horizon')
        demand = tuple(getattr(spec, name) for name in RESOURCE_FIELDS)
        holds.append((start, release, demand))
        rows.append(dict(task_id=spec.task_id, arrival_time=spec.arrival_time.isoformat(),
                         start_time=start.isoformat(), finish_time=finish.isoformat(),
                         release_time=release.isoformat(), deadline=spec.sla_deadline.isoformat(),
                         missed=finish > spec.sla_deadline,
                         wait_minutes=(start - spec.arrival_time).total_seconds() / 60,
                         jct_minutes=(finish - spec.arrival_time).total_seconds() / 60))
    background = [(b.start, b.release, b.demand) for b in case.background]
    peaks = [0.0, 0.0, 0.0]
    for now in pd.date_range(case.start, case.end, freq=STEP, inclusive='left'):
        active = [demand for start, release, demand in holds + background if start <= now < release]
        for resource, cap in enumerate(case.capacity):
            used = math.fsum(d[resource] for d in active)
            if used > cap:
                raise ValueError(f'Original {RESOURCE_NAMES[resource]} capacity exceeded at {now}')
            peaks[resource] = max(peaks[resource], used)
    actual = [math.fsum(getattr(spec, name) * pd.Timedelta(minutes=spec.duration).value / 3.6e12
                       for spec in case.tasks) for name in RESOURCE_FIELDS]
    occupied = [math.fsum(getattr(spec, name) * (release - start).value / 3.6e12
                         for spec, (start, release, _) in zip(case.tasks, holds))
                for name in RESOURCE_FIELDS]
    return dict(verified=True, task_count=len(rows), completed_tasks=len(rows), unfinished_tasks=0,
                miss_count=sum(row['missed'] for row in rows),
                mean_jct_minutes=math.fsum(r['jct_minutes'] for r in rows) / len(rows) if rows else 0,
                maximum_wait_minutes=max((r['wait_minutes'] for r in rows), default=0),
                actual_resource_hours=actual, occupied_resource_hours=occupied,
                peak_used_resources=peaks, background_task_count=len(background),
                drain_release=max((release for _, release, _ in holds), default=case.start).isoformat(),
                tasks=rows, verification_resource_rule='original individual-demand math.fsum <= capacity',
                verification_deadline_rule='actual raw finish <= original deadline')
