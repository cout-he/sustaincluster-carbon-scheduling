"""Synthetic same-time task replication; never a measured higher-load trace."""
from dataclasses import replace
from decimal import Decimal, InvalidOperation, ROUND_FLOOR

from .contracts import content_hash

VERSION = 'e1-2b-nested-arrival-replication-v1'


def stress_workload(base, factor, seed):
    if isinstance(factor, bool) or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('Factor must be numeric; seed must be a nonnegative integer')
    try:
        value = Decimal(str(factor))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError('Invalid load factor') from exc
    if not value.is_finite() or not 1 <= value <= 2:
        raise ValueError('Load factor must be finite within [1,2]')
    ranked = sorted(base.tasks, key=lambda t: (content_hash(dict(seed=seed, task_id=t.task_id)), t.task_id))
    extra = int((len(base.tasks)*(value-1)).to_integral_value(rounding=ROUND_FLOOR))
    copies = tuple(replace(t, task_id=f'{t.task_id}~e1-2b-{seed}-copy1') for t in ranked[:extra])
    ids = [t.task_id for t in base.tasks+copies]
    if len(set(ids)) != len(ids):
        raise ValueError('Synthetic task ID collision')
    workload = replace(base, tasks=base.tasks+copies) if extra else base
    manifest = dict(version=VERSION, synthetic=True, seed=seed, requested_factor=float(value),
        realized_factor=len(workload.tasks)/len(base.tasks) if base.tasks else 1,
        base_tasks=len(base.tasks), extra_tasks=extra, total_tasks=len(workload.tasks),
        base_snapshot_sha256=base.sha256, stressed_snapshot_sha256=workload.sha256,
        rounding='floor(N*(factor-1))', selection='SHA256 canonical(seed,task_id), stable_id tie-break',
        arrival_order='all original tasks in original order, then ranked clone prefix',
        copies=[dict(task_id=c.task_id,parent_task_id=p.task_id) for c,p in zip(copies,ranked)])
    return workload, manifest
