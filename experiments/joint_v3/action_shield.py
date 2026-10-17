"""Capacity, own-deadline and causal reservation masks for the two task actions."""
import math
from dataclasses import dataclass

from experiments.e1.contracts import STEP, utc_time
from experiments.e1.policies import latest_start
from experiments.joint_v2.deadline_guard import legal_task_mask

from .reservation import _resources


@dataclass(frozen=True)
class ShieldResult:
    mask: tuple
    reasons: tuple = ()
    protected_task_ids: tuple = ()
    conflict: bool = False
    wait_masked: bool = False
    start_masked: bool = False
    resource_protection_triggered: bool = False
    risk_unreliable: bool = False
    capacity_wait: bool = False


def shield(candidate, now, available, capacity, pending, running, risk, enabled=True):
    """Always retain a physical action; forced capacity waits are not policy delays.

    Disabling the V3 queue-risk restriction retains the V2 causal reservation,
    own latest-start and capacity rules for the queue-risk ablation. A V2 forced
    reservation wait is not discretionary and remains valid without a reliable
    queue prediction. Conflicting local and queue advice is explicitly counted.
    """
    now = utc_time(now, True)
    available = _resources(available, 'available')
    capacity = _resources(capacity, 'capacity', True)
    if any(value > limit for value, limit in zip(available, capacity)):
        raise ValueError('Available resources exceed capacity')
    demand = _resources((candidate.cores_req, candidate.gpu_req, candidate.mem_req), 'demand')
    pending, running = tuple(pending), tuple(running)
    if candidate.arrival_time > now or any(job.arrival_time > now for job in pending):
        raise ValueError('Action shield must not inspect a future task')
    reliable = bool(getattr(risk, 'risk_estimation_valid', False))
    try:
        confidence = float(getattr(risk, 'risk_confidence', 0.))
        budget = float(getattr(risk, 'safe_wait_budget_minutes', 0.))
        delta = float(getattr(risk, 'delta_deadline_misses', 0.))
        reliable = reliable and 0 < confidence <= 1 and all(map(math.isfinite, (confidence, budget, delta)))
    except (TypeError, ValueError, OverflowError):
        reliable, budget, delta = False, 0., 0.
    unavailable = not all(value <= free for value, free in zip(demand, available))
    if unavailable:
        return ShieldResult((False, True), ('capacity_wait',), start_masked=True,
                            risk_unreliable=not reliable, capacity_wait=True)
    own_wait = now + STEP <= latest_start(candidate)
    guard = legal_task_mask(candidate, now, available, capacity, pending, running, enabled=True)
    if not own_wait:
        return ShieldResult((True, False), ('own_latest_start',), guard.protected_task_ids,
                            guard.conflict, wait_masked=True, risk_unreliable=not reliable)
    queue_wait = reliable and budget >= 15 and delta <= 0
    if not guard.mask[0]:
        # Preserve the existing analytical protection, including on conservative
        # estimator fallback. A conflicting valid predictor is a diagnostic.
        conflict = bool(enabled and reliable and not queue_wait)
        reasons = ('causal_reservation_wait',) + (('reservation_queue_risk_conflict',) if conflict else ())
        return ShieldResult((False, True), reasons, guard.protected_task_ids, conflict,
                            start_masked=True, resource_protection_triggered=True,
                            risk_unreliable=not reliable)
    if not enabled:
        return ShieldResult((True, True), ('queue_risk_ablation',), risk_unreliable=not reliable)
    if not queue_wait:
        reason = 'risk_unreliable_no_discretionary_wait' if not reliable else 'known_queue_risk_wait_blocked'
        return ShieldResult((True, False), (reason,), wait_masked=True, risk_unreliable=not reliable)
    return ShieldResult((True, True), ('conditional_known_queue_wait',))
