"""Small causal reservation rule over known EDF backlog and running releases.

This is a sufficient local blocking check, not a scheduler or a deadline proof.
It never reads future arrivals.  Starting an urgent feasible candidate retains
the V1 no-delay rule even when another reservation would conflict with it.
"""
import math
from dataclasses import dataclass

from experiments.e1.contracts import STEP
from experiments.e1.policies import latest_start


@dataclass(frozen=True)
class GuardResult:
    mask: tuple
    protected_task_ids: tuple = ()
    conflict: bool = False


def legal_task_mask(candidate, now, available, capacity, pending, running, *, enabled=True):
    """Mask a flexible start that would obstruct an earlier known deadline.

    For each earlier-deadline task that cannot fit now, inspect resource use at
    its last legal grid start. Known running completions can only free capacity.
    Reserve only if the task can fit at that boundary without this candidate,
    but cannot fit there with its nonpreemptive occupancy. This checks only
    known jobs and their release boundaries, without scheduling the queue.
    """
    wait = now + STEP <= latest_start(candidate)
    if not enabled:
        return GuardResult((True, wait))
    protected = []
    candidate_release = (now + STEP * (candidate.duration / 15)).ceil('15min')
    names = ('cores_req', 'gpu_req', 'mem_req')
    candidate_demand = tuple(getattr(candidate, name) for name in names)
    for other in pending:
        if other.arrival_time > now:
            raise ValueError('Deadline guard must not inspect a future task')
        if other.task_id == candidate.task_id or other.sla_deadline >= candidate.sla_deadline:
            continue
        demand = tuple(getattr(other, name) for name in names)
        if all(v <= free for v, free in zip(demand, available)):
            continue  # Not a currently blocked reservation.
        bound = latest_start(other).floor('15min')
        if bound <= now or candidate_release <= bound:
            continue  # Already impossible, or the candidate releases in time.
        used_at_bound = tuple(math.fsum(r.demand[i] for r in running
                                        if r.release_time > bound) for i in range(3))
        without = all(math.fsum((u, d)) <= c for u, d, c in zip(used_at_bound, demand, capacity))
        with_candidate = all(math.fsum((u, d, x)) <= c for u, d, x, c in
                             zip(used_at_bound, demand, candidate_demand, capacity))
        if without and not with_candidate:
            protected.append(other.task_id)
    if not protected:
        return GuardResult((True, wait))
    if not wait:
        # Capacity eligibility is enforced by LocalExecutor before this method;
        # preserve its urgent candidate's legal start rather than an empty mask.
        return GuardResult((True, False), tuple(protected), True)
    return GuardResult((False, True), tuple(protected))
