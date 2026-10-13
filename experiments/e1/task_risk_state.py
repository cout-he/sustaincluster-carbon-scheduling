"""Pure online task-matched budget; a heuristic, never a feasibility guarantee.

No workload/environment references, forecast, planner, or offline SLA imports.
The caller supplies only already-observed strict EDF predecessors still pending.
"""
import math
from dataclasses import dataclass

from .cars_state import demand
from .contracts import finite, utc_time
from .policies import latest_start

VERSION = 'e1-6a-task-risk-budget-v1'
RESOURCES = ('cpu', 'gpu', 'mem')


@dataclass(frozen=True)
class BudgetParameters:
    a_u: float = 1
    a_q: float = 1
    a_h: float = 1
    rho0: float = .5
    k: float = 1
    K: int = 8
    cold_buffer_steps: int = 2
    history_minutes: float = 120

    def __post_init__(self):
        for name in ('a_u', 'a_q', 'a_h', 'rho0', 'k', 'history_minutes'):
            object.__setattr__(self, name, finite(getattr(self, name), name, name in ('k', 'history_minutes')))
        for name in ('K', 'cold_buffer_steps'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(name + ' must be a nonnegative integer')


@dataclass(frozen=True)
class Budget:
    weights: tuple
    utilization: tuple
    urgent_pressure: tuple
    history_pressure: tuple
    resource_pressure: tuple
    score: float
    buffer_minutes: int
    solo_slack_minutes: float
    safe_slack_minutes: float
    cold_steps: int
    dominant_resource: str


def occupied_minutes(spec):
    """Same duration rounding as the unchanged ArrivalHistory (grid starts)."""
    return 15 * math.ceil(spec.duration / 15)


def latest_grid_start(spec):
    """Actual finish SLA bound, not deadline minus rounded resource occupancy."""
    return latest_start(spec).floor('15min')


def _budget_from_pressures(spec, now, capacity, utilization, urgent, pressure,
                           parameters, aggregation):
    matched = tuple(q / c for q, c in zip(demand(spec), capacity))
    total = math.fsum(matched)
    weights = tuple(q / total for q in matched) if total else (0., 0., 0.)
    rho = tuple(pressure.rho_resources)
    cold = parameters.cold_buffer_steps if pressure.effective_minutes == 0 else 0
    if len(rho) != 3 or ((pressure.effective_minutes == 0 and not all(v is None for v in rho))
                        or (pressure.effective_minutes > 0 and any(v is None for v in rho))):
        raise ValueError('History unknown/effective duration mismatch')
    terms = tuple(parameters.a_u * u + parameters.a_q * q
                  + (parameters.a_h * max(0., h - parameters.rho0) if h is not None else 0.)
                  for u, q, h in zip(utilization, urgent, rho))
    if aggregation not in ('weighted', 'max_required'):
        raise ValueError('Unknown task-risk aggregation')
    if not total:
        score, buffer, dominant = 0., 0, 'zero'
    else:
        score = (math.fsum(w * x for w, x in zip(weights, terms)) if aggregation == 'weighted'
                 else max(x for q, x in zip(matched, terms) if q > 0))
        buffer = 15 * math.ceil(cold + min(parameters.K, parameters.k * score))
        leaders = [r for r, q in zip(RESOURCES, matched) if q == max(matched)]
        dominant = leaders[0] if len(leaders) == 1 else 'tie'
    slack = (latest_grid_start(spec) - now).total_seconds() / 60
    return Budget(weights, tuple(utilization), tuple(urgent), rho, terms, score,
                  buffer, slack, slack - buffer, cold, dominant)


def calculate_budget(spec, now, capacity, running, predecessors, pressure,
                     parameters, aggregation='weighted'):
    """Correctness reference: scan known predecessors, no per-candidate sorting.

    Caller owns strict EDF rank and removes started tasks; this function filters
    urgency and excludes self. Overdue pending tasks intentionally remain urgent.
    Costs O(running + predecessors), with three dimensions treated as constant.
    """
    now = utc_time(now, True)
    capacity = tuple(finite(c, 'capacity', True) for c in capacity)
    if len(capacity) != 3:
        raise ValueError('CPU/GPU/MEM capacity required')
    predecessors, running = tuple(predecessors), tuple(running)
    if spec.arrival_time > now or any(s.arrival_time > now for s in predecessors):
        raise ValueError('Cannot evaluate future task')
    urgent = tuple(s for s in predecessors if s.task_id != spec.task_id
                   and (latest_grid_start(s) - now).total_seconds() / 60 <= parameters.history_minutes)
    utilization = tuple(math.fsum(r.demand[k] for r in running) / capacity[k] for k in range(3))
    queue = tuple(math.fsum(demand(s)[k] * occupied_minutes(s) for s in urgent)
                  / (capacity[k] * parameters.history_minutes) for k in range(3))
    return _budget_from_pressures(spec, now, capacity, utilization, queue, pressure,
                                 parameters, aggregation)
