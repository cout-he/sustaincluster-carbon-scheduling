"""Offline necessary SLA workload bounds, never an online scheduling policy.

SLA uses actual finish time, while capacity is occupied until the next 15-minute
release. The two clocks are deliberately kept separate. Interval/resource
cardinality bounds are maximized, never summed.
"""
import heapq
import math

import numpy as np
import pandas as pd

from .contracts import STEP, content_hash, finite

RESOURCES = ('cpu', 'gpu', 'mem')
RESOURCE_FIELDS = ('cores_req', 'gpu_req', 'mem_req')
BOUND_SCHEMA = 'e1-5c-offline-sla-necessary-v1'
_STEP_NS = STEP.value


def task_timing(spec):
    """Derive clocks using the same Timedelta construction as Task/Executor."""
    duration = pd.Timedelta(minutes=spec.duration)
    occupied_steps = (duration.value + _STEP_NS - 1) // _STEP_NS
    latest_grid_start = (spec.sla_deadline - duration).floor('15min')
    return dict(arrival=spec.arrival_time, deadline=spec.sla_deadline,
                occupied_minutes=15 * occupied_steps,
                latest_grid_start=latest_grid_start,
                latest_occupied_release=latest_grid_start + occupied_steps * STEP)


class _DemandTree:
    """Descending demand order statistics; each job is inserted once per u.

    The tree estimates the removable cardinality in O(log n). Reported witnesses
    are subsequently certified with math.fsum and the original member tasks.
    """
    def __init__(self, distinct_weights):
        self.values = distinct_weights
        self.weights = [0.0] * (len(distinct_weights) + 1)
        self.counts = [0] * (len(distinct_weights) + 1)

    def add(self, rank, weight):
        if not weight:
            return
        index = rank + 1
        while index < len(self.weights):
            self.weights[index] += weight
            self.counts[index] += 1
            index += index & -index

    def minimum_removed(self, excess):
        if excess <= 0:
            return 0
        length = len(self.values)
        index, removed_sum, removed_count = 0, 0.0, 0
        bit = 1 << (length.bit_length() - 1)
        while bit:
            following = index + bit
            if following <= length and removed_sum + self.weights[following] < excess:
                removed_sum += self.weights[following]
                removed_count += self.counts[following]
                index = following
            bit >>= 1
        if index >= length:
            return removed_count
        needed = math.ceil((excess - removed_sum) / self.values[index])
        return removed_count + max(0, needed)


def _tolerance(required, capacity):
    # A tolerance only weakens a necessary bound. It is never a pool capacity
    # allowance, and it does not modify the executor's exact allocation rule.
    return 1e-10 * max(1.0, abs(required), abs(capacity))


def _witness(candidate, specs, arrivals, deadlines, work, capacity, include_task_ids):
    resource = candidate['resource_index']
    selected = np.flatnonzero((arrivals >= candidate['u']) & (deadlines <= candidate['v']))
    weights = sorted(((float(work[index, resource]), specs[index].task_id)
                      for index in selected), key=lambda item: (-item[0], item[1]))
    required = math.fsum(weight for weight, _ in weights)
    capacity_work = capacity[resource] * (candidate['v'] - candidate['u']) / 60e9
    tolerance = _tolerance(required, capacity_work)
    low, high = 0, len(weights)
    while low < high:
        middle = (low + high) // 2
        removed_work = math.fsum(weight for weight, _ in weights[:middle])
        if required - removed_work <= capacity_work + tolerance:
            high = middle
        else:
            low = middle + 1
    removed = [task_id for _, task_id in weights[:low]]
    removed_work = math.fsum(weight for weight, _ in weights[:low])
    ids = sorted(specs[index].task_id for index in selected)
    result = dict(resource=RESOURCES[resource],
        start=pd.Timestamp(candidate['u'], tz='UTC').isoformat(),
        end=pd.Timestamp(candidate['v'], tz='UTC').isoformat(),
        interval_minutes=(candidate['v'] - candidate['u']) / 60e9,
        task_count=len(selected), task_id_sha256=content_hash(ids),
        required_resource_minutes=required, capacity_resource_minutes=capacity_work,
        overload_resource_minutes=max(0.0, required - capacity_work),
        workload_capacity_ratio=required / capacity_work,
        numerical_tolerance_resource_minutes=tolerance,
        minimum_misses_lower_bound=len(removed),
        greedy_removal_resource_minutes=removed_work)
    if include_task_ids:
        result.update(task_ids=ids, greedy_removal_task_ids=removed)
    return result


def analyze_sla_feasibility(workload, capacity, *, interval_mode='raw_deadline',
                            top_witnesses=5, include_task_ids=False):
    """Exhaust all arrival/deadline endpoint pairs for a finite frozen cohort.

    raw_deadline: J(u,v) uses actual arrival >= u and actual deadline <= v;
    v is restricted to 15-minute grid points. This is not an exhaustive analysis
    of arbitrary real-valued intervals and no-overload does not prove feasibility.

    occupied_deadline is a separately labelled strengthening: every on-time
    grid start has a resource release <= floor(deadline-duration)+ceil(duration).
    The original deadline and all TaskSpecs remain unchanged.
    """
    capacity = tuple(finite(value, 'capacity', True) for value in capacity)
    if len(capacity) != 3:
        raise ValueError('Capacity requires CPU/GPU/MEM')
    if interval_mode not in ('raw_deadline', 'occupied_deadline'):
        raise ValueError('Unknown offline interval mode')
    if isinstance(top_witnesses, bool) or not isinstance(top_witnesses, int) or top_witnesses < 0:
        raise ValueError('top_witnesses must be a nonnegative integer')
    specs = tuple(sorted(workload.tasks, key=lambda spec: spec.task_id))
    timings = [task_timing(spec) for spec in specs]
    intrinsic = sorted(spec.task_id for spec, timing in zip(specs, timings)
                       if timing['latest_grid_start'] < spec.arrival_time or any(
                           getattr(spec, field) > limit
                           for field, limit in zip(RESOURCE_FIELDS, capacity)))
    result = dict(schema=BOUND_SCHEMA, analysis_role='offline_only_no_policy_input',
        workload_sha256=workload.sha256, task_count=len(specs), capacity=list(capacity),
        interval_mode=interval_mode, step_minutes=15,
        occupied_duration_rule='ceil(actual duration / 15min) * 15min',
        sla_rule='actual finish_time <= original raw deadline',
        endpoint_scope='all distinct arrivals and grid-ceiled analysis deadlines',
        combination_rule='max across single intervals/resources and intrinsic unique count; no sum',
        intrinsically_impossible_count=len(intrinsic),
        intrinsically_impossible_task_id_sha256=content_hash(intrinsic),
        checked_interval_count=0, checked_resource_interval_count=0,
        arrival_endpoint_count=0, deadline_endpoint_count=0,
        minimum_deadline_misses_lower_bound=len(intrinsic),
        violation_rate_lower_bound=len(intrinsic) / len(specs) if specs else 0,
        resource_results={}, witnesses=[],
        limitations=['Necessary conditions are not sufficient.',
                     'No detected overload does not establish a feasible schedule.',
                     'EDF misses are not an optimal lower bound.',
                     'This analysis must not be supplied to the online policy.'])
    if include_task_ids:
        result['intrinsically_impossible_task_ids'] = intrinsic
    if not specs:
        result['resource_results'] = {name: dict(minimum_deadline_misses_lower_bound=0,
            overloaded_interval_count=0, maximum_workload_capacity_ratio=0, witness=None)
            for name in RESOURCES}
        return result

    arrivals = np.asarray([spec.arrival_time.value for spec in specs], dtype=np.int64)
    deadlines = np.asarray([(timing['deadline'].ceil('15min') if interval_mode == 'raw_deadline'
                             else timing['latest_occupied_release']).value
                            for timing in timings], dtype=np.int64)
    work = np.asarray([[getattr(spec, field) * timing['occupied_minutes']
                        for field in RESOURCE_FIELDS] for spec, timing in zip(specs, timings)], dtype=float)
    starts, ends = np.unique(arrivals), np.unique(deadlines)
    result['arrival_endpoint_count'], result['deadline_endpoint_count'] = len(starts), len(ends)
    deadline_order = np.argsort(deadlines, kind='stable')
    distinct = [sorted(set(work[:, resource]) - {0.0}, reverse=True) for resource in range(3)]
    ranks = [{weight: index for index, weight in enumerate(weights)} for weights in distinct]
    task_ranks = [[ranks[resource].get(float(work[index, resource]), 0)
                   for resource in range(3)] for index in range(len(specs))]
    best = [None] * 3
    best_scores = [None] * 3
    peaks, overload_counts, top = [0.0] * 3, [0] * 3, []
    for u in starts:
        trees = [_DemandTree(weights) for weights in distinct]
        eligible = deadline_order[arrivals[deadline_order] >= u]
        cursor, total = 0, [0.0, 0.0, 0.0]
        for v in ends:
            while cursor < len(eligible) and deadlines[eligible[cursor]] <= v:
                index = int(eligible[cursor])
                for resource in range(3):
                    weight = float(work[index, resource])
                    total[resource] += weight
                    trees[resource].add(task_ranks[index][resource], weight)
                cursor += 1
            if v <= u:
                continue
            result['checked_interval_count'] += 1
            length = (int(v) - int(u)) / 60e9
            for resource in range(3):
                cap = capacity[resource] * length
                ratio = total[resource] / cap
                peaks[resource] = max(peaks[resource], ratio)
                tolerance = _tolerance(total[resource], cap)
                if total[resource] <= cap + tolerance:
                    continue
                overload_counts[resource] += 1
                lower = trees[resource].minimum_removed(total[resource] - cap - tolerance)
                candidate = dict(u=int(u), v=int(v), resource_index=resource)
                score = (lower, ratio, -int(u), -int(v), -resource)
                if best_scores[resource] is None or score > best_scores[resource]:
                    best[resource], best_scores[resource] = candidate, score
                if top_witnesses:
                    if len(top) < top_witnesses:
                        heapq.heappush(top, (score, candidate))
                    elif score > top[0][0]:
                        heapq.heapreplace(top, (score, candidate))
    result['checked_resource_interval_count'] = 3 * result['checked_interval_count']
    for resource, name in enumerate(RESOURCES):
        certified = (_witness(best[resource], specs, arrivals, deadlines, work, capacity,
                               include_task_ids) if best[resource] is not None else None)
        lower = certified['minimum_misses_lower_bound'] if certified else 0
        result['resource_results'][name] = dict(minimum_deadline_misses_lower_bound=lower,
            overloaded_interval_count=overload_counts[resource],
            maximum_workload_capacity_ratio=peaks[resource], witness=certified)
        result['minimum_deadline_misses_lower_bound'] = max(
            result['minimum_deadline_misses_lower_bound'], lower)
    result['witnesses'] = [_witness(candidate, specs, arrivals, deadlines, work, capacity,
                                   include_task_ids) for _, candidate in sorted(top, reverse=True)]
    result['violation_rate_lower_bound'] = result['minimum_deadline_misses_lower_bound'] / len(specs)
    return result
